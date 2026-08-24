"""Compare face-embedding checkpoints under controlled probe degradation."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from identification_service.modules.extraction.embedding import (  # noqa: E402
    FaceEmbeddingExtractor,
    InceptionResnetBackend,
)


FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]
StringVector = NDArray[np.str_]

MODELS = ("vggface2", "casia-webface")
CONDITIONS = (
    "clean",
    "gaussian_blur_radius_2",
    "resize_down_64_up_250",
    "brightness_0_60",
    "brightness_1_40",
)
TOP_N_VALUES = (1, 3, 5, 10)
CACHE_SCHEMA_VERSION = 1
CACHE_ARRAY_FIELDS = (
    "embeddings",
    "identities",
    "relative_paths",
    "mean_intensity",
    "zero_fraction",
    "saturated_fraction",
)


def clean(image: Image.Image) -> Image.Image:
    return image


def gaussian_blur_radius_2(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=2.0))


def resize_down_64_up_250(image: Image.Image) -> Image.Image:
    reduced = image.resize((64, 64), Image.Resampling.BILINEAR)
    return reduced.resize((250, 250), Image.Resampling.BILINEAR)


def brightness_0_60(image: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(0.60)


def brightness_1_40(image: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(1.40)


TRANSFORMS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "clean": clean,
    "gaussian_blur_radius_2": gaussian_blur_radius_2,
    "resize_down_64_up_250": resize_down_64_up_250,
    "brightness_0_60": brightness_0_60,
    "brightness_1_40": brightness_1_40,
}


def run_comparison(
    *,
    dataset_manifest: Path,
    dataset_audit: Path,
    asset_root: Path,
    output_dir: Path,
    cache_dir: Path,
    models: Sequence[str] = MODELS,
    batch_size: int = 32,
    device: str | None = None,
    max_identities: int | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Run the complete controlled comparison or a deterministic pilot."""

    _validate_run_configuration(models, batch_size, max_identities)
    manifest_rows, manifest_sha256 = _load_and_verify_manifest(
        dataset_manifest, dataset_audit
    )
    selected_rows = select_scope(manifest_rows, max_identities=max_identities)
    gallery_rows = [row for row in selected_rows if row["split"] == "gallery"]
    probe_rows = [row for row in selected_rows if row["split"] == "probe"]

    if not gallery_rows or not probe_rows:
        raise ValueError("selected experiment scope must include gallery and probe images")

    run_scope = "full" if max_identities is None else "pilot"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = output_dir / "comparison_run.json"
    if run_manifest_path.exists() and not force:
        raise FileExistsError(
            "comparison output already exists; pass --force to replace it"
        )

    selection_fingerprint = fingerprint_rows(selected_rows)
    print(f"Run scope: {run_scope}")
    print(f"Gallery images: {len(gallery_rows)}")
    print(f"Gallery identities: {_identity_count(gallery_rows)}")
    print(f"Probe images: {len(probe_rows)}")
    print(f"Probe identities: {_identity_count(probe_rows)}")
    print(f"Models: {tuple(models)}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Selection fingerprint: {selection_fingerprint}")

    metric_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    quality_by_condition: dict[str, dict[str, FloatVector]] = {}
    resolved_devices: dict[str, str] = {}

    for model_name in models:
        print("=" * 80)
        print(f"MODEL: {model_name}")
        print("=" * 80)

        backend = InceptionResnetBackend(pretrained=model_name, device=device)
        resolved_devices[model_name] = str(backend.device)
        extractor = FaceEmbeddingExtractor(backend=backend, image_size=160)

        gallery_cache = compute_or_load_embeddings(
            rows=gallery_rows,
            asset_root=asset_root,
            cache_dir=cache_dir,
            selection_fingerprint=selection_fingerprint,
            model_name=model_name,
            split_name="gallery",
            condition_name="clean",
            extractor=extractor,
            batch_size=batch_size,
            force=force,
        )

        model_results: dict[str, dict[str, object]] = {}
        probe_embeddings_by_condition: dict[str, FloatMatrix] = {}

        for condition_name in CONDITIONS:
            print(f"Condition: {condition_name}")
            probe_cache = compute_or_load_embeddings(
                rows=probe_rows,
                asset_root=asset_root,
                cache_dir=cache_dir,
                selection_fingerprint=selection_fingerprint,
                model_name=model_name,
                split_name="probe",
                condition_name=condition_name,
                extractor=extractor,
                batch_size=batch_size,
                force=force,
            )

            if condition_name not in quality_by_condition:
                quality_by_condition[condition_name] = {
                    "mean_intensity": probe_cache["mean_intensity"],
                    "zero_fraction": probe_cache["zero_fraction"],
                    "saturated_fraction": probe_cache["saturated_fraction"],
                }

            evaluation = evaluate_retrieval(
                gallery_embeddings=gallery_cache["embeddings"],
                gallery_identities=gallery_cache["identities"],
                probe_embeddings=probe_cache["embeddings"],
                probe_identities=probe_cache["identities"],
                probe_paths=probe_cache["relative_paths"],
            )
            probe_embeddings_by_condition[condition_name] = probe_cache[
                "embeddings"
            ]
            model_results[condition_name] = evaluation

        clean_metrics = model_results["clean"]["metrics"]
        clean_probe_rows = model_results["clean"]["probe_rows"]

        for condition_name in CONDITIONS:
            evaluation = model_results[condition_name]
            metrics = evaluation["metrics"]
            metric_rows.append(
                {
                    "model": model_name,
                    "condition": condition_name,
                    "probe_count": metrics["probe_count"],
                    "top_1": metrics["top_1"],
                    "top_3": metrics["top_3"],
                    "top_5": metrics["top_5"],
                    "top_10": metrics["top_10"],
                    "mrr": metrics["mrr"],
                    "top_1_drop_from_clean": clean_metrics["top_1"]
                    - metrics["top_1"],
                    "mrr_drop_from_clean": clean_metrics["mrr"]
                    - metrics["mrr"],
                }
            )

            for probe_row in evaluation["probe_rows"]:
                ranking_rows.append(
                    {
                        "model": model_name,
                        "condition": condition_name,
                        **probe_row,
                    }
                )

            if condition_name != "clean":
                stability_rows.append(
                    calculate_stability(
                        model_name=model_name,
                        condition_name=condition_name,
                        clean_probe_rows=clean_probe_rows,
                        degraded_probe_rows=evaluation["probe_rows"],
                        clean_embeddings=probe_embeddings_by_condition["clean"],
                        degraded_embeddings=probe_embeddings_by_condition[
                            condition_name
                        ],
                    )
                )

        del extractor, backend
        gc.collect()

    quality_rows = summarize_image_quality(quality_by_condition)
    metric_rows.sort(key=lambda row: (str(row["model"]), str(row["condition"])))
    stability_rows.sort(
        key=lambda row: (str(row["model"]), str(row["condition"]))
    )
    ranking_rows.sort(
        key=lambda row: (
            str(row["model"]),
            str(row["condition"]),
            str(row["probe_relative_path"]),
        )
    )

    metrics_path = output_dir / "condition_metrics.csv"
    stability_path = output_dir / "rank_stability.csv"
    rankings_path = output_dir / "probe_rankings.csv"
    quality_path = output_dir / "image_quality_summary.csv"

    _write_csv(metrics_path, metric_rows)
    _write_csv(stability_path, stability_rows)
    _write_csv(rankings_path, ranking_rows)
    _write_csv(quality_path, quality_rows)

    artifact_paths = (metrics_path, stability_path, rankings_path, quality_path)
    run_manifest = {
        "status": "complete",
        "run_scope": run_scope,
        "models": list(models),
        "conditions": list(CONDITIONS),
        "top_n_values": list(TOP_N_VALUES),
        "batch_size": batch_size,
        "device_request": device or "auto",
        "resolved_devices": resolved_devices,
        "gallery_images": len(gallery_rows),
        "gallery_identities": _identity_count(gallery_rows),
        "probe_images": len(probe_rows),
        "probe_identities": _identity_count(probe_rows),
        "dataset_manifest_sha256": manifest_sha256,
        "selection_fingerprint": selection_fingerprint,
        "pipeline": {
            "image_size": 160,
            "preprocessing": "full_image_resize",
            "standardization": "fixed_127_5_div_128",
            "retrieval": "exact_cosine",
            "ranking_unit": "distinct_identity",
        },
        "dependencies": _dependency_versions(),
        "artifacts": {
            path.name: _sha256_file(path) for path in artifact_paths
        },
    }
    _write_json(run_manifest_path, run_manifest)

    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    for row in metric_rows:
        print(
            f"{row['model']:15s} {row['condition']:24s} "
            f"Top-1={row['top_1']:.4f} "
            f"Top-3={row['top_3']:.4f} MRR={row['mrr']:.4f}"
        )
    for path in (*artifact_paths, run_manifest_path):
        print(f"[WRITE] {path}")

    return {
        "run_manifest": run_manifest,
        "artifact_paths": (*artifact_paths, run_manifest_path),
    }


def compute_or_load_embeddings(
    *,
    rows: Sequence[dict[str, str]],
    asset_root: Path,
    cache_dir: Path,
    selection_fingerprint: str,
    model_name: str,
    split_name: str,
    condition_name: str,
    extractor: FaceEmbeddingExtractor,
    batch_size: int,
    force: bool,
) -> dict[str, NDArray]:
    """Load a validated cache or compute embeddings in batches."""

    configuration = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "selection_fingerprint": selection_fingerprint,
        "model": model_name,
        "split": split_name,
        "condition": condition_name,
        "image_size": 160,
        "standardization": "fixed_127_5_div_128",
    }
    configuration_json = json.dumps(configuration, sort_keys=True)
    cache_name = (
        f"{selection_fingerprint[:12]}__{model_name}__"
        f"{split_name}__{condition_name}.npz"
    )
    cache_path = cache_dir / cache_name

    expected_identities = np.asarray(
        [row["identity"] for row in rows], dtype=np.str_
    )
    expected_paths = np.asarray(
        [row["relative_path"] for row in rows], dtype=np.str_
    )

    if cache_path.exists() and not force:
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                if _cache_is_valid(
                    cached=cached,
                    configuration_json=configuration_json,
                    expected_identities=expected_identities,
                    expected_paths=expected_paths,
                ):
                    print(f"[CACHE HIT] {cache_path.name}")
                    return {
                        name: cached[name].copy()
                        for name in ("configuration_json", *CACHE_ARRAY_FIELDS)
                    }
        except (OSError, ValueError, KeyError):
            pass
        print(f"[CACHE STALE] {cache_path.name}")

    transform = TRANSFORMS[condition_name]
    embedding_chunks: list[FloatMatrix] = []
    intensity_values: list[float] = []
    zero_fractions: list[float] = []
    saturated_fractions: list[float] = []

    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        model_images: list[Image.Image] = []

        for row in rows[start:stop]:
            image_path = asset_root / row["relative_path"]
            with Image.open(image_path) as source_image:
                oriented = ImageOps.exif_transpose(source_image).convert("RGB")
                transformed = transform(oriented)
                pixels = np.asarray(transformed, dtype=np.uint8)
                intensity_values.append(float(pixels.mean() / 255.0))
                zero_fractions.append(float(np.mean(pixels == 0)))
                saturated_fractions.append(float(np.mean(pixels == 255)))
                model_images.append(
                    transformed.resize((160, 160), Image.Resampling.BILINEAR)
                )

        embedding_chunks.append(extractor.encode_batch(model_images))
        print(
            f"  {model_name} {split_name} {condition_name}: "
            f"{stop}/{len(rows)}"
        )

    embeddings = np.ascontiguousarray(
        np.vstack(embedding_chunks), dtype=np.float32
    )
    if embeddings.shape != (len(rows), 512):
        raise ValueError(
            f"expected embedding shape {(len(rows), 512)}; "
            f"received {embeddings.shape}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("computed embeddings contain non-finite values")

    payload = {
        "configuration_json": np.asarray(configuration_json),
        "embeddings": embeddings,
        "identities": expected_identities,
        "relative_paths": expected_paths,
        "mean_intensity": np.asarray(intensity_values, dtype=np.float32),
        "zero_fraction": np.asarray(zero_fractions, dtype=np.float32),
        "saturated_fraction": np.asarray(
            saturated_fractions, dtype=np.float32
        ),
    }
    temporary_path = cache_path.with_suffix(".npz.tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary_path.replace(cache_path)
    print(f"[CACHE WRITE] {cache_path.name}")
    return payload


def _cache_is_valid(
    *,
    cached: np.lib.npyio.NpzFile,
    configuration_json: str,
    expected_identities: StringVector,
    expected_paths: StringVector,
) -> bool:
    expected_fields = {"configuration_json", *CACHE_ARRAY_FIELDS}
    if not expected_fields.issubset(cached.files):
        return False
    if str(cached["configuration_json"].item()) != configuration_json:
        return False
    if not np.array_equal(cached["identities"], expected_identities):
        return False
    if not np.array_equal(cached["relative_paths"], expected_paths):
        return False

    row_count = len(expected_identities)
    if cached["embeddings"].shape != (row_count, 512):
        return False
    if not np.isfinite(cached["embeddings"]).all():
        return False
    for field in ("mean_intensity", "zero_fraction", "saturated_fraction"):
        values = cached[field]
        if values.shape != (row_count,) or not np.isfinite(values).all():
            return False
    return True


def evaluate_retrieval(
    *,
    gallery_embeddings: FloatMatrix,
    gallery_identities: StringVector,
    probe_embeddings: FloatMatrix,
    probe_identities: StringVector,
    probe_paths: StringVector,
) -> dict[str, object]:
    """Evaluate exact cosine retrieval after collapsing image-level duplicates."""

    gallery = _normalize_rows(gallery_embeddings)
    probes = _normalize_rows(probe_embeddings)
    similarities = probes @ gallery.T
    image_orders = np.argsort(-similarities, axis=1, kind="stable")

    probe_rows: list[dict[str, object]] = []
    true_ranks: list[int] = []

    for probe_index, image_order in enumerate(image_orders):
        true_identity = str(probe_identities[probe_index])
        seen: set[str] = set()
        top_identities: list[str] = []
        top_similarity: float | None = None
        true_rank: int | None = None
        true_similarity: float | None = None

        for gallery_position in image_order:
            identity = str(gallery_identities[int(gallery_position)])
            if identity in seen:
                continue
            seen.add(identity)
            identity_rank = len(seen)
            similarity = float(similarities[probe_index, int(gallery_position)])

            if top_similarity is None:
                top_similarity = similarity
            if len(top_identities) < max(TOP_N_VALUES):
                top_identities.append(identity)
            if identity == true_identity:
                true_rank = identity_rank
                true_similarity = similarity
            if true_rank is not None and len(top_identities) == max(TOP_N_VALUES):
                break

        if true_rank is None or true_similarity is None or top_similarity is None:
            raise ValueError(
                f"true identity is absent from gallery ranking: {true_identity}"
            )

        true_ranks.append(true_rank)
        row: dict[str, object] = {
            "probe_relative_path": str(probe_paths[probe_index]),
            "true_identity": true_identity,
            "true_identity_rank": true_rank,
            "reciprocal_rank": 1.0 / true_rank,
            "rank_1_similarity": top_similarity,
            "true_identity_similarity": true_similarity,
        }
        for rank in range(1, max(TOP_N_VALUES) + 1):
            row[f"rank_{rank}"] = (
                top_identities[rank - 1]
                if rank <= len(top_identities)
                else ""
            )
        probe_rows.append(row)

    ranks = np.asarray(true_ranks, dtype=np.int32)
    metrics: dict[str, object] = {
        "probe_count": len(probe_rows),
        "mrr": float(np.mean(1.0 / ranks)),
    }
    for top_n in TOP_N_VALUES:
        metrics[f"top_{top_n}"] = float(np.mean(ranks <= top_n))

    return {"metrics": metrics, "probe_rows": probe_rows}


def calculate_stability(
    *,
    model_name: str,
    condition_name: str,
    clean_probe_rows: Sequence[dict[str, object]],
    degraded_probe_rows: Sequence[dict[str, object]],
    clean_embeddings: FloatMatrix,
    degraded_embeddings: FloatMatrix,
) -> dict[str, object]:
    """Measure rank-1 changes, clean-correct retention, and embedding drift."""

    if len(clean_probe_rows) != len(degraded_probe_rows):
        raise ValueError("clean and degraded probe result counts must match")

    rank_1_changed = 0
    clean_correct = 0
    clean_correct_retained = 0

    for clean_row, degraded_row in zip(clean_probe_rows, degraded_probe_rows):
        if clean_row["probe_relative_path"] != degraded_row["probe_relative_path"]:
            raise ValueError("clean and degraded probe ordering must match")
        rank_1_changed += int(clean_row["rank_1"] != degraded_row["rank_1"])
        was_clean_correct = clean_row["true_identity_rank"] == 1
        clean_correct += int(was_clean_correct)
        clean_correct_retained += int(
            was_clean_correct and degraded_row["true_identity_rank"] == 1
        )

    clean_normalized = _normalize_rows(clean_embeddings)
    degraded_normalized = _normalize_rows(degraded_embeddings)
    cosine_drift = 1.0 - np.sum(
        clean_normalized * degraded_normalized, axis=1
    )

    return {
        "model": model_name,
        "condition": condition_name,
        "probe_count": len(clean_probe_rows),
        "rank_1_changed_rate": rank_1_changed / len(clean_probe_rows),
        "clean_correct_count": clean_correct,
        "clean_correct_retained_count": clean_correct_retained,
        "clean_correct_retention": (
            clean_correct_retained / clean_correct if clean_correct else None
        ),
        "mean_cosine_embedding_drift": float(np.mean(cosine_drift)),
    }


def summarize_image_quality(
    quality_by_condition: dict[str, dict[str, FloatVector]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition_name in CONDITIONS:
        values = quality_by_condition[condition_name]
        rows.append(
            {
                "condition": condition_name,
                "probe_count": len(values["mean_intensity"]),
                "mean_intensity": float(np.mean(values["mean_intensity"])),
                "mean_zero_fraction": float(np.mean(values["zero_fraction"])),
                "mean_255_fraction": float(
                    np.mean(values["saturated_fraction"])
                ),
            }
        )
    return rows


def select_scope(
    rows: Sequence[dict[str, str]], *, max_identities: int | None
) -> list[dict[str, str]]:
    """Select all data or a deterministic closed-set pilot identity subset."""

    if max_identities is None:
        return list(rows)

    probe_identities = sorted(
        {row["identity"] for row in rows if row["split"] == "probe"}
    )
    selected_identities = set(probe_identities[:max_identities])
    return [row for row in rows if row["identity"] in selected_identities]


def fingerprint_rows(rows: Sequence[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["split"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(row["identity"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(row["relative_path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(row["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_and_verify_manifest(
    manifest_path: Path, audit_path: Path
) -> tuple[list[dict[str, str]], str]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    actual_sha256 = _sha256_file(manifest_path)
    if actual_sha256 != audit.get("manifest_sha256"):
        raise ValueError("dataset manifest hash does not match dataset audit")

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"split", "identity", "relative_path", "sha256"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("dataset manifest is empty or missing required columns")
    return rows, actual_sha256


def _normalize_rows(values: FloatMatrix) -> FloatMatrix:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("embedding collection must be two-dimensional")
    if not np.isfinite(matrix).all():
        raise ValueError("embedding collection must contain finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding collection cannot contain zero vectors")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _identity_count(rows: Sequence[dict[str, str]]) -> int:
    return len({row["identity"] for row in rows})


def _validate_run_configuration(
    models: Sequence[str], batch_size: int, max_identities: int | None
) -> None:
    if not models or any(model not in MODELS for model in models):
        raise ValueError(f"models must be selected from {MODELS}")
    if isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if max_identities is not None and max_identities <= 0:
        raise ValueError("max_identities must be positive when provided")


def _dependency_versions() -> dict[str, str]:
    return {
        package: version(package)
        for package in (
            "facenet-pytorch",
            "numpy",
            "Pillow",
            "torch",
            "torchvision",
        )
    }


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV artifact: {path.name}")
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare FaceNet checkpoints under controlled probe degradation."
        )
    )
    default_output = Path("experiments/outputs/01_embedding_robustness")
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=default_output / "dataset_manifest.csv",
    )
    parser.add_argument(
        "--dataset-audit",
        type=Path,
        default=default_output / "dataset_audit.json",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("identification_service/storage"),
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("experiments/cache/01_embedding_robustness"),
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=list(MODELS)
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--max-identities", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_comparison(
        dataset_manifest=args.dataset_manifest,
        dataset_audit=args.dataset_audit,
        asset_root=args.asset_root,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        models=args.models,
        batch_size=args.batch_size,
        device=args.device,
        max_identities=args.max_identities,
        force=args.force,
    )


if __name__ == "__main__":
    main()
