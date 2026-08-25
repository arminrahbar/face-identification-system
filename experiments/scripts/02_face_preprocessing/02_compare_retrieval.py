"""Compare full-image and MTCNN crop/fallback identity retrieval.

Experiment 01 fixed the embedding checkpoint at VGGFace2. This Experiment 02
stage therefore changes only preprocessing while keeping the gallery, probes,
embedding model, standardization, retrieval backend, and identity-ranking
policy constant.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from identification_service.modules.extraction.embedding import (  # noqa: E402
    FaceEmbeddingExtractor,
    InceptionResnetBackend,
)


FloatMatrix = NDArray[np.float32]
StringVector = NDArray[np.str_]

MODEL_NAME = "vggface2"
CONDITIONS = ("full_image", "mtcnn_crop_fallback")
TOP_N_VALUES = (1, 3, 5, 10)
CACHE_SCHEMA_VERSION = 1
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_825
COMPARISON_ARTIFACT_NAMES = (
    "condition_metrics.csv",
    "metric_deltas.csv",
    "probe_rankings.csv",
    "probe_case_analysis.csv",
    "top1_outcome_summary.csv",
    "rank_change_summary.csv",
    "comparison_run.json",
)


def run_comparison(
    *,
    dataset_manifest: Path,
    dataset_audit: Path,
    detection_records: Path,
    preprocessing_audit: Path,
    experiment_01_metrics: Path,
    experiment_01_run: Path,
    asset_root: Path,
    crop_root: Path,
    output_dir: Path,
    cache_dir: Path,
    batch_size: int = 32,
    device: str | None = None,
    max_identities: int | None = None,
    force_cache: bool = False,
) -> dict[str, object]:
    """Run the full controlled comparison or a closed-set pilot."""

    _validate_run_configuration(batch_size, max_identities)
    inputs = load_verified_inputs(
        dataset_manifest=dataset_manifest,
        dataset_audit=dataset_audit,
        detection_records=detection_records,
        preprocessing_audit=preprocessing_audit,
    )
    all_rows = inputs["dataset_rows"]
    selected_rows = select_scope(all_rows, max_identities=max_identities)
    gallery_rows = [row for row in selected_rows if row["split"] == "gallery"]
    probe_rows = [row for row in selected_rows if row["split"] == "probe"]
    if not gallery_rows or not probe_rows:
        raise ValueError("selected scope must contain gallery and probe images")

    run_scope = "full" if max_identities is None else "pilot"
    selection_fingerprint = fingerprint_rows(selected_rows)
    detection_by_path = inputs["detection_by_path"]
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir / ".comparison.incomplete"
    final_paths = [output_dir / name for name in COMPARISON_ARTIFACT_NAMES]
    occupied = [path for path in final_paths if path.exists()]
    if occupied:
        raise FileExistsError(
            "comparison outputs already exist: "
            + ", ".join(str(path) for path in occupied)
        )
    if staging_dir.exists():
        raise FileExistsError(
            "incomplete comparison staging directory already exists: "
            f"{staging_dir}"
        )
    staging_dir.mkdir()

    print(f"Run scope: {run_scope}")
    print(f"Model: {MODEL_NAME}")
    print(f"Gallery images: {len(gallery_rows)}")
    print(f"Gallery identities: {_identity_count(gallery_rows)}")
    print(f"Probe images: {len(probe_rows)}")
    print(f"Probe identities: {_identity_count(probe_rows)}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Selection fingerprint: {selection_fingerprint}")

    try:
        backend = InceptionResnetBackend(pretrained=MODEL_NAME, device=device)
        extractor = FaceEmbeddingExtractor(backend=backend, image_size=160)
        caches: dict[tuple[str, str], dict[str, NDArray]] = {}

        for condition in CONDITIONS:
            print("=" * 80)
            print(f"CONDITION: {condition}")
            print("=" * 80)
            for split_name, rows in (
                ("gallery", gallery_rows),
                ("probe", probe_rows),
            ):
                caches[(condition, split_name)] = compute_or_load_embeddings(
                    rows=rows,
                    asset_root=asset_root,
                    crop_root=crop_root,
                    detection_by_path=detection_by_path,
                    cache_dir=cache_dir,
                    selection_fingerprint=selection_fingerprint,
                    preprocessing_audit_sha256=inputs[
                        "preprocessing_audit_sha256"
                    ],
                    condition=condition,
                    split_name=split_name,
                    extractor=extractor,
                    batch_size=batch_size,
                    force=force_cache,
                )

        metric_rows: list[dict[str, object]] = []
        ranking_rows: list[dict[str, object]] = []
        evaluations: dict[str, dict[str, object]] = {}

        for condition in CONDITIONS:
            gallery_cache = caches[(condition, "gallery")]
            probe_cache = caches[(condition, "probe")]
            evaluation = evaluate_retrieval(
                gallery_embeddings=gallery_cache["embeddings"],
                gallery_identities=gallery_cache["identities"],
                probe_embeddings=probe_cache["embeddings"],
                probe_identities=probe_cache["identities"],
                probe_paths=probe_cache["relative_paths"],
            )
            evaluations[condition] = evaluation
            metric_rows.append(
                {
                    "model": MODEL_NAME,
                    "condition": condition,
                    "gallery_images": len(gallery_rows),
                    "gallery_identities": _identity_count(gallery_rows),
                    "probe_images": len(probe_rows),
                    "probe_identities": _identity_count(probe_rows),
                    **evaluation["metrics"],
                }
            )
            for row in evaluation["probe_rows"]:
                ranking_rows.append(
                    {"model": MODEL_NAME, "condition": condition, **row}
                )

        full_metrics = metric_rows[0]
        crop_metrics = metric_rows[1]
        delta_row = {
            "model": MODEL_NAME,
            "comparison": "mtcnn_crop_fallback_minus_full_image",
            **{
                f"{metric}_delta": float(crop_metrics[metric])
                - float(full_metrics[metric])
                for metric in ("top_1", "top_3", "top_5", "top_10", "mrr")
            },
        }

        paired = build_paired_analysis(
            full_rows=evaluations["full_image"]["probe_rows"],
            crop_rows=evaluations["mtcnn_crop_fallback"]["probe_rows"],
            detection_by_path=detection_by_path,
            full_probe_embeddings=caches[("full_image", "probe")]["embeddings"],
            crop_probe_embeddings=caches[("mtcnn_crop_fallback", "probe")][
                "embeddings"
            ],
        )
        coverage_rate = float(inputs["preprocessing_audit"]["results"]["coverage_rate"])
        decision = determine_recommendation(
            top_1_delta=float(delta_row["top_1_delta"]),
            mrr_delta=float(delta_row["mrr_delta"]),
            bootstrap_ci_low=float(
                paired["outcome_summary"]["top_1_delta_ci_95_low"]
            ),
            bootstrap_ci_high=float(
                paired["outcome_summary"]["top_1_delta_ci_95_high"]
            ),
            mcnemar_p_value=float(
                paired["outcome_summary"]["mcnemar_exact_two_sided_p_value"]
            ),
            coverage_rate=coverage_rate,
        )

        baseline_check = verify_experiment_01_baseline(
            run_scope=run_scope,
            current_metrics=full_metrics,
            experiment_01_metrics=experiment_01_metrics,
            experiment_01_run=experiment_01_run,
        )

        artifact_rows = {
            "condition_metrics.csv": metric_rows,
            "metric_deltas.csv": [delta_row],
            "probe_rankings.csv": ranking_rows,
            "probe_case_analysis.csv": paired["case_rows"],
            "top1_outcome_summary.csv": [paired["outcome_summary"]],
            "rank_change_summary.csv": [paired["rank_summary"]],
        }
        for artifact_name, rows in artifact_rows.items():
            _write_csv(staging_dir / artifact_name, rows)

        artifact_metadata = {
            name: _artifact_metadata(staging_dir / name, len(rows))
            for name, rows in artifact_rows.items()
        }
        run_manifest = {
            "status": "complete",
            "run_scope": run_scope,
            "model": MODEL_NAME,
            "device_request": device or "auto",
            "resolved_device": str(backend.device),
            "dataset": {
                "manifest_sha256": inputs["dataset_manifest_sha256"],
                "selection_fingerprint": selection_fingerprint,
                "gallery_images": len(gallery_rows),
                "gallery_identities": _identity_count(gallery_rows),
                "probe_images": len(probe_rows),
                "probe_identities": _identity_count(probe_rows),
            },
            "preprocessing_audit": {
                "sha256": inputs["preprocessing_audit_sha256"],
                "detection_records_sha256": inputs[
                    "detection_records_sha256"
                ],
                "coverage_rate": coverage_rate,
                "detected_images": inputs["preprocessing_audit"]["results"][
                    "detected_images"
                ],
                "fallback_images": inputs["preprocessing_audit"]["results"][
                    "fallback_images"
                ],
            },
            "pipeline": {
                "conditions": list(CONDITIONS),
                "image_size": 160,
                "crop_margin": 20,
                "standardization": "fixed_127_5_div_128",
                "retrieval": "exact_cosine",
                "ranking_unit": "distinct_identity",
                "top_n_values": list(TOP_N_VALUES),
            },
            "decision_policy": {
                "primary_metric": "paired_top_1_delta",
                "requirements_to_adopt_crop_fallback": [
                    "top_1_delta_greater_than_zero",
                    "paired_bootstrap_95_percent_ci_lower_bound_greater_than_zero",
                    "mcnemar_exact_two_sided_p_value_below_0_05",
                    "mrr_delta_greater_than_zero",
                    "coverage_rate_equals_one",
                ],
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            },
            "result": {
                "recommendation": decision,
                "top_1_delta": delta_row["top_1_delta"],
                "mrr_delta": delta_row["mrr_delta"],
                "top_1_delta_ci_95_low": paired["outcome_summary"][
                    "top_1_delta_ci_95_low"
                ],
                "top_1_delta_ci_95_high": paired["outcome_summary"][
                    "top_1_delta_ci_95_high"
                ],
                "mcnemar_exact_two_sided_p_value": paired[
                    "outcome_summary"
                ]["mcnemar_exact_two_sided_p_value"],
            },
            "experiment_01_baseline_check": baseline_check,
            "dependencies": _dependency_versions(),
            "artifacts": artifact_metadata,
        }
        _write_json(staging_dir / "comparison_run.json", run_manifest)

        for artifact_name in COMPARISON_ARTIFACT_NAMES:
            (staging_dir / artifact_name).replace(output_dir / artifact_name)
        staging_dir.rmdir()
    except BaseException:
        print(f"[INCOMPLETE] Comparison staging retained at: {staging_dir}")
        raise

    print("=" * 80)
    print("PREPROCESSING RETRIEVAL COMPARISON")
    print("=" * 80)
    for row in metric_rows:
        print(
            f"{row['condition']:24s} Top-1={row['top_1']:.4f} "
            f"Top-5={row['top_5']:.4f} MRR={row['mrr']:.4f}"
        )
    print(f"Top-1 delta: {delta_row['top_1_delta']:+.4f}")
    print(f"MRR delta: {delta_row['mrr_delta']:+.4f}")
    print(
        "Paired Top-1 outcomes: "
        f"helps={paired['outcome_summary']['cropping_helps_count']}, "
        f"hurts={paired['outcome_summary']['cropping_hurts_count']}"
    )
    print(f"Recommendation: {decision}")
    for artifact_name in COMPARISON_ARTIFACT_NAMES:
        print(f"[WRITE] {output_dir / artifact_name}")

    return {
        "run_manifest": run_manifest,
        "artifact_paths": tuple(output_dir / name for name in COMPARISON_ARTIFACT_NAMES),
    }


def load_verified_inputs(
    *,
    dataset_manifest: Path,
    dataset_audit: Path,
    detection_records: Path,
    preprocessing_audit: Path,
) -> dict[str, object]:
    """Verify the dataset and preprocessing audit before reuse."""

    dataset_audit_payload = json.loads(dataset_audit.read_text(encoding="utf-8"))
    dataset_manifest_sha256 = _sha256_file(dataset_manifest)
    if dataset_audit_payload.get("status") != "complete":
        raise ValueError("dataset audit must be complete")
    if dataset_manifest_sha256 != dataset_audit_payload.get("manifest_sha256"):
        raise ValueError("dataset manifest hash does not match dataset audit")

    with dataset_manifest.open(newline="", encoding="utf-8") as handle:
        dataset_rows = list(csv.DictReader(handle))
    required_dataset_fields = {"split", "identity", "relative_path", "sha256"}
    if not dataset_rows or not required_dataset_fields.issubset(dataset_rows[0]):
        raise ValueError("dataset manifest is empty or missing required fields")

    preprocessing_payload = json.loads(
        preprocessing_audit.read_text(encoding="utf-8")
    )
    if (
        preprocessing_payload.get("status") != "complete"
        or preprocessing_payload.get("run_scope") != "full"
    ):
        raise ValueError("preprocessing audit must describe a complete full run")
    if (
        preprocessing_payload["dataset"]["manifest_sha256"]
        != dataset_manifest_sha256
    ):
        raise ValueError("preprocessing audit references a different dataset")

    detection_records_sha256 = _sha256_file(detection_records)
    expected_metadata = preprocessing_payload["artifacts"].get(
        detection_records.name
    )
    if expected_metadata is None:
        raise ValueError("preprocessing audit does not register detection records")
    if detection_records_sha256 != expected_metadata.get("sha256"):
        raise ValueError("detection records hash does not match preprocessing audit")

    with detection_records.open(newline="", encoding="utf-8") as handle:
        detection_rows = list(csv.DictReader(handle))
    required_detection_fields = {
        "split",
        "identity",
        "source_relative_path",
        "source_sha256",
        "face_detected",
        "confidence",
        "crop_area_ratio",
        "fallback_reason",
        "crop_relative_path",
        "crop_sha256",
    }
    if not detection_rows or not required_detection_fields.issubset(
        detection_rows[0]
    ):
        raise ValueError("detection records are empty or missing required fields")
    if len(detection_rows) != len(dataset_rows):
        raise ValueError("dataset and detection record counts must match")

    detection_by_path: dict[str, dict[str, str]] = {}
    for row in detection_rows:
        relative_path = row["source_relative_path"]
        if relative_path in detection_by_path:
            raise ValueError(f"duplicate detection record: {relative_path}")
        _parse_bool(row["face_detected"])
        detection_by_path[relative_path] = row

    for row in dataset_rows:
        relative_path = row["relative_path"]
        detection = detection_by_path.get(relative_path)
        if detection is None:
            raise ValueError(f"missing detection record: {relative_path}")
        if (
            detection["split"] != row["split"]
            or detection["identity"] != row["identity"]
            or detection["source_sha256"] != row["sha256"]
        ):
            raise ValueError(f"detection metadata mismatch: {relative_path}")
        detected = _parse_bool(detection["face_detected"])
        if detected and (
            not detection["crop_relative_path"] or not detection["crop_sha256"]
        ):
            raise ValueError(f"detected face is missing crop evidence: {relative_path}")
        if not detected and detection["fallback_reason"] not in {
            "face_not_detected",
            "detector_error",
        }:
            raise ValueError(f"fallback reason is invalid: {relative_path}")

    return {
        "dataset_rows": dataset_rows,
        "detection_by_path": detection_by_path,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "preprocessing_audit": preprocessing_payload,
        "preprocessing_audit_sha256": _sha256_file(preprocessing_audit),
        "detection_records_sha256": detection_records_sha256,
    }


def compute_or_load_embeddings(
    *,
    rows: Sequence[dict[str, str]],
    asset_root: Path,
    crop_root: Path,
    detection_by_path: dict[str, dict[str, str]],
    cache_dir: Path,
    selection_fingerprint: str,
    preprocessing_audit_sha256: str,
    condition: str,
    split_name: str,
    extractor: FaceEmbeddingExtractor,
    batch_size: int,
    force: bool,
) -> dict[str, NDArray]:
    """Load a validated embedding cache or compute it in batches."""

    configuration = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "selection_fingerprint": selection_fingerprint,
        "preprocessing_audit_sha256": preprocessing_audit_sha256,
        "model": MODEL_NAME,
        "condition": condition,
        "split": split_name,
        "image_size": 160,
        "standardization": "fixed_127_5_div_128",
    }
    configuration_json = json.dumps(configuration, sort_keys=True)
    cache_path = cache_dir / (
        f"{selection_fingerprint[:12]}__{MODEL_NAME}__{split_name}__{condition}.npz"
    )
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
                        for name in (
                            "configuration_json",
                            "embeddings",
                            "identities",
                            "relative_paths",
                        )
                    }
        except (OSError, ValueError, KeyError):
            pass
        print(f"[CACHE STALE] {cache_path.name}")

    chunks: list[FloatMatrix] = []
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        images = [
            load_model_image(
                row=row,
                condition=condition,
                asset_root=asset_root,
                crop_root=crop_root,
                detection=detection_by_path[row["relative_path"]],
            )
            for row in rows[start:stop]
        ]
        chunks.append(extractor.encode_batch(images))
        print(f"  {condition} {split_name}: {stop}/{len(rows)}")

    embeddings = np.ascontiguousarray(np.vstack(chunks), dtype=np.float32)
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
    }
    temporary_path = cache_path.with_suffix(".npz.tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary_path.replace(cache_path)
    print(f"[CACHE WRITE] {cache_path.name}")
    return payload


def load_model_image(
    *,
    row: dict[str, str],
    condition: str,
    asset_root: Path,
    crop_root: Path,
    detection: dict[str, str],
) -> Image.Image:
    """Load one verified model input under the requested condition."""

    if condition not in CONDITIONS:
        raise ValueError(f"unsupported preprocessing condition: {condition}")
    source_path = _resolve_within(asset_root, row["relative_path"])
    if _sha256_file(source_path) != row["sha256"]:
        raise ValueError(f"source image hash mismatch: {row['relative_path']}")

    detected = _parse_bool(detection["face_detected"])
    if condition == "mtcnn_crop_fallback" and detected:
        crop_relative_path = detection["crop_relative_path"]
        crop_path = _resolve_within(crop_root, crop_relative_path)
        if _sha256_file(crop_path) != detection["crop_sha256"]:
            raise ValueError(f"crop image hash mismatch: {crop_relative_path}")
        with Image.open(crop_path) as crop_image:
            prepared = crop_image.convert("RGB")
            if prepared.size != (160, 160):
                raise ValueError(f"crop image must be 160x160: {crop_relative_path}")
            return prepared.copy()

    with Image.open(source_path) as source_image:
        oriented = ImageOps.exif_transpose(source_image).convert("RGB")
        return oriented.resize((160, 160), Image.Resampling.BILINEAR)


def evaluate_retrieval(
    *,
    gallery_embeddings: FloatMatrix,
    gallery_identities: StringVector,
    probe_embeddings: FloatMatrix,
    probe_identities: StringVector,
    probe_paths: StringVector,
) -> dict[str, object]:
    """Evaluate exact cosine retrieval after collapsing image duplicates."""

    gallery = _normalize_rows(gallery_embeddings)
    probes = _normalize_rows(probe_embeddings)
    similarities = probes @ gallery.T
    image_orders = np.argsort(-similarities, axis=1, kind="stable")
    result_rows: list[dict[str, object]] = []
    true_ranks: list[int] = []

    for probe_index, image_order in enumerate(image_orders):
        true_identity = str(probe_identities[probe_index])
        seen: set[str] = set()
        top_identities: list[str] = []
        true_rank: int | None = None
        true_similarity: float | None = None
        rank_1_similarity: float | None = None

        for gallery_position in image_order:
            position = int(gallery_position)
            identity = str(gallery_identities[position])
            if identity in seen:
                continue
            seen.add(identity)
            identity_rank = len(seen)
            similarity = float(similarities[probe_index, position])
            if rank_1_similarity is None:
                rank_1_similarity = similarity
            if len(top_identities) < max(TOP_N_VALUES):
                top_identities.append(identity)
            if identity == true_identity:
                true_rank = identity_rank
                true_similarity = similarity
            if true_rank is not None and len(top_identities) == max(TOP_N_VALUES):
                break

        if true_rank is None or true_similarity is None or rank_1_similarity is None:
            raise ValueError(
                f"true identity is absent from gallery ranking: {true_identity}"
            )
        true_ranks.append(true_rank)
        row: dict[str, object] = {
            "probe_relative_path": str(probe_paths[probe_index]),
            "true_identity": true_identity,
            "true_identity_rank": true_rank,
            "reciprocal_rank": 1.0 / true_rank,
            "rank_1_similarity": rank_1_similarity,
            "true_identity_similarity": true_similarity,
        }
        for rank in range(1, max(TOP_N_VALUES) + 1):
            row[f"rank_{rank}"] = (
                top_identities[rank - 1]
                if rank <= len(top_identities)
                else ""
            )
        result_rows.append(row)

    ranks = np.asarray(true_ranks, dtype=np.int32)
    metrics: dict[str, object] = {
        "probe_count": len(result_rows),
        "mrr": float(np.mean(1.0 / ranks)),
    }
    for top_n in TOP_N_VALUES:
        metrics[f"top_{top_n}"] = float(np.mean(ranks <= top_n))
    return {"metrics": metrics, "probe_rows": result_rows}


def build_paired_analysis(
    *,
    full_rows: Sequence[dict[str, object]],
    crop_rows: Sequence[dict[str, object]],
    detection_by_path: dict[str, dict[str, str]],
    full_probe_embeddings: FloatMatrix,
    crop_probe_embeddings: FloatMatrix,
) -> dict[str, object]:
    """Build probe-level cases and paired statistical summaries."""

    if len(full_rows) != len(crop_rows) or not full_rows:
        raise ValueError("paired result collections must be nonempty and equal")
    case_rows: list[dict[str, object]] = []
    full_correct: list[bool] = []
    crop_correct: list[bool] = []
    full_ranks: list[int] = []
    crop_ranks: list[int] = []

    for full, crop in zip(full_rows, crop_rows):
        if (
            full["probe_relative_path"] != crop["probe_relative_path"]
            or full["true_identity"] != crop["true_identity"]
        ):
            raise ValueError("paired probe ordering must match")
        path = str(full["probe_relative_path"])
        detection = detection_by_path[path]
        full_is_correct = int(full["true_identity_rank"]) == 1
        crop_is_correct = int(crop["true_identity_rank"]) == 1
        if full_is_correct and crop_is_correct:
            case_type = "both_correct"
        elif not full_is_correct and crop_is_correct:
            case_type = "cropping_helps"
        elif full_is_correct and not crop_is_correct:
            case_type = "cropping_hurts"
        else:
            case_type = "neither_correct"

        full_rank = int(full["true_identity_rank"])
        crop_rank = int(crop["true_identity_rank"])
        full_correct.append(full_is_correct)
        crop_correct.append(crop_is_correct)
        full_ranks.append(full_rank)
        crop_ranks.append(crop_rank)
        case_rows.append(
            {
                "probe_relative_path": path,
                "true_identity": full["true_identity"],
                "full_image_rank_1": full["rank_1"],
                "crop_fallback_rank_1": crop["rank_1"],
                "full_image_true_identity_rank": full_rank,
                "crop_fallback_true_identity_rank": crop_rank,
                "rank_delta_crop_minus_full": crop_rank - full_rank,
                "full_image_top_1_correct": full_is_correct,
                "crop_fallback_top_1_correct": crop_is_correct,
                "case_type": case_type,
                "mtcnn_face_detected": _parse_bool(detection["face_detected"]),
                "mtcnn_confidence": detection["confidence"],
                "crop_area_ratio": detection["crop_area_ratio"],
                "fallback_reason": detection["fallback_reason"],
            }
        )

    full_correct_array = np.asarray(full_correct, dtype=np.int8)
    crop_correct_array = np.asarray(crop_correct, dtype=np.int8)
    case_counts = {
        label: sum(row["case_type"] == label for row in case_rows)
        for label in (
            "both_correct",
            "cropping_helps",
            "cropping_hurts",
            "neither_correct",
        )
    }
    ci_low, ci_high = paired_bootstrap_top_1_interval(
        full_correct_array,
        crop_correct_array,
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    outcome_summary = {
        "probe_count": len(case_rows),
        "both_correct_count": case_counts["both_correct"],
        "cropping_helps_count": case_counts["cropping_helps"],
        "cropping_hurts_count": case_counts["cropping_hurts"],
        "neither_correct_count": case_counts["neither_correct"],
        "net_top_1_wins": case_counts["cropping_helps"]
        - case_counts["cropping_hurts"],
        "top_1_delta": float(
            np.mean(crop_correct_array) - np.mean(full_correct_array)
        ),
        "top_1_delta_ci_95_low": ci_low,
        "top_1_delta_ci_95_high": ci_high,
        "mcnemar_exact_two_sided_p_value": exact_mcnemar_p_value(
            cropping_helps=case_counts["cropping_helps"],
            cropping_hurts=case_counts["cropping_hurts"],
        ),
    }

    full_rank_array = np.asarray(full_ranks, dtype=np.int32)
    crop_rank_array = np.asarray(crop_ranks, dtype=np.int32)
    rank_delta = crop_rank_array - full_rank_array
    full_normalized = _normalize_rows(full_probe_embeddings)
    crop_normalized = _normalize_rows(crop_probe_embeddings)
    cosine_drift = 1.0 - np.sum(full_normalized * crop_normalized, axis=1)
    rank_summary = {
        "probe_count": len(case_rows),
        "rank_improved_count": int(np.sum(rank_delta < 0)),
        "rank_worsened_count": int(np.sum(rank_delta > 0)),
        "rank_unchanged_count": int(np.sum(rank_delta == 0)),
        "mean_rank_delta_crop_minus_full": float(np.mean(rank_delta)),
        "median_rank_delta_crop_minus_full": float(np.median(rank_delta)),
        "mean_probe_embedding_cosine_drift": float(np.mean(cosine_drift)),
        "median_probe_embedding_cosine_drift": float(np.median(cosine_drift)),
    }
    return {
        "case_rows": case_rows,
        "outcome_summary": outcome_summary,
        "rank_summary": rank_summary,
    }


def paired_bootstrap_top_1_interval(
    full_correct: NDArray[np.int8],
    crop_correct: NDArray[np.int8],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Return a deterministic percentile interval for paired Top-1 delta."""

    if full_correct.shape != crop_correct.shape or full_correct.ndim != 1:
        raise ValueError("paired correctness arrays must be one-dimensional and equal")
    if len(full_correct) == 0 or samples <= 0:
        raise ValueError("paired bootstrap requires observations and samples")
    differences = crop_correct.astype(np.float64) - full_correct.astype(np.float64)
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    chunk_size = 500
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        indices = rng.integers(
            0,
            len(differences),
            size=(stop - start, len(differences)),
        )
        bootstrap_means[start:stop] = differences[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(low), float(high)


def exact_mcnemar_p_value(*, cropping_helps: int, cropping_hurts: int) -> float:
    """Return the two-sided exact McNemar/binomial p-value."""

    if cropping_helps < 0 or cropping_hurts < 0:
        raise ValueError("paired outcome counts cannot be negative")
    discordant = cropping_helps + cropping_hurts
    if discordant == 0:
        return 1.0
    tail = min(cropping_helps, cropping_hurts)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * probability)


def determine_recommendation(
    *,
    top_1_delta: float,
    mrr_delta: float,
    bootstrap_ci_low: float,
    bootstrap_ci_high: float,
    mcnemar_p_value: float,
    coverage_rate: float,
) -> str:
    """Apply the declared preprocessing adoption policy."""

    if (
        top_1_delta > 0
        and bootstrap_ci_low > 0
        and mcnemar_p_value < 0.05
        and mrr_delta > 0
        and coverage_rate == 1.0
    ):
        return "adopt_mtcnn_crop_fallback"
    if (
        top_1_delta < 0
        and bootstrap_ci_high < 0
        and mcnemar_p_value < 0.05
    ):
        return "retain_full_image"
    return "inconclusive"


def verify_experiment_01_baseline(
    *,
    run_scope: str,
    current_metrics: dict[str, object],
    experiment_01_metrics: Path,
    experiment_01_run: Path,
) -> dict[str, object]:
    """Confirm the full-image condition reproduces Experiment 01 clean results."""

    if run_scope != "full":
        return {"status": "not_applicable_to_pilot"}
    run = json.loads(experiment_01_run.read_text(encoding="utf-8"))
    if run.get("status") != "complete" or run.get("run_scope") != "full":
        raise ValueError("Experiment 01 run manifest must be complete and full")
    if run.get("selection", {}).get("selected_model") != MODEL_NAME:
        raise ValueError("Experiment 01 did not select the expected model")
    expected_hash = run.get("artifacts", {}).get(experiment_01_metrics.name)
    if expected_hash != _sha256_file(experiment_01_metrics):
        raise ValueError("Experiment 01 metric hash does not match its run manifest")

    with experiment_01_metrics.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [
        row
        for row in rows
        if row["model"] == MODEL_NAME and row["condition"] == "clean"
    ]
    if len(matches) != 1:
        raise ValueError("Experiment 01 clean VGGFace2 baseline is ambiguous")
    expected = matches[0]
    metrics = ("top_1", "top_3", "top_5", "top_10", "mrr")
    differences = {
        metric: float(current_metrics[metric]) - float(expected[metric])
        for metric in metrics
    }
    if any(abs(value) > 1e-12 for value in differences.values()):
        raise ValueError(
            "full-image metrics do not reproduce Experiment 01 clean baseline: "
            f"{differences}"
        )
    return {
        "status": "matched",
        "experiment_01_run_sha256": _sha256_file(experiment_01_run),
        "experiment_01_metrics_sha256": _sha256_file(experiment_01_metrics),
        "metric_differences": differences,
    }


def select_scope(
    rows: Sequence[dict[str, str]], *, max_identities: int | None
) -> list[dict[str, str]]:
    """Select a deterministic closed-set identity subset for a pilot."""

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
        for field in ("split", "identity", "relative_path", "sha256"):
            digest.update(row[field].encode("utf-8"))
            digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()


def _cache_is_valid(
    *,
    cached: np.lib.npyio.NpzFile,
    configuration_json: str,
    expected_identities: StringVector,
    expected_paths: StringVector,
) -> bool:
    required = {"configuration_json", "embeddings", "identities", "relative_paths"}
    if not required.issubset(cached.files):
        return False
    if str(cached["configuration_json"].item()) != configuration_json:
        return False
    if not np.array_equal(cached["identities"], expected_identities):
        return False
    if not np.array_equal(cached["relative_paths"], expected_paths):
        return False
    embeddings = cached["embeddings"]
    return embeddings.shape == (len(expected_paths), 512) and np.isfinite(
        embeddings
    ).all()


def _normalize_rows(values: FloatMatrix) -> FloatMatrix:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("embedding matrix must be two-dimensional and finite")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding matrix cannot contain zero vectors")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _resolve_within(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    candidate = (root / Path(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes configured root: {relative_path}") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"required image not found: {relative_path}")
    return candidate


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected serialized boolean, received: {value!r}")


def _identity_count(rows: Sequence[dict[str, str]]) -> int:
    return len({row["identity"] for row in rows})


def _validate_run_configuration(
    batch_size: int, max_identities: int | None
) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if max_identities is not None and (
        isinstance(max_identities, bool)
        or not isinstance(max_identities, int)
        or max_identities <= 0
    ):
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


def _artifact_metadata(path: Path, row_count: int) -> dict[str, object]:
    return {
        "rows": row_count,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty artifact: {path.name}")
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full-image and MTCNN crop/fallback retrieval."
    )
    experiment_01 = Path("experiments/outputs/01_embedding_robustness")
    experiment_02 = Path("experiments/outputs/02_face_preprocessing")
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=experiment_01 / "dataset_manifest.csv",
    )
    parser.add_argument(
        "--dataset-audit",
        type=Path,
        default=experiment_01 / "dataset_audit.json",
    )
    parser.add_argument(
        "--detection-records",
        type=Path,
        default=experiment_02 / "detection_records.csv",
    )
    parser.add_argument(
        "--preprocessing-audit",
        type=Path,
        default=experiment_02 / "preprocessing_audit.json",
    )
    parser.add_argument(
        "--experiment-01-metrics",
        type=Path,
        default=experiment_01 / "condition_metrics.csv",
    )
    parser.add_argument(
        "--experiment-01-run",
        type=Path,
        default=experiment_01 / "comparison_run.json",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("identification_service/storage"),
    )
    parser.add_argument(
        "--crop-root",
        type=Path,
        default=Path("experiments/cache/02_face_preprocessing/crops"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiment_02,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("experiments/cache/02_face_preprocessing/embeddings"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument("--max-identities", type=int)
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_comparison(
        dataset_manifest=args.dataset_manifest,
        dataset_audit=args.dataset_audit,
        detection_records=args.detection_records,
        preprocessing_audit=args.preprocessing_audit,
        experiment_01_metrics=args.experiment_01_metrics,
        experiment_01_run=args.experiment_01_run,
        asset_root=args.asset_root,
        crop_root=args.crop_root,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        device=args.device,
        max_identities=args.max_identities,
        force_cache=args.force_cache,
    )


if __name__ == "__main__":
    main()
