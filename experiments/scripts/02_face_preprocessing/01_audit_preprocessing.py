"""Audit MTCNN face crops across the complete evaluation corpus.

This first Experiment 02 stage applies the service preprocessing policy to
every gallery and probe image. It records detection and crop decisions while
storing lossless crop images in the ignored experiment cache. Public artifacts
contain repository-relative identifiers only; local asset paths are never
written to the result tables.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from statistics import mean, median
import sys
from typing import Protocol, Sequence

from PIL import Image, ImageOps


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from identification_service.modules.extraction.preprocessing import (
    FacePreprocessor,
    PreprocessedFace,
)  # noqa: E402


@dataclass(frozen=True, slots=True)
class DatasetRow:
    """Validated manifest fields needed by the preprocessing audit."""

    split: str
    identity: str
    relative_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class DetectionRecord:
    """Portable evidence for one preprocessing decision."""

    split: str
    identity: str
    source_relative_path: str
    source_sha256: str
    image_width: int
    image_height: int
    face_detected: bool
    confidence: float | None
    detection_left: float | None
    detection_top: float | None
    detection_right: float | None
    detection_bottom: float | None
    crop_left: int
    crop_top: int
    crop_right: int
    crop_bottom: int
    crop_width: int
    crop_height: int
    crop_area_ratio: float
    fallback_reason: str | None
    crop_relative_path: str
    crop_sha256: str


class FaceProcessor(Protocol):
    """Behavior required from the runtime preprocessor."""

    def process(self, image: Image.Image) -> PreprocessedFace: ...


def run_preprocessing_audit(
    *,
    asset_root: Path,
    dataset_manifest: Path,
    dataset_audit: Path,
    output_dir: Path,
    cache_root: Path,
    max_images_per_split: int | None = None,
    expected_gallery_images: int | None = None,
    expected_probe_images: int | None = None,
    progress_interval: int = 100,
    preprocessor: FaceProcessor | None = None,
) -> dict[str, object]:
    """Apply and audit the crop/fallback policy for a deterministic scope."""

    _validate_positive_optional("max_images_per_split", max_images_per_split)
    _validate_positive_optional("expected_gallery_images", expected_gallery_images)
    _validate_positive_optional("expected_probe_images", expected_probe_images)
    if isinstance(progress_interval, bool) or not isinstance(progress_interval, int):
        raise TypeError("progress_interval must be an integer")
    if progress_interval <= 0:
        raise ValueError("progress_interval must be greater than zero")

    asset_root = asset_root.resolve()
    rows, manifest_sha256 = load_verified_dataset(
        dataset_manifest=dataset_manifest,
        dataset_audit=dataset_audit,
    )
    split_counts = _split_counts(rows)
    _verify_expected_count(
        "gallery", split_counts.get("gallery", 0), expected_gallery_images
    )
    _verify_expected_count(
        "probe", split_counts.get("probe", 0), expected_probe_images
    )

    selected_rows = select_scope(rows, max_images_per_split=max_images_per_split)
    scope = "pilot" if max_images_per_split is not None else "full"
    selected_counts = _split_counts(selected_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    final_crop_dir = cache_root / "crops"
    working_crop_dir = cache_root / ".crops.incomplete"

    output_paths = {
        "records": output_dir / "detection_records.csv",
        "summary": output_dir / "detection_summary.csv",
        "audit": output_dir / "preprocessing_audit.json",
    }
    occupied = [path for path in output_paths.values() if path.exists()]
    if occupied:
        raise FileExistsError(
            "preprocessing audit outputs already exist: "
            + ", ".join(str(path) for path in occupied)
        )
    if final_crop_dir.exists() or working_crop_dir.exists():
        raise FileExistsError(
            "crop cache target already exists; use a new cache root or remove the "
            "verified stale run before retrying"
        )

    working_crop_dir.mkdir()
    processor = preprocessor if preprocessor is not None else FacePreprocessor()
    records: list[DetectionRecord] = []

    try:
        total = len(selected_rows)
        for position, row in enumerate(selected_rows, start=1):
            source_path = _resolve_source(asset_root, row.relative_path)
            actual_sha256 = _sha256_file(source_path)
            if actual_sha256 != row.sha256:
                raise ValueError(
                    "source image hash does not match dataset manifest: "
                    f"{row.relative_path}"
                )

            with Image.open(source_path) as source_image:
                oriented = ImageOps.exif_transpose(source_image).convert("RGB")
                image_width, image_height = oriented.size
                result = processor.process(oriented)

            crop_relative_path = ""
            crop_sha256 = ""
            if result.face_detected:
                crop_relative_path = _crop_relative_path(row)
                crop_path = working_crop_dir / Path(crop_relative_path)
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                result.image.save(crop_path, format="PNG", optimize=False)
                crop_sha256 = _sha256_file(crop_path)

            records.append(
                _build_detection_record(
                    row=row,
                    image_width=image_width,
                    image_height=image_height,
                    result=result,
                    crop_relative_path=crop_relative_path,
                    crop_sha256=crop_sha256,
                )
            )

            if position % progress_interval == 0 or position == total:
                detected = sum(record.face_detected for record in records)
                fallback = len(records) - detected
                print(
                    f"[MTCNN] Processed {position}/{total} images | "
                    f"detected={detected} | fallback={fallback}"
                )

        summary_rows = summarize_detections(records)
        record_rows = [asdict(record) for record in records]
        _write_csv(
            output_paths["records"],
            rows=record_rows,
            fieldnames=list(record_rows[0]),
        )
        _write_csv(
            output_paths["summary"],
            rows=summary_rows,
            fieldnames=list(summary_rows[0]),
        )

        working_crop_dir.replace(final_crop_dir)
        crop_count = sum(record.face_detected for record in records)
        fallback_count = len(records) - crop_count
        audit_payload = {
            "status": "complete",
            "run_scope": scope,
            "protocol": {
                "detector": "facenet-pytorch MTCNN",
                "face_selection": "largest detected face",
                "image_size": 160,
                "margin": 20,
                "fallback_policy": "resize full image when no face is detected or detector execution fails",
                "crop_encoding": "lossless PNG",
            },
            "dataset": {
                "manifest_sha256": manifest_sha256,
                "source_gallery_images": split_counts.get("gallery", 0),
                "source_probe_images": split_counts.get("probe", 0),
                "selected_gallery_images": selected_counts.get("gallery", 0),
                "selected_probe_images": selected_counts.get("probe", 0),
                "selected_rows_sha256": fingerprint_rows(selected_rows),
            },
            "results": {
                "processed_images": len(records),
                "detected_images": crop_count,
                "fallback_images": fallback_count,
                "coverage_rate": 1.0,
            },
            "artifacts": {
                output_paths["records"].name: _artifact_metadata(
                    output_paths["records"], len(record_rows)
                ),
                output_paths["summary"].name: _artifact_metadata(
                    output_paths["summary"], len(summary_rows)
                ),
            },
        }
        _write_json(output_paths["audit"], audit_payload)
    except BaseException:
        print(f"[INCOMPLETE] Partial crop cache retained at: {working_crop_dir}")
        raise

    return {
        "audit": audit_payload,
        "records": tuple(records),
        "summary": tuple(summary_rows),
        "artifact_paths": tuple(output_paths.values()),
        "crop_dir": final_crop_dir,
    }


def load_verified_dataset(
    *, dataset_manifest: Path, dataset_audit: Path
) -> tuple[tuple[DatasetRow, ...], str]:
    """Load Experiment 01 inventory only after verifying its manifest hash."""

    if not dataset_manifest.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {dataset_manifest}")
    if not dataset_audit.is_file():
        raise FileNotFoundError(f"dataset audit not found: {dataset_audit}")

    audit = json.loads(dataset_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "complete" or audit.get("closed_set") is not True:
        raise ValueError("dataset audit must describe a complete closed-set inventory")

    manifest_sha256 = _sha256_file(dataset_manifest)
    if manifest_sha256 != audit.get("manifest_sha256"):
        raise ValueError("dataset manifest hash does not match dataset audit")

    with dataset_manifest.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    required = {"split", "identity", "relative_path", "sha256"}
    if not source_rows or not required.issubset(source_rows[0]):
        raise ValueError("dataset manifest is empty or missing required columns")

    rows = tuple(
        DatasetRow(
            split=row["split"],
            identity=row["identity"],
            relative_path=row["relative_path"],
            sha256=row["sha256"],
        )
        for row in source_rows
    )
    if {row.split for row in rows} != {"gallery", "probe"}:
        raise ValueError("dataset manifest must contain gallery and probe rows")
    if len({row.relative_path for row in rows}) != len(rows):
        raise ValueError("dataset manifest contains duplicate relative paths")
    return rows, manifest_sha256


def select_scope(
    rows: Sequence[DatasetRow], *, max_images_per_split: int | None
) -> tuple[DatasetRow, ...]:
    """Select a deterministic pilot or retain the complete manifest."""

    ordered = sorted(
        rows,
        key=lambda row: (row.split, row.identity, row.relative_path),
    )
    if max_images_per_split is None:
        return tuple(ordered)

    selected: list[DatasetRow] = []
    selected_counts: dict[str, int] = {}
    for row in ordered:
        count = selected_counts.get(row.split, 0)
        if count >= max_images_per_split:
            continue
        selected.append(row)
        selected_counts[row.split] = count + 1
    return tuple(selected)


def summarize_detections(
    records: Sequence[DetectionRecord],
) -> list[dict[str, object]]:
    """Summarize detection reliability and crop geometry by split."""

    summaries: list[dict[str, object]] = []
    for split in ("gallery", "probe", "all"):
        group = (
            list(records)
            if split == "all"
            else [record for record in records if record.split == split]
        )
        if not group:
            continue

        detected = [record for record in group if record.face_detected]
        confidences = [
            float(record.confidence)
            for record in detected
            if record.confidence is not None
        ]
        area_ratios = [record.crop_area_ratio for record in detected]
        detector_errors = sum(
            record.fallback_reason == "detector_error" for record in group
        )
        no_face = sum(
            record.fallback_reason == "face_not_detected" for record in group
        )

        summaries.append(
            {
                "split": split,
                "image_count": len(group),
                "detected_count": len(detected),
                "fallback_count": len(group) - len(detected),
                "detector_error_count": detector_errors,
                "face_not_detected_count": no_face,
                "detection_rate": len(detected) / len(group),
                "fallback_rate": (len(group) - len(detected)) / len(group),
                "coverage_rate": 1.0,
                "mean_detection_confidence": mean(confidences)
                if confidences
                else None,
                "median_detection_confidence": median(confidences)
                if confidences
                else None,
                "minimum_detection_confidence": min(confidences)
                if confidences
                else None,
                "mean_crop_area_ratio": mean(area_ratios)
                if area_ratios
                else None,
                "median_crop_area_ratio": median(area_ratios)
                if area_ratios
                else None,
            }
        )
    return summaries


def fingerprint_rows(rows: Sequence[DatasetRow]) -> str:
    """Fingerprint the ordered experiment scope."""

    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.split.encode("utf-8"))
        digest.update(b"\0")
        digest.update(row.identity.encode("utf-8"))
        digest.update(b"\0")
        digest.update(row.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(row.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _build_detection_record(
    *,
    row: DatasetRow,
    image_width: int,
    image_height: int,
    result: PreprocessedFace,
    crop_relative_path: str,
    crop_sha256: str,
) -> DetectionRecord:
    detection = result.detection_box or (None, None, None, None)
    crop_left, crop_top, crop_right, crop_bottom = result.crop_box
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(f"preprocessor produced an invalid crop for {row.relative_path}")

    return DetectionRecord(
        split=row.split,
        identity=row.identity,
        source_relative_path=row.relative_path,
        source_sha256=row.sha256,
        image_width=image_width,
        image_height=image_height,
        face_detected=result.face_detected,
        confidence=result.confidence,
        detection_left=detection[0],
        detection_top=detection[1],
        detection_right=detection[2],
        detection_bottom=detection[3],
        crop_left=crop_left,
        crop_top=crop_top,
        crop_right=crop_right,
        crop_bottom=crop_bottom,
        crop_width=crop_width,
        crop_height=crop_height,
        crop_area_ratio=(crop_width * crop_height) / (image_width * image_height),
        fallback_reason=result.fallback_reason,
        crop_relative_path=crop_relative_path,
        crop_sha256=crop_sha256,
    )


def _crop_relative_path(row: DatasetRow) -> str:
    digest = hashlib.sha256(row.relative_path.encode("utf-8")).hexdigest()[:12]
    return f"{row.split}/{row.identity}/{digest}.png"


def _resolve_source(asset_root: Path, relative_path: str) -> Path:
    candidate = (asset_root / Path(relative_path)).resolve()
    try:
        candidate.relative_to(asset_root)
    except ValueError as error:
        raise ValueError(
            f"dataset path escapes the configured asset root: {relative_path}"
        ) from error
    if not candidate.is_file():
        raise FileNotFoundError(f"dataset image not found: {relative_path}")
    return candidate


def _split_counts(rows: Sequence[DatasetRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.split] = counts.get(row.split, 0) + 1
    return counts


def _verify_expected_count(
    split: str, actual: int, expected: int | None
) -> None:
    if expected is not None and actual != expected:
        raise ValueError(
            f"expected {expected} {split} images; manifest contains {actual}"
        )


def _validate_positive_optional(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _artifact_metadata(path: Path, row_count: int) -> dict[str, object]:
    return {
        "rows": row_count,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _write_csv(
    path: Path,
    *,
    rows: Sequence[dict[str, object]],
    fieldnames: Sequence[str],
) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
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
        description="Audit MTCNN crop and fallback behavior for Experiment 02."
    )
    experiment_01 = Path("experiments/outputs/01_embedding_robustness")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("identification_service/storage"),
    )
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
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/02_face_preprocessing"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("experiments/cache/02_face_preprocessing"),
    )
    parser.add_argument("--max-images-per-split", type=int)
    parser.add_argument("--expected-gallery-images", type=int)
    parser.add_argument("--expected-probe-images", type=int)
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_preprocessing_audit(
        asset_root=args.asset_root,
        dataset_manifest=args.dataset_manifest,
        dataset_audit=args.dataset_audit,
        output_dir=args.output_dir,
        cache_root=args.cache_root,
        max_images_per_split=args.max_images_per_split,
        expected_gallery_images=args.expected_gallery_images,
        expected_probe_images=args.expected_probe_images,
        progress_interval=args.progress_interval,
    )
    audit = result["audit"]
    print("\nPREPROCESSING AUDIT")
    print(f"Run scope: {audit['run_scope']}")
    print(f"Processed images: {audit['results']['processed_images']}")
    print(f"Detected images: {audit['results']['detected_images']}")
    print(f"Fallback images: {audit['results']['fallback_images']}")
    print(f"Coverage rate: {audit['results']['coverage_rate']:.3f}")
    for artifact_path in result["artifact_paths"]:
        print(f"[WRITE] {artifact_path}")
    print(f"[CACHE] {result['crop_dir']}")


if __name__ == "__main__":
    main()
