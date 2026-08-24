"""Validate and inventory the dataset used by the embedding robustness study."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Sequence

from PIL import Image


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """Portable metadata for one validated dataset image."""

    split: str
    identity: str
    relative_path: str
    filename: str
    width: int
    height: int
    image_format: str
    color_mode: str
    file_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SplitInventory:
    """Validated image records and exclusion counts for one split."""

    records: tuple[ImageRecord, ...]
    ignored_image_like_files: int
    ignored_other_files: int


def build_dataset_inventory(
    asset_root: Path,
    output_dir: Path,
    *,
    expected_gallery_images: int | None = None,
    expected_probe_images: int | None = None,
) -> dict[str, object]:
    """Validate both splits and write deterministic inventory artifacts."""

    asset_root = asset_root.resolve()
    gallery_root = asset_root / "multi_image_gallery"
    probe_root = asset_root / "probe"

    gallery = collect_split(gallery_root, asset_root, "gallery")
    probe = collect_split(probe_root, asset_root, "probe")

    _verify_expected_count(
        "gallery", len(gallery.records), expected_gallery_images
    )
    _verify_expected_count("probe", len(probe.records), expected_probe_images)

    gallery_identities = {record.identity for record in gallery.records}
    probe_identities = {record.identity for record in probe.records}
    missing_probe_identities = sorted(probe_identities - gallery_identities)
    if missing_probe_identities:
        preview = ", ".join(missing_probe_identities[:5])
        raise ValueError(
            "probe identities must all appear in the gallery; "
            f"missing {len(missing_probe_identities)}: {preview}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "dataset_manifest.csv"
    summary_path = output_dir / "dataset_summary.csv"
    distribution_path = output_dir / "gallery_image_count_distribution.csv"
    audit_path = output_dir / "dataset_audit.json"

    all_records = tuple(
        sorted(
            gallery.records + probe.records,
            key=lambda record: (
                record.split,
                record.identity,
                record.relative_path,
            ),
        )
    )
    _write_csv(
        manifest_path,
        rows=[asdict(record) for record in all_records],
        fieldnames=list(asdict(all_records[0])),
    )

    summary_rows = [
        summarize_split("gallery", gallery),
        summarize_split("probe", probe),
    ]
    _write_csv(
        summary_path,
        rows=summary_rows,
        fieldnames=list(summary_rows[0]),
    )

    gallery_counts = _identity_counts(gallery.records)
    distribution_rows = [
        {
            "gallery_images_per_identity": image_count,
            "identity_count": sum(
                1 for count in gallery_counts.values() if count == image_count
            ),
        }
        for image_count in sorted(set(gallery_counts.values()))
    ]
    _write_csv(
        distribution_path,
        rows=distribution_rows,
        fieldnames=["gallery_images_per_identity", "identity_count"],
    )

    manifest_sha256 = _sha256_file(manifest_path)
    audit = {
        "status": "complete",
        "closed_set": True,
        "gallery_images": len(gallery.records),
        "gallery_identities": len(gallery_identities),
        "probe_images": len(probe.records),
        "probe_identities": len(probe_identities),
        "probe_identities_missing_from_gallery": 0,
        "manifest_sha256": manifest_sha256,
        "artifacts": {
            manifest_path.name: _sha256_file(manifest_path),
            summary_path.name: _sha256_file(summary_path),
            distribution_path.name: _sha256_file(distribution_path),
        },
    }
    _write_json(audit_path, audit)

    return {
        "audit": audit,
        "artifact_paths": (
            manifest_path,
            summary_path,
            distribution_path,
            audit_path,
        ),
    }


def collect_split(
    split_root: Path, asset_root: Path, split_name: str
) -> SplitInventory:
    """Collect validated images from identity directories in one split."""

    if not split_root.is_dir():
        raise FileNotFoundError(f"split directory not found: {split_root}")

    records: list[ImageRecord] = []
    ignored_image_like_files = 0
    ignored_other_files = 0

    for entry in sorted(split_root.iterdir(), key=lambda path: path.name):
        if entry.name.startswith("."):
            ignored_image_like_files += int(
                entry.is_file()
                and entry.suffix.lower() in VALID_IMAGE_EXTENSIONS
            )
            ignored_other_files += int(
                entry.is_file()
                and entry.suffix.lower() not in VALID_IMAGE_EXTENSIONS
            )
            continue
        if entry.is_file():
            ignored_other_files += 1
            continue
        if not entry.is_dir():
            raise ValueError(f"unsupported split entry: {entry}")

        identity = entry.name.strip()
        if not identity:
            raise ValueError(f"identity directory cannot be empty: {entry}")

        for image_path in sorted(entry.iterdir(), key=lambda path: path.name):
            if image_path.is_symlink():
                raise ValueError(f"symbolic links are not supported: {image_path}")
            if image_path.name.startswith("."):
                ignored_image_like_files += int(
                    image_path.is_file()
                    and image_path.suffix.lower() in VALID_IMAGE_EXTENSIONS
                )
                ignored_other_files += int(
                    image_path.is_file()
                    and image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS
                )
                continue
            if image_path.is_dir():
                raise ValueError(
                    "nested directories are not supported inside identity folders: "
                    f"{image_path}"
                )
            if image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                ignored_other_files += int(image_path.is_file())
                continue
            if not image_path.is_file():
                raise ValueError(f"unsupported image entry: {image_path}")

            records.append(
                inspect_image(
                    image_path,
                    asset_root=asset_root,
                    split_name=split_name,
                    identity=identity,
                )
            )

    if not records:
        raise ValueError(f"no usable images found in split: {split_root}")

    return SplitInventory(
        records=tuple(records),
        ignored_image_like_files=ignored_image_like_files,
        ignored_other_files=ignored_other_files,
    )


def inspect_image(
    image_path: Path,
    *,
    asset_root: Path,
    split_name: str,
    identity: str,
) -> ImageRecord:
    """Decode one image, capture metadata, and compute its content hash."""

    try:
        with Image.open(image_path) as image:
            width, height = image.size
            image_format = image.format or "unknown"
            color_mode = image.mode
            image.verify()
    except Exception as error:
        relative_path = image_path.relative_to(asset_root).as_posix()
        raise ValueError(f"unreadable image: {relative_path}") from error

    return ImageRecord(
        split=split_name,
        identity=identity,
        relative_path=image_path.relative_to(asset_root).as_posix(),
        filename=image_path.name,
        width=width,
        height=height,
        image_format=image_format,
        color_mode=color_mode,
        file_bytes=image_path.stat().st_size,
        sha256=_sha256_file(image_path),
    )


def summarize_split(split_name: str, inventory: SplitInventory) -> dict[str, object]:
    """Return one tabular summary row for a split inventory."""

    counts = tuple(_identity_counts(inventory.records).values())
    return {
        "split": split_name,
        "image_count": len(inventory.records),
        "identity_count": len(counts),
        "min_images_per_identity": min(counts),
        "median_images_per_identity": median(counts),
        "max_images_per_identity": max(counts),
        "ignored_image_like_files": inventory.ignored_image_like_files,
        "ignored_other_files": inventory.ignored_other_files,
    }


def _identity_counts(records: Sequence[ImageRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.identity] = counts.get(record.identity, 0) + 1
    return counts


def _verify_expected_count(
    split_name: str, actual: int, expected: int | None
) -> None:
    if expected is not None and actual != expected:
        raise ValueError(
            f"expected {expected} {split_name} images; discovered {actual}"
        )


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
        description="Validate and inventory face-identification experiment data."
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("identification_service/storage"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/01_embedding_robustness"),
    )
    parser.add_argument("--expected-gallery-images", type=int)
    parser.add_argument("--expected-probe-images", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_dataset_inventory(
        args.asset_root,
        args.output_dir,
        expected_gallery_images=args.expected_gallery_images,
        expected_probe_images=args.expected_probe_images,
    )
    audit = result["audit"]

    print(f"Gallery images: {audit['gallery_images']}")
    print(f"Gallery identities: {audit['gallery_identities']}")
    print(f"Probe images: {audit['probe_images']}")
    print(f"Probe identities: {audit['probe_identities']}")
    print(f"Closed-set validation: {audit['closed_set']}")
    print(f"Manifest SHA-256: {audit['manifest_sha256']}")
    for artifact_path in result["artifact_paths"]:
        print(f"[WRITE] {artifact_path}")


if __name__ == "__main__":
    main()
