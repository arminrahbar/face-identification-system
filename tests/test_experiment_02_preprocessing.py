from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from identification_service.modules.extraction.preprocessing import (
    PreprocessedFace,
)


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "scripts"
    / "02_face_preprocessing"
    / "01_audit_preprocessing.py"
)
SPEC = importlib.util.spec_from_file_location(
    "experiment_02_audit_preprocessing", SCRIPT_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load Experiment 02 script: {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class StubPreprocessor:
    def process(self, image: Image.Image) -> PreprocessedFace:
        if image.getpixel((0, 0))[0] == 0:
            return PreprocessedFace(
                image=image.resize((160, 160)),
                face_detected=False,
                confidence=None,
                detection_box=None,
                crop_box=(0, 0, image.width, image.height),
                fallback_reason="face_not_detected",
            )
        crop_box = (2, 1, image.width - 2, image.height - 1)
        return PreprocessedFace(
            image=image.crop(crop_box).resize((160, 160)),
            face_detected=True,
            confidence=0.95,
            detection_box=(3.0, 2.0, float(image.width - 3), float(image.height - 2)),
            crop_box=crop_box,
            fallback_reason=None,
        )


class Experiment02PreprocessingAuditTests(unittest.TestCase):
    def test_audit_writes_portable_verified_artifacts_and_lossless_crops(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            asset_root = root / "assets"
            output_dir = root / "outputs"
            cache_root = root / "cache"
            rows = []

            specifications = (
                ("gallery", "alice", "alice_1.jpg", (255, 0, 0)),
                ("gallery", "bob", "bob_1.jpg", (0, 0, 0)),
                ("probe", "alice", "alice_2.jpg", (255, 255, 255)),
                ("probe", "bob", "bob_2.jpg", (0, 0, 0)),
            )
            for split, identity, filename, color in specifications:
                path = asset_root / split / identity / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (20, 10), color).save(path)
                rows.append(
                    {
                        "split": split,
                        "identity": identity,
                        "relative_path": path.relative_to(asset_root).as_posix(),
                        "sha256": sha256(path),
                    }
                )

            manifest_path = root / "dataset_manifest.csv"
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["split", "identity", "relative_path", "sha256"],
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)

            audit_path = root / "dataset_audit.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "closed_set": True,
                        "manifest_sha256": sha256(manifest_path),
                    }
                ),
                encoding="utf-8",
            )

            result = MODULE.run_preprocessing_audit(
                asset_root=asset_root,
                dataset_manifest=manifest_path,
                dataset_audit=audit_path,
                output_dir=output_dir,
                cache_root=cache_root,
                expected_gallery_images=2,
                expected_probe_images=2,
                progress_interval=1,
                preprocessor=StubPreprocessor(),
            )

            self.assertEqual(result["audit"]["run_scope"], "full")
            self.assertEqual(result["audit"]["results"]["processed_images"], 4)
            self.assertEqual(result["audit"]["results"]["detected_images"], 2)
            self.assertEqual(result["audit"]["results"]["fallback_images"], 2)
            self.assertFalse((cache_root / ".crops.incomplete").exists())

            with (output_dir / "detection_records.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                records = list(csv.DictReader(handle))

            self.assertEqual(len(records), 4)
            self.assertTrue(
                all(not Path(row["source_relative_path"]).is_absolute() for row in records)
            )
            detected = [row for row in records if row["face_detected"] == "True"]
            fallback = [row for row in records if row["face_detected"] == "False"]
            self.assertEqual(len(detected), 2)
            self.assertTrue(all(row["crop_relative_path"] for row in detected))
            self.assertTrue(all(row["crop_sha256"] for row in detected))
            self.assertTrue(all(not row["crop_relative_path"] for row in fallback))
            self.assertTrue(
                all(
                    (cache_root / "crops" / row["crop_relative_path"]).is_file()
                    for row in detected
                )
            )

            manifest = json.loads(
                (output_dir / "preprocessing_audit.json").read_text(encoding="utf-8")
            )
            for artifact_name, metadata in manifest["artifacts"].items():
                self.assertEqual(
                    sha256(output_dir / artifact_name), metadata["sha256"]
                )

    def test_pilot_scope_selects_each_split_deterministically(self) -> None:
        rows = (
            MODULE.DatasetRow("probe", "bob", "probe/bob/2.jpg", "b"),
            MODULE.DatasetRow("gallery", "bob", "gallery/bob/2.jpg", "c"),
            MODULE.DatasetRow("probe", "alice", "probe/alice/1.jpg", "a"),
            MODULE.DatasetRow("gallery", "alice", "gallery/alice/1.jpg", "d"),
        )

        selected = MODULE.select_scope(rows, max_images_per_split=1)

        self.assertEqual(
            [(row.split, row.identity) for row in selected],
            [("gallery", "alice"), ("probe", "alice")],
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()
