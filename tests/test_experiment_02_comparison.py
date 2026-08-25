from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "scripts"
    / "02_face_preprocessing"
    / "02_compare_retrieval.py"
)
SPEC = spec_from_file_location("face_preprocessing_comparison", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load Experiment 02 script: {SCRIPT_PATH}")
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Experiment02ComparisonTests(unittest.TestCase):
    def test_exact_retrieval_collapses_duplicate_gallery_identities(self) -> None:
        result = MODULE.evaluate_retrieval(
            gallery_embeddings=np.asarray(
                [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32
            ),
            gallery_identities=np.asarray(
                ["alice", "alice", "bob"], dtype=np.str_
            ),
            probe_embeddings=np.asarray(
                [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
            ),
            probe_identities=np.asarray(["alice", "bob"], dtype=np.str_),
            probe_paths=np.asarray(
                ["probe/alice/a.jpg", "probe/bob/b.jpg"], dtype=np.str_
            ),
        )

        self.assertEqual(result["metrics"]["top_1"], 1.0)
        self.assertEqual(result["probe_rows"][0]["rank_1"], "alice")
        self.assertEqual(result["probe_rows"][0]["rank_2"], "bob")

    def test_paired_analysis_counts_help_harm_and_rank_movement(self) -> None:
        full_rows = make_probe_rows(
            ranks=(1, 2, 1, 3),
            rank_ones=("a", "x", "c", "x"),
        )
        crop_rows = make_probe_rows(
            ranks=(1, 1, 2, 3),
            rank_ones=("a", "b", "x", "x"),
        )
        detections = {
            f"probe/{identity}/{identity}.jpg": {
                "face_detected": "True",
                "confidence": "0.99",
                "crop_area_ratio": "0.25",
                "fallback_reason": "",
            }
            for identity in ("a", "b", "c", "d")
        }

        result = MODULE.build_paired_analysis(
            full_rows=full_rows,
            crop_rows=crop_rows,
            detection_by_path=detections,
            full_probe_embeddings=np.eye(4, dtype=np.float32),
            crop_probe_embeddings=np.eye(4, dtype=np.float32),
        )

        summary = result["outcome_summary"]
        ranks = result["rank_summary"]
        self.assertEqual(summary["both_correct_count"], 1)
        self.assertEqual(summary["cropping_helps_count"], 1)
        self.assertEqual(summary["cropping_hurts_count"], 1)
        self.assertEqual(summary["neither_correct_count"], 1)
        self.assertEqual(summary["net_top_1_wins"], 0)
        self.assertEqual(summary["mcnemar_exact_two_sided_p_value"], 1.0)
        self.assertEqual(ranks["rank_improved_count"], 1)
        self.assertEqual(ranks["rank_worsened_count"], 1)
        self.assertEqual(ranks["rank_unchanged_count"], 2)

    def test_recommendation_requires_every_declared_adoption_condition(self) -> None:
        adopted = MODULE.determine_recommendation(
            top_1_delta=0.1,
            mrr_delta=0.08,
            bootstrap_ci_low=0.05,
            bootstrap_ci_high=0.15,
            mcnemar_p_value=0.001,
            coverage_rate=1.0,
        )
        inconclusive = MODULE.determine_recommendation(
            top_1_delta=0.1,
            mrr_delta=-0.01,
            bootstrap_ci_low=0.05,
            bootstrap_ci_high=0.15,
            mcnemar_p_value=0.001,
            coverage_rate=1.0,
        )

        self.assertEqual(adopted, "adopt_mtcnn_crop_fallback")
        self.assertEqual(inconclusive, "inconclusive")

    def test_crop_condition_uses_verified_crop_and_fallback_uses_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            asset_root = root / "assets"
            crop_root = root / "crops"
            source_path = asset_root / "probe" / "alice" / "source.jpg"
            crop_path = crop_root / "probe" / "alice" / "crop.png"
            source_path.parent.mkdir(parents=True)
            crop_path.parent.mkdir(parents=True)
            Image.new("RGB", (20, 10), "red").save(source_path)
            Image.new("RGB", (160, 160), "blue").save(crop_path)
            row = {
                "relative_path": "probe/alice/source.jpg",
                "sha256": MODULE._sha256_file(source_path),
            }
            detected = {
                "face_detected": "True",
                "crop_relative_path": "probe/alice/crop.png",
                "crop_sha256": MODULE._sha256_file(crop_path),
            }
            fallback = {
                "face_detected": "False",
                "crop_relative_path": "",
                "crop_sha256": "",
            }

            cropped = MODULE.load_model_image(
                row=row,
                condition="mtcnn_crop_fallback",
                asset_root=asset_root,
                crop_root=crop_root,
                detection=detected,
            )
            fallback_image = MODULE.load_model_image(
                row=row,
                condition="mtcnn_crop_fallback",
                asset_root=asset_root,
                crop_root=crop_root,
                detection=fallback,
            )

            self.assertEqual(cropped.size, (160, 160))
            self.assertEqual(cropped.getpixel((0, 0)), (0, 0, 255))
            self.assertEqual(fallback_image.size, (160, 160))
            self.assertEqual(fallback_image.getpixel((0, 0)), (254, 0, 0))

    def test_pilot_scope_retains_matching_gallery_images(self) -> None:
        rows = [
            {"split": "gallery", "identity": "a"},
            {"split": "gallery", "identity": "a"},
            {"split": "gallery", "identity": "b"},
            {"split": "gallery", "identity": "gallery_only"},
            {"split": "probe", "identity": "a"},
            {"split": "probe", "identity": "b"},
        ]

        selected = MODULE.select_scope(rows, max_identities=1)

        self.assertEqual({row["identity"] for row in selected}, {"a"})
        self.assertEqual(len(selected), 3)


def make_probe_rows(
    *, ranks: tuple[int, ...], rank_ones: tuple[str, ...]
) -> list[dict[str, object]]:
    identities = ("a", "b", "c", "d")
    rows = []
    for identity, rank, rank_one in zip(identities, ranks, rank_ones):
        row: dict[str, object] = {
            "probe_relative_path": f"probe/{identity}/{identity}.jpg",
            "true_identity": identity,
            "true_identity_rank": rank,
            "rank_1": rank_one,
        }
        rows.append(row)
    return rows


if __name__ == "__main__":
    unittest.main()
