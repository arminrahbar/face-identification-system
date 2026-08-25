from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "scripts"
    / "03_retrieval_configuration"
    / "01_analyze_configuration.py"
)
SPEC = spec_from_file_location("retrieval_configuration_analysis", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load Experiment 03 script: {SCRIPT_PATH}")
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Experiment03ConfigurationTests(unittest.TestCase):
    def test_retrieval_collapses_duplicate_images_into_distinct_identities(
        self,
    ) -> None:
        result = MODULE.evaluate_identity_retrieval(
            gallery_embeddings=np.asarray(
                [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32
            ),
            gallery_identities=np.asarray(
                ["alice", "alice", "bob"], dtype=np.str_
            ),
            probe_embeddings=np.asarray(
                [[0.98, 0.02], [0.0, 1.0]], dtype=np.float32
            ),
            probe_identities=np.asarray(["alice", "bob"], dtype=np.str_),
            probe_paths=np.asarray(
                ["probe/alice.jpg", "probe/bob.jpg"], dtype=np.str_
            ),
            max_candidates=2,
        )

        self.assertEqual(result["true_ranks"].tolist(), [1, 1])
        self.assertEqual(result["ranking_rows"][0]["rank_1"], "alice")
        self.assertEqual(result["ranking_rows"][0]["rank_2"], "bob")

    def test_candidate_policy_selects_shortest_list_meeting_target(self) -> None:
        curve = MODULE.build_topn_curve(
            np.asarray([1, 1, 2, 4], dtype=np.int32), max_candidates=4
        )

        selected = MODULE.select_candidate_count(curve, target_accuracy=0.75)

        self.assertEqual(selected, 2)
        self.assertEqual(curve[0]["newly_recovered_probes"], 2)
        self.assertEqual(curve[1]["newly_recovered_probes"], 1)
        self.assertEqual(curve[3]["candidate_review_slots"], 16)

    def test_preprocessing_baseline_must_match_exactly(self) -> None:
        curve = MODULE.build_topn_curve(
            np.asarray([1, 1, 3, 8], dtype=np.int32), max_candidates=10
        )
        expected = {
            "top_1": 0.5,
            "top_3": 0.75,
            "top_5": 0.75,
            "top_10": 1.0,
            "mrr": (1.0 + 1.0 + 1 / 3 + 1 / 8) / 4,
        }

        result = MODULE.verify_preprocessing_baseline(
            topn_curve=curve,
            mrr=float(expected["mrr"]),
            expected_metrics=expected,
        )

        self.assertEqual(result["status"], "matched")
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            MODULE.verify_preprocessing_baseline(
                topn_curve=curve,
                mrr=0.0,
                expected_metrics=expected,
            )

    def test_nested_samples_only_add_images_as_m_increases(self) -> None:
        samples = MODULE.build_nested_gallery_samples(
            gallery_positions={
                "alice": [0, 1, 2, 3, 4],
                "bob": [5, 6, 7, 8, 9],
            },
            identities=["alice", "bob"],
            max_gallery_images=5,
            rng=np.random.default_rng(1234),
        )

        for image_count in range(1, 5):
            self.assertTrue(
                set(samples[image_count]).issubset(
                    set(samples[image_count + 1])
                )
            )
            self.assertEqual(len(samples[image_count]), image_count * 2)

    def test_gallery_policy_selects_smallest_m_within_tolerance(self) -> None:
        rows = [
            {"gallery_images_per_identity": 1, "top_1_mean": 0.88},
            {"gallery_images_per_identity": 2, "top_1_mean": 0.94},
            {"gallery_images_per_identity": 3, "top_1_mean": 0.972},
            {"gallery_images_per_identity": 4, "top_1_mean": 0.978},
            {"gallery_images_per_identity": 5, "top_1_mean": 0.980},
        ]

        selected = MODULE.select_gallery_depth(rows, tolerance=0.01)

        self.assertEqual(selected, 3)

    def test_trial_summary_reports_incremental_gains(self) -> None:
        trial_rows = []
        for image_count, values in ((1, (0.8, 0.9)), (2, (0.9, 1.0))):
            for trial_index, top_1 in enumerate(values):
                trial_rows.append(
                    {
                        "gallery_images_per_identity": image_count,
                        "identity_count": 2,
                        "gallery_image_count": image_count * 2,
                        "top_1": top_1,
                        "top_3": 1.0,
                        "top_5": 1.0,
                        "top_10": 1.0,
                        "mrr": top_1,
                        "mean_true_identity_rank": 1.0,
                        "median_true_identity_rank": 1.0,
                        "trial_index": trial_index,
                        "seed": 10 + trial_index,
                        "probe_count": 2,
                    }
                )

        summary = MODULE.summarize_gallery_trials(
            trial_rows,
            min_gallery_images=1,
            max_gallery_images=2,
        )

        self.assertEqual(summary[0]["incremental_top_1_gain"], "")
        self.assertAlmostEqual(summary[1]["incremental_top_1_gain"], 0.1)
        self.assertGreater(summary[0]["top_1_std"], 0.0)


if __name__ == "__main__":
    unittest.main()
