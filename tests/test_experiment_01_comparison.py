from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "scripts"
    / "01_embedding_robustness"
    / "02_run_comparison.py"
)
SPEC = spec_from_file_location("embedding_robustness_comparison", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load experiment script: {SCRIPT_PATH}")
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EmbeddingRobustnessComparisonTests(unittest.TestCase):
    def test_resize_degradation_has_the_documented_intermediate_size(self) -> None:
        transformed = MODULE.resize_down_64_up_250(
            Image.new("RGB", (320, 240), "white")
        )

        self.assertEqual(transformed.size, (250, 250))

    def test_exact_retrieval_collapses_duplicate_gallery_identities(self) -> None:
        evaluation = MODULE.evaluate_retrieval(
            gallery_embeddings=np.asarray(
                [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]],
                dtype=np.float32,
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

        metrics = evaluation["metrics"]
        probe_rows = evaluation["probe_rows"]
        self.assertEqual(metrics["top_1"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertEqual(probe_rows[0]["rank_1"], "alice")
        self.assertEqual(probe_rows[0]["rank_2"], "bob")

    def test_stability_reports_rank_changes_retention_and_drift(self) -> None:
        clean_rows = [
            {
                "probe_relative_path": "probe/alice/a.jpg",
                "rank_1": "alice",
                "true_identity_rank": 1,
            },
            {
                "probe_relative_path": "probe/bob/b.jpg",
                "rank_1": "alice",
                "true_identity_rank": 2,
            },
        ]
        degraded_rows = [
            {
                "probe_relative_path": "probe/alice/a.jpg",
                "rank_1": "bob",
                "true_identity_rank": 2,
            },
            {
                "probe_relative_path": "probe/bob/b.jpg",
                "rank_1": "bob",
                "true_identity_rank": 1,
            },
        ]

        result = MODULE.calculate_stability(
            model_name="vggface2",
            condition_name="brightness_1_40",
            clean_probe_rows=clean_rows,
            degraded_probe_rows=degraded_rows,
            clean_embeddings=np.asarray(
                [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
            ),
            degraded_embeddings=np.asarray(
                [[0.0, 1.0], [0.0, 1.0]], dtype=np.float32
            ),
        )

        self.assertEqual(result["rank_1_changed_rate"], 1.0)
        self.assertEqual(result["clean_correct_count"], 1)
        self.assertEqual(result["clean_correct_retained_count"], 0)
        self.assertEqual(result["clean_correct_retention"], 0.0)
        self.assertAlmostEqual(result["mean_cosine_embedding_drift"], 0.5)

    def test_pilot_scope_selects_probe_identities_and_matching_gallery(self) -> None:
        rows = [
            {"split": "gallery", "identity": "alice"},
            {"split": "gallery", "identity": "bob"},
            {"split": "gallery", "identity": "gallery_only"},
            {"split": "probe", "identity": "alice"},
            {"split": "probe", "identity": "bob"},
        ]

        selected = MODULE.select_scope(rows, max_identities=1)

        self.assertEqual({row["identity"] for row in selected}, {"alice"})
        self.assertEqual(len(selected), 2)

    def test_image_quality_uses_mean_intensity_terminology(self) -> None:
        quality = {
            condition: {
                "mean_intensity": np.asarray([0.25, 0.75], dtype=np.float32),
                "zero_fraction": np.asarray([0.1, 0.2], dtype=np.float32),
                "saturated_fraction": np.asarray(
                    [0.01, 0.03], dtype=np.float32
                ),
            }
            for condition in MODULE.CONDITIONS
        }

        rows = MODULE.summarize_image_quality(quality)

        self.assertEqual(rows[0]["mean_intensity"], 0.5)
        self.assertNotIn("mean_luminance", rows[0])

    def test_model_selection_uses_declared_robustness_policy(self) -> None:
        metric_rows = []
        stability_rows = []
        scores = {
            "vggface2": [0.9, 0.8, 0.8, 0.7, 0.8],
            "casia-webface": [0.7, 0.6, 0.5, 0.6, 0.6],
        }

        for model_name, top_1_values in scores.items():
            for condition, top_1 in zip(MODULE.CONDITIONS, top_1_values):
                metric_rows.append(
                    {
                        "model": model_name,
                        "condition": condition,
                        "top_1": top_1,
                        "mrr": top_1 + 0.05,
                    }
                )
                if condition != "clean":
                    stability_rows.append(
                        {
                            "model": model_name,
                            "condition": condition,
                            "rank_1_changed_rate": 0.1,
                            "mean_cosine_embedding_drift": 0.05,
                        }
                    )

        summaries, selected_model = MODULE.summarize_models(
            metric_rows=metric_rows,
            stability_rows=stability_rows,
        )

        self.assertEqual(selected_model, "vggface2")
        self.assertEqual(summaries[0]["selection_rank"], 1)
        self.assertTrue(summaries[0]["selected"])
        self.assertEqual(summaries[0]["worst_condition_top_1"], 0.7)

    def test_model_selection_reports_an_exact_tie(self) -> None:
        metric_rows = []
        stability_rows = []
        for model_name in MODULE.MODELS:
            for condition in MODULE.CONDITIONS:
                metric_rows.append(
                    {
                        "model": model_name,
                        "condition": condition,
                        "top_1": 0.8,
                        "mrr": 0.9,
                    }
                )
                if condition != "clean":
                    stability_rows.append(
                        {
                            "model": model_name,
                            "condition": condition,
                            "rank_1_changed_rate": 0.1,
                            "mean_cosine_embedding_drift": 0.05,
                        }
                    )

        summaries, selected_model = MODULE.summarize_models(
            metric_rows=metric_rows,
            stability_rows=stability_rows,
        )

        self.assertIsNone(selected_model)
        self.assertFalse(any(row["selected"] for row in summaries))


if __name__ == "__main__":
    unittest.main()
