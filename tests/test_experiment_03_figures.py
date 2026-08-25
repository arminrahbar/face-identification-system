from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "scripts"
    / "03_retrieval_configuration"
    / "02_generate_figures.py"
)
SPEC = spec_from_file_location("retrieval_configuration_figures", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load Experiment 03 figure script: {SCRIPT_PATH}")
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RetrievalConfigurationFigureTests(unittest.TestCase):
    def test_canonical_results_pass_figure_input_validation(self) -> None:
        results = MODULE.load_verified_results(
            REPOSITORY_ROOT
            / "experiments"
            / "outputs"
            / "03_retrieval_configuration"
        )

        self.assertEqual(
            results["manifest"]["candidate_list"]["selected_candidate_count"],
            5,
        )
        self.assertEqual(
            results["manifest"]["gallery_depth"][
                "selected_gallery_images_per_identity"
            ],
            2,
        )
        self.assertEqual(len(results["gallery_m_trial_metrics"]), 150)

    def test_figure_set_contains_three_numbered_outputs(self) -> None:
        self.assertEqual(
            MODULE.FIGURE_FILENAMES,
            (
                "01_candidate_list_decision.png",
                "02_gallery_depth_decision.png",
                "03_gallery_population_context.png",
            ),
        )


if __name__ == "__main__":
    unittest.main()
