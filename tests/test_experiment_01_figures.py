from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "scripts"
    / "01_embedding_robustness"
    / "03_generate_figures.py"
)
SPEC = spec_from_file_location("embedding_robustness_figures", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load experiment script: {SCRIPT_PATH}")
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EmbeddingRobustnessFigureTests(unittest.TestCase):
    def test_canonical_results_pass_figure_input_validation(self) -> None:
        results = MODULE.load_verified_results(
            REPOSITORY_ROOT
            / "experiments"
            / "outputs"
            / "01_embedding_robustness"
        )

        self.assertEqual(len(results["condition_metrics"]), 10)
        self.assertEqual(len(results["probe_rankings"]), 9990)
        self.assertEqual(
            results["manifest"]["selection"]["selected_model"],
            "vggface2",
        )

    def test_figure_set_contains_three_numbered_outputs(self) -> None:
        self.assertEqual(
            MODULE.FIGURE_FILENAMES,
            (
                "01_checkpoint_comparison.png",
                "02_vggface2_robustness.png",
                "03_transformation_diagnostics.png",
            ),
        )


if __name__ == "__main__":
    unittest.main()
