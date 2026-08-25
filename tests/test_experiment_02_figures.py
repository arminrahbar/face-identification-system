from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "scripts"
    / "02_face_preprocessing"
    / "03_generate_figures.py"
)
SPEC = spec_from_file_location("face_preprocessing_figures", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load experiment script: {SCRIPT_PATH}")
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FacePreprocessingFigureTests(unittest.TestCase):
    def test_canonical_results_pass_figure_input_validation(self) -> None:
        results = MODULE.load_verified_results(
            REPOSITORY_ROOT
            / "experiments"
            / "outputs"
            / "02_face_preprocessing"
        )

        self.assertEqual(len(results["detection_records"]), 3264)
        self.assertEqual(len(results["probe_case_analysis"]), 999)
        self.assertEqual(
            results["comparison_run"]["result"]["recommendation"],
            "adopt_mtcnn_crop_fallback",
        )

    def test_figure_set_contains_three_numbered_outputs(self) -> None:
        self.assertEqual(
            MODULE.FIGURE_FILENAMES,
            (
                "01_preprocessing_decision.png",
                "02_detection_audit.png",
                "03_rank_movement.png",
            ),
        )


if __name__ == "__main__":
    unittest.main()
