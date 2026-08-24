from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from PIL import Image


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "scripts"
    / "01_embedding_robustness"
    / "01_prepare_dataset.py"
)
SPEC = spec_from_file_location("embedding_dataset_preparation", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load experiment script: {SCRIPT_PATH}")
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DatasetPreparationTests(unittest.TestCase):
    def test_inventory_is_closed_set_and_excludes_appledouble_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            output = root / "output"

            self._write_image(
                assets / "multi_image_gallery" / "alice" / "alice_1.jpg"
            )
            self._write_image(
                assets / "multi_image_gallery" / "alice" / "alice_2.png"
            )
            self._write_image(assets / "probe" / "alice" / "alice_3.jpg")
            self._write_bytes(
                assets / "multi_image_gallery" / "alice" / "._alice_1.jpg",
                b"metadata",
            )

            result = MODULE.build_dataset_inventory(
                assets,
                output,
                expected_gallery_images=2,
                expected_probe_images=1,
            )

            audit = result["audit"]
            self.assertTrue(audit["closed_set"])
            self.assertEqual(audit["gallery_images"], 2)
            self.assertEqual(audit["probe_images"], 1)
            self.assertEqual(
                {path.name for path in result["artifact_paths"]},
                {
                    "dataset_manifest.csv",
                    "dataset_summary.csv",
                    "gallery_image_count_distribution.csv",
                    "dataset_audit.json",
                },
            )

            summary = (output / "dataset_summary.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("ignored_image_like_files", summary)
            self.assertNotIn("._alice_1.jpg", summary)

    def test_probe_identity_missing_from_gallery_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            assets = Path(directory) / "assets"
            self._write_image(
                assets / "multi_image_gallery" / "alice" / "alice.jpg"
            )
            self._write_image(assets / "probe" / "bob" / "bob.jpg")

            with self.assertRaisesRegex(ValueError, "all appear in the gallery"):
                MODULE.build_dataset_inventory(
                    assets, Path(directory) / "output"
                )

    def test_unreadable_image_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            assets = Path(directory) / "assets"
            self._write_bytes(
                assets / "multi_image_gallery" / "alice" / "alice.jpg",
                b"not an image",
            )
            self._write_image(assets / "probe" / "alice" / "alice.jpg")

            with self.assertRaisesRegex(ValueError, "unreadable image"):
                MODULE.build_dataset_inventory(
                    assets, Path(directory) / "output"
                )

    @staticmethod
    def _write_image(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), "navy").save(path)

    @staticmethod
    def _write_bytes(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


if __name__ == "__main__":
    unittest.main()
