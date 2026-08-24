import unittest

import numpy as np
from PIL import Image

from identification_service.modules.extraction.preprocessing import (
    FacePreprocessor,
)


class StubDetector:
    def __init__(self, boxes=None, probabilities=None, error=None) -> None:
        self.boxes = boxes
        self.probabilities = probabilities
        self.error = error

    def detect(self, image):
        if self.error is not None:
            raise self.error
        return self.boxes, self.probabilities


class FacePreprocessorTests(unittest.TestCase):
    def test_largest_face_is_selected_and_resized(self) -> None:
        detector = StubDetector(
            boxes=np.array(
                [
                    [20.0, 20.0, 60.0, 60.0],
                    [50.0, 30.0, 150.0, 130.0],
                ]
            ),
            probabilities=np.array([0.99, 0.91]),
        )
        preprocessor = FacePreprocessor(detector=detector)

        result = preprocessor.process(Image.new("RGB", (200, 160), "white"))

        self.assertTrue(result.face_detected)
        self.assertEqual(result.image.size, (160, 160))
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.detection_box, (50.0, 30.0, 150.0, 130.0))
        self.assertEqual(result.crop_box, (42, 22, 157, 137))
        self.assertAlmostEqual(result.confidence, 0.91)
        self.assertIsNone(result.fallback_reason)

    def test_missing_detection_uses_full_image_fallback(self) -> None:
        preprocessor = FacePreprocessor(detector=StubDetector())

        result = preprocessor.process(Image.new("L", (80, 40), 128))

        self.assertFalse(result.face_detected)
        self.assertEqual(result.image.size, (160, 160))
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.crop_box, (0, 0, 80, 40))
        self.assertEqual(result.fallback_reason, "face_not_detected")

    def test_detector_error_is_reported_as_a_distinct_fallback(self) -> None:
        preprocessor = FacePreprocessor(
            detector=StubDetector(error=RuntimeError("detector unavailable"))
        )

        result = preprocessor.process(Image.new("RGB", (32, 48), "black"))

        self.assertFalse(result.face_detected)
        self.assertEqual(result.fallback_reason, "detector_error")
        self.assertEqual(result.crop_box, (0, 0, 32, 48))

    def test_malformed_detector_output_is_rejected(self) -> None:
        detector = StubDetector(
            boxes=np.array([[10.0, 10.0, 30.0]]),
            probabilities=np.array([0.9]),
        )
        preprocessor = FacePreprocessor(detector=detector)

        with self.assertRaisesRegex(ValueError, r"shape \(n, 4\)"):
            preprocessor.process(Image.new("RGB", (50, 50), "white"))

    def test_out_of_range_detection_probability_is_rejected(self) -> None:
        detector = StubDetector(
            boxes=np.array([[10.0, 10.0, 30.0, 30.0]]),
            probabilities=np.array([1.1]),
        )
        preprocessor = FacePreprocessor(detector=detector)

        with self.assertRaisesRegex(ValueError, "between zero and one"):
            preprocessor.process(Image.new("RGB", (50, 50), "white"))

    def test_face_box_outside_source_image_is_rejected(self) -> None:
        detector = StubDetector(
            boxes=np.array([[60.0, 60.0, 90.0, 90.0]]),
            probabilities=np.array([0.9]),
        )
        preprocessor = FacePreprocessor(detector=detector)

        with self.assertRaisesRegex(ValueError, "does not intersect"):
            preprocessor.process(Image.new("RGB", (50, 50), "white"))

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "smaller than image_size"):
            FacePreprocessor(
                detector=StubDetector(),
                image_size=160,
                margin=160,
            )


if __name__ == "__main__":
    unittest.main()
