import unittest

import numpy as np
from PIL import Image

from identification_service.embedding import FaceEmbeddingExtractor


class RecordingBackend:
    dimension = 512
    model_name = "test-backend"

    def __init__(self, output=None) -> None:
        self.output = (
            output
            if output is not None
            else np.ones((1, self.dimension), dtype=np.float32)
        )
        self.received_batch = None

    def encode(self, image_batch):
        self.received_batch = image_batch.copy()
        return self.output


class FaceEmbeddingExtractorTests(unittest.TestCase):
    def test_image_is_standardized_and_reordered_to_nchw(self) -> None:
        backend = RecordingBackend()
        extractor = FaceEmbeddingExtractor(backend=backend)
        image = Image.new("RGB", (160, 160), (0, 127, 255))

        embedding = extractor.encode(image)

        self.assertEqual(embedding.shape, (512,))
        self.assertEqual(embedding.dtype, np.float32)
        self.assertEqual(backend.received_batch.shape, (1, 3, 160, 160))
        np.testing.assert_allclose(
            backend.received_batch[0, :, 0, 0],
            np.array(
                [-0.99609375, -0.00390625, 0.99609375],
                dtype=np.float32,
            ),
        )

    def test_non_rgb_image_is_converted_before_encoding(self) -> None:
        backend = RecordingBackend()
        extractor = FaceEmbeddingExtractor(backend=backend)

        extractor.encode(Image.new("L", (160, 160), 255))

        self.assertEqual(backend.received_batch.shape, (1, 3, 160, 160))
        np.testing.assert_allclose(
            backend.received_batch[0, :, 0, 0],
            np.full(3, 0.99609375, dtype=np.float32),
        )

    def test_incorrect_image_size_is_rejected(self) -> None:
        extractor = FaceEmbeddingExtractor(backend=RecordingBackend())

        with self.assertRaisesRegex(ValueError, "must be 160x160"):
            extractor.encode(Image.new("RGB", (100, 160), "white"))

    def test_backend_output_shape_is_validated(self) -> None:
        backend = RecordingBackend(
            output=np.ones((2, 512), dtype=np.float32)
        )
        extractor = FaceEmbeddingExtractor(backend=backend)

        with self.assertRaisesRegex(ValueError, "must return shape"):
            extractor.encode(Image.new("RGB", (160, 160), "white"))

    def test_non_finite_embedding_is_rejected(self) -> None:
        output = np.ones((1, 512), dtype=np.float32)
        output[0, 0] = np.nan
        extractor = FaceEmbeddingExtractor(
            backend=RecordingBackend(output=output)
        )

        with self.assertRaisesRegex(ValueError, "non-finite"):
            extractor.encode(Image.new("RGB", (160, 160), "white"))

    def test_zero_embedding_is_rejected(self) -> None:
        extractor = FaceEmbeddingExtractor(
            backend=RecordingBackend(
                output=np.zeros((1, 512), dtype=np.float32)
            )
        )

        with self.assertRaisesRegex(ValueError, "zero-length"):
            extractor.encode(Image.new("RGB", (160, 160), "white"))

    def test_backend_dimension_must_match_facenet(self) -> None:
        backend = RecordingBackend()
        backend.dimension = 128

        with self.assertRaisesRegex(ValueError, "512 dimensions"):
            FaceEmbeddingExtractor(backend=backend)


if __name__ == "__main__":
    unittest.main()
