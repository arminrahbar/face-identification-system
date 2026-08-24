import importlib.util
from io import BytesIO
import unittest

import numpy as np
from PIL import Image

from identification_service.modules.extraction.preprocessing import (
    PreprocessedFace,
)
from identification_service.modules.retrieval.index.bruteforce import (
    ExactCosineIndex,
)


FLASK_AVAILABLE = importlib.util.find_spec("flask") is not None


class StubPreprocessor:
    def process(self, image: Image.Image) -> PreprocessedFace:
        prepared = image.convert("RGB").resize((160, 160))
        return PreprocessedFace(
            image=prepared,
            face_detected=True,
            confidence=0.99,
            detection_box=(0.0, 0.0, float(image.width), float(image.height)),
            crop_box=(0, 0, image.width, image.height),
            fallback_reason=None,
        )


class StubEmbeddingExtractor:
    def encode(self, image: Image.Image) -> np.ndarray:
        red, green, _ = np.asarray(image, dtype=np.float32).mean(axis=(0, 1))
        if red >= green:
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)


@unittest.skipUnless(FLASK_AVAILABLE, "Flask is not installed")
class IdentificationApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        from identification_service.app import (
            IdentificationService,
            create_app,
        )

        self.index = ExactCosineIndex(dimension=2)
        service = IdentificationService(
            preprocessor=StubPreprocessor(),
            extractor=StubEmbeddingExtractor(),
            index=self.index,
        )
        application = create_app(service)
        application.config.update(TESTING=True)
        self.client = application.test_client()

    def test_enrollment_accepts_multiple_images_for_one_identity(self) -> None:
        first = self.client.post(
            "/add",
            data={"identity": "alice", "image": self._image_upload("red")},
        )
        second = self.client.post(
            "/add",
            data={"identity": "alice", "image": self._image_upload("red")},
        )

        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)
        self.assertEqual(second.get_json()["gallery_images"], 2)
        self.assertEqual(second.get_json()["distinct_identities"], 1)

    def test_identification_returns_ranked_distinct_identities(self) -> None:
        for identity, color in (
            ("alice", "red"),
            ("alice", "red"),
            ("bob", "green"),
        ):
            response = self.client.post(
                "/add",
                data={"identity": identity, "image": self._image_upload(color)},
            )
            self.assertEqual(response.status_code, 201)

        response = self.client.post(
            "/identify",
            data={"probe": self._image_upload("red"), "k": "2"},
        )
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["requested_top_k"], 2)
        self.assertEqual(payload["returned_matches"], 2)
        self.assertEqual(
            [match["identity"] for match in payload["matches"]],
            ["alice", "bob"],
        )

    def test_identification_rejects_an_empty_gallery(self) -> None:
        response = self.client.post(
            "/identify",
            data={"probe": self._image_upload("red")},
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()["error"]["code"], "empty_gallery")

    def test_invalid_top_k_is_rejected(self) -> None:
        for value in ("0", "-1", "not-an-integer"):
            with self.subTest(value=value):
                response = self.client.post(
                    "/identify",
                    data={"probe": self._image_upload("red"), "k": value},
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(
                    response.get_json()["error"]["code"], "invalid_top_k"
                )

    def test_missing_and_unreadable_images_are_rejected(self) -> None:
        missing = self.client.post("/add", data={"identity": "alice"})
        unreadable = self.client.post(
            "/add",
            data={
                "identity": "alice",
                "image": (BytesIO(b"not an image"), "face.jpg"),
            },
        )

        self.assertEqual(missing.status_code, 400)
        self.assertEqual(missing.get_json()["error"]["code"], "missing_image")
        self.assertEqual(unreadable.status_code, 400)
        self.assertEqual(
            unreadable.get_json()["error"]["code"], "invalid_image"
        )

    @staticmethod
    def _image_upload(color: str) -> tuple[BytesIO, str]:
        buffer = BytesIO()
        Image.new("RGB", (24, 24), color=color).save(buffer, format="PNG")
        buffer.seek(0)
        return buffer, "face.png"


if __name__ == "__main__":
    unittest.main()
