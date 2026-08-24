"""FaceNet embedding extraction for preprocessed face images."""

from __future__ import annotations

from typing import Literal, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image


FloatArray = NDArray[np.float32]
PretrainedModel = Literal["vggface2", "casia-webface"]


class EmbeddingBackend(Protocol):
    """Model interface consumed by the image-to-embedding pipeline."""

    dimension: int
    model_name: str

    def encode(self, image_batch: FloatArray) -> FloatArray: ...


class InceptionResnetBackend:
    """PyTorch adapter for a pretrained InceptionResnetV1 model."""

    dimension = 512

    def __init__(
        self,
        *,
        pretrained: PretrainedModel = "vggface2",
        device: str | None = None,
    ) -> None:
        if pretrained not in ("vggface2", "casia-webface"):
            raise ValueError(
                "pretrained must be 'vggface2' or 'casia-webface'"
            )

        import torch
        from facenet_pytorch import InceptionResnetV1

        selected_device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = torch.device(selected_device)
        self.model_name = pretrained
        self._torch = torch
        self._model = (
            InceptionResnetV1(pretrained=pretrained)
            .eval()
            .to(self.device)
        )

    def encode(self, image_batch: FloatArray) -> FloatArray:
        """Run one standardized NCHW image batch through FaceNet."""

        tensor = self._torch.from_numpy(image_batch).to(self.device)
        with self._torch.inference_mode():
            embeddings = self._model(tensor)
        return embeddings.detach().cpu().numpy().astype(np.float32)


class FaceEmbeddingExtractor:
    """Convert a prepared face image into a validated FaceNet embedding."""

    def __init__(
        self,
        backend: EmbeddingBackend | None = None,
        *,
        image_size: int = 160,
    ) -> None:
        if isinstance(image_size, bool) or not isinstance(image_size, int):
            raise TypeError("image_size must be an integer")
        if image_size <= 0:
            raise ValueError("image_size must be greater than zero")

        self.backend = (
            backend if backend is not None else InceptionResnetBackend()
        )
        if self.backend.dimension != 512:
            raise ValueError("embedding backend must produce 512 dimensions")
        self.image_size = image_size

    @property
    def model_name(self) -> str:
        """Return the configured pretrained model identifier."""

        return self.backend.model_name

    def encode(self, image: Image.Image) -> FloatArray:
        """Standardize one RGB face image and return its embedding vector."""

        return np.ascontiguousarray(self.encode_batch([image])[0])

    def encode_batch(self, images: Sequence[Image.Image]) -> FloatArray:
        """Standardize a nonempty image batch and return one embedding per image."""

        if isinstance(images, (str, bytes)) or not isinstance(images, Sequence):
            raise TypeError("images must be a sequence of PIL images")
        if not images:
            raise ValueError("images cannot be empty")

        standardized_images: list[FloatArray] = []

        for image in images:
            standardized_images.append(self._standardize_image(image))

        image_batch = np.ascontiguousarray(
            np.stack(standardized_images, axis=0), dtype=np.float32
        )

        embeddings = np.asarray(
            self.backend.encode(image_batch), dtype=np.float32
        )
        expected_shape = (len(images), self.backend.dimension)
        if embeddings.shape != expected_shape:
            raise ValueError(
                "embedding backend must return shape "
                f"{expected_shape}; received {embeddings.shape}"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("embedding backend returned non-finite values")

        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(norms == 0):
            raise ValueError("embedding backend returned a zero-length vector")
        return np.ascontiguousarray(embeddings, dtype=np.float32)

    def _standardize_image(self, image: Image.Image) -> FloatArray:
        """Convert one model-sized image to a standardized CHW array."""

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL image")
        if image.size != (self.image_size, self.image_size):
            raise ValueError(
                f"image must be {self.image_size}x{self.image_size} pixels; "
                f"received {image.width}x{image.height}"
            )

        rgb_image = image.convert("RGB")
        pixels = np.asarray(rgb_image, dtype=np.float32)
        standardized = (pixels - 127.5) / 128.0
        return np.ascontiguousarray(
            standardized.transpose(2, 0, 1), dtype=np.float32
        )
