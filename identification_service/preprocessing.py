"""Face localization and crop preparation for embedding extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps


BoxMatrix = NDArray[np.float64]
ProbabilityVector = NDArray[np.float64]


class FaceDetector(Protocol):
    """Minimal detector interface required by the preprocessor."""

    def detect(
        self, image: Image.Image
    ) -> tuple[BoxMatrix | None, ProbabilityVector | None]: ...


class MTCNNFaceDetector:
    """Adapter around the face detector provided by facenet-pytorch."""

    def __init__(self, *, device: str | None = None) -> None:
        from facenet_pytorch import MTCNN

        self._detector = MTCNN(keep_all=True, device=device)

    def detect(
        self, image: Image.Image
    ) -> tuple[BoxMatrix | None, ProbabilityVector | None]:
        boxes, probabilities = self._detector.detect(image)
        return boxes, probabilities


@dataclass(frozen=True, slots=True)
class PreprocessedFace:
    """A model-sized RGB crop and the decision that produced it."""

    image: Image.Image
    face_detected: bool
    confidence: float | None
    detection_box: tuple[float, float, float, float] | None
    crop_box: tuple[int, int, int, int]
    fallback_reason: str | None


class FacePreprocessor:
    """Select the largest detected face or fall back to the full image.

    The crop policy mirrors the evaluated configuration: a 160-by-160 output
    with a 20-pixel margin expressed in output-image coordinates. Detection and
    cropping remain separate from tensor standardization, which belongs to the
    embedding model boundary.
    """

    def __init__(
        self,
        detector: FaceDetector | None = None,
        *,
        image_size: int = 160,
        margin: int = 20,
    ) -> None:
        if isinstance(image_size, bool) or not isinstance(image_size, int):
            raise TypeError("image_size must be an integer")
        if image_size <= 0:
            raise ValueError("image_size must be greater than zero")
        if isinstance(margin, bool) or not isinstance(margin, int):
            raise TypeError("margin must be an integer")
        if margin < 0 or margin >= image_size:
            raise ValueError("margin must be non-negative and smaller than image_size")

        self.detector = detector if detector is not None else MTCNNFaceDetector()
        self.image_size = image_size
        self.margin = margin

    def process(self, image: Image.Image) -> PreprocessedFace:
        """Orient, convert, localize, crop, and resize one source image."""

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL image")

        rgb_image = ImageOps.exif_transpose(image).convert("RGB")

        try:
            boxes, probabilities = self.detector.detect(rgb_image)
        except Exception:
            return self._fallback(rgb_image, reason="detector_error")

        if boxes is None or probabilities is None:
            return self._fallback(rgb_image, reason="face_not_detected")

        validated_boxes, validated_probabilities = self._validate_detections(
            boxes, probabilities
        )
        selected_index = self._largest_box_index(validated_boxes)
        detection_box = validated_boxes[selected_index]
        crop_box = self._crop_box(
            detection_box,
            image_width=rgb_image.width,
            image_height=rgb_image.height,
        )
        cropped = rgb_image.crop(crop_box).resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )

        return PreprocessedFace(
            image=cropped,
            face_detected=True,
            confidence=float(validated_probabilities[selected_index]),
            detection_box=tuple(float(value) for value in detection_box),
            crop_box=crop_box,
            fallback_reason=None,
        )

    def _fallback(self, image: Image.Image, *, reason: str) -> PreprocessedFace:
        full_image_box = (0, 0, image.width, image.height)
        resized = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )
        return PreprocessedFace(
            image=resized,
            face_detected=False,
            confidence=None,
            detection_box=None,
            crop_box=full_image_box,
            fallback_reason=reason,
        )

    @staticmethod
    def _validate_detections(
        boxes: BoxMatrix, probabilities: ProbabilityVector
    ) -> tuple[BoxMatrix, ProbabilityVector]:
        box_array = np.asarray(boxes, dtype=np.float64)
        probability_array = np.asarray(probabilities, dtype=np.float64)

        if box_array.ndim != 2 or box_array.shape[1] != 4:
            raise ValueError("detector boxes must have shape (n, 4)")
        if probability_array.ndim != 1:
            raise ValueError("detector probabilities must have shape (n,)")
        if box_array.shape[0] == 0:
            raise ValueError("detector returned an empty box collection")
        if box_array.shape[0] != probability_array.shape[0]:
            raise ValueError("detector box and probability counts must match")
        if not np.isfinite(box_array).all() or not np.isfinite(
            probability_array
        ).all():
            raise ValueError("detector results must contain only finite values")

        widths = box_array[:, 2] - box_array[:, 0]
        heights = box_array[:, 3] - box_array[:, 1]
        if np.any(widths <= 0) or np.any(heights <= 0):
            raise ValueError("detector boxes must have positive width and height")
        if np.any(probability_array < 0) or np.any(probability_array > 1):
            raise ValueError("detector probabilities must be between zero and one")

        return box_array, probability_array

    @staticmethod
    def _largest_box_index(boxes: BoxMatrix) -> int:
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        return int(np.argmax(widths * heights))

    def _crop_box(
        self,
        detection_box: NDArray[np.float64],
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        left, top, right, bottom = detection_box
        width = right - left
        height = bottom - top
        horizontal_margin = self.margin * width / (self.image_size - self.margin)
        vertical_margin = self.margin * height / (self.image_size - self.margin)

        crop_box = (
            int(max(left - horizontal_margin / 2, 0)),
            int(max(top - vertical_margin / 2, 0)),
            int(min(right + horizontal_margin / 2, image_width)),
            int(min(bottom + vertical_margin / 2, image_height)),
        )
        crop_left, crop_top, crop_right, crop_bottom = crop_box
        if crop_right <= crop_left or crop_bottom <= crop_top:
            raise ValueError("selected face box does not intersect the source image")
        return crop_box
