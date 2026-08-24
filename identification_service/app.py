"""HTTP interface for gallery enrollment and probe identification."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import BinaryIO, Protocol, Sequence

from flask import Flask, jsonify, request
from numpy.typing import ArrayLike
from PIL import Image, UnidentifiedImageError

from identification_service.modules.extraction.embedding import (
    FaceEmbeddingExtractor,
)
from identification_service.modules.extraction.preprocessing import (
    FacePreprocessor,
    PreprocessedFace,
)
from identification_service.modules.retrieval.index.bruteforce import (
    ExactCosineIndex,
)
from identification_service.modules.retrieval.search import (
    IdentityMatch,
    ImageSearchIndex,
    rank_identities,
)


DEFAULT_TOP_K = 3
EMBEDDING_DIMENSION = 512


class FacePreparation(Protocol):
    """Image-preparation behavior required by the application workflow."""

    def process(self, image: Image.Image) -> PreprocessedFace: ...


class EmbeddingExtraction(Protocol):
    """Embedding behavior required by the application workflow."""

    def encode(self, image: Image.Image) -> ArrayLike: ...


class GalleryIndex(ImageSearchIndex, Protocol):
    """Mutable gallery index required by enrollment and identification."""

    @property
    def size(self) -> int: ...

    @property
    def identity_count(self) -> int: ...

    def add(self, embeddings: ArrayLike, identities: Sequence[str]) -> None: ...


@dataclass(frozen=True, slots=True)
class EnrollmentResult:
    """Outcome of adding one gallery image to the index."""

    preprocessing: PreprocessedFace
    gallery_images: int
    distinct_identities: int


@dataclass(frozen=True, slots=True)
class IdentificationResult:
    """Outcome of ranking the gallery for one probe image."""

    preprocessing: PreprocessedFace
    matches: tuple[IdentityMatch, ...]


class EmptyGalleryError(RuntimeError):
    """Raised when identification is requested before enrollment."""


class IdentificationService:
    """Coordinate preprocessing, embedding extraction, and retrieval."""

    def __init__(
        self,
        preprocessor: FacePreparation,
        extractor: EmbeddingExtraction,
        index: GalleryIndex,
    ) -> None:
        self.preprocessor = preprocessor
        self.extractor = extractor
        self.index = index
        self._inference_lock = Lock()

    def enroll(self, image: Image.Image, identity: str) -> EnrollmentResult:
        """Extract and index one gallery image for an identity."""

        prepared, embedding = self._prepare_embedding(image)
        self.index.add(embedding, [identity])
        return EnrollmentResult(
            preprocessing=prepared,
            gallery_images=self.index.size,
            distinct_identities=self.index.identity_count,
        )

    def identify(
        self, image: Image.Image, *, top_k: int = DEFAULT_TOP_K
    ) -> IdentificationResult:
        """Return the best distinct gallery identities for a probe image."""

        if self.index.size == 0:
            raise EmptyGalleryError("gallery contains no enrolled images")

        prepared, embedding = self._prepare_embedding(image)
        matches = rank_identities(self.index, embedding, top_k=top_k)
        return IdentificationResult(preprocessing=prepared, matches=matches)

    def _prepare_embedding(
        self, image: Image.Image
    ) -> tuple[PreprocessedFace, ArrayLike]:
        with self._inference_lock:
            prepared = self.preprocessor.process(image)
            embedding = self.extractor.encode(prepared.image)
        return prepared, embedding


def create_default_service() -> IdentificationService:
    """Build the production service with the selected runtime components."""

    return IdentificationService(
        preprocessor=FacePreprocessor(image_size=160, margin=20),
        extractor=FaceEmbeddingExtractor(image_size=160),
        index=ExactCosineIndex(dimension=EMBEDDING_DIMENSION),
    )


def create_app(service: IdentificationService | None = None) -> Flask:
    """Create the Flask application, optionally with injected dependencies."""

    application = Flask(__name__)
    service_lock = Lock()
    configured_service = service

    def get_service() -> IdentificationService:
        nonlocal configured_service

        if configured_service is None:
            with service_lock:
                if configured_service is None:
                    configured_service = create_default_service()
        return configured_service

    @application.post("/add")
    def add_gallery_image():
        uploaded_image = request.files.get("image")
        if uploaded_image is None:
            return _error_response(
                "missing_image", "multipart field 'image' is required", 400
            )
        if not uploaded_image.filename:
            return _error_response(
                "empty_filename", "an image file must be selected", 400
            )

        identity = request.form.get("identity") or request.form.get("name")
        if identity is None or not identity.strip():
            return _error_response(
                "missing_identity",
                "form field 'identity' is required",
                400,
            )
        identity = identity.strip()

        try:
            image = _read_image(uploaded_image.stream)
        except (UnidentifiedImageError, OSError, ValueError):
            return _error_response(
                "invalid_image", "uploaded file is not a readable image", 400
            )

        try:
            result = get_service().enroll(image, identity)
        except Exception:
            application.logger.exception("gallery enrollment failed")
            return _error_response(
                "enrollment_failed", "unable to enroll gallery image", 500
            )

        return (
            jsonify(
                {
                    "identity": identity,
                    "gallery_images": result.gallery_images,
                    "distinct_identities": result.distinct_identities,
                    "preprocessing": _preprocessing_payload(
                        result.preprocessing
                    ),
                }
            ),
            201,
        )

    @application.post("/identify")
    def identify_probe():
        uploaded_image = request.files.get("probe")
        if uploaded_image is None:
            uploaded_image = request.files.get("image")
        if uploaded_image is None:
            return _error_response(
                "missing_probe",
                "multipart field 'probe' is required",
                400,
            )
        if not uploaded_image.filename:
            return _error_response(
                "empty_filename", "an image file must be selected", 400
            )

        try:
            top_k = _parse_top_k(request.form.get("k"))
        except ValueError as error:
            return _error_response("invalid_top_k", str(error), 400)

        try:
            image = _read_image(uploaded_image.stream)
        except (UnidentifiedImageError, OSError, ValueError):
            return _error_response(
                "invalid_image", "uploaded file is not a readable image", 400
            )

        try:
            result = get_service().identify(image, top_k=top_k)
        except EmptyGalleryError:
            return _error_response(
                "empty_gallery",
                "enroll at least one gallery image before identification",
                409,
            )
        except Exception:
            application.logger.exception("probe identification failed")
            return _error_response(
                "identification_failed", "unable to identify probe image", 500
            )

        return jsonify(
            {
                "requested_top_k": top_k,
                "returned_matches": len(result.matches),
                "matches": [
                    {
                        "rank": match.rank,
                        "identity": match.identity,
                        "similarity": match.similarity,
                    }
                    for match in result.matches
                ],
                "preprocessing": _preprocessing_payload(result.preprocessing),
            }
        )

    return application


def _read_image(stream: BinaryIO) -> Image.Image:
    with Image.open(stream) as source_image:
        source_image.load()
        return source_image.copy()


def _parse_top_k(raw_value: str | None) -> int:
    if raw_value is None:
        return DEFAULT_TOP_K
    try:
        top_k = int(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError("form field 'k' must be a positive integer") from error
    if top_k <= 0:
        raise ValueError("form field 'k' must be a positive integer")
    return top_k


def _preprocessing_payload(result: PreprocessedFace) -> dict[str, object]:
    return {
        "face_detected": result.face_detected,
        "confidence": result.confidence,
        "fallback_reason": result.fallback_reason,
    }


def _error_response(code: str, message: str, status: int):
    return jsonify({"error": {"code": code, "message": message}}), status


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
