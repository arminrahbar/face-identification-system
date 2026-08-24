"""In-memory embedding index used by the identification service."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatMatrix = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class ImageMatch:
    """One gallery-image match returned by the embedding index."""

    gallery_position: int
    identity: str
    similarity: float


class ExactCosineIndex:
    """Store normalized embeddings and rank them by exact cosine similarity.

    The index keeps one row per gallery image. Multiple rows may therefore share
    the same identity, which lets enrollment retain the additional evidence from
    multiple reference images instead of overwriting an existing identity.
    """

    def __init__(self, dimension: int) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError("dimension must be an integer")
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")

        self.dimension = dimension
        self._embeddings = np.empty((0, dimension), dtype=np.float32)
        self._identities: list[str] = []
        self._lock = RLock()

    @property
    def size(self) -> int:
        """Return the number of indexed gallery images."""

        with self._lock:
            return len(self._identities)

    @property
    def identity_count(self) -> int:
        """Return the number of distinct indexed identities."""

        with self._lock:
            return len(set(self._identities))

    def add(self, embeddings: ArrayLike, identities: Sequence[str]) -> None:
        """Add one or more gallery embeddings and their identity labels.

        The complete batch is validated before the index changes, so a malformed
        row cannot leave embeddings and identity metadata out of sync.
        """

        normalized = self._normalize_embeddings(embeddings, label="embeddings")
        validated_identities = self._validate_identities(identities)

        if normalized.shape[0] != len(validated_identities):
            raise ValueError(
                "the number of embeddings must match the number of identities"
            )

        with self._lock:
            self._embeddings = np.concatenate(
                (self._embeddings, normalized), axis=0
            )
            self._identities.extend(validated_identities)

    def nearest_images(
        self, probe_embedding: ArrayLike, limit: int | None = None
    ) -> tuple[ImageMatch, ...]:
        """Return gallery-image matches ordered by descending similarity."""

        probe = self._normalize_embeddings(
            probe_embedding, label="probe_embedding"
        )
        if probe.shape[0] != 1:
            raise ValueError("probe_embedding must contain exactly one embedding")

        with self._lock:
            if not self._identities:
                raise ValueError("cannot search an empty index")
            gallery = self._embeddings.copy()
            identities = tuple(self._identities)

        result_limit = self._validate_limit(limit, len(identities))
        similarities = gallery @ probe[0]
        positions = np.argsort(-similarities, kind="stable")[:result_limit]

        return tuple(
            ImageMatch(
                gallery_position=int(position),
                identity=identities[int(position)],
                similarity=float(similarities[int(position)]),
            )
            for position in positions
        )

    def _normalize_embeddings(
        self, values: ArrayLike, *, label: str
    ) -> FloatMatrix:
        embeddings = np.asarray(values, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.ndim != 2:
            raise ValueError(f"{label} must be a one- or two-dimensional array")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"{label} dimension must be {self.dimension}; "
                f"received {embeddings.shape[1]}"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError(f"{label} must contain only finite values")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError(f"{label} cannot contain a zero-length vector")

        return np.ascontiguousarray(embeddings / norms, dtype=np.float32)

    @staticmethod
    def _validate_identities(identities: Sequence[str]) -> list[str]:
        if isinstance(identities, (str, bytes)):
            raise TypeError("identities must be a sequence of identity strings")

        validated = list(identities)
        if not validated:
            raise ValueError("identities cannot be empty")
        if any(not isinstance(identity, str) for identity in validated):
            raise TypeError("every identity must be a string")
        if any(not identity.strip() for identity in validated):
            raise ValueError("identity values cannot be empty or whitespace")
        return validated

    @staticmethod
    def _validate_limit(limit: int | None, available: int) -> int:
        if limit is None:
            return available
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer or None")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        return min(limit, available)
