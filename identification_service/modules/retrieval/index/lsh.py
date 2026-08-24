"""Approximate embedding retrieval backed by FAISS locality-sensitive hashing."""

from __future__ import annotations

from threading import RLock
from typing import Sequence

from numpy.typing import ArrayLike

from identification_service.modules.retrieval.index.bruteforce import (
    ImageMatch,
    _normalize_embeddings,
    _validate_dimension,
    _validate_identities,
    _validate_limit,
)


class LSHCosineIndex:
    """Search normalized embeddings through compact binary hash signatures."""

    def __init__(self, dimension: int, *, hash_bits: int = 128) -> None:
        import faiss

        self.dimension = _validate_dimension(dimension)
        self.hash_bits = self._validate_hash_bits(hash_bits)
        self._index = faiss.IndexLSH(self.dimension, self.hash_bits)
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
        """Add a validated batch of normalized gallery embeddings."""

        normalized = _normalize_embeddings(
            embeddings, dimension=self.dimension, label="embeddings"
        )
        validated_identities = _validate_identities(identities)
        if normalized.shape[0] != len(validated_identities):
            raise ValueError(
                "the number of embeddings must match the number of identities"
            )

        with self._lock:
            self._index.add(normalized)
            self._identities.extend(validated_identities)

    def nearest_images(
        self, probe_embedding: ArrayLike, limit: int | None = None
    ) -> tuple[ImageMatch, ...]:
        """Return image matches ordered by binary-signature agreement."""

        probe = _normalize_embeddings(
            probe_embedding,
            dimension=self.dimension,
            label="probe_embedding",
        )
        if probe.shape[0] != 1:
            raise ValueError("probe_embedding must contain exactly one embedding")

        with self._lock:
            if not self._identities:
                raise ValueError("cannot search an empty index")
            result_limit = _validate_limit(limit, len(self._identities))
            distances, positions = self._index.search(probe, result_limit)
            identities = tuple(self._identities)

        return tuple(
            ImageMatch(
                gallery_position=int(position),
                identity=identities[int(position)],
                similarity=1.0 - (float(distance) / self.hash_bits),
            )
            for position, distance in zip(positions[0], distances[0])
        )

    @staticmethod
    def _validate_hash_bits(hash_bits: int) -> int:
        if isinstance(hash_bits, bool) or not isinstance(hash_bits, int):
            raise TypeError("hash_bits must be an integer")
        if hash_bits <= 0:
            raise ValueError("hash_bits must be greater than zero")
        return hash_bits
