"""Approximate cosine retrieval backed by a FAISS HNSW graph."""

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


class HNSWCosineIndex:
    """Search normalized embeddings with a hierarchical navigable graph."""

    def __init__(
        self,
        dimension: int,
        *,
        graph_degree: int = 32,
        construction_ef: int = 40,
        search_ef: int = 16,
    ) -> None:
        import faiss

        self.dimension = _validate_dimension(dimension)
        self.graph_degree = self._validate_setting(
            graph_degree, label="graph_degree"
        )
        self.construction_ef = self._validate_setting(
            construction_ef, label="construction_ef"
        )
        self.search_ef = self._validate_setting(search_ef, label="search_ef")

        self._index = faiss.IndexHNSWFlat(
            self.dimension,
            self.graph_degree,
            faiss.METRIC_INNER_PRODUCT,
        )
        self._index.hnsw.efConstruction = self.construction_ef
        self._index.hnsw.efSearch = self.search_ef
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
        """Return approximate image matches ordered by cosine similarity."""

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
            similarities, positions = self._index.search(probe, result_limit)
            identities = tuple(self._identities)

        return tuple(
            ImageMatch(
                gallery_position=int(position),
                identity=identities[int(position)],
                similarity=float(similarity),
            )
            for position, similarity in zip(positions[0], similarities[0])
        )

    @staticmethod
    def _validate_setting(value: int, *, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{label} must be an integer")
        if value <= 0:
            raise ValueError(f"{label} must be greater than zero")
        return value
