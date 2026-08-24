"""Identity-level ranking built on image-level embedding matches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from numpy.typing import ArrayLike

from identification_service.modules.retrieval.index.bruteforce import (
    ImageMatch,
)


class ImageSearchIndex(Protocol):
    """Common search behavior provided by every retrieval backend."""

    def nearest_images(
        self, probe_embedding: ArrayLike, limit: int | None = None
    ) -> tuple[ImageMatch, ...]: ...


@dataclass(frozen=True, slots=True)
class IdentityMatch:
    """The best gallery evidence found for one identity."""

    rank: int
    identity: str
    similarity: float
    gallery_position: int


def rank_identities(
    index: ImageSearchIndex,
    probe_embedding: ArrayLike,
    *,
    top_k: int = 3,
) -> tuple[IdentityMatch, ...]:
    """Return the highest-scoring distinct identities for a probe embedding.

    Image-level matches are already ordered by cosine similarity. Keeping the
    first occurrence of each identity is therefore equivalent to assigning that
    identity its best gallery-image score.
    """

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    ranked: list[IdentityMatch] = []
    seen_identities: set[str] = set()

    for image_match in index.nearest_images(probe_embedding):
        if image_match.identity in seen_identities:
            continue

        seen_identities.add(image_match.identity)
        ranked.append(
            IdentityMatch(
                rank=len(ranked) + 1,
                identity=image_match.identity,
                similarity=image_match.similarity,
                gallery_position=image_match.gallery_position,
            )
        )

        if len(ranked) == top_k:
            break

    return tuple(ranked)
