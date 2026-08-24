import unittest

from identification_service.indexing import ExactCosineIndex
from identification_service.retrieval import rank_identities


class IdentityRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = ExactCosineIndex(dimension=2)
        self.index.add(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.8, 0.2],
                [0.0, 1.0],
            ],
            ["alice", "alice", "bob", "charlie"],
        )

    def test_duplicate_image_matches_collapse_to_distinct_identities(self) -> None:
        results = rank_identities(self.index, [1.0, 0.0], top_k=3)

        self.assertEqual(
            [result.identity for result in results],
            ["alice", "bob", "charlie"],
        )
        self.assertEqual([result.rank for result in results], [1, 2, 3])
        self.assertEqual(results[0].gallery_position, 0)

    def test_top_k_is_capped_by_available_identity_count(self) -> None:
        results = rank_identities(self.index, [1.0, 0.0], top_k=20)

        self.assertEqual(len(results), 3)

    def test_invalid_top_k_is_rejected(self) -> None:
        for top_k in (0, -1):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(ValueError, "greater than zero"):
                    rank_identities(self.index, [1.0, 0.0], top_k=top_k)


if __name__ == "__main__":
    unittest.main()
