import importlib.util
import unittest


FAISS_AVAILABLE = importlib.util.find_spec("faiss") is not None


@unittest.skipUnless(FAISS_AVAILABLE, "faiss-cpu is not installed")
class LSHCosineIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        from identification_service.modules.retrieval.index.lsh import (
            LSHCosineIndex,
        )

        self.index = LSHCosineIndex(dimension=4, hash_bits=64)

    def test_self_query_is_ranked_first_with_full_hash_agreement(self) -> None:
        self.index.add(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            ["alice", "bob", "charlie"],
        )

        matches = self.index.nearest_images([1.0, 0.0, 0.0, 0.0])

        self.assertEqual(matches[0].identity, "alice")
        self.assertAlmostEqual(matches[0].similarity, 1.0, places=6)

    def test_limit_and_identity_metadata_are_preserved(self) -> None:
        self.index.add(
            [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
            ["alice", "alice"],
        )

        matches = self.index.nearest_images(
            [1.0, 0.0, 0.0, 0.0], limit=1
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(self.index.size, 2)
        self.assertEqual(self.index.identity_count, 1)

    def test_identity_ranking_accepts_lsh_backend(self) -> None:
        from identification_service.modules.retrieval.search import rank_identities

        self.index.add(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.99, 0.01, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            ["alice", "alice", "bob"],
        )

        results = rank_identities(
            self.index, [1.0, 0.0, 0.0, 0.0], top_k=2
        )

        self.assertEqual([result.identity for result in results], ["alice", "bob"])


if __name__ == "__main__":
    unittest.main()
