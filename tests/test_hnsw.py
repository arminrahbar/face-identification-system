import importlib.util
import unittest


FAISS_AVAILABLE = importlib.util.find_spec("faiss") is not None


@unittest.skipUnless(FAISS_AVAILABLE, "faiss-cpu is not installed")
class HNSWCosineIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        from identification_service.modules.retrieval.index.hnsw import (
            HNSWCosineIndex,
        )

        self.index = HNSWCosineIndex(
            dimension=3,
            graph_degree=8,
            construction_ef=20,
            search_ef=20,
        )

    def test_self_query_is_ranked_first(self) -> None:
        self.index.add(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ["alice", "bob", "charlie"],
        )

        matches = self.index.nearest_images([1.0, 0.0, 0.0])

        self.assertEqual(matches[0].identity, "alice")
        self.assertAlmostEqual(matches[0].similarity, 1.0, places=6)

    def test_index_tracks_images_and_distinct_identities(self) -> None:
        self.index.add(
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
            ["alice", "alice"],
        )

        self.assertEqual(self.index.size, 2)
        self.assertEqual(self.index.identity_count, 1)

    def test_identity_ranking_accepts_hnsw_backend(self) -> None:
        from identification_service.modules.retrieval.search import rank_identities

        self.index.add(
            [[1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [0.0, 1.0, 0.0]],
            ["alice", "alice", "bob"],
        )

        results = rank_identities(self.index, [1.0, 0.0, 0.0], top_k=2)

        self.assertEqual([result.identity for result in results], ["alice", "bob"])


if __name__ == "__main__":
    unittest.main()
