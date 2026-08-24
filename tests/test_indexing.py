import unittest

import numpy as np

from identification_service.indexing import ExactCosineIndex


class ExactCosineIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = ExactCosineIndex(dimension=3)

    def test_search_orders_gallery_images_by_cosine_similarity(self) -> None:
        self.index.add(
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            ["alice", "bob", "charlie"],
        )

        matches = self.index.nearest_images([1.0, 0.0, 0.0])

        self.assertEqual([match.identity for match in matches], ["alice", "bob", "charlie"])
        self.assertAlmostEqual(matches[0].similarity, 1.0, places=6)
        self.assertAlmostEqual(matches[1].similarity, 2**-0.5, places=6)
        self.assertAlmostEqual(matches[2].similarity, 0.0, places=6)

    def test_multiple_gallery_images_can_share_an_identity(self) -> None:
        self.index.add(
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
            ["alice", "alice"],
        )

        self.assertEqual(self.index.size, 2)
        self.assertEqual(self.index.identity_count, 1)

    def test_add_is_atomic_when_identity_count_does_not_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "number of embeddings"):
            self.index.add(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                ["alice"],
            )

        self.assertEqual(self.index.size, 0)

    def test_invalid_embeddings_are_rejected(self) -> None:
        invalid_cases = (
            ([[1.0, 0.0]], "dimension"),
            ([[0.0, 0.0, 0.0]], "zero-length"),
            ([[np.nan, 0.0, 0.0]], "finite"),
        )

        for embeddings, expected_message in invalid_cases:
            with self.subTest(embeddings=embeddings):
                with self.assertRaisesRegex(ValueError, expected_message):
                    self.index.add(embeddings, ["alice"])

    def test_empty_index_cannot_be_searched(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty index"):
            self.index.nearest_images([1.0, 0.0, 0.0])

    def test_limit_caps_the_number_of_image_matches(self) -> None:
        self.index.add(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ["alice", "bob"],
        )

        matches = self.index.nearest_images([1.0, 0.0, 0.0], limit=1)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].identity, "alice")


if __name__ == "__main__":
    unittest.main()
