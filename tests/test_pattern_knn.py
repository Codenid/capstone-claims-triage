import unittest

import numpy as np

from src.models import pattern_knn


class VoteTests(unittest.TestCase):
    def test_weighted_vote_and_nearest_tie_break(self):
        similarities = np.array(
            [
                [0.9, 0.8, 0.7],  # two votes for 1 (1.5) outweigh one for 0 (0.9)
                [0.9, 0.4, 0.4],  # the nearest alone (0.9) beats the two others (0.8)
                [0.5, 0.5, 0.1],  # exact tie between 2 and 3: the nearest wins
            ]
        )
        labels = np.array([[0, 1, 1], [3, 0, 0], [2, 3, 0]])

        self.assertEqual(
            pattern_knn.vote(similarities, labels, 3, 4).tolist(), [1, 3, 2]
        )
        self.assertEqual(
            pattern_knn.vote(similarities, labels, 1, 4).tolist(), [0, 3, 2]
        )
        with self.assertRaises(ValueError):
            pattern_knn.vote(similarities, labels, 4, 4)

    def test_novelty_threshold_and_agreement(self):
        top1 = np.linspace(0.5, 1.0, 101)
        self.assertAlmostEqual(pattern_knn.novelty_threshold(top1, 0.01), 0.505)
        self.assertEqual(
            pattern_knn.agreement(np.array([0, 1, 2, 3]), np.array([0, 1, 0, 3])), 0.75
        )
        with self.assertRaises(ValueError):
            pattern_knn.agreement(np.array([0]), np.array([0, 1]))

    def test_core_and_border_split_the_agreement(self):
        frozen = np.array([0, 0, 1, 1])
        reprojected = np.array([0, 0, 1, 0])
        candidate = np.array([0, 1, 1, 1])

        summary = pattern_knn.core_and_border(candidate, frozen, reprojected)

        self.assertEqual(summary["stable_fraction"], 0.75)
        self.assertAlmostEqual(summary["agreement_stable"], 2 / 3)
        self.assertEqual(summary["agreement_border"], 1.0)

    def test_decision_follows_the_registered_rule(self):
        self.assertTrue(
            pattern_knn.decide({"agreement_knn": 0.92, "agreement_reprojected": 0.915})[
                "adopted"
            ]
        )
        self.assertFalse(
            pattern_knn.decide({"agreement_knn": 0.90, "agreement_reprojected": 0.915})[
                "adopted"
            ]
        )


if __name__ == "__main__":
    unittest.main()
