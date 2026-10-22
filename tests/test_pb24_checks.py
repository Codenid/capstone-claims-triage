import unittest

import numpy as np

from src.models import pattern_margin
from src.models.weekly_counts import discount_curve


class DiscountCurveTests(unittest.TestCase):
    def test_shape_reports_both_arms_of_the_u(self):
        scores = [
            {"discount": 0.1, "wis": 30.0},
            {"discount": 0.5, "wis": 20.0},
            {"discount": 0.9, "wis": 26.0},
        ]
        result = discount_curve.shape(scores)
        self.assertEqual(result["argmin_discount"], 0.5)
        self.assertTrue(result["u_shaped"])

        boundary = discount_curve.shape(scores[1:])
        self.assertFalse(boundary["left_arm_observed"])
        self.assertFalse(boundary["u_shaped"])


class PatternMarginTests(unittest.TestCase):
    def test_probability_margins_use_the_two_largest(self):
        probabilities = np.array([[0.7, 0.2, 0.1], [0.5, 0.45, 0.05]])
        top, gap = pattern_margin.probability_margins(probabilities)
        np.testing.assert_allclose(top, [0.7, 0.5])
        np.testing.assert_allclose(gap, [0.5, 0.05])
        with self.assertRaises(ValueError):
            pattern_margin.probability_margins(np.array([[1.0]]))

    def test_distance_margin_is_zero_halfway_between_centres(self):
        centres = np.array([[0.0, 0.0], [10.0, 0.0]])
        values = np.array([[5.0, 0.0], [1.0, 0.0]])
        margin = pattern_margin.distance_margins(values, centres)
        self.assertAlmostEqual(margin[0], 0.0)
        self.assertAlmostEqual(margin[1], 8.0)

    def test_boundary_table_counts_per_pattern(self):
        labels = np.array([0, 0, 1, 1])
        top = np.array([0.9, 0.55, 0.8, 0.95])
        gap = np.array([0.8, 0.1, 0.6, 0.9])
        geometric = np.array([1.0, 0.05, 0.5, 2.0])
        table = pattern_margin.boundary_table(labels, top, gap, geometric, 3)
        self.assertEqual(table[0]["rows"], 2)
        self.assertEqual(table[0]["boundary_fraction_probability"], 0.5)
        self.assertEqual(table[0]["uncertain_fraction"], 0.5)
        self.assertEqual(table[0]["boundary_fraction_distance"], 0.5)
        self.assertEqual(table[1]["boundary_fraction_probability"], 0.0)
        self.assertEqual(table[2], {"cluster_id": 2, "rows": 0})


if __name__ == "__main__":
    unittest.main()
