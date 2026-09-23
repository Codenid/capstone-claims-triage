import unittest

import pandas as pd

from src.models.frequency_baselines import complete_top_three, choose_threshold


class FrequencyBaselineTests(unittest.TestCase):
    def test_completes_product_ranking_with_global_classes(self):
        ranking = complete_top_three(["B", "B"], ["A", "B", "C"])

        self.assertEqual(ranking, ("B", "A", "C"))

    def test_selects_threshold_with_best_calibration_f1(self):
        truth = pd.Series([False, False, True, True], dtype=object)
        scores = pd.Series([0.1, 0.4, 0.35, 0.8])

        threshold, best_f1 = choose_threshold(truth, scores)

        self.assertAlmostEqual(threshold, 0.35)
        self.assertAlmostEqual(best_f1, 0.8)


if __name__ == "__main__":
    unittest.main()
