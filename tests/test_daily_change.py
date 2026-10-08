import unittest

import numpy as np

from src.models.daily_change import (
    DailyExpectation,
    burst_factors,
    decide,
    detection_summary,
    false_alarm_rate,
    weekly_view,
)
import pandas as pd


class DailyChangeTests(unittest.TestCase):
    def test_false_alarm_rate_and_burst_shapes(self):
        self.assertAlmostEqual(false_alarm_rate(4, 40), 4 / (40 * 365.25 / 12))
        factors = burst_factors({"kind": "burst", "size": 2.0, "days": 3}, 6)
        self.assertEqual(factors.tolist(), [2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
        self.assertEqual(burst_factors({"kind": "none"}, 3).tolist(), [1.0] * 3)

    def test_expected_counts_keep_the_daily_total(self):
        rng = np.random.default_rng(1)
        counts = rng.multinomial(500, [0.5, 0.3, 0.2], size=10)
        dow = np.arange(10) % 7
        model = DailyExpectation(0.8, rng.normal(0, 0.3, size=(3, 7)))

        expected = model.expected(counts, dow)

        self.assertTrue(np.isnan(expected[0]).all())
        self.assertTrue(np.allclose(expected[1:].sum(axis=1), counts[1:].sum(axis=1)))

    def test_zero_inflation_deflates_the_mean_and_enters_the_excess(self):
        rng = np.random.default_rng(2)
        counts = rng.multinomial(300, [0.5, 0.3, 0.2], size=6)
        dow = np.arange(6) % 7
        beta = np.zeros((3, 7))
        zero = np.array([0.0, 0.2, 0.0])
        plain = DailyExpectation(0.8, beta)
        inflated = DailyExpectation(0.8, beta, zero)
        alpha = np.full((4, 3), 10.0)

        nb_mean = inflated.nb_mean(counts, dow)
        expected = inflated.expected(counts, dow)
        ids = np.array([1, 1])
        observed = np.array([0, 40])
        means = nb_mean[[2, 3], 1]
        z_plain = plain.excess(observed, means, alpha, ids)
        z_inflated = inflated.excess(observed, means, alpha, ids)

        same = np.allclose(nb_mean, plain.nb_mean(counts, dow), equal_nan=True)
        self.assertTrue(same)
        self.assertTrue(np.allclose(expected[1:, 1], 0.8 * nb_mean[1:, 1]))
        self.assertTrue(np.allclose(expected[1:, 0], nb_mean[1:, 0]))
        # The zero-inflated CDF is pi + (1 - pi) F_NB, above F_NB for every
        # count: each score moves up, the observed zero by the most.
        self.assertTrue((z_inflated > z_plain).all())
        gain = z_inflated - z_plain
        self.assertGreater(gain[0], gain[1])

    def test_weekly_view_keeps_complete_weeks(self):
        days = np.array(pd.date_range("2024-01-03", periods=19, freq="D"))
        counts = np.ones((19, 2), dtype=int)

        weekly, weeks = weekly_view(counts, days)

        self.assertEqual(weekly.shape, (2, 2))
        self.assertEqual(weekly[0].tolist(), [7, 7])
        self.assertEqual(str(np.datetime64(weeks[0], "D")), "2024-01-08")

    def test_decision_needs_detection_and_delay(self):
        runs = pd.DataFrame(
            {
                "scenario": ["a"] * 4 + ["b"] * 4,
                "cusum_delay": [1, 2, np.nan, 1, 1, 1, 2, 1],
                "daily_delay": [np.nan, np.nan, np.nan, 1, 1, np.nan, np.nan, np.nan],
            }
        )
        settings = {
            "deciding_scenarios": ["a", "b"],
            "minimum_detection": 0.5,
            "maximum_median_delay": 2,
        }

        summary = detection_summary(runs)
        decision = decide(summary, settings)

        self.assertEqual(summary["a"]["cusum"]["detected"], 0.75)
        self.assertTrue(decision["adopted"])
        self.assertEqual(decision["rule"], "cusum")


if __name__ == "__main__":
    unittest.main()
