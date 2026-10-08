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
