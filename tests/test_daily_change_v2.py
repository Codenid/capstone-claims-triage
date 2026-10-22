import unittest

import numpy as np
import pandas as pd

from src.models import daily_change_v2 as v2
from src.models.daily_change import (
    DailyExpectation,
    detection_summary,
    false_alarm_rate,
)
from src.models.persistent_change import cusum


class ScanAndThresholdTests(unittest.TestCase):
    def test_scan_statistic_sums_the_window_over_its_root(self):
        z = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 3.0], [-1.0, 0.0]])

        scan = v2.scan_statistic(z, 3)

        self.assertTrue(np.isnan(scan[:2]).all())
        np.testing.assert_allclose(scan[2], [3 / np.sqrt(3), 3 / np.sqrt(3)])
        np.testing.assert_allclose(scan[3], [1 / np.sqrt(3), 3 / np.sqrt(3)])
        with self.assertRaises(ValueError):
            v2.scan_statistic(z, 0)

    def test_empirical_thresholds_meet_the_budget_on_fit(self):
        rng = np.random.default_rng(3)
        z = rng.standard_t(5, size=(2000, 40))
        rate = false_alarm_rate(4, 40)

        thresholds = v2.empirical_thresholds(z, rate, 0.5, 3)

        self.assertAlmostEqual((z > thresholds["daily"]).mean(), rate, places=3)
        scan = v2.scan_statistic(z, 3)
        self.assertAlmostEqual(np.nanmean(scan > thresholds["scan"]), rate, places=3)
        self.assertLessEqual(cusum(z, 0.5, thresholds["cusum"])[1].mean(), rate)
        self.assertGreater(thresholds["daily"], 2.5)


class AlarmTests(unittest.TestCase):
    def test_add_alarms_marks_the_three_rules(self):
        days = pd.date_range("2024-01-08", periods=5, freq="D")
        rows = pd.DataFrame(
            {
                "split": "fit",
                "day": np.repeat(days, 2),
                "cluster_id": np.tile([0, 1], 5),
                "z": [0, 0, 0, 0, 3, 0, 3, 0, 3, 0],
            }
        )
        thresholds = {"daily": 2.7, "scan": 4.0, "cusum": 3.9}

        scored = v2.add_alarms(rows, thresholds, 0.5, 3)
        summary = v2.alarm_summary(scored)

        self.assertEqual(int(scored["daily_alarm"].sum()), 3)
        self.assertEqual(int(scored["scan_alarm"].sum()), 1)  # 9 / sqrt(3) on day 5
        self.assertEqual(summary["fit"]["days"], 5)
        self.assertEqual(summary["fit"]["daily"], 3)

    def test_injected_window_scores_align_with_the_horizon(self):
        rng = np.random.default_rng(4)
        counts = rng.multinomial(400, [0.5, 0.3, 0.2], size=30)
        dow = np.arange(30) % 7
        model = DailyExpectation(0.8, np.zeros((3, 7)))
        alpha = np.full((20, 3), 10.0)
        factors = np.array([2.0, 2.0, 2.0, 1.0, 1.0])

        scores, increased = v2.injected_window_scores(
            counts, dow, alpha, model, 2, 10, factors, 3
        )

        self.assertEqual(scores["daily"].shape, (5,))
        self.assertEqual(scores["scan"].shape, (5,))
        self.assertFalse(np.isnan(scores["scan"]).any())
        self.assertEqual(int(increased[10, 2]), int(round(counts[10, 2] * 2)))
        self.assertTrue((increased[:10] == counts[:10]).all())
        with self.assertRaises(ValueError):
            v2.injected_window_scores(counts, dow, alpha, model, 2, 1, factors, 3)


class DecisionTests(unittest.TestCase):
    def settings(self):
        return {
            "deciding_scenarios": ["a", "b"],
            "minimum_detection": 0.5,
            "maximum_median_delay": 2,
            "false_alarms_per_month": 4,
            "calibration_budget_factor": 1.5,
        }

    def runs(self, scan_delays):
        return pd.DataFrame(
            {
                "scenario": ["a"] * 4 + ["b"] * 4,
                "daily_delay": [np.nan] * 8,
                "scan_delay": scan_delays,
                "cusum_delay": [1, 1, 1, np.nan, 1, 1, np.nan, np.nan],
            }
        )

    def test_scan_rule_adopted_when_it_detects_within_budget(self):
        summary = detection_summary(self.runs([1, 1, 2, np.nan, 1, 2, 1, np.nan]))
        alarms = {"daily_per_month": 1.0, "scan_per_month": 5.0, "cusum_per_month": 3.0}

        decision = v2.decide(summary, alarms, self.settings())

        self.assertTrue(decision["adopted"])
        self.assertEqual(decision["rule"], "scan")
        self.assertEqual(decision["calibration_budget_limit"], 6.0)

    def test_budget_check_can_veto_a_detecting_rule(self):
        summary = detection_summary(self.runs([1, 1, 2, np.nan, 1, 2, 1, np.nan]))
        alarms = {"daily_per_month": 1.0, "scan_per_month": 9.0, "cusum_per_month": 3.0}

        decision = v2.decide(summary, alarms, self.settings())

        self.assertFalse(decision["checks"]["budget:scan"])
        self.assertEqual(decision["rule"], "cusum")  # the scan is vetoed, CUSUM passes


if __name__ == "__main__":
    unittest.main()
