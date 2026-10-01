import unittest

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm

from src.evaluation.experiment import load_experiment_config
from src.models.persistent_change import (
    WEEKS_PER_MONTH,
    add_alarms,
    alarm_summary,
    choose_rule,
    cusum,
    cusum_threshold,
    detection_summary,
    excess_scores,
    false_alarm_rate,
    first_alarm,
    increased_counts,
    injected_scores,
    recent_shares,
    scenario_factors,
    windowed_rows,
)
from src.models.weekly_composition.data import composition_panel
from src.models.weekly_counts.data import prepare_model_frame

SPLIT_WEEKS = {"fit": 8, "calibration": 3, "validation": 2}


def weekly_panel():
    rng = np.random.default_rng(11)
    splits = [split for split, weeks in SPLIT_WEEKS.items() for _ in range(weeks)]
    rows = []
    for week, split in zip(pd.date_range("2023-01-02", periods=13, freq="7D"), splits):
        counts = rng.poisson([60, 30, 10])
        for cluster_id, count in enumerate(counts):
            rows.append(
                {
                    "split": split,
                    "week": week,
                    "cluster_id": cluster_id,
                    "complaint_count": int(count),
                    "is_complete_week": True,
                    "weekly_total": int(counts.sum()),
                }
            )
    frame, _ = prepare_model_frame(pd.DataFrame(rows), 3, SPLIT_WEEKS, 365.25)
    return composition_panel(frame, 3)


class PersistentChangeTests(unittest.TestCase):
    def test_budget_matches_the_preregistered_numbers(self):
        rate = false_alarm_rate(1, 40)
        self.assertAlmostEqual(rate * WEEKS_PER_MONTH * 40, 1.0)
        self.assertAlmostEqual(rate, 0.00575, places=5)
        self.assertAlmostEqual(float(norm.ppf(1 - rate)), 2.53, places=2)

    def test_config_decides_with_configured_scenarios(self):
        settings = load_experiment_config()["persistent_change"]
        self.assertLessEqual(
            set(settings["deciding_scenarios"]), set(settings["scenarios"])
        )

    def test_recent_shares_match_m9(self):
        panel = weekly_panel()
        np.testing.assert_allclose(
            recent_shares(panel["counts"]), panel["recent_share"]
        )

    def test_excess_scores_average_the_cdf_over_draws(self):
        observed = np.array([5])
        mu = np.array([4.0])
        alpha = np.array([[2.0], [8.0]])
        cdf = [
            (nbinom.cdf(4, a, a / (a + 4.0)) + nbinom.cdf(5, a, a / (a + 4.0))) / 2
            for a in (2.0, 8.0)
        ]
        expected = norm.ppf(np.mean(cdf))
        self.assertAlmostEqual(excess_scores(observed, mu, alpha)[0], expected)

    def test_excess_scores_are_standard_normal_under_the_model(self):
        alpha, mu = 5.0, 50.0
        observed = np.random.default_rng(3).negative_binomial(
            alpha, alpha / (alpha + mu), 20000
        )
        z = excess_scores(observed, np.full(len(observed), mu), np.array([[alpha]]))
        self.assertAlmostEqual(z.mean(), 0.0, delta=0.05)
        self.assertAlmostEqual(z.std(), 1.0, delta=0.05)

    def test_cusum_accumulates_and_restarts_after_an_alarm(self):
        z = np.array([1.0, 1.0, 1.0, 1.0, -2.0, 3.0])
        statistic, alarms = cusum(z, k=0.5, h=1.2)
        np.testing.assert_allclose(statistic, [0.5, 1.0, 1.5, 0.5, 0.0, 2.5])
        np.testing.assert_array_equal(alarms, [False, False, True, False, False, True])

    def test_cusum_threshold_gives_the_target_rate(self):
        rate = 0.02
        h = cusum_threshold(rate, 0.5, series=400, weeks=2000, seed=1)
        z = np.random.default_rng(1).standard_normal((2000, 400))
        realized = cusum(z, 0.5, h)[1].mean()
        self.assertLessEqual(realized, rate)
        self.assertGreater(realized, 0.9 * rate)

    def test_scenario_factors(self):
        growth = scenario_factors("growth", 0.1, 3)
        np.testing.assert_allclose(growth, [1.1, 1.21, 1.331])
        np.testing.assert_allclose(scenario_factors("step", 0.5, 3), [1.5, 1.5, 1.5])
        np.testing.assert_allclose(scenario_factors("none", 0.0, 3), [1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            scenario_factors("drop", 0.5, 3)

    def test_increase_changes_only_its_pattern_and_weeks(self):
        counts = np.array([[10, 20], [11, 21], [12, 22], [13, 23]])
        increased = increased_counts(counts, 1, 1, np.array([1.5, 2.0]))
        np.testing.assert_array_equal(
            increased, [[10, 20], [11, 32], [12, 44], [13, 23]]
        )
        np.testing.assert_array_equal(counts[:, 1], [20, 21, 22, 23])

    def test_injection_without_increase_gives_the_real_scores(self):
        panel = weekly_panel()
        alpha = np.full((4, 3), 20.0)
        rows = windowed_rows(panel, alpha)
        real = rows.loc[rows["cluster_id"] == 1, "z"].to_numpy()[:3]
        injected = injected_scores(panel["counts"], alpha, 1, 4, np.ones(3))
        np.testing.assert_allclose(injected, real)

    def test_add_alarms_runs_one_cusum_per_pattern(self):
        rows = pd.DataFrame(
            {
                "split": ["fit"] * 6,
                "week": np.repeat(pd.date_range("2024-01-01", periods=3, freq="7D"), 2),
                "cluster_id": [0, 1] * 3,
                "z": [1.5, 0.0, 1.5, 0.0, 3.0, 0.0],
            }
        )
        alarmed = add_alarms(rows, {"cusum": 1.5, "weekly": 2.5}, k=0.5)
        np.testing.assert_array_equal(
            alarmed["cusum_alarm"], [False, False, True, False, True, False]
        )
        np.testing.assert_array_equal(
            alarmed["weekly_alarm"], [False, False, False, False, True, False]
        )
        summary = alarm_summary(alarmed)["fit"]
        counts = (summary["weeks"], summary["cusum"], summary["weekly"])
        self.assertEqual(counts, (3, 2, 1))
        self.assertAlmostEqual(summary["cusum_per_month"], 2 / 3 * WEEKS_PER_MONTH)

    def test_first_alarm_counts_the_first_week_as_one(self):
        self.assertEqual(first_alarm(np.array([False, False, True, True])), 3.0)
        self.assertTrue(np.isnan(first_alarm(np.array([False, False]))))

    def test_rule_needs_a_clear_gain_in_every_deciding_scenario(self):
        runs = pd.DataFrame(
            {
                "scenario": ["growth_10"] * 4 + ["growth_20"] * 4,
                "cusum_delay": [1, np.nan, 3, 2, 1, 1, 2, 2],
                "weekly_delay": [np.nan, np.nan, 1, np.nan, 1, 1, 2, 2],
            }
        )
        summary = detection_summary(runs)
        growth_10 = summary["growth_10"]
        self.assertEqual(growth_10["cusum"], {"detected": 0.75, "median_delay": 2.0})
        self.assertEqual(growth_10["weekly"], {"detected": 0.25, "median_delay": 1.0})
        settings = {"deciding_scenarios": ["growth_10"], "minimum_detection_gain": 0.05}
        self.assertEqual(choose_rule(summary, settings)["rule"], "cusum")
        settings["deciding_scenarios"] = ["growth_10", "growth_20"]
        self.assertEqual(choose_rule(summary, settings)["rule"], "weekly")


if __name__ == "__main__":
    unittest.main()
