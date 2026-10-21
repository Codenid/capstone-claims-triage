import unittest

import numpy as np

from src.models import persistent_change_farr as farr
from src.models.persistent_change import excess_scores, windowed_rows
from src.models.weekly_counts.discounted_reference import shares_from_counts

CLUSTERS = 3
WEEKS = 12


def panel():
    rng = np.random.default_rng(5)
    counts = rng.poisson([60, 30, 10], size=(WEEKS, CLUSTERS)).astype(float)
    alpha = rng.gamma(20, 1, size=(50, CLUSTERS))
    return counts, alpha


class ReweightedSharesTests(unittest.TestCase):
    def test_weights_are_one_up_to_the_threshold_then_inverse_square(self):
        z = np.array([-1.0, 2.58, 3.0, 5.16])
        weights = farr.excess_weights(z, 2.58)
        np.testing.assert_allclose(weights, [1.0, 1.0, (2.58 / 3.0) ** 2, 0.25])

    def test_infinite_threshold_reproduces_c_a_memory(self):
        counts, alpha = panel()
        shares = farr.ReweightedShares(alpha, 0.5, np.inf)
        np.testing.assert_allclose(
            shares(counts), shares_from_counts(counts, 0.5, None)
        )

    def test_a_burst_enters_the_memory_with_less_weight(self):
        counts, alpha = panel()
        burst = counts.copy()
        burst[6, 2] *= 12
        plain = shares_from_counts(burst, 0.5, None)
        reweighted = farr.ReweightedShares(alpha, 0.5, 2.58)(burst)
        np.testing.assert_allclose(reweighted[:7], plain[:7])
        self.assertLess(reweighted[7, 2], plain[7, 2])
        self.assertAlmostEqual(reweighted[7].sum(), 1.0)

    def test_resumed_panel_matches_a_fresh_computation(self):
        counts, alpha = panel()
        shares = farr.ReweightedShares(alpha, 0.5, 2.58)
        baseline = shares(counts)
        increased = counts.copy()
        increased[5:9, 1] = np.rint(increased[5:9, 1] * 1.5)
        fresh = farr.ReweightedShares(alpha, 0.5, 2.58)(increased)

        resumed = shares.partial(increased, 9)

        np.testing.assert_allclose(resumed[:9], fresh[:9])
        self.assertTrue(np.isnan(resumed[9:]).all())
        np.testing.assert_allclose(shares(counts), baseline)
        np.testing.assert_allclose(shares(increased), fresh)

    def test_reweighting_lowers_the_excess_after_a_burst(self):
        counts, alpha = panel()
        burst = counts.copy()
        burst[6, 2] *= 12
        plain = shares_from_counts(burst, 0.5, None)
        reweighted = farr.ReweightedShares(alpha, 0.5, 2.58)(burst)
        total = burst[7].sum()
        z_plain = excess_scores(burst[7], total * plain[7], alpha)
        z_farr = excess_scores(burst[7], total * reweighted[7], alpha)
        self.assertGreater(z_farr[2], z_plain[2])


class DetectionAndDecisionTests(unittest.TestCase):
    def test_detection_runs_cover_every_scenario_pattern_and_start(self):
        counts, alpha = panel()
        weeks = np.array([f"2023-{week + 1:02d}-01" for week in range(WEEKS)])
        table = {"counts": counts, "week": weeks, "split": np.array(["fit"] * WEEKS)}
        settings = {
            "horizon_weeks": 3,
            "cusum_k": 0.5,
            "scenarios": {"growth_20": {"kind": "growth", "size": 0.2}},
        }
        shares = farr.ReweightedShares(alpha, 0.5, 2.58)
        windowed_rows(table, alpha, shares)

        runs = farr.detection_runs(
            table, alpha, {"cusum": 3.37, "weekly": 2.53}, settings, shares
        )

        self.assertEqual(len(runs), CLUSTERS * (WEEKS - 3 - 4 + 1))
        self.assertEqual(set(runs["scenario"]), {"growth_20"})

    def test_plugin_wis_is_finite_and_penalizes_a_wrong_share(self):
        counts, alpha = panel()
        shares = shares_from_counts(counts, 0.5, None)
        weeks = np.arange(4, WEEKS)
        good = farr.plugin_wis(counts, shares, alpha, weeks, 5, 0)
        wrong = farr.plugin_wis(counts, shares[:, ::-1], alpha, weeks, 5, 0)
        self.assertTrue(np.isfinite(good))
        self.assertLess(good, wrong)

    def test_decision_needs_both_conditions(self):
        reference = {
            "detection": {"growth_10": {"cusum": {"detected": 0.42}}},
            "real_alarms": {
                "fit": {"cusum_per_month": 2.25},
                "calibration": {"cusum_per_month": 1.45},
            },
        }
        alarms = {
            "fit": {"cusum_per_month": 2.0},
            "calibration": {"cusum_per_month": 1.2},
        }
        kept = {"growth_10": {"cusum": {"detected": 0.40}}}
        lost = {"growth_10": {"cusum": {"detected": 0.30}}}
        self.assertTrue(
            farr.decide(kept, alarms, reference, ["growth_10"], 0.05)["adopted"]
        )
        self.assertFalse(
            farr.decide(lost, alarms, reference, ["growth_10"], 0.05)["adopted"]
        )
        same = {
            "fit": {"cusum_per_month": 2.25},
            "calibration": {"cusum_per_month": 1.2},
        }
        self.assertFalse(
            farr.decide(kept, same, reference, ["growth_10"], 0.05)["adopted"]
        )


if __name__ == "__main__":
    unittest.main()
