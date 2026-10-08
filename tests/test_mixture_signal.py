import unittest

import numpy as np
import pandas as pd

from src.models.weekly_composition.mixture_signal import (
    largest_excess,
    mixture_log_score,
    score_weeks,
    summarize,
    week_surprise,
)


def panel(weeks: int, clusters: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shares = rng.dirichlet(np.full(clusters, 5.0), size=weeks)
    totals = np.full(weeks, 2000)
    counts = np.vstack([rng.multinomial(total, p) for total, p in zip(totals, shares)])
    return {
        "week": np.array(
            [f"2024-10-{7 + 7 * i:02d}" for i in range(weeks)], dtype="datetime64[D]"
        ),
        "split": np.array(["fit"] * (weeks - 1) + ["calibration"]),
        "weekly_total": totals,
        "counts": counts,
        "recent_share": shares,
    }


class MixtureSignalTests(unittest.TestCase):
    def test_a_typical_week_is_not_rare_and_a_distorted_one_is(self):
        rng = np.random.default_rng(0)
        clusters = 8
        shares = np.full(clusters, 1 / clusters)
        kappa = np.full(50, 800.0)
        concentration = kappa[:, None] * shares[None, :]
        typical = rng.multinomial(2000, shares)
        distorted = np.array([1500] + [500 // (clusters - 1)] * (clusters - 1))
        distorted[-1] += 2000 - distorted.sum()

        _, typical_p = week_surprise(typical, 2000, concentration, 40, rng)
        _, distorted_p = week_surprise(distorted, 2000, concentration, 40, rng)

        self.assertGreater(typical_p, 0.05)
        self.assertLess(distorted_p, 0.01)

    def test_mixture_score_matches_a_single_draw(self):
        counts = np.array([10, 20, 30])
        concentration = np.array([[2.0, 4.0, 6.0]])

        score = mixture_log_score(counts, concentration)
        batch = mixture_log_score(np.vstack([counts, counts]), concentration)

        self.assertEqual(batch.shape, (2,))
        self.assertAlmostEqual(float(batch[0]), float(score))

    def test_largest_excess_names_the_pattern_that_moved_most(self):
        counts = np.array([700, 200, 100])
        recent = np.array([0.4, 0.4, 0.2])

        cluster, share, expected = largest_excess(counts, 1000, recent)

        self.assertEqual(cluster, 0)
        self.assertAlmostEqual(share, 0.7)
        self.assertAlmostEqual(expected, 0.4)

    def test_scores_and_summary_cover_every_week(self):
        data = panel(weeks=4, clusters=6, seed=1)
        kappa = np.full(20, 300.0)

        table = score_weeks(data, kappa, simulations=10, seed=42)
        summary = summarize(table, alert_p_value=0.01)

        self.assertEqual(len(table), 4)
        self.assertTrue(((table["p_value"] > 0) & (table["p_value"] <= 1)).all())
        self.assertEqual(summary["splits"]["fit"]["weeks"], 3)
        self.assertEqual(summary["splits"]["calibration"]["weeks"], 1)
        self.assertIsInstance(summary["flagged_weeks"], list)
        self.assertIsInstance(table, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
