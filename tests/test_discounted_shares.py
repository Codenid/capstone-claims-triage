import unittest

import numpy as np
import pandas as pd

from src.models.weekly_counts.discount_grid import (
    grid_scores,
    plug_in_wis,
    quantile_level,
    scored_rows,
)
from src.models.weekly_counts.discounted_reference import (
    discounted_shares,
    row_discounted_shares,
)


def panel(counts: list[list[int]], splits: list[str] | None = None) -> pd.DataFrame:
    """Weekly rows of a 2-cluster panel, one list of counts per week."""
    weeks = pd.date_range("2023-01-02", periods=len(counts), freq="7D")
    splits = splits or ["fit"] * len(counts)
    rows = [
        {
            "split": split,
            "week": week,
            "cluster_id": cluster,
            "complaint_count": count,
            "weekly_total": sum(week_counts),
        }
        for week, split, week_counts in zip(weeks, splits, counts)
        for cluster, count in enumerate(week_counts)
    ]
    return pd.DataFrame(rows)


class DiscountedSharesTests(unittest.TestCase):
    def test_first_week_has_no_share_and_later_weeks_use_only_the_past(self):
        frame = panel([[30, 10], [20, 20], [10, 30]])
        changed = frame.copy()
        changed.loc[changed["week"] == changed["week"].max(), "complaint_count"] = 99

        shares = discounted_shares(frame, 2, discount=0.5, cap=None)

        self.assertTrue(np.isnan(shares.iloc[0]).all())
        pd.testing.assert_frame_equal(
            shares, discounted_shares(changed, 2, discount=0.5, cap=None)
        )

    def test_weights_decay_by_the_discount(self):
        frame = panel([[30, 10], [20, 20], [10, 30]])

        shares = discounted_shares(frame, 2, discount=0.5, cap=None)

        memory = 0.5 * np.array([30, 10]) + np.array([20, 20])
        np.testing.assert_allclose(shares.iloc[2], (memory + 1) / (memory.sum() + 2))

    def test_without_decay_or_cap_it_is_the_cumulative_share(self):
        frame = panel([[30, 10], [20, 20], [10, 30], [5, 5]])

        shares = discounted_shares(frame, 2, discount=1.0, cap=None)

        cumulative = np.array([60, 60])
        np.testing.assert_allclose(shares.iloc[3], (cumulative + 1) / 122)

    def test_the_cap_keeps_a_burst_out_of_the_memory(self):
        frame = panel([[50, 50], [400, 50], [50, 50]])

        free = discounted_shares(frame, 2, discount=0.5, cap=None)
        capped = discounted_shares(frame, 2, discount=0.5, cap=1.5)

        expected = free.iloc[1].to_numpy() * 1.5 * 450
        burst = np.minimum([400, 50], expected)
        memory = 0.5 * np.array([50, 50]) + burst
        np.testing.assert_allclose(capped.iloc[2], (memory + 1) / (memory.sum() + 2))
        self.assertLess(capped.iloc[2, 0], free.iloc[2, 0])

    def test_rows_follow_the_frame_order(self):
        frame = panel([[30, 10], [20, 20]]).iloc[::-1].reset_index(drop=True)

        values = row_discounted_shares(frame, 2, discount=0.5, cap=None)

        self.assertAlmostEqual(values[0], 11 / 42)
        self.assertTrue(np.isnan(values[-1]))

    def test_rejects_invalid_settings(self):
        frame = panel([[1, 1], [1, 1]])
        with self.assertRaisesRegex(ValueError, "discount"):
            discounted_shares(frame, 2, discount=0.0, cap=None)
        with self.assertRaisesRegex(ValueError, "cap"):
            discounted_shares(frame, 2, discount=0.5, cap=0.5)


class DiscountGridTests(unittest.TestCase):
    def test_quantile_labels_map_to_levels(self):
        self.assertEqual(quantile_level("p025"), 0.025)
        self.assertEqual(quantile_level("p10"), 0.10)
        self.assertEqual(quantile_level("p975"), 0.975)

    def test_scores_only_fit_rows_after_the_warmup(self):
        frame = panel([[1, 1]] * 6, splits=["fit"] * 5 + ["calibration"])

        mask = scored_rows(frame, warmup_weeks=4)

        self.assertEqual(int(mask.sum()), 2)
        self.assertTrue((frame.loc[mask, "split"] == "fit").all())

    def test_a_sharper_prediction_scores_better(self):
        frame = panel([[60, 40]] * 3)
        alpha = np.array([50.0, 50.0])

        right = plug_in_wis(frame, np.tile([0.6, 0.4], 3), alpha)
        wrong = plug_in_wis(frame, np.tile([0.3, 0.7], 3), alpha)

        self.assertLess(right, wrong)

    def test_grid_scores_every_pair(self):
        rng = np.random.default_rng(0)
        frame = panel(rng.poisson([60, 40], size=(12, 2)).tolist())
        settings = {"warmup_weeks": 4, "discounts": [0.5, 0.9], "caps": [2, None]}

        scores, v3_wis = grid_scores(frame, 2, np.array([20.0, 20.0]), settings)

        self.assertEqual(len(scores), 4)
        self.assertTrue(all(score["wis"] > 0 for score in scores))
        self.assertGreater(v3_wis, 0)


if __name__ == "__main__":
    unittest.main()
