import unittest

import numpy as np
import pandas as pd

from src.models.weekly_counts.challenge import candidate_intervals, decide
from src.models.weekly_counts.foundation_series import (
    calibration_metrics,
    decile_quantiles,
    prediction_rows,
)


def target(counts: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": "calibration",
            "week": "2024-10-07",
            "cluster_id": range(len(counts)),
            "complaint_count": counts,
            "weekly_total": sum(counts),
        }
    )


class FoundationSeriesTests(unittest.TestCase):
    def test_deciles_give_interpolated_quartiles_and_no_tails(self):
        deciles = np.array([np.arange(10, 100, 10, dtype=float)])

        quantiles = decile_quantiles(deciles)

        self.assertAlmostEqual(quantiles["p25"][0], 25.0)
        self.assertEqual(quantiles["p50"][0], 50.0)
        self.assertAlmostEqual(quantiles["p75"][0], 75.0)
        self.assertTrue(np.isnan(quantiles["p975"][0]))

    def test_rows_clip_negative_quantiles(self):
        rows = prediction_rows(
            target([3, 7]),
            {
                "mean": np.array([2.0, 6.0]),
                "p025": np.array([-1.0, 4.0]),
                "p10": np.array([0.0, 5.0]),
                "p25": np.array([1.0, 6.0]),
                "p50": np.array([2.0, 7.0]),
                "p75": np.array([3.0, 8.0]),
                "p90": np.array([4.0, 9.0]),
                "p975": np.array([5.0, 11.0]),
            },
        )

        self.assertEqual(rows["model_p025"].tolist(), [0.0, 4.0])
        self.assertAlmostEqual(rows["model_impossible_probability"].iloc[1], 1 / 7)

    def test_metrics_skip_the_95_interval_when_absent(self):
        rows = prediction_rows(
            target([3, 7]),
            decile_quantiles(np.array([np.arange(1, 10.0), np.arange(3, 12.0)]))
            | {"mean": np.array([5.0, 7.0])},
        )

        metrics = calibration_metrics(rows, [50, 80])

        self.assertIsNone(metrics["coverage_95"])
        self.assertEqual(metrics["coverage_80"], 1.0)

    def test_challenge_uses_the_candidate_intervals_and_coverages(self):
        intervals = candidate_intervals({"interval_widths": [50, 80]})
        comparison = {
            "wis_gain": 0.1,
            "difference": {
                "wis": {"p975": -1.0},
                "wape": {"p025": -0.1},
                "mae": {"p025": -0.1},
            },
        }
        acceptance = {
            "coverage_80_minimum": 0.70,
            "coverage_80_maximum": 0.90,
            "coverage_95_minimum": 0.88,
            "coverage_95_maximum": 0.99,
        }

        result = decide(
            comparison, {"coverage_80": 0.8, "coverage_95": None}, acceptance, 0.05
        )

        self.assertEqual([alpha for alpha, *_ in intervals], [0.5, 0.2])
        self.assertNotIn("coverage_95", result["checks"])
        self.assertTrue(result["wins"])


if __name__ == "__main__":
    unittest.main()
