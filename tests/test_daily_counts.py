import unittest

import numpy as np
import pandas as pd

from src.models.daily_counts.baselines import ROLLING_NAME, rolling_columns
from src.models.daily_counts.data import (
    as_weekly_view,
    panel_arrays,
    panel_from_assignments,
    validate_daily_counts,
)
from src.models.daily_counts.models import nb_daily_hierarchical_v1, nb_daily_no_dow_v1
from src.models.daily_counts.references import (
    discounted_daily_shares,
    static_daily_shares,
    warm_rows,
    windowed_daily_shares,
)
from src.models.daily_counts.run import build_predictions, flatten

CLUSTERS = 3
PERIODS = {
    "fit": {"start": "2024-01-01", "end": "2024-01-14"},
    "calibration": {"start": "2024-01-15", "end": "2024-01-18"},
    "validation": {"start": "2024-01-19", "end": "2024-01-20"},
}


def assignments() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for split, period in PERIODS.items():
        for day in pd.date_range(period["start"], period["end"], freq="D"):
            for _ in range(int(rng.integers(20, 40))):
                rows.append(
                    {
                        "Complaint ID": str(len(rows)),
                        "Date received": day.date(),
                        "cluster_id": int(rng.choice(CLUSTERS, p=[0.6, 0.3, 0.1])),
                        "is_novel": False,
                        "split": split,
                    }
                )
    return pd.DataFrame(rows)


def validated_panel() -> pd.DataFrame:
    return validate_daily_counts(
        panel_from_assignments(assignments(), CLUSTERS, PERIODS), CLUSTERS
    )


def settings(discount: float = 0.8) -> dict:
    return {
        "clusters": CLUSTERS,
        "share": {"discount": discount, "warmup_days": 3},
        "priors": {
            "log_alpha_global_mean": 2.3,
            "log_alpha_global_sigma": 1.0,
            "log_alpha_sigma": 0.75,
            "beta_sigma": 0.5,
        },
    }


class DailyPanelTests(unittest.TestCase):
    def test_panel_is_a_complete_grid_with_daily_totals(self):
        panel = panel_from_assignments(assignments(), CLUSTERS, PERIODS)

        days = {"fit": 14, "calibration": 4, "validation": 2}
        validated = validate_daily_counts(panel, CLUSTERS, days)
        self.assertEqual(len(validated), sum(days.values()) * CLUSTERS)
        grouped = validated.groupby(["split", "day"])
        sums = grouped["complaint_count"].sum()
        self.assertTrue((sums == grouped["daily_total"].first()).all())
        self.assertTrue(set(validated["day_of_week"].unique()) <= set(range(7)))
        self.assertIn("week", as_weekly_view(validated).columns)

    def test_shares_use_previous_days_only(self):
        panel = validated_panel()
        fit = panel.loc[panel["split"] == "fit"].reset_index(drop=True)

        discounted = discounted_daily_shares(fit, CLUSTERS, 0.8)
        windowed = windowed_daily_shares(fit, CLUSTERS, 7)
        static = static_daily_shares(fit, CLUSTERS)

        first_day = fit["day"] == fit["day"].min()
        self.assertTrue(np.isnan(discounted[first_day]).all())
        self.assertTrue(np.isnan(windowed[first_day]).all())
        self.assertTrue(np.isfinite(discounted[~first_day]).all())
        shares = pd.DataFrame({"day": fit["day"], "s": discounted}).dropna()
        by_day = shares.groupby("day")["s"].sum()
        self.assertTrue(np.allclose(by_day, 1.0))
        self.assertAlmostEqual(float(static.sum()), 1.0)
        self.assertEqual(int(warm_rows(fit, 3).sum()), (14 - 3) * CLUSTERS)

    def test_models_build_and_draw_from_the_prior(self):
        import pymc as pm

        panel = validated_panel()
        frame = nb_daily_hierarchical_v1.prepare_frame(panel, settings())
        fit = frame.loc[frame["split"] == "fit"].reset_index(drop=True)
        arrays = panel_arrays(fit, CLUSTERS)

        self.assertEqual(arrays["counts"].shape, (11, CLUSTERS))
        for module in (nb_daily_hierarchical_v1, nb_daily_no_dow_v1):
            with module.build_model(fit, settings()):
                prior = pm.sample_prior_predictive(draws=5, random_seed=1)
            observed = prior.prior_predictive["observed"].to_numpy()
            self.assertEqual(observed.shape[-2:], (11, CLUSTERS))
            expected = prior.prior["mu"].to_numpy().reshape(-1, 11, CLUSTERS)
            totals = arrays["totals"][None, :]
            self.assertTrue(np.allclose(expected.sum(axis=2), totals))

    def test_predictions_and_rolling_baseline_align_with_rows(self):
        panel = validated_panel()
        frame = nb_daily_hierarchical_v1.prepare_frame(panel, settings())
        rows = len(frame)
        draws = np.random.default_rng(0).poisson(10, size=(6, rows))

        predictions = build_predictions(frame, draws.astype(float), draws, draws)
        rolling = rolling_columns(frame, panel, CLUSTERS, 6, 0)

        self.assertEqual(len(predictions), rows)
        self.assertIn("model_p50", predictions.columns)
        self.assertIn(f"{ROLLING_NAME}_p975", rolling.columns)
        self.assertEqual(flatten(np.zeros((2, 4, 3))).shape, (2, 12))


if __name__ == "__main__":
    unittest.main()
