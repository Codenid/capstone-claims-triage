import unittest

import numpy as np
import pandas as pd

from src.models.weekly_counts.baselines import baseline_predictions
from src.models.weekly_counts.comparison import (
    compare_with_baselines,
    paired_bootstrap,
    weekly_sums,
)
from src.models.weekly_counts.data import prepare_model_frame
from src.models.weekly_counts.metrics import (
    BOOTSTRAP_RULE,
    evaluate_predictions,
    run_acceptance,
)
from src.models.weekly_counts.prediction import build_predictions, summarize_draws
from src.models.weekly_counts.reporting import backtest_metrics, comparison_metrics
from src.models.weekly_counts.rolling_reference import (
    ROLLING_WINDOWS,
    expected_counts,
    rolling_columns,
    rolling_shares,
)


def weekly_panel(fit_weeks=14, calibration_weeks=3, validation_weeks=2):
    rng = np.random.default_rng(7)
    weeks = pd.date_range(
        "2023-01-02", periods=fit_weeks + calibration_weeks + validation_weeks, freq="7D"
    )
    splits = (
        ["fit"] * fit_weeks
        + ["calibration"] * calibration_weeks
        + ["validation"] * validation_weeks
    )
    rows = []
    for week, split in zip(weeks, splits):
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
    frame, _ = prepare_model_frame(
        pd.DataFrame(rows),
        clusters=3,
        expected_complete_weeks={
            "fit": fit_weeks,
            "calibration": calibration_weeks,
            "validation": validation_weeks,
        },
        time_scale_days=365.25,
    )
    return frame


class RollingReferenceTests(unittest.TestCase):
    def setUp(self):
        self.frame = weekly_panel()
        self.counts = self.frame.pivot(
            index="week", columns="cluster_id", values="complaint_count"
        )

    def test_shares_use_previous_weeks_with_one_pseudo_count(self):
        shares = rolling_shares(self.frame, clusters=3, window=2)

        previous = self.counts.iloc[0:2].sum()
        np.testing.assert_allclose(
            shares.iloc[2].to_numpy(),
            ((previous + 1) / (previous.sum() + 3)).to_numpy(),
        )
        self.assertTrue(shares.iloc[:2].isna().all().all())
        np.testing.assert_allclose(shares.iloc[2:].sum(axis=1), 1.0)

    def test_a_week_only_changes_the_shares_of_later_weeks(self):
        changed = self.frame.copy()
        third_week = changed["week"] == self.counts.index[2]
        changed.loc[third_week, "complaint_count"] += 50

        before = rolling_shares(self.frame, clusters=3, window=2)
        after = rolling_shares(changed, clusters=3, window=2)

        np.testing.assert_array_equal(before.iloc[:3], after.iloc[:3])
        self.assertFalse(np.allclose(before.iloc[3], after.iloc[3]))

    def test_expected_counts_scale_shares_by_the_weekly_total(self):
        mu = expected_counts(self.frame, clusters=3, window=4)
        shares = rolling_shares(self.frame, clusters=3, window=4)
        row = self.frame.index[self.frame["split"] == "calibration"][0]
        week = self.frame.loc[row, "week"]
        cluster = self.frame.loc[row, "cluster_id"]

        self.assertAlmostEqual(
            mu[row], self.frame.loc[row, "weekly_total"] * shares.loc[week, cluster]
        )
        first_week = (self.frame["week"] == self.counts.index[0]).to_numpy()
        self.assertTrue(np.isnan(mu[first_week]).all())

    def test_rolling_columns_cover_later_splits_and_are_seeded(self):
        first = rolling_columns(self.frame, clusters=3, draws=50, seed=44)
        second = rolling_columns(self.frame, clusters=3, draws=50, seed=44)

        pd.testing.assert_frame_equal(first, second)
        later = self.frame["split"] != "fit"
        for prefix in ROLLING_WINDOWS:
            self.assertFalse(first.loc[later, f"{prefix}_p50"].isna().any())
            self.assertIn(f"{prefix}_impossible_probability", first)
        first_week = self.frame["week"] == self.counts.index[0]
        self.assertTrue(first.loc[first_week, "b1_rolling_4_p50"].isna().all())

    def test_baseline_predictions_keep_later_rows_and_own_columns(self):
        columns = rolling_columns(self.frame, clusters=3, draws=20, seed=44)

        predictions = baseline_predictions(self.frame, columns, "b1_rolling_4")

        self.assertEqual(set(predictions["split"]), {"calibration", "validation"})
        self.assertIn("b1_rolling_4_p50", predictions)
        self.assertFalse(any(name.startswith("b1_rolling_13") for name in predictions))


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        weeks = np.repeat(pd.date_range("2024-10-07", periods=12, freq="7D"), 5)
        observed = rng.poisson(50, size=len(weeks))
        frame = pd.DataFrame(
            {"split": "calibration", "week": weeks, "complaint_count": observed}
        )
        self.predictions = pd.concat(
            [
                frame,
                summarize_draws(rng.poisson(observed, size=(300, len(weeks))), "model"),
                summarize_draws(
                    rng.poisson(observed * 1.6, size=(300, len(weeks))), "baseline"
                ),
                summarize_draws(
                    rng.poisson(observed * 2.2, size=(300, len(weeks))), "other"
                ),
            ],
            axis=1,
        )

    def test_identical_predictors_have_no_difference(self):
        weekly = weekly_sums(self.predictions, "model")

        result = paired_bootstrap(weekly, weekly, draws=200, seed=42)

        for values in result.values():
            self.assertEqual(set(values.values()), {0.0})

    def test_bootstrap_is_seeded(self):
        model = weekly_sums(self.predictions, "model")
        baseline = weekly_sums(self.predictions, "baseline")

        self.assertEqual(
            paired_bootstrap(model, baseline, draws=200, seed=42),
            paired_bootstrap(model, baseline, draws=200, seed=42),
        )

    def test_candidate_is_compared_with_the_lowest_wis_baseline(self):
        comparison = compare_with_baselines(self.predictions, ("other", "baseline"), 42)

        self.assertEqual(comparison["best_baseline"], "baseline")
        self.assertEqual(comparison["weeks"], 12)
        self.assertGreater(comparison["wis_gain"], 0.05)
        self.assertLess(comparison["difference"]["wis"]["p975"], 0)
        self.assertAlmostEqual(
            comparison["difference"]["wis"]["estimate"],
            comparison["metrics"]["model"]["wis"]
            - comparison["metrics"]["baseline"]["wis"],
        )


class PromotionRuleTests(unittest.TestCase):
    diagnostics = {"rhat_max": 1.0, "ess_bulk_min": 900, "ess_tail_min": 900, "divergences": 0}
    calibration = {
        "model": {"wis": 90.0, "wape": 0.36, "coverage_80": 0.80, "coverage_95": 0.94},
        "baseline": {"wis": 100.0, "wape": 0.34},
    }

    def settings(self, bootstrap_rule=True):
        acceptance = {
            "maximum_rhat": 1.01,
            "minimum_ess": 400,
            "coverage_80_minimum": 0.70,
            "coverage_80_maximum": 0.90,
            "coverage_95_minimum": 0.88,
            "coverage_95_maximum": 0.99,
        }
        if bootstrap_rule:
            acceptance.update({"rule": BOOTSTRAP_RULE, "minimum_wis_gain": 0.05})
        return {"acceptance": acceptance}

    def comparison(
        self,
        gain=0.10,
        wis=(-12.0, -8.0, -4.0),
        wape=(-0.01, 0.005, 0.02),
        mae=(-3.0, 1.0, 5.0),
    ):
        def interval(values):
            return dict(zip(("p025", "p50", "p975"), values))

        return {
            "wis_gain": gain,
            "difference": {"wis": interval(wis), "wape": interval(wape), "mae": interval(mae)},
        }

    def test_clear_wis_gain_without_significant_loss_is_accepted(self):
        result = run_acceptance(
            self.diagnostics, self.calibration, self.comparison(), self.settings()
        )
        legacy = run_acceptance(
            self.diagnostics,
            self.calibration,
            self.comparison(),
            self.settings(bootstrap_rule=False),
        )

        self.assertTrue(result["accepted"])
        # Configs without the rule keep the strict comparison with the fixed Poisson.
        self.assertFalse(legacy["checks"]["wape"])

    def test_small_uncertain_gains_and_clear_losses_are_rejected(self):
        cases = {
            "wis_gain": self.comparison(gain=0.03),
            "wis_bootstrap": self.comparison(wis=(-12.0, -5.0, 1.0)),
            "wape": self.comparison(wape=(0.001, 0.01, 0.02)),
            "mae": self.comparison(mae=(0.5, 2.0, 4.0)),
        }
        for failed, comparison in cases.items():
            result = run_acceptance(
                self.diagnostics, self.calibration, comparison, self.settings()
            )

            self.assertFalse(result["accepted"], failed)
            self.assertEqual(
                [name for name, passed in result["checks"].items() if not passed],
                [failed],
            )


class ValidationStaysClosedTests(unittest.TestCase):
    def test_backtest_and_mlflow_metrics_skip_validation(self):
        frame = weekly_panel()
        observed = frame["complaint_count"].to_numpy()
        draws = np.tile(observed, (20, 1))
        predictions = build_predictions(frame, draws.astype(float), draws, draws + 1)

        metrics, backtest = evaluate_predictions(predictions, ("fit", "calibration"))

        self.assertEqual(set(backtest), {"fit", "calibration"})
        self.assertNotIn("validation", set(metrics["split"]))
        self.assertFalse(
            any(name.startswith("validation_") for name in backtest_metrics(backtest))
        )

    def test_comparison_metrics_do_not_repeat_backtest_names(self):
        comparison = {
            "split": "calibration",
            "best_baseline": "b1_rolling_4",
            "wis_gain": 0.1,
            "metrics": {
                "model": {"wis": 1.0},
                "baseline": {"wis": 2.0},
                "b1_rolling_4": {"wis": 1.5},
            },
            "difference": {
                "wis": {"estimate": -0.5, "p025": -0.7, "p50": -0.5, "p975": -0.2}
            },
        }
        backtest = {"calibration": {"model": {}, "baseline": {}}}

        metrics = comparison_metrics(comparison, backtest)

        self.assertEqual(metrics["calibration_b1_rolling_4_wis"], 1.5)
        self.assertEqual(metrics["calibration_best_baseline_wis"], 1.5)
        self.assertNotIn("calibration_model_wis", metrics)
        self.assertEqual(metrics["calibration_bootstrap_wis_difference_p975"], -0.2)


if __name__ == "__main__":
    unittest.main()
