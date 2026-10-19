import unittest

import numpy as np
import pandas as pd

from src.evaluation.experiment import load_experiment_config
from src.models.daily_change import DailyExpectation
from src.pipeline import counts, signals
from src.pipeline.contract import (
    CANONICAL_PRODUCT_COLUMN,
    NORMALIZED_COLUMN,
    prepare_table,
    validate_input,
)
from src.pipeline.frozen import frozen_artifacts, manifest, missing_artifacts

CLUSTERS = 4


def synthetic_table(rows: int = 200, start: str = "2025-08-04") -> pd.DataFrame:
    """Two hundred complaints after the taxonomy freeze, four columns only."""
    rng = np.random.default_rng(7)
    days = pd.date_range(start, periods=28, freq="D")
    products = ["Credit card", "Mortgage", "Checking or savings account"]
    return pd.DataFrame(
        {
            "Complaint ID": [str(10_000 + i) for i in range(rows)],
            "Date received": rng.choice(days, size=rows),
            "Product": rng.choice(products, size=rows),
            "Consumer complaint narrative": [
                f"On XXXX my card was charged twice {i}" for i in range(rows)
            ],
        }
    )


def assignments(rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    table = synthetic_table(rows)
    table["cluster_id"] = rng.integers(0, CLUSTERS, size=rows)
    table["is_novel"] = rng.random(rows) < 0.02
    return table


class ContractTests(unittest.TestCase):
    def test_validate_fills_optional_columns_and_formats_dates(self):
        validated = validate_input(synthetic_table())

        self.assertEqual(len(validated.columns), 16)
        self.assertTrue(
            validated["Date received"].str.match(r"\d{4}-\d{2}-\d{2}").all()
        )
        self.assertTrue(validated["Issue"].isna().all())
        with self.assertRaises(ValueError):
            validate_input(synthetic_table().drop(columns=["Product"]))
        duplicated = pd.concat([synthetic_table(3)] * 2)
        with self.assertRaises(ValueError):
            validate_input(duplicated)

    def test_prepare_applies_normalization_and_taxonomy(self):
        prepared = prepare_table(synthetic_table(20))

        self.assertIn(NORMALIZED_COLUMN, prepared.columns)
        self.assertTrue(prepared[NORMALIZED_COLUMN].str.contains("<redacted>").all())
        self.assertIn(CANONICAL_PRODUCT_COLUMN, prepared.columns)
        self.assertTrue(prepared["known_product"].all())
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(prepared["Date received"]))
        unknown = synthetic_table(5)
        unknown["Product"] = "Something new"
        self.assertEqual(
            prepare_table(unknown)[CANONICAL_PRODUCT_COLUMN].unique().tolist(),
            ["__UNKNOWN__"],
        )

    def test_rows_before_the_freeze_need_their_issue(self):
        old = synthetic_table(5, start="2024-03-04")
        with self.assertRaises(ValueError):
            prepare_table(old)


class CountsTests(unittest.TestCase):
    def test_period_counts_complete_the_grid(self):
        weekly = counts.period_counts(assignments(), CLUSTERS, "week")
        daily = counts.period_counts(assignments(), CLUSTERS, "day")

        self.assertEqual(weekly["cluster_id"].nunique(), CLUSTERS)
        self.assertEqual(len(weekly) % CLUSTERS, 0)
        self.assertEqual(int(weekly["complaint_count"].sum()), 200)
        self.assertEqual(int(daily["complaint_count"].sum()), 200)
        self.assertTrue((weekly.groupby("period")["total"].nunique() == 1).all())

    def test_extend_history_replaces_overlapping_periods(self):
        history = pd.DataFrame(
            {
                "period": np.repeat(
                    pd.date_range("2025-07-21", periods=4, freq="7D"), CLUSTERS
                ),
                "cluster_id": np.tile(range(CLUSTERS), 4),
                "complaint_count": 5,
            }
        )
        new = counts.period_counts(assignments(), CLUSTERS, "week")

        extended = counts.extend_history(history, new, CLUSTERS)

        self.assertEqual(extended["period"].nunique(), 2 + new["period"].nunique())
        overlapping = extended.loc[extended["period"].isin(new["period"])]
        self.assertTrue((overlapping["source"] == "new").all())
        self.assertEqual(int(overlapping["complaint_count"].sum()), 200)

    def test_weekly_predictive_scores_after_the_first_week(self):
        rng = np.random.default_rng(0)
        panel = counts.period_counts(assignments(), CLUSTERS, "week")
        alpha = np.full((50, CLUSTERS), 8.0)

        scored = counts.weekly_predictive(
            panel,
            alpha,
            {"discount": 0.5, "cap": None},
            {"cusum": 3.37, "weekly": 2.53},
            0.5,
            rng,
        )

        first = scored["period"] == scored["period"].min()
        self.assertFalse(scored.loc[first, "scored"].any())
        self.assertTrue(scored.loc[~first, "scored"].all())
        self.assertTrue(
            (scored.loc[~first, "model_p025"] <= scored.loc[~first, "model_p975"]).all()
        )
        self.assertTrue(np.isfinite(scored.loc[~first, "z"]).all())
        self.assertIn("cusum_alarm", scored.columns)

    def test_weekly_mixture_gives_p_values_after_four_weeks(self):
        panel = counts.period_counts(assignments(400), CLUSTERS, "week")
        kappa = np.full(10, 200.0)

        mixture = counts.weekly_mixture(panel, kappa, 5, 0.01, 1)

        self.assertEqual(len(mixture), panel["period"].nunique())
        self.assertTrue(mixture["p_value"].iloc[:4].isna().all())
        later = mixture["p_value"].iloc[4:]
        self.assertTrue(((later > 0) & (later <= 1)).all())

    def test_daily_predictive_handles_zero_inflation(self):
        rng = np.random.default_rng(0)
        panel = counts.period_counts(assignments(), CLUSTERS, "day")
        alpha = np.full((40, CLUSTERS), 10.0)
        model = DailyExpectation(
            0.8, np.zeros((CLUSTERS, 7)), np.array([0.0, 0.3, 0.0, 0.0])
        )

        scored = counts.daily_predictive(panel, alpha, model, rng)

        self.assertEqual(len(scored), len(panel))
        later = scored.loc[scored["scored"]]
        self.assertTrue((later["model_p50"] >= 0).all())
        self.assertTrue((later["expected"] <= later["total"] + 1e-9).all())


class SignalsTests(unittest.TestCase):
    def test_signals_warn_on_novelty_coverage_and_macro_f1(self):
        complaints = assignments()
        complaints["is_novel"] = True
        complaints["Issue canonical"] = "a"
        complaints["issue_top1"] = "b"
        weekly = pd.DataFrame(
            {
                "period": pd.Timestamp("2025-08-04"),
                "complaint_count": [100] * 4,
                "model_p10": [0] * 4,
                "model_p90": [1] * 4,
                "scored": True,
                "is_complete": True,
            }
        )
        mixture = pd.DataFrame(
            {
                "period": [pd.Timestamp("2025-08-04")],
                "p_value": [0.001],
                "rare_mixture": [True],
            }
        )

        result = signals.retraining_signals(complaints, weekly, mixture, 0.27)

        self.assertTrue(result["retrain_recommended"])
        self.assertEqual(set(result["warnings"]), {"novelty", "coverage", "macro_f1"})
        self.assertEqual(result["mixture"]["rare_weeks"], ["2025-08-04"])


class FrozenTests(unittest.TestCase):
    def test_manifest_names_the_frozen_runs(self):
        config = load_experiment_config()
        frozen = frozen_artifacts(config)

        record = manifest(frozen, config)
        missing = missing_artifacts(frozen)

        self.assertTrue(record["m9_run"].startswith("nb_discounted_hierarchical_v1/"))
        self.assertTrue(record["daily_run"].startswith("zinb_daily_hierarchical_v1/"))
        self.assertIsInstance(missing, list)


if __name__ == "__main__":
    unittest.main()
