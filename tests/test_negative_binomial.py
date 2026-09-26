import unittest

import numpy as np
import pandas as pd

from src.models.weekly_counts.data import prepare_model_frame, validate_weekly_counts
from src.models.weekly_counts.fixed_poisson_reference import (
    expected_counts as reference_expected_counts,
    fit_baseline_rates,
    predictive_draws as reference_predictive_draws,
)
from src.models.weekly_counts.metrics import (
    calibration_acceptance,
    predictive_metrics,
    weighted_interval_score,
)
from src.models.weekly_counts.negative_binomial import (
    expected_counts,
    negative_binomial_draws,
    zero_sum_basis,
)
from src.models.weekly_counts.models import nb_softmax_linear_v2
from src.models.weekly_counts.prediction import build_predictions
from src.models.weekly_counts.reporting import source_manifest_sha256
from src.models.weekly_counts.sampling import sample_prior_predictive


class NegativeBinomialTests(unittest.TestCase):
    def setUp(self):
        rows = []
        specifications = [
            ("fit", "2023-01-02", True, (7, 3)),
            ("fit", "2023-01-09", True, (8, 2)),
            ("fit", "2023-01-16", False, (5, 5)),
            ("calibration", "2023-02-06", True, (6, 4)),
            ("validation", "2023-03-06", True, (9, 1)),
        ]
        for split, week, complete, counts in specifications:
            for cluster_id, count in enumerate(counts):
                rows.append(
                    {
                        "split": split,
                        "week": week,
                        "cluster_id": cluster_id,
                        "complaint_count": count,
                        "is_complete_week": complete,
                        "weekly_total": sum(counts),
                    }
                )
        self.frame = pd.DataFrame(rows)
        self.expected_weeks = {"fit": 2, "calibration": 1, "validation": 1}

    def prepared(self):
        return prepare_model_frame(
            self.frame,
            clusters=2,
            expected_complete_weeks=self.expected_weeks,
            time_scale_days=365.25,
        )[0]


    def test_excludes_partial_weeks_and_checks_frozen_counts(self):
        complete = validate_weekly_counts(
            self.frame,
            clusters=2,
            expected_complete_weeks=self.expected_weeks,
        )

        self.assertEqual(len(complete), 8)
        self.assertNotIn(pd.Timestamp("2023-01-16"), set(complete["week"]))

    def test_rejects_a_week_missing_a_cluster(self):
        invalid = self.frame.drop(index=0)

        with self.assertRaisesRegex(ValueError, "every frozen cluster"):
            validate_weekly_counts(invalid, clusters=2)

    def test_rejects_counts_that_do_not_sum_to_total(self):
        invalid = self.frame.copy()
        invalid.loc[0, "complaint_count"] = 6

        with self.assertRaisesRegex(ValueError, "sum to weekly_total"):
            validate_weekly_counts(invalid, clusters=2)

    def test_time_center_uses_only_fit_weeks(self):
        _, first_center = prepare_model_frame(
            self.frame,
            clusters=2,
            expected_complete_weeks=self.expected_weeks,
            time_scale_days=365.25,
        )
        changed = self.frame.copy()
        changed.loc[changed["split"] == "validation", "week"] = "2030-01-07"
        _, second_center = prepare_model_frame(
            changed,
            clusters=2,
            expected_complete_weeks=self.expected_weeks,
            time_scale_days=365.25,
        )

        self.assertEqual(first_center, second_center)
        self.assertEqual(first_center, pd.Timestamp("2023-01-05 12:00:00"))

    def test_baseline_uses_only_fit_counts(self):
        prepared = self.prepared()
        first = fit_baseline_rates(prepared, clusters=2)
        changed = prepared.copy()
        mask = changed["split"] == "calibration"
        changed.loc[mask & (changed["cluster_id"] == 0), "complaint_count"] = 1
        changed.loc[mask & (changed["cluster_id"] == 1), "complaint_count"] = 9
        second = fit_baseline_rates(changed, clusters=2)

        np.testing.assert_allclose(first, [0.75, 0.25])
        np.testing.assert_array_equal(first, second)

    def test_zero_sum_basis_has_unit_marginal_variance(self):
        basis = zero_sum_basis(4)

        self.assertEqual(basis.shape, (4, 3))
        np.testing.assert_allclose(basis.sum(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.square(basis).sum(axis=1), 1.0)

    def test_exposure_doubles_expected_count(self):
        log_rate = np.log(np.array([[0.25, 0.75]]))
        trend = np.zeros((1, 2))
        clusters = np.array([0, 0], dtype=np.int64)
        totals = np.array([100.0, 200.0])
        times = np.zeros(2)

        values = expected_counts(log_rate, trend, clusters, totals, times)

        np.testing.assert_allclose(values, [[25.0, 50.0]])

    def test_expected_counts_sum_to_weekly_total(self):
        log_rate = np.log(np.array([[0.25, 0.75]]))
        trend = np.array([[0.1, -0.1]])
        clusters = np.array([0, 1, 0, 1], dtype=np.int64)
        totals = np.full(4, 100.0)
        times = np.array([0.0, 0.0, 1.0, 1.0])

        values = expected_counts(log_rate, trend, clusters, totals, times)

        np.testing.assert_allclose(values.reshape(1, 2, 2).sum(axis=2), 100.0)

    def test_poisson_draws_are_seeded_and_non_negative(self):
        mu = np.array([10.0, 20.0, 30.0])

        first = reference_predictive_draws(mu, draws=4, seed=42)
        second = reference_predictive_draws(mu, draws=4, seed=42)

        self.assertEqual(first.shape, (4, 3))
        self.assertTrue((first >= 0).all())
        np.testing.assert_array_equal(first, second)

    def test_negative_binomial_draws_are_seeded_and_non_negative(self):
        mu = np.full((4, 3), 10.0)
        alpha = np.full((4, 3), 5.0)

        first = negative_binomial_draws(mu, alpha, seed=42)
        second = negative_binomial_draws(mu, alpha, seed=42)

        np.testing.assert_array_equal(first, second)
        self.assertTrue((first >= 0).all())

    def test_prior_predictive_is_seeded_and_matches_fit_shape(self):
        prepared = self.prepared()
        settings = {
            "clusters": 2,
            "priors": {
                "log_rate_sigma": 0.2,
                "annual_trend_sigma": 0.1,
                "log_alpha_global_mean": 1.0,
                "log_alpha_global_sigma": 0.2,
                "log_alpha_sigma": 0.2,
            },
            "sampling": {"prior_draws": 5},
        }

        model = nb_softmax_linear_v2.build_model(prepared, settings)
        first = sample_prior_predictive(model, 5, "observed", seed=42)
        second = sample_prior_predictive(model, 5, "observed", seed=42)

        self.assertEqual(first.shape, (5, len(prepared)))
        self.assertTrue((first >= 0).all())
        np.testing.assert_array_equal(first, second)

    def test_weighted_interval_score_is_zero_for_perfect_prediction(self):
        observed = np.array([10.0])
        intervals = [
            (0.50, np.array([10.0]), np.array([10.0])),
            (0.20, np.array([10.0]), np.array([10.0])),
            (0.05, np.array([10.0]), np.array([10.0])),
        ]

        score = weighted_interval_score(observed, observed, intervals)

        np.testing.assert_array_equal(score, [0.0])

    def test_generates_all_predictive_quantiles(self):
        prepared = self.prepared()
        draws = 20
        observed = prepared["complaint_count"].to_numpy(dtype=float)
        expected = np.tile(observed, (draws, 1))
        predictive = np.tile(observed, (draws, 1)).astype(np.int64)
        reference = predictive.copy()
        predictions = build_predictions(prepared, expected, predictive, reference)

        for column in (
            "model_p025",
            "model_p10",
            "model_p25",
            "model_p50",
            "model_p75",
            "model_p90",
            "model_p975",
            "expected_mean",
            "baseline_p50",
        ):
            self.assertIn(column, predictions)
        self.assertEqual(len(predictions), len(prepared))

    def test_predictive_metrics_are_finite(self):
        frame = pd.DataFrame(
            {
                "complaint_count": [10, 20],
                "model_p025": [5, 10],
                "model_p10": [7, 14],
                "model_p25": [8, 17],
                "model_p50": [10, 20],
                "model_p75": [12, 23],
                "model_p90": [13, 26],
                "model_p975": [15, 30],
                "model_impossible_probability": [0.0, 0.0],
            }
        )

        metrics = predictive_metrics(frame, "model")

        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["coverage_95"], 1.0)
        self.assertTrue(all(np.isfinite(list(metrics.values()))))

    def test_acceptance_uses_calibration_metrics_and_diagnostics(self):
        diagnostics = {
            "rhat_max": 1.0,
            "ess_bulk_min": 500,
            "ess_tail_min": 500,
            "divergences": 0,
        }
        calibration = {
            "model": {
                "wis": 4.0,
                "wape": 0.10,
                "coverage_80": 0.80,
                "coverage_95": 0.95,
            },
            "baseline": {"wis": 5.0, "wape": 0.12},
        }
        settings = {
            "acceptance": {
                "maximum_rhat": 1.01,
                "minimum_ess": 400,
                "coverage_80_minimum": 0.70,
                "coverage_80_maximum": 0.90,
                "coverage_95_minimum": 0.88,
                "coverage_95_maximum": 0.99,
            }
        }

        result = calibration_acceptance(diagnostics, calibration, settings)

        self.assertTrue(result["accepted"])
        self.assertTrue(all(result["checks"].values()))

    def test_source_manifest_hash_is_order_independent(self):
        first = source_manifest_sha256({"b.py": "2", "a.py": "1"})
        second = source_manifest_sha256({"a.py": "1", "b.py": "2"})

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)


if __name__ == "__main__":
    unittest.main()
