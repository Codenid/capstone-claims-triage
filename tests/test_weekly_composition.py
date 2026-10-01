import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import multinomial
import yaml

from src.models.weekly_composition.baselines import (
    baseline_draws,
    baseline_scores,
    static_concentration,
)
from src.models.weekly_composition.data import (
    composition_panel,
    split_weeks,
    windowed_weeks,
)
from src.models.weekly_composition.evaluation import (
    COMPOSITION_RULE,
    compare_with_baselines,
    composition_acceptance,
    long_predictions,
    total_variation,
)
from src.models.weekly_composition.models import (
    dirichlet_multinomial_rolling_4_v1,
    dirichlet_multinomial_static_v1,
)
from src.models.weekly_composition.registry import get_model
from src.models.weekly_composition.run import model_panel
from src.models.weekly_composition.scores import (
    dirichlet_multinomial_draws,
    dirichlet_multinomial_logpmf,
    joint_log_score,
    multinomial_draws,
    multinomial_logpmf,
)
from src.models.weekly_counts.data import prepare_model_frame
from src.models.weekly_counts.reporting import file_sha256
from src.models.weekly_counts.rolling_reference import rolling_shares
from src.models.weekly_counts.sampling import sample_predictions

CONFIG_DIR = Path("configs/weekly_composition")
MODULES = (dirichlet_multinomial_static_v1, dirichlet_multinomial_rolling_4_v1)
ROLES = {
    dirichlet_multinomial_static_v1.MODEL_ID: "reference",
    dirichlet_multinomial_rolling_4_v1.MODEL_ID: "candidate",
}
SPLIT_WEEKS = {"fit": 8, "calibration": 3, "validation": 2}


def weekly_frame():
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
    return frame


def load_settings(module):
    path = CONFIG_DIR / f"{module.MODEL_ID}.yaml"
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings["clusters"] = 3
    return settings


def fit_weeks(module, frame):
    return split_weeks(model_panel(module, composition_panel(frame, 3)), "fit")


def assert_draws_sum_to_totals(draws, totals):
    expected = np.broadcast_to(totals, draws.shape[:-1])
    np.testing.assert_array_equal(draws.sum(axis=-1), expected)


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.frame = weekly_frame()
        self.panel = composition_panel(self.frame, 3)

    def test_builds_one_row_per_week_that_sums_to_the_total(self):
        first = self.frame.loc[self.frame["week"] == self.frame["week"].min()]

        self.assertEqual(self.panel["counts"].shape, (13, 3))
        assert_draws_sum_to_totals(self.panel["counts"], self.panel["weekly_total"])
        np.testing.assert_array_equal(
            self.panel["counts"][0], first.sort_values("cluster_id")["complaint_count"]
        )
        self.assertEqual(list(self.panel["split"][:8]), ["fit"] * 8)

    def test_recent_shares_are_the_b1_r4_shares(self):
        expected = rolling_shares(self.frame, 3, 4).to_numpy()

        np.testing.assert_array_equal(self.panel["recent_share"], expected)
        self.assertTrue(np.isnan(self.panel["recent_share"][:4]).all())
        self.assertEqual(len(windowed_weeks(self.panel)["week"]), 13 - 4)

    def test_rejects_counts_that_do_not_sum_to_the_total(self):
        broken = self.frame.copy()
        broken.loc[0, "complaint_count"] += 1

        with self.assertRaises(ValueError):
            composition_panel(broken, 3)

    def test_only_the_rolling_model_drops_weeks_without_a_window(self):
        static = fit_weeks(dirichlet_multinomial_static_v1, self.frame)
        rolling = fit_weeks(dirichlet_multinomial_rolling_4_v1, self.frame)

        self.assertEqual(len(static["week"]), 8)
        self.assertEqual(len(rolling["week"]), 4)


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.counts = np.array([[5, 3, 2], [0, 4, 6]])
        self.concentration = np.array([[2.0, 1.5, 0.5], [0.3, 1.0, 4.0]])
        self.shares = np.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]])

    def test_multinomial_matches_scipy(self):
        expected = [
            multinomial.logpmf(row, row.sum(), share)
            for row, share in zip(self.counts, self.shares)
        ]

        actual = multinomial_logpmf(self.counts, self.shares)
        np.testing.assert_allclose(actual, expected)

    def test_dirichlet_multinomial_matches_pymc(self):
        for row, concentration in zip(self.counts, self.concentration):
            distribution = pm.DirichletMultinomial.dist(n=row.sum(), a=concentration)
            expected = float(pm.logp(distribution, row).eval())

            actual = float(dirichlet_multinomial_logpmf(row, concentration))
            self.assertAlmostEqual(actual, expected)

    def test_large_concentration_approaches_the_multinomial(self):
        share = self.shares[0]

        np.testing.assert_allclose(
            dirichlet_multinomial_logpmf(self.counts, 1e9 * share),
            multinomial_logpmf(self.counts, share),
            atol=1e-6,
        )

    def test_joint_score_of_identical_draws_is_the_single_score(self):
        draws = np.broadcast_to(self.concentration, (5, 2, 3))

        np.testing.assert_allclose(
            joint_log_score(self.counts, draws),
            dirichlet_multinomial_logpmf(self.counts, self.concentration),
        )

    def test_draws_sum_to_the_week_total_and_repeat_with_a_seed(self):
        totals = np.array([10, 20])
        concentration = np.broadcast_to(self.concentration, (6, 2, 3))

        def multinomial_sample(seed):
            rng = np.random.default_rng(seed)
            return multinomial_draws(totals, self.shares, 6, rng)

        def dirichlet_sample(seed):
            rng = np.random.default_rng(seed)
            return dirichlet_multinomial_draws(totals, concentration, rng)

        for sample in (multinomial_sample, dirichlet_sample):
            values = sample(42)
            self.assertEqual(values.shape, (6, 2, 3))
            assert_draws_sum_to_totals(values, totals)
            np.testing.assert_array_equal(values, sample(42))


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.panel = composition_panel(weekly_frame(), 3)
        self.evaluation = windowed_weeks(self.panel)
        self.fit_counts = split_weeks(self.panel, "fit")["counts"]
        self.concentration = static_concentration(self.fit_counts, 1.0)

    def test_b2_posterior_adds_every_fit_count_to_the_prior(self):
        expected = 1.0 + self.fit_counts.sum(axis=0)

        np.testing.assert_array_equal(self.concentration, expected)

    def test_baseline_scores_are_finite_and_use_the_recent_shares(self):
        scores = baseline_scores(self.evaluation, self.concentration)
        counts, shares = self.evaluation["counts"], self.evaluation["recent_share"]

        self.assertTrue(all(np.isfinite(values).all() for values in scores.values()))
        np.testing.assert_allclose(
            scores["b2_rolling_4"], multinomial_logpmf(counts, shares)
        )

    def test_baseline_draws_sum_to_the_total_and_repeat_with_a_seed(self):
        first = baseline_draws(self.evaluation, self.concentration, 7, seed=43)
        second = baseline_draws(self.evaluation, self.concentration, 7, seed=43)

        for name, values in first.items():
            assert_draws_sum_to_totals(values, self.evaluation["weekly_total"])
            np.testing.assert_array_equal(values, second[name])


class CompositionModelTests(unittest.TestCase):
    def setUp(self):
        self.frame = weekly_frame()
        self.evaluation = windowed_weeks(composition_panel(self.frame, 3))

    def test_registry_and_one_positive_kappa(self):
        for module in MODULES:
            with self.subTest(module.MODEL_ID):
                settings = load_settings(module)
                model = module.build_model(fit_weeks(module, self.frame), settings)
                kappa, rho = pm.draw(
                    [model["kappa"], model["rho"]], draws=200, random_seed=42
                )

                self.assertIs(get_model(module.MODEL_ID), module)
                self.assertEqual(settings["model_id"], module.MODEL_ID)
                self.assertIsInstance(model, pm.Model)
                self.assertTrue((kappa > 0).all())
                np.testing.assert_allclose(rho, 1 / (kappa + 1))

    def test_concentration_sums_to_kappa_every_week(self):
        for module in MODULES:
            with self.subTest(module.MODEL_ID):
                fit = fit_weeks(module, self.frame)
                model = module.build_model(fit, load_settings(module))
                kappa, concentration = pm.draw(
                    [model["kappa"], model["concentration"]], random_seed=42
                )

                np.testing.assert_allclose(concentration.sum(axis=1), kappa)
                if module is dirichlet_multinomial_rolling_4_v1:
                    expected = kappa * fit["recent_share"]
                    np.testing.assert_allclose(concentration, expected)

    def test_prior_draws_sum_to_the_week_total(self):
        for module in MODULES:
            with self.subTest(module.MODEL_ID):
                fit = fit_weeks(module, self.frame)
                model = module.build_model(fit, load_settings(module))
                observed = pm.draw(model["observed"], draws=5, random_seed=42)

                assert_draws_sum_to_totals(observed, fit["weekly_total"])

    def test_predictions_use_the_weeks_of_the_prediction_panel(self):
        for module in MODULES:
            with self.subTest(module.MODEL_ID):
                settings = load_settings(module)
                with module.build_model(fit_weeks(module, self.frame), settings):
                    idata = pm.sample_prior_predictive(draws=4, random_seed=42)
                posterior = idata.prior.stack(sample=("chain", "draw"))
                # The runner drops week-sized variables in the same way.
                weekly = [
                    name for name, values in posterior.items() if "week" in values.dims
                ]
                prediction_model = module.build_model(self.evaluation, settings)

                concentration, draws = sample_predictions(
                    prediction_model,
                    posterior.drop_vars(weekly),
                    "concentration",
                    "observed",
                    seed=42,
                )

                self.assertEqual(draws.shape, (4, 9, 3))
                assert_draws_sum_to_totals(draws, self.evaluation["weekly_total"])
                if module is dirichlet_multinomial_rolling_4_v1:
                    kappa = posterior["kappa"].to_numpy()[:, None, None]
                    expected = kappa * self.evaluation["recent_share"][None]
                    np.testing.assert_allclose(concentration, expected)

    def test_fit_ignores_calibration_and_validation_weeks(self):
        changed = self.frame.copy()
        later = changed["split"] != "fit"
        changed.loc[later, ["complaint_count", "weekly_total"]] *= 7

        for module in MODULES:
            with self.subTest(module.MODEL_ID):
                settings = load_settings(module)
                model = module.build_model(fit_weeks(module, self.frame), settings)
                other = module.build_model(fit_weeks(module, changed), settings)
                point = model.initial_point()

                self.assertEqual(
                    model.compile_logp()(point), other.compile_logp()(point)
                )

    def test_release_freezes_lf_source_and_config(self):
        path = CONFIG_DIR / "releases.json"
        releases = json.loads(path.read_text(encoding="utf-8"))

        for module in MODULES:
            with self.subTest(module.MODEL_ID):
                release = releases[module.MODEL_ID]
                source = Path(module.__file__)
                config = CONFIG_DIR / f"{module.MODEL_ID}.yaml"

                self.assertEqual(release["model_source_sha256"], file_sha256(source))
                self.assertEqual(release["config_sha256"], file_sha256(config))
                self.assertEqual(release["source_status"], "native_versioned_source")
                self.assertIsNone(release["historical_run_id"])
                self.assertEqual(release["candidate_role"], ROLES[module.MODEL_ID])
                for hashed in (source, config):
                    self.assertNotIn(b"\r", hashed.read_bytes(), hashed)


class EvaluationTests(unittest.TestCase):
    def test_long_predictions_keep_the_week_and_cluster_order(self):
        panel = windowed_weeks(composition_panel(weekly_frame(), 3))
        counts = panel["counts"].reshape(-1)
        draws = {"model": np.broadcast_to(panel["counts"], (5, 9, 3))}

        frame = long_predictions(panel, draws)

        self.assertEqual(len(frame), 9 * 3)
        np.testing.assert_array_equal(frame["complaint_count"], counts)
        np.testing.assert_array_equal(frame["model_p50"], counts)
        self.assertEqual(frame["cluster_id"].tolist()[:4], [0, 1, 2, 0])
        self.assertEqual(total_variation(panel, draws["model"]), 0.0)

    def test_compares_with_the_best_baseline_reproducibly(self):
        scores = {
            "model": np.array([-10.0, -11.0, -9.0, -10.5]),
            "b2_static": np.array([-30.0, -31.0, -29.0, -32.0]),
            "b2_rolling_4": np.array([-20.0, -19.0, -21.0, -20.0]),
        }
        baselines = ("b2_static", "b2_rolling_4")

        first = compare_with_baselines(scores, baselines, seed=42)
        second = compare_with_baselines(scores, baselines, seed=42)

        self.assertEqual(first["best_baseline"], "b2_rolling_4")
        # Weekly differences against B2-R4 are 10, 8, 12 and 9.5.
        self.assertAlmostEqual(first["difference"]["estimate"], 9.875)
        self.assertGreater(first["difference"]["p025"], 0)
        self.assertEqual(first, second)

    def test_promotion_needs_a_gain_with_an_interval_above_zero(self):
        settings = {
            "acceptance": {
                "rule": COMPOSITION_RULE,
                "maximum_rhat": 1.01,
                "minimum_ess": 400,
                "coverage_80_minimum": 0.70,
                "coverage_80_maximum": 0.90,
                "coverage_95_minimum": 0.88,
                "coverage_95_maximum": 0.99,
            }
        }
        diagnostics = {
            "rhat_max": 1.0,
            "ess_bulk_min": 900,
            "ess_tail_min": 800,
            "divergences": 0,
        }
        calibration = {"coverage_80": 0.82, "coverage_95": 0.95}
        clear = {"difference": {"estimate": 3.0, "p025": 1.0}}
        noisy = {"difference": {"estimate": 3.0, "p025": -0.5}}

        accepted = composition_acceptance(diagnostics, calibration, clear, settings)
        rejected = composition_acceptance(diagnostics, calibration, noisy, settings)

        self.assertTrue(accepted["accepted"])
        self.assertFalse(rejected["accepted"])
        self.assertFalse(rejected["checks"]["log_score_bootstrap"])
        settings["acceptance"]["rule"] = "best_baseline_bootstrap_v1"
        with self.assertRaises(ValueError):
            composition_acceptance(diagnostics, calibration, clear, settings)


if __name__ == "__main__":
    unittest.main()
