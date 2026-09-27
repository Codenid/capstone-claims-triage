import json
from pathlib import Path
import unittest

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
import yaml

from src.models.weekly_counts.data import fit_rows
from src.models.weekly_counts.diagnostics import (
    posterior_diagnostics,
    prior_check_summary,
)
from src.models.weekly_counts.metrics import candidate_status, rejection_reason
from src.models.weekly_counts.registry import DEFAULT_MODEL_ID, MODELS, get_model
from src.models.weekly_counts.reporting import file_sha256
from src.models.weekly_counts.run import (
    PILOT_SAMPLING,
    load_candidate_role,
    sampling_for_run_mode,
)
from src.models.weekly_counts.sampling import (
    sample_predictions,
    sample_prior_predictive,
    sample_prior_variables,
    subsample_posterior,
)


CONFIG_DIR = Path("configs/weekly_counts")


class WeeklyCountModelTests(unittest.TestCase):
    def setUp(self):
        self.fit = pd.DataFrame(
            {
                "cluster_id": [0, 1, 0, 1],
                "complaint_count": [7, 3, 8, 2],
                "weekly_total": [10, 10, 10, 10],
                "time_years": [-0.01, -0.01, 0.01, 0.01],
            }
        )

    def settings(self, model_id):
        path = CONFIG_DIR / f"{model_id}.yaml"
        settings = yaml.safe_load(path.read_text(encoding="utf-8"))
        settings["clusters"] = 2
        settings["sampling"]["prior_draws"] = 5
        return settings

    def test_registry_contains_only_versioned_pymc_models(self):
        self.assertEqual(
            set(MODELS),
            {
                "nb_independent_linear_v1",
                "nb_softmax_linear_v2",
                "nb_static_global_v3",
                "poisson_static_pymc_v1",
            },
        )
        self.assertEqual(DEFAULT_MODEL_ID, "nb_softmax_linear_v2")

        for model_id, model_module in MODELS.items():
            model = model_module.build_model(self.fit, self.settings(model_id))
            self.assertIsInstance(model, pm.Model)
            self.assertIn("observed", model.named_vars)
            self.assertIs(get_model(model_id), model_module)

    def test_models_generate_seeded_prior_draws(self):
        for model_id, model_module in MODELS.items():
            settings = self.settings(model_id)
            model = model_module.build_model(self.fit, settings)
            first = sample_prior_predictive(
                model, 5, model_module.OBSERVED_VARIABLE, 42
            )
            second = sample_prior_predictive(
                model, 5, model_module.OBSERVED_VARIABLE, 42
            )

            self.assertEqual(first.shape, (5, len(self.fit)))
            np.testing.assert_array_equal(first, second)

    def test_models_generate_seeded_posterior_predictions(self):
        for model_id, model_module in MODELS.items():
            settings = self.settings(model_id)
            model = model_module.build_model(self.fit, settings)
            with model:
                idata = pm.sample_prior_predictive(draws=3, random_seed=42)
            posterior = getattr(idata, "prior").stack(sample=("chain", "draw"))

            first = sample_predictions(
                model,
                posterior,
                model_module.EXPECTED_VARIABLE,
                model_module.OBSERVED_VARIABLE,
                42,
            )
            second = sample_predictions(
                model,
                posterior,
                model_module.EXPECTED_VARIABLE,
                model_module.OBSERVED_VARIABLE,
                42,
            )

            self.assertEqual(first[0].shape, (3, len(self.fit)))
            self.assertEqual(first[1].shape, (3, len(self.fit)))
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1], second[1])

    def test_release_manifest_freezes_model_sources_and_configs(self):
        releases = json.loads(
            (CONFIG_DIR / "releases.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(releases), set(MODELS))

        for model_id, model_module in MODELS.items():
            release = releases[model_id]
            self.assertEqual(
                release["model_source_sha256"],
                file_sha256(Path(model_module.__file__)),
            )
            self.assertEqual(
                release["config_sha256"],
                file_sha256(CONFIG_DIR / f"{model_id}.yaml"),
            )
            self.assertEqual(release["source_status"], model_module.SOURCE_STATUS)
            self.assertEqual(
                release["historical_run_id"], model_module.HISTORICAL_RUN_ID
            )

    def test_posterior_subsampling_uses_shared_draw_indices(self):
        idata = type(
            "InferenceData",
            (),
            {
                "posterior": xr.Dataset(
                    {
                        "first": (("chain", "draw"), [[0, 1, 2], [3, 4, 5]]),
                        "second": (("chain", "draw"), [[10, 11, 12], [13, 14, 15]]),
                    }
                )
            },
        )()

        posterior = subsample_posterior(idata, draws=3, seed=42)

        np.testing.assert_array_equal(
            posterior["second"].to_numpy() - posterior["first"].to_numpy(),
            10,
        )
        self.assertEqual(posterior.sizes["sample"], 3)


class WeeklyCountRunTests(unittest.TestCase):
    def acceptance(self, **failed):
        checks = {
            "rhat": True,
            "ess_bulk": True,
            "ess_tail": True,
            "divergences": True,
            "wis": True,
            "wape": True,
            "coverage_80": True,
            "coverage_95": True,
        }
        checks.update({name: False for name in failed})
        return {"accepted": all(checks.values()), "checks": checks}

    def test_pilot_mode_only_shortens_sampling(self):
        path = CONFIG_DIR / "poisson_static_pymc_v1.yaml"
        full = yaml.safe_load(path.read_text(encoding="utf-8"))["sampling"]
        original = dict(full)

        pilot = sampling_for_run_mode(full, "pilot")

        self.assertEqual(
            {name: pilot[name] for name in PILOT_SAMPLING},
            {"chains": 2, "tune": 250, "draws": 250, "prediction_draws": 500},
        )
        for name in set(full) - set(PILOT_SAMPLING):
            self.assertEqual(pilot[name], full[name])
        self.assertEqual(sampling_for_run_mode(full, "full"), full)
        self.assertEqual(full, original)

    def test_candidate_status_uses_plan_states(self):
        self.assertEqual(candidate_status("full", self.acceptance()), "accepted")
        self.assertEqual(
            candidate_status("pilot", self.acceptance(rhat=False)), "pilot_only"
        )
        self.assertEqual(
            candidate_status("full", self.acceptance(ess_bulk=False, wis=False)),
            "rejected_convergence",
        )
        self.assertEqual(
            candidate_status("full", self.acceptance(coverage_95=False)),
            "rejected_predictive",
        )
        self.assertEqual(
            candidate_status("full", self.acceptance(wape=False)),
            "rejected_no_practical_gain",
        )

    def test_rejection_reason_lists_failed_checks(self):
        failed = self.acceptance(ess_bulk=False, coverage_95=False)

        self.assertEqual(rejection_reason("full", failed), "ess_bulk,coverage_95")
        self.assertEqual(rejection_reason("pilot", failed), "none")
        self.assertEqual(rejection_reason("full", self.acceptance()), "none")

    def test_releases_mark_b1_as_pipeline_baseline(self):
        self.assertEqual(
            load_candidate_role(
                CONFIG_DIR / "poisson_static_pymc_v1.yaml",
                "poisson_static_pymc_v1",
            ),
            "pipeline_baseline",
        )
        self.assertEqual(
            load_candidate_role(
                CONFIG_DIR / "nb_softmax_linear_v2.yaml",
                "nb_softmax_linear_v2",
            ),
            "candidate",
        )

    def test_fit_rows_exclude_calibration_and_validation(self):
        frame = pd.DataFrame(
            {"split": ["fit", "calibration", "fit", "validation"], "value": range(4)}
        )

        fit = fit_rows(frame)

        self.assertEqual(fit["split"].unique().tolist(), ["fit"])
        self.assertEqual(fit["value"].tolist(), [0, 2])

    def test_prior_check_compares_prior_draws_with_fit(self):
        fit = pd.DataFrame(
            {
                "week": pd.to_datetime(["2023-01-02"] * 3 + ["2023-01-09"] * 3),
                "cluster_id": [0, 1, 2] * 2,
                "complaint_count": [6, 3, 1, 5, 5, 0],
                "weekly_total": [10] * 6,
            }
        )
        observed = np.array([[6, 3, 1, 5, 5, 0], [2, 2, 2, 0, 0, 20]])
        expected = np.full(observed.shape, 5.0)

        summary = prior_check_summary(observed, expected, np.array([2.0, 4.0]), fit)

        self.assertEqual(summary["negative_count_fraction"], 0.0)
        self.assertEqual(summary["counts"]["impossible_fraction"], 1 / 12)
        self.assertEqual(summary["intervals_near_mean"]["5"]["rows"], 12.0)
        self.assertEqual(summary["intervals_near_mean"]["100"], {"rows": 0.0})
        # Weekly draw sums are 10, 10, 6 and 20 against a total of 10.
        self.assertAlmostEqual(
            summary["draw_sum_relative_deviation"]["p50"],
            np.median([0.0, 0.0, -0.4, 1.0]),
        )
        # Observed fit maxima are 0.6 and 0.5; cluster 2 has one zero week.
        self.assertAlmostEqual(
            summary["fit_observed"]["max_weekly_share"]["p50"], 0.55
        )
        self.assertEqual(
            summary["fit_observed"]["zero_fraction_by_cluster"]["p50"], 0.0
        )
        # Variance-to-mean ratios at mu=20 are 1 + 20/2 and 1 + 20/4.
        self.assertEqual(
            summary["dispersion"]["variance_to_mean"]["20"]["p50"], 8.5
        )
        self.assertEqual(
            set(summary["prior"]["share_by_fit_volume"]),
            {"small", "medium", "large"},
        )

    def test_posterior_diagnostics_are_not_rounded(self):
        rng = np.random.default_rng(42)
        idata = az.from_dict(
            posterior={"x": 0.1234567 + 0.001 * rng.standard_normal((4, 200))},
            sample_stats={
                "diverging": np.zeros((4, 200), dtype=bool),
                "energy": rng.standard_normal((4, 200)),
            },
        )

        summary, diagnostics = posterior_diagnostics(idata, ("x",), ("x",))

        # Default ArviZ rounding would give two decimals for R-hat.
        self.assertNotEqual(diagnostics["rhat_max"], round(diagnostics["rhat_max"], 2))
        self.assertNotEqual(summary.loc["x", "mean"], round(summary.loc["x", "mean"], 3))
        self.assertAlmostEqual(summary.loc["x", "mean"], 0.1234567, places=4)

    def test_prior_variables_are_seeded(self):
        model_module = MODELS["poisson_static_pymc_v1"]
        fit = pd.DataFrame(
            {
                "cluster_id": [0, 1, 0, 1],
                "complaint_count": [7, 3, 8, 2],
                "weekly_total": [10, 10, 10, 10],
            }
        )
        settings = {"clusters": 2, "priors": {"rate_concentration": 1.0}}
        model = model_module.build_model(fit, settings)

        first = sample_prior_variables(model, 4, ["mu", "observed"], seed=42)
        second = sample_prior_variables(model, 4, ["mu", "observed"], seed=42)

        self.assertEqual(first["mu"].shape, (4, 4))
        self.assertEqual(first["observed"].shape, (4, 4))
        np.testing.assert_array_equal(first["observed"], second["observed"])


if __name__ == "__main__":
    unittest.main()
