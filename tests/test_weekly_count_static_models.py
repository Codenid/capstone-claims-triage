import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import pymc as pm
import yaml

from src.models.weekly_counts.data import fit_rows, prepare_model_frame
from src.models.weekly_counts.models import nb_static_global_v3
from src.models.weekly_counts.registry import get_model
from src.models.weekly_counts.reporting import file_sha256
from src.models.weekly_counts.sampling import (
    sample_predictions,
    sample_prior_predictive,
)

CONFIG_DIR = Path("configs/weekly_counts")
MODEL_ID = "nb_static_global_v3"


def weekly_panel() -> pd.DataFrame:
    rows = []
    specifications = [
        ("fit", "2023-01-02", (60, 30, 10)),
        ("fit", "2023-01-09", (55, 35, 10)),
        ("fit", "2023-01-16", (70, 20, 10)),
        ("calibration", "2023-02-06", (40, 40, 20)),
        ("validation", "2023-03-06", (90, 5, 5)),
    ]
    for split, week, counts in specifications:
        for cluster_id, count in enumerate(counts):
            rows.append(
                {
                    "split": split,
                    "week": week,
                    "cluster_id": cluster_id,
                    "complaint_count": count,
                    "is_complete_week": True,
                    "weekly_total": sum(counts),
                }
            )
    frame, _ = prepare_model_frame(
        pd.DataFrame(rows),
        clusters=3,
        expected_complete_weeks={"fit": 3, "calibration": 1, "validation": 1},
        time_scale_days=365.25,
    )
    return frame


class StaticGlobalNegativeBinomialTests(unittest.TestCase):
    def setUp(self):
        path = CONFIG_DIR / f"{MODEL_ID}.yaml"
        self.settings = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.settings["clusters"] = 3
        self.frame = weekly_panel()
        self.fit = fit_rows(self.frame)
        self.model = nb_static_global_v3.build_model(self.fit, self.settings)

    def test_registry_returns_the_nb_v3_module(self):
        self.assertIs(get_model(MODEL_ID), nb_static_global_v3)
        self.assertEqual(nb_static_global_v3.MODEL_ID, MODEL_ID)
        self.assertEqual(self.settings["model_id"], MODEL_ID)

    def test_build_model_returns_a_pymc_model(self):
        self.assertIsInstance(self.model, pm.Model)
        self.assertIn(nb_static_global_v3.OBSERVED_VARIABLE, self.model.named_vars)

    def test_alpha_is_one_positive_scalar(self):
        self.assertEqual(self.model["alpha"].ndim, 0)

        alpha = pm.draw(self.model["alpha"], draws=200, random_seed=42)

        self.assertEqual(alpha.shape, (200,))
        self.assertTrue((alpha > 0).all())

    def test_mu_has_one_entry_per_observation(self):
        mu = pm.draw(self.model["mu"], random_seed=42)

        self.assertEqual(mu.shape, (len(self.fit),))

    def test_means_sum_to_weekly_total_each_week(self):
        mu = pm.draw(self.model["mu"], draws=50, random_seed=42)
        week_index = pd.factorize(self.fit["week"], sort=True)[0]
        weekly_mu = np.zeros((len(mu), week_index.max() + 1))
        np.add.at(weekly_mu.T, week_index, mu.T)
        weekly_total = self.fit.groupby("week")["weekly_total"].first().to_numpy()

        np.testing.assert_allclose(
            weekly_mu, np.broadcast_to(weekly_total, weekly_mu.shape), rtol=0, atol=1e-10
        )

    def test_variance_uses_the_pymc_alpha_parameterization(self):
        share = np.array([0.6, 0.3, 0.1])
        fixed = pm.do(self.model, {"share": share, "log_alpha": np.log(2.0)})

        draws = pm.draw(fixed["observed"], draws=20000, random_seed=42)
        mu = self.fit["weekly_total"].to_numpy() * share[self.fit["cluster_id"]]

        np.testing.assert_allclose(draws.mean(axis=0), mu, rtol=0.05)
        np.testing.assert_allclose(draws.var(axis=0), mu + mu**2 / 2.0, rtol=0.1)

    def test_prior_predictive_is_reproducible_with_a_seed(self):
        first = sample_prior_predictive(self.model, 5, "observed", seed=42)
        second = sample_prior_predictive(self.model, 5, "observed", seed=42)

        self.assertEqual(first.shape, (5, len(self.fit)))
        self.assertTrue((first >= 0).all())
        np.testing.assert_array_equal(first, second)

    def test_posterior_predictive_uses_posterior_shares_for_all_splits(self):
        with self.model:
            idata = pm.sample_prior_predictive(draws=4, random_seed=42)
        posterior = idata.prior.stack(sample=("chain", "draw"))
        prediction_model = nb_static_global_v3.build_model(self.frame, self.settings)

        expected, predictive = sample_predictions(
            prediction_model, posterior, "mu", "observed", seed=42
        )

        self.assertEqual(expected.shape, (4, len(self.frame)))
        self.assertEqual(predictive.shape, (4, len(self.frame)))
        share = posterior["share"].transpose("sample", "cluster").to_numpy()
        np.testing.assert_allclose(
            expected,
            self.frame["weekly_total"].to_numpy()
            * share[:, self.frame["cluster_id"].to_numpy()],
        )

    def test_fit_ignores_calibration_and_validation_rows(self):
        changed = self.frame.copy()
        later = changed["split"] != "fit"
        changed.loc[later, "complaint_count"] = changed.loc[later, "complaint_count"] * 7
        changed_model = nb_static_global_v3.build_model(fit_rows(changed), self.settings)
        point = self.model.initial_point()

        observed = self.model.rvs_to_values[self.model["observed"]].data
        np.testing.assert_array_equal(observed, self.fit["complaint_count"].to_numpy())
        self.assertEqual(
            self.model.compile_logp()(point), changed_model.compile_logp()(point)
        )

    def test_release_freezes_source_and_config(self):
        releases = json.loads(
            (CONFIG_DIR / "releases.json").read_text(encoding="utf-8")
        )
        release = releases[MODEL_ID]

        self.assertEqual(
            release["model_source_sha256"],
            file_sha256(Path(nb_static_global_v3.__file__)),
        )
        self.assertEqual(
            release["config_sha256"], file_sha256(CONFIG_DIR / f"{MODEL_ID}.yaml")
        )
        self.assertEqual(release["source_status"], "native_versioned_source")
        self.assertIsNone(release["historical_run_id"])
        self.assertEqual(release["candidate_role"], "candidate")

    def test_hashed_files_use_lf_on_every_platform(self):
        # Hashes can match on Windows and Linux only if no CRLF reaches them.
        for path in (
            Path(nb_static_global_v3.__file__),
            CONFIG_DIR / f"{MODEL_ID}.yaml",
            CONFIG_DIR / "releases.json",
        ):
            self.assertNotIn(b"\r", path.read_bytes(), path)


if __name__ == "__main__":
    unittest.main()
