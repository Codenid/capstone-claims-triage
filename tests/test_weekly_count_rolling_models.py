import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import pymc as pm
import yaml

from src.models.weekly_counts.data import fit_rows, prepare_model_frame
from src.models.weekly_counts.models import (
    nb_rolling_4_global_v1,
    nb_rolling_4_hierarchical_v1,
    nb_static_global_v3,
)
from src.models.weekly_counts.registry import get_model
from src.models.weekly_counts.reporting import file_sha256
from src.models.weekly_counts.rolling_reference import row_shares
from src.models.weekly_counts.run import model_frame
from src.models.weekly_counts.sampling import (
    sample_predictions,
    sample_prior_predictive,
)

CONFIG_DIR = Path("configs/weekly_counts")
MODEL_ID = "nb_rolling_4_global_v1"


def weekly_panel(fit_weeks=8, calibration_weeks=3, validation_weeks=2):
    rng = np.random.default_rng(11)
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


class RollingNegativeBinomialTests(unittest.TestCase):
    def setUp(self):
        path = CONFIG_DIR / f"{MODEL_ID}.yaml"
        self.settings = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.settings["clusters"] = 3
        self.panel = weekly_panel()
        self.frame = nb_rolling_4_global_v1.prepare_frame(self.panel, self.settings)
        self.fit = fit_rows(self.frame)
        self.model = nb_rolling_4_global_v1.build_model(self.fit, self.settings)

    def test_registry_returns_the_nb_r4_module(self):
        self.assertIs(get_model(MODEL_ID), nb_rolling_4_global_v1)
        self.assertEqual(self.settings["model_id"], MODEL_ID)

    def test_prepare_frame_keeps_rows_with_a_full_window(self):
        first_weeks = self.panel["week"].drop_duplicates().iloc[:4]

        self.assertFalse(self.frame["week"].isin(first_weeks).any())
        self.assertEqual(len(self.fit), (8 - 4) * 3)
        # The panel is sorted by week, so the first 4 weeks are its first rows.
        np.testing.assert_allclose(
            self.frame["recent_share"], row_shares(self.panel, 3, 4)[4 * 3 :]
        )

    def test_recent_share_uses_only_previous_weeks(self):
        weeks = self.panel["week"].drop_duplicates().to_numpy()
        changed = self.panel.copy()
        changed.loc[changed["week"] == weeks[5], "complaint_count"] += 40

        before = nb_rolling_4_global_v1.prepare_frame(self.panel, self.settings)
        after = nb_rolling_4_global_v1.prepare_frame(changed, self.settings)

        same_week = before["week"] == weeks[5]
        next_week = before["week"] == weeks[6]
        np.testing.assert_array_equal(
            before.loc[same_week, "recent_share"], after.loc[same_week, "recent_share"]
        )
        self.assertFalse(
            np.allclose(
                before.loc[next_week, "recent_share"], after.loc[next_week, "recent_share"]
            )
        )

    def test_alpha_is_one_positive_scalar(self):
        self.assertIsInstance(self.model, pm.Model)
        self.assertEqual(self.model["alpha"].ndim, 0)

        alpha = pm.draw(self.model["alpha"], draws=200, random_seed=42)

        self.assertTrue((alpha > 0).all())

    def test_means_follow_recent_shares_and_sum_to_weekly_total(self):
        mu = pm.draw(self.model["mu"], random_seed=42)
        weekly = pd.Series(mu).groupby(self.fit["week"].to_numpy()).sum()
        totals = self.fit.groupby("week")["weekly_total"].first()

        np.testing.assert_allclose(
            mu, self.fit["weekly_total"] * self.fit["recent_share"], rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(weekly.to_numpy(), totals.to_numpy(), rtol=0, atol=1e-10)

    def test_prior_predictive_is_reproducible_with_a_seed(self):
        first = sample_prior_predictive(self.model, 5, "observed", seed=42)
        second = sample_prior_predictive(self.model, 5, "observed", seed=42)

        self.assertEqual(first.shape, (5, len(self.fit)))
        np.testing.assert_array_equal(first, second)

    def test_posterior_predictive_uses_the_shares_of_each_prediction_row(self):
        with self.model:
            idata = pm.sample_prior_predictive(draws=4, random_seed=42)
        posterior = idata.prior.stack(sample=("chain", "draw"))
        prediction_model = nb_rolling_4_global_v1.build_model(self.frame, self.settings)

        expected, predictive = sample_predictions(
            prediction_model, posterior, "mu", "observed", seed=42
        )

        self.assertEqual(predictive.shape, (4, len(self.frame)))
        np.testing.assert_allclose(
            expected,
            np.broadcast_to(
                self.frame["weekly_total"] * self.frame["recent_share"], expected.shape
            ),
        )

    def test_fit_ignores_calibration_and_validation_rows(self):
        changed = self.panel.copy()
        later = changed["split"] != "fit"
        changed.loc[later, "complaint_count"] = changed.loc[later, "complaint_count"] * 7
        changed_fit = fit_rows(nb_rolling_4_global_v1.prepare_frame(changed, self.settings))
        changed_model = nb_rolling_4_global_v1.build_model(changed_fit, self.settings)
        point = self.model.initial_point()

        pd.testing.assert_frame_equal(changed_fit, self.fit)
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
            file_sha256(Path(nb_rolling_4_global_v1.__file__)),
        )
        self.assertEqual(
            release["config_sha256"], file_sha256(CONFIG_DIR / f"{MODEL_ID}.yaml")
        )
        self.assertEqual(release["source_status"], "native_versioned_source")
        self.assertIsNone(release["historical_run_id"])
        self.assertEqual(release["candidate_role"], "candidate")

    def test_hashed_files_use_lf_on_every_platform(self):
        for path in (
            Path(nb_rolling_4_global_v1.__file__),
            CONFIG_DIR / f"{MODEL_ID}.yaml",
        ):
            self.assertNotIn(b"\r", path.read_bytes(), path)


class RollingHierarchicalNegativeBinomialTests(unittest.TestCase):
    model_id = "nb_rolling_4_hierarchical_v1"

    def setUp(self):
        path = CONFIG_DIR / f"{self.model_id}.yaml"
        self.settings = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.settings["clusters"] = 3
        self.panel = weekly_panel()
        self.frame = nb_rolling_4_hierarchical_v1.prepare_frame(self.panel, self.settings)
        self.fit = fit_rows(self.frame)
        self.model = nb_rolling_4_hierarchical_v1.build_model(self.fit, self.settings)

    def test_registry_and_one_positive_alpha_per_cluster(self):
        self.assertIs(get_model(self.model_id), nb_rolling_4_hierarchical_v1)

        alpha = pm.draw(self.model["alpha"], draws=100, random_seed=42)

        self.assertEqual(alpha.shape, (100, 3))
        self.assertTrue((alpha > 0).all())

    def test_cluster_deviations_sum_to_zero(self):
        global_draws, alpha = pm.draw(
            [self.model["log_alpha_global"], self.model["alpha"]],
            draws=50,
            random_seed=42,
        )

        deviations = np.log(alpha) - global_draws[:, None]
        np.testing.assert_allclose(deviations.sum(axis=1), 0.0, atol=1e-10)

    def test_zero_sigma_reduces_to_one_global_alpha(self):
        fixed = pm.do(self.model, {"log_alpha_sigma": 0.0})

        alpha = pm.draw(fixed["alpha"], draws=20, random_seed=42)

        np.testing.assert_allclose(alpha, alpha[:, :1] * np.ones((1, 3)))

    def test_mean_is_the_same_as_nb_r4(self):
        global_frame = nb_rolling_4_global_v1.prepare_frame(self.panel, self.settings)
        mu = pm.draw(self.model["mu"], random_seed=42)

        pd.testing.assert_frame_equal(self.frame, global_frame)
        np.testing.assert_allclose(
            mu, self.fit["weekly_total"] * self.fit["recent_share"], rtol=0, atol=1e-12
        )

    def test_release_freezes_lf_source_and_config(self):
        releases = json.loads(
            (CONFIG_DIR / "releases.json").read_text(encoding="utf-8")
        )
        release = releases[self.model_id]
        source = Path(nb_rolling_4_hierarchical_v1.__file__)
        config = CONFIG_DIR / f"{self.model_id}.yaml"

        self.assertEqual(release["model_source_sha256"], file_sha256(source))
        self.assertEqual(release["config_sha256"], file_sha256(config))
        self.assertEqual(release["candidate_role"], "candidate")
        self.assertIsNone(release["historical_run_id"])
        for path in (source, config):
            self.assertNotIn(b"\r", path.read_bytes(), path)


class ModelFrameTests(unittest.TestCase):
    def test_model_frame_is_the_panel_unless_the_model_prepares_one(self):
        panel = weekly_panel()
        settings = {"clusters": 3}

        self.assertIs(model_frame(nb_static_global_v3, panel, settings), panel)
        self.assertEqual(
            len(model_frame(nb_rolling_4_global_v1, panel, settings)),
            len(panel) - 4 * 3,
        )


if __name__ == "__main__":
    unittest.main()
