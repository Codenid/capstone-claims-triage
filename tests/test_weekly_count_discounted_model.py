import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import pymc as pm
import yaml

from src.models.weekly_counts.data import fit_rows, prepare_model_frame
from src.models.weekly_counts.discounted_reference import row_discounted_shares
from src.models.weekly_counts.models import (
    nb_discounted_hierarchical_v1,
    nb_rolling_4_hierarchical_v3,
)
from src.models.weekly_counts.registry import get_model
from src.models.weekly_counts.reporting import file_sha256

CONFIG_DIR = Path("configs/weekly_counts")
MODULE = nb_discounted_hierarchical_v1


def weekly_panel(fit_weeks=8, calibration_weeks=3, validation_weeks=2):
    rng = np.random.default_rng(11)
    total_weeks = fit_weeks + calibration_weeks + validation_weeks
    weeks = pd.date_range("2023-01-02", periods=total_weeks, freq="7D")
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


class DiscountedHierarchicalTests(unittest.TestCase):
    def setUp(self):
        path = CONFIG_DIR / f"{MODULE.MODEL_ID}.yaml"
        self.settings = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.settings["clusters"] = 3
        self.panel = weekly_panel()
        self.frame = MODULE.prepare_frame(self.panel, self.settings)
        self.fit = fit_rows(self.frame)
        self.model = MODULE.build_model(self.fit, self.settings)

    def test_registry_and_frozen_share_settings(self):
        self.assertIs(get_model(MODULE.MODEL_ID), MODULE)
        self.assertEqual(self.settings["model_id"], MODULE.MODEL_ID)
        self.assertEqual(self.settings["share"], {"discount": 0.5, "cap": None})

    def test_scores_the_same_rows_as_v3_with_its_own_shares(self):
        v3 = nb_rolling_4_hierarchical_v3.prepare_frame(self.panel, self.settings)
        share = self.settings["share"]

        columns = ["split", "week", "cluster_id"]
        pd.testing.assert_frame_equal(self.frame[columns], v3[columns])
        np.testing.assert_allclose(
            self.frame["recent_share"],
            row_discounted_shares(self.panel, 3, share["discount"], share["cap"])[
                4 * 3 :
            ],
        )

    def test_means_follow_the_shares_and_sum_to_the_weekly_total(self):
        mu = pm.draw(self.model["mu"], random_seed=42)
        weekly = pd.Series(mu).groupby(self.fit["week"].to_numpy()).sum()
        totals = self.fit.groupby("week")["weekly_total"].first()

        np.testing.assert_allclose(
            mu, self.fit["weekly_total"] * self.fit["recent_share"], atol=1e-12
        )
        np.testing.assert_allclose(weekly.to_numpy(), totals.to_numpy(), atol=1e-10)

    def test_one_positive_dispersion_per_cluster(self):
        alpha = pm.draw(self.model["alpha"], draws=100, random_seed=42)

        self.assertEqual(alpha.shape, (100, 3))
        self.assertTrue((alpha > 0).all())

    def test_fit_ignores_calibration_and_validation_rows(self):
        changed = self.panel.copy()
        later = changed["split"] != "fit"
        changed.loc[later, "complaint_count"] *= 7
        changed_fit = fit_rows(MODULE.prepare_frame(changed, self.settings))
        point = self.model.initial_point()

        pd.testing.assert_frame_equal(changed_fit, self.fit)
        self.assertEqual(
            self.model.compile_logp()(point),
            MODULE.build_model(changed_fit, self.settings).compile_logp()(point),
        )

    def test_release_freezes_lf_source_and_config(self):
        releases = json.loads(
            (CONFIG_DIR / "releases.json").read_text(encoding="utf-8")
        )
        release = releases[MODULE.MODEL_ID]
        source = Path(MODULE.__file__)
        config = CONFIG_DIR / f"{MODULE.MODEL_ID}.yaml"

        self.assertEqual(release["model_source_sha256"], file_sha256(source))
        self.assertEqual(release["config_sha256"], file_sha256(config))
        self.assertEqual(release["candidate_role"], "candidate")
        for path in (source, config):
            self.assertNotIn(b"\r", path.read_bytes(), path)


if __name__ == "__main__":
    unittest.main()
