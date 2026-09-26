import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
import yaml

from src.models.weekly_counts.registry import DEFAULT_MODEL_ID, MODELS, get_model
from src.models.weekly_counts.reporting import file_sha256
from src.models.weekly_counts.sampling import (
    sample_predictions,
    sample_prior_predictive,
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


if __name__ == "__main__":
    unittest.main()
