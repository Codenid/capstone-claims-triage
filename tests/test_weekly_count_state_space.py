from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from src.models.weekly_counts.negative_binomial import zero_sum_basis
from src.models.weekly_counts.state_space import (
    build_model,
    one_step_draws,
    week_matrix,
)

PRIORS = {
    "tau_sigma": 0.5,
    "nu_alpha": 2.0,
    "nu_beta": 0.1,
    "initial_sigma": 3.0,
    "log_alpha_global_mean": 2.302585093,
    "log_alpha_global_sigma": 1.0,
    "log_alpha_sigma": 0.75,
}


def counts(weeks: int = 6) -> np.ndarray:
    return np.random.default_rng(0).poisson([60, 30, 10], size=(weeks, 3))


class StateSpaceTests(unittest.TestCase):
    def test_week_matrix_orders_weeks_and_clusters(self):
        frame = pd.DataFrame(
            {
                "week": ["2023-01-09", "2023-01-02", "2023-01-09", "2023-01-02"],
                "cluster_id": [1, 1, 0, 0],
                "complaint_count": [4, 2, 3, 1],
            }
        )

        values, totals = week_matrix(frame, clusters=2)

        np.testing.assert_array_equal(values, [[1, 2], [3, 4]])
        np.testing.assert_array_equal(totals, [3, 7])

    def test_means_are_shares_of_the_weekly_total(self):
        observed = counts()
        totals = observed.sum(axis=1).astype(float)
        model = build_model(observed, totals, PRIORS)

        mu = pm.draw(model["mu"], draws=5, random_seed=1)

        self.assertEqual(mu.shape, (5, 6, 3))
        np.testing.assert_allclose(mu.sum(axis=2), np.broadcast_to(totals, (5, 6)))

    def test_fixed_hyperparameters_leave_only_the_states_free(self):
        observed = counts()
        fixed = {"tau": 0.2, "nu": 5.0, "alpha": np.array([10.0, 10.0, 10.0])}

        model = build_model(observed, observed.sum(axis=1).astype(float), PRIORS, fixed)

        free = {variable.name for variable in model.free_RVs}
        self.assertEqual(free, {"z"})

    def test_one_step_draws_center_on_the_last_shares(self):
        z_last = np.tile(np.array([[0.5, -0.2]]), (4000, 1))
        idata = SimpleNamespace(
            posterior=xr.Dataset(
                {"z_last": (("chain", "draw", "contrast"), z_last[None])}
            )
        )
        fixed = {"tau": 1e-6, "nu": 5.0, "alpha": np.array([1e6, 1e6, 1e6])}

        draws = one_step_draws(idata, 1000.0, fixed, np.random.default_rng(0))

        eta = z_last[0] @ zero_sum_basis(3).T
        expected = 1000 * np.exp(eta) / np.exp(eta).sum()
        np.testing.assert_allclose(draws.mean(axis=0), expected, rtol=0.02)


if __name__ == "__main__":
    unittest.main()
