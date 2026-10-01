"""B2 and B2-R4, the Multinomial baselines of M10 (models_plan.md §7.6, §7.9).

Neither needs MCMC. B2-R4 has no parameters, and B2 is conjugate: with a
Dirichlet(prior) share and Multinomial weeks, the posterior is
Dirichlet(prior + every fit count) and its predictive is Dirichlet-Multinomial.
"""

from __future__ import annotations

import numpy as np

from src.models.weekly_composition.scores import (
    dirichlet_multinomial_draws,
    dirichlet_multinomial_logpmf,
    multinomial_draws,
    multinomial_logpmf,
)

BASELINES = ("b2_static", "b2_rolling_4")


def static_concentration(
    fit_counts: np.ndarray,
    share_concentration: float,
) -> np.ndarray:
    """B2 posterior concentration from every fit week."""
    return share_concentration + fit_counts.sum(axis=0)


def baseline_scores(
    panel: dict[str, np.ndarray],
    concentration: np.ndarray,
) -> dict[str, np.ndarray]:
    """Exact joint log score of each week under both baselines."""
    counts = panel["counts"]
    return {
        "b2_static": dirichlet_multinomial_logpmf(counts, concentration[None, :]),
        "b2_rolling_4": multinomial_logpmf(counts, panel["recent_share"]),
    }


def baseline_draws(
    panel: dict[str, np.ndarray],
    concentration: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """(draws, weeks, clusters) predictive compositions of both baselines."""
    rng = np.random.default_rng(seed)
    totals = panel["weekly_total"]
    static = np.broadcast_to(concentration, (draws, len(totals), len(concentration)))
    return {
        "b2_static": dirichlet_multinomial_draws(totals, static, rng),
        "b2_rolling_4": multinomial_draws(totals, panel["recent_share"], draws, rng),
    }
