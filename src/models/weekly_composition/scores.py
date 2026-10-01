"""Joint log scores and predictive draws of weekly compositions (§12.3)."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp, xlogy


def multinomial_logpmf(counts: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """log p(counts | N, p) over the last axis; N is the sum of the counts."""
    total = counts.sum(axis=-1)
    return (
        gammaln(total + 1)
        - gammaln(counts + 1).sum(axis=-1)
        + xlogy(counts, probabilities).sum(axis=-1)
    )


def dirichlet_multinomial_logpmf(
    counts: np.ndarray,
    concentration: np.ndarray,
) -> np.ndarray:
    """log p(counts | N, a) over the last axis; N is the sum of the counts."""
    total = counts.sum(axis=-1)
    concentration_total = concentration.sum(axis=-1)
    return (
        gammaln(total + 1)
        - gammaln(counts + 1).sum(axis=-1)
        + gammaln(concentration_total)
        - gammaln(total + concentration_total)
        + (gammaln(counts + concentration) - gammaln(concentration)).sum(axis=-1)
    )


def joint_log_score(counts: np.ndarray, concentration: np.ndarray) -> np.ndarray:
    """log p(y_t | data) per week, averaging posterior draws with logsumexp.

    counts is (weeks, clusters) and concentration is (draws, weeks, clusters).
    """
    per_draw = dirichlet_multinomial_logpmf(counts[None], concentration)
    return logsumexp(per_draw, axis=0) - np.log(len(concentration))


def multinomial_draws(
    totals: np.ndarray,
    probabilities: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """(draws, weeks, clusters) compositions that each sum to the week total."""
    size = (draws, *probabilities.shape)
    return rng.multinomial(totals[None, :], np.broadcast_to(probabilities, size))


def dirichlet_multinomial_draws(
    totals: np.ndarray,
    concentration: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """One composition per (draw, week): Dirichlet shares, then Multinomial."""
    gamma = rng.standard_gamma(concentration)
    shares = gamma / gamma.sum(axis=-1, keepdims=True)
    return rng.multinomial(np.broadcast_to(totals, shares.shape[:-1]), shares)
