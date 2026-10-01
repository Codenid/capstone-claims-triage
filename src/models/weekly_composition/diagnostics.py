"""Prior predictive summary of the composition models (models_plan.md §10)."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.models.weekly_composition.data import windowed_weeks
from src.models.weekly_counts.diagnostics import (
    VOLUME_GROUPS,
    composition_summary,
    fit_volume_groups,
    quantile_summary,
)

REFERENCE_SHARES = (0.005, 0.025, 0.1)


def implied_share_variation(kappa: np.ndarray) -> dict[str, dict[str, float]]:
    """Relative sd of a share p under Dirichlet(kappa * p).

    It is sqrt((1 - p) / (p * (kappa + 1))), before Multinomial noise.
    """
    return {
        str(share): quantile_summary(np.sqrt((1 - share) / (share * (kappa + 1))))
        for share in REFERENCE_SHARES
    }


def observed_share_variation(fit: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """How far each fit share moved from its recent share, by fit volume group."""
    weeks = windowed_weeks(fit)
    shares = weeks["counts"] / weeks["weekly_total"][:, None]
    change = np.abs(shares / weeks["recent_share"] - 1)
    groups = fit_volume_groups(fit["counts"][None])
    return {
        group: quantile_summary(change[:, groups == group]) for group in VOLUME_GROUPS
    }


def prior_check_summary(
    draws: dict[str, np.ndarray],
    fit: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compare prior draws with the same statistics observed in fit."""
    counts = draws["observed"]
    totals = fit["weekly_total"]
    groups = fit_volume_groups(fit["counts"][None])
    return {
        "kappa": quantile_summary(draws["kappa"]),
        "rho": quantile_summary(draws["rho"]),
        "implied_relative_share_sd": implied_share_variation(draws["kappa"]),
        "fit_relative_share_change": observed_share_variation(fit),
        "draw_sums_match_weekly_total": bool((counts.sum(axis=2) == totals).all()),
        "negative_count_fraction": float((counts < 0).mean()),
        "prior": composition_summary(counts, totals, groups),
        "fit_observed": composition_summary(fit["counts"][None], totals, groups),
    }
