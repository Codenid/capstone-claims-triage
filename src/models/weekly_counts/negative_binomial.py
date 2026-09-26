"""Negative Binomial helpers for weekly count models."""

from __future__ import annotations

import math

import numpy as np


def zero_sum_basis(clusters: int) -> np.ndarray:
    if clusters < 2:
        raise ValueError("M9 requires at least two clusters.")
    basis = np.zeros((clusters, clusters - 1), dtype=float)
    for column in range(clusters - 1):
        denominator = math.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1 / denominator
        basis[column + 1, column] = -(column + 1) / denominator
    return basis * math.sqrt(clusters / (clusters - 1))


def expected_counts(
    log_rate: np.ndarray,
    annual_trend: np.ndarray,
    cluster_index: np.ndarray,
    weekly_total: np.ndarray,
    time_years: np.ndarray,
) -> np.ndarray:
    unique_times, time_index = np.unique(time_years, return_inverse=True)
    logits = (
        log_rate[:, :, None]
        + annual_trend[:, :, None] * unique_times[None, None, :]
    )
    maximum = logits.max(axis=1)
    log_normalizer = maximum + np.log(
        np.exp(logits - maximum[:, None, :]).sum(axis=1)
    )
    log_mu = (
        np.log(weekly_total)[None, :]
        + log_rate[:, cluster_index]
        + annual_trend[:, cluster_index] * time_years[None, :]
        - log_normalizer[:, time_index]
    )
    values = np.exp(log_mu)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("M9 expected counts must be positive and finite.")
    return values


def negative_binomial_draws(
    mu: np.ndarray,
    alpha: np.ndarray,
    seed: int,
) -> np.ndarray:
    if mu.shape != alpha.shape:
        raise ValueError("M9 mu and alpha draws must have the same shape.")
    probability = alpha / (alpha + mu)
    if (
        not np.isfinite(probability).all()
        or (probability <= 0).any()
        or (probability >= 1).any()
    ):
        raise ValueError("M9 Negative Binomial probabilities are invalid.")
    return np.random.default_rng(seed).negative_binomial(alpha, probability)
