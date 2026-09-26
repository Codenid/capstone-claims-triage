"""Compatibility exports for the modular weekly count model package."""

from src.models.weekly_counts.contracts import (
    INTERVALS,
    PREDICTIVE_QUANTILES,
    REQUIRED_COLUMNS,
    SPLITS,
)
from src.models.weekly_counts.data import (
    _coerce_complete_week,
    prepare_model_frame,
    validate_weekly_counts,
)
from src.models.weekly_counts.diagnostics import (
    posterior_diagnostics,
    prior_predictive_summary,
)
from src.models.weekly_counts.metrics import (
    calibration_acceptance,
    evaluate_predictions,
    predictive_metrics,
    weighted_interval_score,
)
from src.models.weekly_counts.negative_binomial import (
    expected_counts,
    negative_binomial_draws,
    zero_sum_basis,
)
from src.models.weekly_counts.poisson import fit_baseline_rates
from src.models.weekly_counts.prediction import (
    _quantile_label,
    generate_predictions,
    summarize_draws,
)
from src.models.weekly_counts.reporting import (
    config_fingerprint,
    file_sha256,
    save_offline_record,
    save_plots,
)
from src.models.weekly_counts.run import main
from src.models.weekly_counts.sampling import (
    fit_model,
    posterior_arrays,
    simulate_prior_predictive,
)

__all__ = [
    "INTERVALS",
    "PREDICTIVE_QUANTILES",
    "REQUIRED_COLUMNS",
    "SPLITS",
    "calibration_acceptance",
    "config_fingerprint",
    "evaluate_predictions",
    "expected_counts",
    "file_sha256",
    "fit_baseline_rates",
    "fit_model",
    "generate_predictions",
    "main",
    "negative_binomial_draws",
    "posterior_arrays",
    "posterior_diagnostics",
    "predictive_metrics",
    "prepare_model_frame",
    "prior_predictive_summary",
    "save_offline_record",
    "save_plots",
    "simulate_prior_predictive",
    "summarize_draws",
    "validate_weekly_counts",
    "weighted_interval_score",
    "zero_sum_basis",
]


if __name__ == "__main__":
    main()
