"""Run the M9 hierarchical Negative Binomial weekly volume model."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from src.evaluation.experiment import (
    PROJECT_ROOT,
    load_experiment_config,
    set_seed,
)
from src.models.semantic_space import load_dvc_hash, maximum_rss_gib
from src.models.weekly_counts.contracts import SPLITS
from src.models.weekly_counts.data import prepare_model_frame
from src.models.weekly_counts.diagnostics import (
    posterior_diagnostics,
    prior_predictive_summary,
)
from src.models.weekly_counts.metrics import calibration_acceptance, evaluate_predictions
from src.models.weekly_counts.poisson import fit_baseline_rates
from src.models.weekly_counts.prediction import generate_predictions
from src.models.weekly_counts.reporting import (
    config_fingerprint,
    file_sha256,
    save_offline_record,
    save_plots,
)
from src.models.weekly_counts.sampling import fit_model, posterior_arrays


def main() -> None:
    started = time.perf_counter()
    config = load_experiment_config()
    settings = config["negative_binomial"]
    seed = config["experiment"]["seed"]
    set_seed(seed)

    paths = {
        "input": PROJECT_ROOT / config["paths"]["weekly_counts"],
        "report": PROJECT_ROOT / config["paths"]["negative_binomial_report"],
        "predictions": PROJECT_ROOT / config["paths"]["negative_binomial_predictions"],
        "metrics": PROJECT_ROOT / config["paths"]["negative_binomial_metrics"],
        "summary": PROJECT_ROOT / config["paths"]["negative_binomial_summary"],
        "backtest": PROJECT_ROOT / config["paths"]["negative_binomial_backtest_plot"],
        "coverage": PROJECT_ROOT / config["paths"]["negative_binomial_coverage_plot"],
        "residuals": PROJECT_ROOT / config["paths"]["negative_binomial_residual_plot"],
        "prior": PROJECT_ROOT / config["paths"]["negative_binomial_prior_plot"],
        "trace": PROJECT_ROOT / config["paths"]["negative_binomial_trace_plot"],
        "output": PROJECT_ROOT / config["paths"]["negative_binomial_artifacts"],
        "staging": PROJECT_ROOT / config["paths"]["negative_binomial_staging"],
    }
    if (paths["output"] / "_SUCCESS").exists():
        print(f"M9 is already complete: {paths['output'].relative_to(PROJECT_ROOT)}")
        return
    if paths["output"].exists():
        raise ValueError("M9 final output exists without _SUCCESS.")
    if paths["staging"].exists():
        raise ValueError("M9 staging exists from an incomplete run; inspect it first.")
    paths["staging"].mkdir(parents=True)
    paths["report"].parent.mkdir(parents=True, exist_ok=True)

    source_hash = load_dvc_hash(
        PROJECT_ROOT / config["paths"]["weekly_patterns_artifacts_dvc"]
    )
    if source_hash != settings["source_dvc_hash"]:
        raise ValueError("M9 weekly patterns DVC hash is not frozen.")

    raw = pd.read_csv(paths["input"])
    frame, time_center = prepare_model_frame(
        raw,
        settings["clusters"],
        settings["expected_complete_weeks"],
        settings["time_scale_days"],
    )
    fit = frame.loc[frame["split"] == "fit"].reset_index(drop=True)
    baseline_rates = fit_baseline_rates(frame, settings["clusters"])

    idata, prior_draws, versions = fit_model(fit, settings, seed)
    idata.to_netcdf(paths["staging"] / "posterior.nc")
    summary, diagnostics = posterior_diagnostics(
        idata,
        settings["sampling"]["max_treedepth"],
    )
    summary_frame = summary.reset_index().rename(columns={"index": "parameter"})
    summary_frame.to_csv(paths["summary"], index=False)
    prior_summary = prior_predictive_summary(prior_draws, fit)

    log_rate, annual_trend, alpha = posterior_arrays(
        idata,
        settings["sampling"]["prediction_draws"],
        seed,
    )
    predictions = generate_predictions(
        frame,
        log_rate,
        annual_trend,
        alpha,
        baseline_rates,
        seed,
    )
    predictions.to_csv(paths["predictions"], index=False)
    metrics, backtest = evaluate_predictions(predictions)
    metrics.to_csv(paths["metrics"], index=False)
    normalization_by_split = {}
    for split in SPLITS:
        split_predictions = predictions.loc[predictions["split"] == split]
        weekly = split_predictions.groupby("week", observed=True).agg(
            expected_sum=("expected_mean", "sum"),
            weekly_total=("weekly_total", "first"),
        )
        absolute_error = np.abs(weekly["expected_sum"] / weekly["weekly_total"] - 1)
        normalization_by_split[split] = {
            "maximum_abs_error": float(absolute_error.max()),
            "mean_abs_error": float(absolute_error.mean()),
        }
    mean_normalization = {
        "maximum_abs_error": max(
            values["maximum_abs_error"] for values in normalization_by_split.values()
        ),
        "by_split": normalization_by_split,
    }
    acceptance = calibration_acceptance(
        diagnostics,
        backtest["calibration"],
        settings,
    )

    save_plots(
        predictions,
        metrics,
        prior_draws,
        fit,
        idata,
        paths,
        settings["plot_clusters"],
    )

    formula = (
        "complaint_count[c,t] ~ NegativeBinomial(mu[c,t], alpha[c]); "
        "eta[c,t] = log_rate[c] + annual_trend[c] * time_years[t]; "
        "share[c,t] = softmax_c(eta[:,t]); "
        "mu[c,t] = weekly_total[t] * share[c,t]"
    )
    model_spec = {
        "formula": formula,
        "model_version": settings["model_version"],
        "negative_binomial_variance": "mu + mu**2 / alpha",
        "cluster_interpretation": "operational partition, not natural categories",
        "weekly_total_usage": "observed exposure; this is a conditional forecast",
        "mean_link": "softmax",
        "mean_sum_constrained": True,
        "predictive_draw_sum_constrained": False,
        "zero_sum_parameterization": "scaled Helmert contrasts with K-1 dimensions",
        "time_center": time_center.isoformat(),
        "time_scale_days": settings["time_scale_days"],
        "priors": settings["priors"],
        "sampling": settings["sampling"],
    }
    (paths["staging"] / "model_spec.json").write_text(
        json.dumps(model_spec, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    elapsed_seconds = time.perf_counter() - started
    report = {
        "stage": "M9",
        "model_version": settings["model_version"],
        "source_dvc_hash": source_hash,
        "weekly_counts_sha256": file_sha256(paths["input"]),
        "config_sha256": config_fingerprint(config),
        "formula": formula,
        "fit_only_training": True,
        "validation_used_for_selection": False,
        "complete_weeks": settings["expected_complete_weeks"],
        "rows": {
            split: int((frame["split"] == split).sum()) for split in SPLITS
        },
        "time_center": time_center.isoformat(),
        "priors": settings["priors"],
        "sampling": settings["sampling"],
        "versions": versions,
        "diagnostics": diagnostics,
        "prior_predictive": prior_summary,
        "mean_normalization": mean_normalization,
        "backtest": backtest,
        "calibration_acceptance": acceptance,
        "resources": {
            "elapsed_seconds": elapsed_seconds,
            "maximum_rss_gib": maximum_rss_gib(),
            "cpu_count": os.cpu_count(),
        },
    }
    encoded_report = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    paths["report"].write_text(encoded_report, encoding="utf-8")
    (paths["staging"] / "metadata.json").write_text(
        encoded_report,
        encoding="utf-8",
    )
    (paths["staging"] / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    paths["staging"].replace(paths["output"])
    save_offline_record(config, report)

    print(f"Fit rows: {len(fit):,}")
    print(f"Maximum R-hat: {diagnostics['rhat_max']:.4f}")
    print(f"Divergences: {diagnostics['divergences']}")
    print(f"Calibration accepted: {acceptance['accepted']}")
    print(f"Report: {paths['report'].relative_to(PROJECT_ROOT)}")
    print(f"Artifacts: {paths['output'].relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
