"""Single final confirmation of the frozen models on validation (§22).

Each frozen model must still beat its reference with a paired weekly bootstrap
interval entirely in its favor; M9 and M10 also need coverage in range.
--rehearsal runs the same code on calibration, where every point estimate must
match its published result, and saves nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import arviz as az
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import yaml

from src.data.apply_taxonomy import CANONICAL_PRODUCT_COLUMN
from src.data.normalize_text import NORMALIZED_COLUMN
from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    load_experiment_config,
    save_run_record,
    set_seed,
)
from src.models.bge_sample import bge_feature_matrices, load_source_rows
from src.models.representation_comparison import (
    DIFFERENCE_QUANTILES,
    attach_embedding_rows,
    bootstrap_metric,
    check_new_outputs,
    load_embeddings,
    primary_metric,
    product_rule_probabilities,
    product_rule_t1,
    read_json,
    t1_outputs,
    week_resamples,
    write_json,
)
from src.models.tfidf_models import (
    TARGETS,
    binary_metrics_for_view,
    eligible_mask,
    split_frames,
    t1_metrics_for_view,
    validate_counts,
)
from src.models.weekly_composition import registry as composition_registry
from src.models.weekly_composition.baselines import (
    baseline_scores,
    static_concentration,
)
from src.models.weekly_composition.data import (
    composition_panel,
    split_weeks,
    windowed_weeks,
)
from src.models.weekly_composition.evaluation import (
    compare_with_baselines as compare_compositions,
)
from src.models.weekly_composition.scores import joint_log_score
from src.models.weekly_counts.comparison import (
    compare_with_baselines as compare_counts,
)
from src.models.weekly_counts.metrics import coverage_checks, predictive_metrics
from src.models.weekly_counts.run import load_frozen_frame
from src.models.weekly_counts.sampling import sample_predictions, subsample_posterior

# Frozen by M5B (models_plan.md §21); frozen_selection checks its report.
CLASSIFIERS = {
    "T1": "bge_product",
    "T2": "tfidf_text",
    "T3": "tfidf_text",
    "T4": "tfidf_product",
}
REFERENCE = "product_rule"
# M4 already evaluated the T4 model on validation (§21).
NOT_INDEPENDENT = {"T4"}
M9_MODEL = "nb_rolling_4_hierarchical_v3"
M9_RUN_KEY = "20260927T161316.494762Z-full-a3ecfab3-9105ce1f"
M10_MODEL = "dirichlet_multinomial_rolling_4_v1"
M10_RUN_KEY = "20261001T015410.667626Z-full-cff27014-1fa6f95c"
M9_REPORT = PROJECT_ROOT / "reports/modeling/weekly_counts" / M9_MODEL / M9_RUN_KEY
M10_RUN = Path(M10_MODEL) / M10_RUN_KEY
M10_REPORT = PROJECT_ROOT / "reports/modeling/weekly_composition" / M10_RUN
M10_ARTIFACT = PROJECT_ROOT / "artifacts/models/weekly_composition" / M10_RUN
REPORT_PATH = PROJECT_ROOT / "reports/modeling/final_confirmation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rehearsal",
        action="store_true",
        help="Run on calibration, check the published results and save nothing.",
    )
    return parser.parse_args()


def load_settings(stage_dir: str, model_id: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "configs" / stage_dir / f"{model_id}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def frozen_selection(config: dict[str, Any]) -> dict[str, Any]:
    """M5B results, after checking that they still name the frozen classifiers."""
    report = read_json(PROJECT_ROOT / config["paths"]["representation_report"])
    results = report["results"]
    for target, representation in CLASSIFIERS.items():
        if results[target]["selection"]["selected"] != representation:
            raise ValueError(f"M5B no longer selects {representation} for {target}.")
    if not results["T1"]["m5_sample_check"]["passed"]:
        raise ValueError("T1 would use the M5 sample model, which is not loaded here.")
    return results


def model_path(representation: str, target: str, config: dict[str, Any]) -> Path:
    """M4 trained TF-IDF + product; M5B trained the other representations."""
    if representation == "tfidf_product":
        directory = PROJECT_ROOT / config["paths"]["tfidf_models"]
        return directory / f"{target.lower()}_model.joblib"
    directory = PROJECT_ROOT / config["paths"]["representation_artifacts"]
    return directory / f"{representation}_{target.lower()}.joblib"


def load_split(config: dict[str, Any], split: str) -> dict[str, pd.DataFrame]:
    """Fit rows for the product rule, and the split rows with weeks and M8A rows."""
    contract = read_json(PROJECT_ROOT / config["paths"]["evaluation_contract"])
    source = load_source_rows(PROJECT_ROOT / config["paths"]["input_data"], config)
    frames = split_frames(source)
    validate_counts(frames, contract)
    rows = frames[split]
    t1 = eligible_mask(rows, "T1", "complete")
    for target in TARGETS:
        if (eligible_mask(rows, target, "complete") & ~t1).any():
            raise ValueError(f"{target} has {split} rows without an M8A embedding.")
    frames = {"fit": frames["fit"], split: rows.loc[t1].reset_index(drop=True)}
    manifest = PROJECT_ROOT / config["paths"]["bge_full_artifacts"] / "manifest.parquet"
    attach_embedding_rows({split: frames[split]}, manifest)
    return frames


def split_features(
    frame: pd.DataFrame,
    split: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Inputs of the frozen classifiers, built with M4's encoders as in M5B."""
    tfidf_dir = PROJECT_ROOT / config["paths"]["tfidf_models"]
    vectorizer = joblib.load(tfidf_dir / "vectorizer.joblib")
    encoder = joblib.load(tfidf_dir / "product_encoder.joblib")
    text = vectorizer.transform(frame[NORMALIZED_COLUMN])
    product = encoder.transform(frame[[CANONICAL_PRODUCT_COLUMN]])
    embeddings_dir = PROJECT_ROOT / config["paths"]["bge_full_artifacts"]
    embeddings = load_embeddings(embeddings_dir, {split: frame})
    return {
        "tfidf_text": text,
        "tfidf_product": hstack([text, product], format="csr", dtype=np.float32),
        "bge_product": bge_feature_matrices(embeddings, {split: frame}, encoder)[split],
    }


def view_metrics(
    target: str,
    outputs: dict[str, np.ndarray],
    truth: np.ndarray,
    threshold: float | None,
    selected: np.ndarray,
) -> dict[str, float | int]:
    """M4 metrics; binary thresholds stay the ones chosen on calibration."""
    if target == "T1":
        metrics, _ = t1_metrics_for_view(
            truth, outputs["prediction"], outputs["top_three"], selected
        )
        return metrics
    if threshold is None:
        raise ValueError(f"{target} needs the threshold chosen on calibration.")
    return binary_metrics_for_view(truth, outputs["probability"], threshold, selected)


def paired_difference(
    model_draws: np.ndarray,
    reference_draws: np.ndarray,
    estimate: float,
) -> dict[str, float]:
    quantiles = np.quantile(
        model_draws - reference_draws, list(DIFFERENCE_QUANTILES.values())
    )
    interval = {
        label: float(value) for label, value in zip(DIFFERENCE_QUANTILES, quantiles)
    }
    return {"estimate": float(estimate), **interval}


def classifier_outputs(
    target: str,
    model: Any,
    inputs: Any,
    fit: pd.DataFrame,
    rows: pd.DataFrame,
    truth: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    if target == "T1":
        return {
            "model": t1_outputs(model, inputs, truth),
            REFERENCE: product_rule_t1(fit, rows, truth),
        }
    return {
        "model": {"probability": model.predict_proba(inputs)[:, 1]},
        REFERENCE: {"probability": product_rule_probabilities(fit, rows, target)},
    }


def confirm_classifier(
    target: str,
    frames: dict[str, pd.DataFrame],
    split: str,
    features: dict[str, Any],
    selection: dict[str, Any],
    classes: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Frozen classifier against the product rule; the decision uses no shared text."""
    representation = CLASSIFIERS[target]
    frame = frames[split]
    eligible = eligible_mask(frame, target, "complete")
    rows = frame.loc[eligible]
    labels = rows[target]
    if target == "T1":
        truth = labels.astype(str).to_numpy()
    else:
        truth = labels.to_numpy(dtype=bool)
    model = joblib.load(model_path(representation, target, config))
    inputs = features[representation][eligible]
    outputs = classifier_outputs(target, model, inputs, frames["fit"], rows, truth)

    published = selection[target]["metrics"]
    thresholds = {
        "model": published[representation].get("threshold"),
        REFERENCE: published[REFERENCE].get("threshold"),
    }
    no_shared = rows["no_shared_text"].to_numpy(dtype=bool)
    views = {"complete": np.ones(len(rows), dtype=bool), "no_shared_text": no_shared}
    metrics = {
        view: {
            name: view_metrics(target, values, truth, thresholds[name], selected)
            for name, values in outputs.items()
        }
        for view, selected in views.items()
    }

    primary = config["evaluation"]["metrics"][target]["primary"]
    weeks = rows.loc[no_shared, "week"].to_numpy()
    seed = config["experiment"]["seed"]
    counts, positions = week_resamples(
        weeks, config["representations"]["bootstrap_draws"], seed
    )
    draws = {
        name: bootstrap_metric(
            primary_metric(
                target,
                {key: array[no_shared] for key, array in values.items()},
                truth[no_shared],
                classes,
            ),
            counts,
            positions,
        )
        for name, values in outputs.items()
    }
    selected = metrics["no_shared_text"]
    estimate = selected["model"][primary] - selected[REFERENCE][primary]
    difference = paired_difference(draws["model"], draws[REFERENCE], estimate)
    return {
        "representation": representation,
        "primary_metric": primary,
        "validation_independent": target not in NOT_INDEPENDENT,
        "weeks": counts.shape[1],
        "metrics": metrics,
        "difference": difference,
        "confirmed": difference["p025"] > 0,
    }


def confirm_weekly_volume(split: str, seed: int) -> dict[str, Any]:
    """NB-R4-H v3 against B1-R4, from the predictions saved by its full run."""
    settings = load_settings("weekly_counts", M9_MODEL)
    predictions = pd.read_csv(M9_REPORT / "predictions.csv")
    rows = predictions.loc[predictions["split"] == split]
    comparison = compare_counts(predictions, ("b1_rolling_4",), seed, split=split)
    model = predictive_metrics(rows, "model")
    checks = {
        "wis_bootstrap": comparison["difference"]["wis"]["p975"] < 0,
        **coverage_checks(model, settings["acceptance"]),
    }
    reference = predictive_metrics(rows, "b1_rolling_4")
    return {
        "model_id": M9_MODEL,
        "run_key": M9_RUN_KEY,
        "weeks": comparison["weeks"],
        "metrics": {"model": model, "b1_rolling_4": reference},
        "difference": comparison["difference"],
        "checks": checks,
        "confirmed": all(checks.values()),
    }


def confirm_weekly_composition(
    config: dict[str, Any],
    split: str,
    seed: int,
) -> dict[str, Any]:
    """DM-R4 against B2-R4: log scores from its posterior, coverage from its run."""
    settings = load_settings("weekly_composition", M10_MODEL)
    module = composition_registry.get_model(M10_MODEL)
    frame, _, _, _ = load_frozen_frame(config, settings)
    panel = composition_panel(frame, settings["clusters"])
    weeks = split_weeks(windowed_weeks(panel), split)
    idata = az.from_netcdf(M10_ARTIFACT / "posterior.nc")
    # The same draws the run used for its predictions.
    draws = settings["sampling"]["prediction_draws"]
    posterior = subsample_posterior(idata, draws, seed)
    concentration, _ = sample_predictions(
        module.build_model(weeks, settings),
        posterior,
        "concentration",
        "observed",
        seed,
    )
    b2 = static_concentration(
        split_weeks(panel, "fit")["counts"], settings["priors"]["share_concentration"]
    )
    scores = {
        "model": joint_log_score(weeks["counts"], concentration),
        "b2_rolling_4": baseline_scores(weeks, b2)["b2_rolling_4"],
    }
    comparison = compare_compositions(scores, ("b2_rolling_4",), seed)
    predictions = pd.read_csv(M10_REPORT / "predictions.csv")
    rows = predictions.loc[predictions["split"] == split]
    model = predictive_metrics(rows, "model")
    checks = {
        "log_score_bootstrap": comparison["difference"]["p025"] > 0,
        **coverage_checks(model, settings["acceptance"]),
    }
    reference = predictive_metrics(rows, "b2_rolling_4")
    return {
        "model_id": M10_MODEL,
        "run_key": M10_RUN_KEY,
        "weeks": comparison["weeks"],
        "log_score": comparison["log_score"],
        "metrics": {"model": model, "b2_rolling_4": reference},
        "difference": comparison["difference"],
        "checks": checks,
        "confirmed": all(checks.values()),
    }


def check_rehearsal(report: dict[str, Any], selection: dict[str, Any]) -> None:
    """On calibration, every point estimate must match its published value."""
    pairs = {}
    for target, result in report["classifiers"].items():
        primary = result["primary_metric"]
        published = selection[target]["metrics"]
        measured = result["metrics"]["no_shared_text"]
        pairs[f"{target} model"] = (
            measured["model"][primary],
            published[CLASSIFIERS[target]][primary],
        )
        pairs[f"{target} reference"] = (
            measured[REFERENCE][primary],
            published[REFERENCE][primary],
        )
    m9 = read_json(M9_REPORT / "bootstrap.json")
    volume = report["weekly_volume"]
    pairs["M9 WIS"] = (volume["metrics"]["model"]["wis"], m9["metrics"]["model"]["wis"])
    pairs["M9 WIS difference"] = (
        volume["difference"]["wis"]["p975"],
        m9["difference"]["wis"]["p975"],
    )
    m10 = read_json(M10_REPORT / "results.json")["comparison"]["difference"]
    composition = report["weekly_composition"]["difference"]
    pairs["M10 log score difference"] = (composition["estimate"], m10["estimate"])
    pairs["M10 log score difference p025"] = (composition["p025"], m10["p025"])
    mismatched = {
        name: values
        for name, values in pairs.items()
        if not np.isclose(values[0], values[1], rtol=0, atol=1e-9)
    }
    if mismatched:
        raise ValueError(f"The rehearsal does not reproduce: {mismatched}")


def flatten(prefix: str, value: Any) -> dict[str, float]:
    """Nested numbers as metric names joined with underscores."""
    if isinstance(value, dict):
        return {
            name: number
            for key, item in value.items()
            for name, number in flatten(f"{prefix}_{key}", item).items()
        }
    return {prefix: float(value)}


def save_record(config: dict[str, Any], report: dict[str, Any]) -> None:
    """The single MLflow record of the confirmation."""
    metrics: dict[str, float] = {}
    for target, result in report["classifiers"].items():
        metrics.update(flatten(target, result["metrics"]))
        metrics.update(flatten(f"{target}_difference", result["difference"]))
    weekly = {"m9": report["weekly_volume"], "m10": report["weekly_composition"]}
    for key, part in weekly.items():
        for section in ("metrics", "difference", "checks"):
            metrics.update(flatten(f"{key}_{section}", part[section]))
    metrics.update(flatten("confirmed", report["confirmed"]))
    record = build_run_record(
        config=config,
        run_name="final-validation-confirmation",
        stage="FINAL",
        target="T1_T4_M9_M10",
        split="validation",
        view="no_shared_text_and_complete_weeks",
        features=config["features"]["initial"],
        parameters={
            "rule": "models_plan.md §22",
            "classifiers": json.dumps(CLASSIFIERS),
            "m9_run_key": M9_RUN_KEY,
            "m10_run_key": M10_RUN_KEY,
            "bootstrap_draws": config["representations"]["bootstrap_draws"],
        },
        metrics=metrics,
        artifacts=[REPORT_PATH.relative_to(PROJECT_ROOT).as_posix()],
    )
    record["tags"].update(
        {
            "run_mode": "full",
            "validation_used_for_selection": "false",
            **{
                f"confirmed_{component}": str(passed).lower()
                for component, passed in report["confirmed"].items()
            },
        }
    )
    path = PROJECT_ROOT / config["paths"]["offline_runs"] / "final_confirmation"
    save_run_record(record, path / "run.json")


def print_summary(report: dict[str, Any]) -> None:
    for target, result in report["classifiers"].items():
        primary = result["primary_metric"]
        values = result["metrics"]["no_shared_text"]
        difference = result["difference"]
        print(
            f"{target} {primary}: model={values['model'][primary]:.4f}, "
            f"rule={values[REFERENCE][primary]:.4f}, "
            f"CI=[{difference['p025']:+.4f}, {difference['p975']:+.4f}], "
            f"confirmed={result['confirmed']}"
        )
    for name, key in (("M9", "weekly_volume"), ("M10", "weekly_composition")):
        part = report[key]
        print(f"{name}: checks={part['checks']}, confirmed={part['confirmed']}")


def main() -> None:
    args = parse_args()
    config = load_experiment_config()
    seed = config["experiment"]["seed"]
    set_seed(seed)
    split = "calibration" if args.rehearsal else "validation"
    if not args.rehearsal:
        check_new_outputs([REPORT_PATH])
    selection = frozen_selection(config)

    frames = load_split(config, split)
    features = split_features(frames[split], split, config)
    fit = frames["fit"]
    classes = np.unique(
        fit.loc[eligible_mask(fit, "T1", "complete"), "T1"].astype(str).to_numpy()
    )
    classifiers = {
        target: confirm_classifier(
            target, frames, split, features, selection, classes, config
        )
        for target in TARGETS
    }
    print("Classifiers done.", flush=True)
    report = {
        "stage": "FINAL",
        "split": split,
        "seed": seed,
        "rule": "models_plan.md §22",
        "classifiers": classifiers,
        "weekly_volume": confirm_weekly_volume(split, seed),
        "weekly_composition": confirm_weekly_composition(config, split, seed),
    }
    report["confirmed"] = {
        **{target: result["confirmed"] for target, result in classifiers.items()},
        "M9": report["weekly_volume"]["confirmed"],
        "M10": report["weekly_composition"]["confirmed"],
    }
    print_summary(report)
    if args.rehearsal:
        check_rehearsal(report, selection)
        print("Rehearsal: every calibration value matches its published result.")
        return
    write_json(REPORT_PATH, report)
    save_record(config, report)
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
