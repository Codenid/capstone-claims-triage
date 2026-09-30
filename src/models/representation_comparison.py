"""Compare T1-T4 representations trained on every fit row (M5B).

models_plan.md §21: fit trains, calibration rows without shared text evaluate,
and validation stays closed until the single final check.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn import __version__ as sklearn_version
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score

from src.data.apply_taxonomy import CANONICAL_PRODUCT_COLUMN
from src.data.normalize_text import NORMALIZED_COLUMN
from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    load_experiment_config,
    save_run_record,
    set_seed,
)
from src.models.bge_full import load_embedding_array
from src.models.bge_sample import (
    ID_COLUMN,
    bge_feature_matrices,
    load_source_rows,
    sample_frames,
)
from src.models.frequency_baselines import (
    binary_scores,
    choose_threshold,
    complete_top_three,
    t1_rankings,
)
from src.models.semantic_space import load_dvc_hash
from src.models.tfidf_models import (
    DATE_COLUMN,
    TARGETS,
    binary_metrics_for_view,
    eligible_mask,
    new_classifier,
    split_frames,
    t1_metrics_for_view,
    top_three_correct,
    validate_counts,
)

# Simplest first: the tie rule of models_plan.md §21 walks this order.
REPRESENTATIONS = (
    "product_rule",
    "tfidf_text",
    "tfidf_product",
    "bge_text",
    "bge_product",
)
TRAINED = ("tfidf_text", "bge_text", "bge_product")
M5_SAMPLE = "bge_product_m5_sample"
DIFFERENCE_QUANTILES = {"p025": 0.025, "p50": 0.50, "p975": 0.975}
SMOKE_ROWS = {"fit": 20_000, "calibration": 20_000, "validation": 1}
SMOKE_BOOTSTRAP_DRAWS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the whole comparison on a small sample and save nothing.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def check_new_outputs(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"M5B never overwrites results: {', '.join(existing)}")


def load_frames(config: dict[str, Any], smoke: bool) -> dict[str, pd.DataFrame]:
    """Every fit row and the calibration rows without shared text."""
    contract = read_json(PROJECT_ROOT / config["paths"]["evaluation_contract"])
    source = load_source_rows(PROJECT_ROOT / config["paths"]["input_data"], config)
    frames = split_frames(source)
    validate_counts(frames, contract)
    if smoke:
        frames = sample_frames(source, SMOKE_ROWS, config["experiment"]["seed"])
    calibration = frames["calibration"]
    no_shared = calibration.loc[calibration["no_shared_text"]]
    return {"fit": frames["fit"], "calibration": no_shared.reset_index(drop=True)}


def attach_embedding_rows(
    frames: dict[str, pd.DataFrame],
    manifest_path: Path,
) -> None:
    """Add each row's position in the M8A arrays and its week."""
    for split, frame in frames.items():
        manifest = pd.read_parquet(
            manifest_path,
            columns=[ID_COLUMN, DATE_COLUMN, "embedding_row"],
            filters=[("split", "=", split)],
        )
        aligned = manifest.set_index(manifest[ID_COLUMN].astype(str)).reindex(
            frame[ID_COLUMN].astype(str)
        )
        if aligned["embedding_row"].isna().to_numpy().any():
            raise ValueError(f"M5B found {split} rows without an M8A embedding.")
        frame["embedding_row"] = aligned["embedding_row"].to_numpy(dtype=np.int64)
        dates = pd.to_datetime(aligned[DATE_COLUMN])
        frame["week"] = dates.dt.to_period("W-SUN").dt.start_time.to_numpy()


def load_embeddings(
    directory: Path,
    frames: dict[str, pd.DataFrame],
) -> dict[str, np.ndarray]:
    """M8A embeddings in the row order of each frame."""
    embeddings = {}
    for split, frame in frames.items():
        values = load_embedding_array(directory, split)
        embeddings[split] = values[:][frame["embedding_row"].to_numpy()]
        values.close()
    return embeddings


def load_reference_models(config: dict[str, Any]) -> dict[str, Any]:
    """The M4 encoders and models, and the M5 sample model for T1."""
    tfidf_dir = PROJECT_ROOT / config["paths"]["tfidf_models"]
    sample_dir = PROJECT_ROOT / config["paths"]["bge_artifacts"]
    return {
        "vectorizer": joblib.load(tfidf_dir / "vectorizer.joblib"),
        "encoder": joblib.load(tfidf_dir / "product_encoder.joblib"),
        "tfidf_product": {
            target: joblib.load(tfidf_dir / f"{target.lower()}_model.joblib")
            for target in TARGETS
        },
        "m5_encoder": joblib.load(sample_dir / "product_encoder.joblib"),
        "m5_t1": joblib.load(sample_dir / "bge_t1.joblib"),
    }


def feature_matrices(
    frames: dict[str, pd.DataFrame],
    embeddings: dict[str, np.ndarray],
    references: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Inputs of each representation; new models use M4's product encoder."""
    calibration = frames["calibration"]
    text = {
        split: references["vectorizer"].transform(frame[NORMALIZED_COLUMN])
        for split, frame in frames.items()
    }
    product = references["encoder"].transform(calibration[[CANONICAL_PRODUCT_COLUMN]])
    m4_inputs = hstack([text["calibration"], product], format="csr", dtype=np.float32)
    return {
        "tfidf_text": text,
        # M4 and M5 models already exist, so they only need calibration inputs.
        "tfidf_product": {"calibration": m4_inputs},
        "bge_text": embeddings,
        "bge_product": bge_feature_matrices(embeddings, frames, references["encoder"]),
        M5_SAMPLE: bge_feature_matrices(
            {"calibration": embeddings["calibration"]},
            {"calibration": calibration},
            references["m5_encoder"],
        ),
    }


def calibration_truth(calibration: pd.DataFrame, target: str) -> np.ndarray:
    values = calibration.loc[eligible_mask(calibration, target, "complete"), target]
    if target == "T1":
        return values.astype(str).to_numpy()
    return values.to_numpy(dtype=bool)


def t1_outputs(model: Any, features: Any, truth: np.ndarray) -> dict[str, np.ndarray]:
    """Top class and top-3 hit of a fitted T1 classifier, as M4 scores them."""
    decisions = model.decision_function(features)
    classes = np.asarray(model.classes_)
    truth_codes = pd.Categorical(truth, categories=classes).codes
    if np.any(truth_codes < 0):
        raise ValueError("M5B calibration contains a T1 class unseen during fit.")
    return {
        "prediction": classes[np.argmax(decisions, axis=1)],
        "top_three": top_three_correct(decisions, truth_codes),
    }


def product_rule_t1(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    truth: np.ndarray,
) -> dict[str, np.ndarray]:
    """M3 rule: the three most frequent fit classes of each product."""
    global_ranking, product_rankings = t1_rankings(fit)
    fallback = complete_top_three(global_ranking, global_ranking)
    rankings = [
        product_rankings.get(str(product), fallback)
        for product in calibration[CANONICAL_PRODUCT_COLUMN]
    ]
    return {
        "prediction": np.array([ranking[0] for ranking in rankings]),
        "top_three": np.array(
            [label in ranking for label, ranking in zip(truth, rankings)]
        ),
    }


def product_rule_probabilities(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    target: str,
) -> np.ndarray:
    """M3 rule: the fit positive rate of each product."""
    _, global_rate, product_rates = binary_scores(fit, target, "product_frequency")
    products = calibration[CANONICAL_PRODUCT_COLUMN].astype(str)
    return np.array([product_rates.get(product, global_rate) for product in products])


def reference_outputs(
    target: str,
    frames: dict[str, pd.DataFrame],
    features: dict[str, dict[str, Any]],
    references: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    """Calibration outputs of the models that already exist: M3, M4 and M5."""
    fit, calibration = frames["fit"], frames["calibration"]
    rows = eligible_mask(calibration, target, "complete")
    truth = calibration_truth(calibration, target)
    m4_model = references["tfidf_product"][target]
    m4_features = features["tfidf_product"]["calibration"][rows]
    if target == "T1":
        m5_features = features[M5_SAMPLE]["calibration"][rows]
        return {
            "product_rule": product_rule_t1(fit, calibration.loc[rows], truth),
            "tfidf_product": t1_outputs(m4_model, m4_features, truth),
            M5_SAMPLE: t1_outputs(references["m5_t1"], m5_features, truth),
        }
    rule = product_rule_probabilities(fit, calibration.loc[rows], target)
    return {
        "product_rule": {"probability": rule},
        "tfidf_product": {"probability": m4_model.predict_proba(m4_features)[:, 1]},
    }


def classifier_config(config: dict[str, Any], representation: str) -> dict[str, Any]:
    """M5 gave BGE more SGD iterations; everything else follows M4."""
    if not representation.startswith("bge"):
        return config
    updated = deepcopy(config)
    updated["tfidf"]["max_iter"] = config["bge"]["linear_max_iter"]
    return updated


def train_binary(
    config: dict[str, Any],
    target: str,
    fit_features: Any,
    fit_truth: np.ndarray,
    calibration_features: Any,
    calibration_truth: np.ndarray,
) -> CalibratedClassifierCV:
    """Fit on fit rows, then calibrate on calibration rows, as M4 does."""
    base = new_classifier(config, target).fit(fit_features, fit_truth)
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(base),
        method=config["tfidf"]["calibration_method"],
    )
    return calibrated.fit(calibration_features, calibration_truth)


def trained_outputs(
    target: str,
    frames: dict[str, pd.DataFrame],
    features: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Train the new representations on every eligible fit row."""
    fit, calibration = frames["fit"], frames["calibration"]
    fit_rows = eligible_mask(fit, target, "complete")
    rows = eligible_mask(calibration, target, "complete")
    truth = calibration_truth(calibration, target)
    outputs: dict[str, dict[str, np.ndarray]] = {}
    models: dict[str, Any] = {}
    for name in TRAINED:
        settings = classifier_config(config, name)
        fit_features = features[name]["fit"][fit_rows]
        calibration_features = features[name]["calibration"][rows]
        if target == "T1":
            labels = fit.loc[fit_rows, target].astype(str)
            model = new_classifier(settings, target).fit(fit_features, labels)
            outputs[name] = t1_outputs(model, calibration_features, truth)
        else:
            model = train_binary(
                settings,
                target,
                fit_features,
                fit.loc[fit_rows, target].to_numpy(dtype=bool),
                calibration_features,
                truth,
            )
            probability = model.predict_proba(calibration_features)[:, 1]
            outputs[name] = {"probability": probability}
        models[f"{name}_{target.lower()}"] = model
        print(f"{target} {name}: trained on {int(fit_rows.sum()):,} rows", flush=True)
    return outputs, models


def representation_metrics(
    target: str,
    outputs: dict[str, np.ndarray],
    truth: np.ndarray,
) -> dict[str, float | int]:
    """M4 calibration metrics; binary thresholds maximize F1 on these rows."""
    every_row = np.ones(len(truth), dtype=bool)
    if target == "T1":
        metrics, _ = t1_metrics_for_view(
            truth, outputs["prediction"], outputs["top_three"], every_row
        )
        return metrics
    probability = outputs["probability"]
    threshold, _ = choose_threshold(pd.Series(truth), pd.Series(probability))
    return {
        "threshold": threshold,
        **binary_metrics_for_view(truth, probability, threshold, every_row),
    }


def check_reproduced(
    outputs: dict[str, dict[str, dict[str, np.ndarray]]],
    calibration: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """M3 and M4 must reproduce their published calibration metric on these rows."""
    baseline = read_json(PROJECT_ROOT / config["paths"]["baseline_report"])
    tfidf = read_json(PROJECT_ROOT / config["paths"]["tfidf_report"])
    for target in TARGETS:
        primary = config["evaluation"]["metrics"][target]["primary"]
        truth = calibration_truth(calibration, target)
        published = {
            "product_rule": baseline["results"][target]["product_frequency"],
            "tfidf_product": tfidf["results"][target],
        }
        for name, result in published.items():
            expected = result["metrics"]["calibration"]["no_shared_text"][primary]
            actual = representation_metrics(target, outputs[target][name], truth)
            if not np.isclose(actual[primary], expected, rtol=0, atol=1e-9):
                raise ValueError(
                    f"M5B does not reproduce {name} for {target}: "
                    f"{actual[primary]} vs {expected}."
                )


def apply_m5_check(
    outputs: dict[str, dict[str, np.ndarray]],
    truth: np.ndarray,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """§21 keeps full-fit BGE + product for T1 only if it beats the M5 sample."""
    full = representation_metrics("T1", outputs["bge_product"], truth)["macro_f1"]
    sample = representation_metrics("T1", outputs[M5_SAMPLE], truth)["macro_f1"]
    check = {"bge_product": full, "m5_sample": sample, "passed": bool(full > sample)}
    if not check["passed"]:
        outputs = {**outputs, "bge_product": outputs[M5_SAMPLE]}
    return outputs, check


def weighted_macro_f1(
    truth_codes: np.ndarray,
    prediction_codes: np.ndarray,
    weights: np.ndarray,
    classes: int,
) -> float:
    """Macro-F1 over the classes present in the truth, with row weights."""
    correct = weights * (truth_codes == prediction_codes)
    hits = np.bincount(truth_codes, correct, classes)
    true_total = np.bincount(truth_codes, weights, classes)
    predicted_total = np.bincount(prediction_codes, weights, classes)
    present = true_total > 0
    scores = 2 * hits[present] / (true_total[present] + predicted_total[present])
    return float(scores.mean())


def primary_metric(
    target: str,
    outputs: dict[str, np.ndarray],
    truth: np.ndarray,
    classes: np.ndarray,
) -> Callable[[np.ndarray], float]:
    """The primary metric of one representation as a function of row weights."""
    if target == "T1":
        truth_codes = pd.Categorical(truth, categories=classes).codes
        prediction_codes = pd.Categorical(outputs["prediction"], categories=classes).codes
        return lambda weights: weighted_macro_f1(
            truth_codes, prediction_codes, weights, len(classes)
        )
    probability = outputs["probability"]
    return lambda weights: float(
        average_precision_score(truth, probability, sample_weight=weights)
    )


def week_resamples(
    weeks: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """How often each week appears in every draw, and the week of each row."""
    labels, positions = np.unique(weeks, return_inverse=True)
    rng = np.random.default_rng(seed)
    index = rng.integers(len(labels), size=(draws, len(labels)))
    counts = np.stack([np.bincount(draw, minlength=len(labels)) for draw in index])
    return counts, positions


def bootstrap_metric(
    metric: Callable[[np.ndarray], float],
    counts: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    return np.array([metric(week_counts[positions]) for week_counts in counts])


def select_representation(
    estimates: dict[str, float],
    draws: dict[str, np.ndarray],
    minimum_gain: float,
) -> dict[str, Any]:
    """Keep the simplest representation unless a more complex one clearly wins."""
    selected = REPRESENTATIONS[0]
    steps = []
    for challenger in REPRESENTATIONS[1:]:
        difference = draws[challenger] - draws[selected]
        quantiles = np.quantile(difference, list(DIFFERENCE_QUANTILES.values()))
        interval = {
            label: float(value) for label, value in zip(DIFFERENCE_QUANTILES, quantiles)
        }
        gain = estimates[challenger] / estimates[selected] - 1
        replaces = bool(gain >= minimum_gain and interval["p025"] > 0)
        steps.append(
            {
                "current": selected,
                "challenger": challenger,
                "relative_gain": float(gain),
                "difference": {
                    "estimate": float(estimates[challenger] - estimates[selected]),
                    **interval,
                },
                "replaces": replaces,
            }
        )
        if replaces:
            selected = challenger
    return {"selected": selected, "steps": steps}


def compare_target(
    target: str,
    outputs: dict[str, dict[str, np.ndarray]],
    truth: np.ndarray,
    weeks: np.ndarray,
    classes: np.ndarray,
    config: dict[str, Any],
    draws: int,
) -> dict[str, Any]:
    """Metrics of every representation and the tie-rule choice among them."""
    primary = config["evaluation"]["metrics"][target]["primary"]
    metrics = {
        name: representation_metrics(target, values, truth)
        for name, values in outputs.items()
    }
    counts, positions = week_resamples(weeks, draws, config["experiment"]["seed"])
    samples = {
        name: bootstrap_metric(
            primary_metric(target, outputs[name], truth, classes), counts, positions
        )
        for name in REPRESENTATIONS
    }
    estimates = {name: float(metrics[name][primary]) for name in REPRESENTATIONS}
    minimum_gain = config["representations"]["minimum_relative_gain"]
    return {
        "primary_metric": primary,
        "calibration_rows": len(truth),
        "weeks": counts.shape[1],
        "metrics": metrics,
        "selection": select_representation(estimates, samples, minimum_gain),
    }


def prediction_frame(
    rows: pd.DataFrame,
    truth: np.ndarray,
    outputs: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    columns: dict[str, Any] = {
        ID_COLUMN: rows[ID_COLUMN].astype(str).to_numpy(),
        "week": rows["week"].to_numpy(),
        "truth": truth,
    }
    for name, values in outputs.items():
        for key, array in values.items():
            columns[f"{name}_{key}"] = array
    return pd.DataFrame(columns)


def save_outputs(
    staging_dir: Path,
    output_dir: Path,
    models: dict[str, Any],
    predictions: dict[str, pd.DataFrame],
    metadata: dict[str, Any],
) -> None:
    """Write into staging and publish the directory only when it is complete."""
    staging_dir.mkdir(parents=True)
    for name, model in models.items():
        joblib.dump(model, staging_dir / f"{name}.joblib", compress=3)
    for target, frame in predictions.items():
        path = staging_dir / f"calibration_{target.lower()}.parquet"
        frame.to_parquet(path, index=False)
    write_json(staging_dir / "metadata.json", metadata)
    (staging_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    staging_dir.replace(output_dir)


def record_metrics(result: dict[str, Any]) -> dict[str, float]:
    metrics = {
        f"calibration_no_shared_text_{name}_{metric}": float(value)
        for name, values in result["metrics"].items()
        for metric, value in values.items()
        if metric not in ("eligible", "positive")
    }
    for step in result["selection"]["steps"]:
        prefix = f"bootstrap_{step['challenger']}"
        metrics[f"{prefix}_relative_gain"] = step["relative_gain"]
        for label, value in step["difference"].items():
            metrics[f"{prefix}_difference_{label}"] = value
    return metrics


def save_records(config: dict[str, Any], results: dict[str, Any]) -> None:
    """One offline MLflow record per target, with the representations as metrics."""
    run_root = PROJECT_ROOT / config["paths"]["offline_runs"] / "m5b"
    settings = config["representations"]
    for target, result in results.items():
        record = build_run_record(
            config=config,
            run_name=f"m5b-{target.lower()}-representations",
            stage="M5B",
            target=target,
            split="fit_calibration",
            view="no_shared_text",
            features=config["features"]["initial"],
            parameters={
                "representations": json.dumps(REPRESENTATIONS),
                "minimum_relative_gain": settings["minimum_relative_gain"],
                "bootstrap_unit": "week",
                "bootstrap_draws": settings["bootstrap_draws"],
                "bootstrap_weeks": result["weeks"],
                "fit_rows": result["fit_rows"],
                "calibration_rows": result["calibration_rows"],
                "class_weight": str(config["tfidf"]["class_weight"][target]),
                "tfidf_max_iter": config["tfidf"]["max_iter"],
                "bge_max_iter": config["bge"]["linear_max_iter"],
                "bge_revision": config["bge"]["revision"],
                "model_artifact": config["paths"]["representation_artifacts"],
            },
            metrics=record_metrics(result),
            artifacts=[config["paths"]["representation_report"]],
        )
        record["tags"].update(
            {
                "run_mode": "full",
                "selected_representation": result["selection"]["selected"],
            }
        )
        if "m5_sample_check" in result:
            passed = result["m5_sample_check"]["passed"]
            record["tags"]["bge_product_source"] = "full_fit" if passed else "m5_sample"
        save_run_record(record, run_root / target.lower() / "run.json")


def print_summary(results: dict[str, Any]) -> None:
    for target, result in results.items():
        primary = result["primary_metric"]
        values = ", ".join(
            f"{name}={result['metrics'][name][primary]:.4f}" for name in REPRESENTATIONS
        )
        selected = result["selection"]["selected"]
        print(f"{target} {primary}: {values}; selected={selected}")


def main() -> None:
    args = parse_args()
    config = load_experiment_config()
    set_seed(config["experiment"]["seed"])
    paths = config["paths"]
    report_path = PROJECT_ROOT / paths["representation_report"]
    output_dir = PROJECT_ROOT / paths["representation_artifacts"]
    staging_dir = PROJECT_ROOT / paths["representation_staging"]
    embeddings_dir = PROJECT_ROOT / paths["bge_full_artifacts"]
    if not args.smoke:
        check_new_outputs([report_path, output_dir, staging_dir])
    embeddings_hash = load_dvc_hash(PROJECT_ROOT / paths["bge_full_artifacts_dvc"])
    if embeddings_hash != config["representations"]["source_dvc_hash"]:
        raise ValueError("M5B embeddings do not match the frozen M8A hash.")

    frames = load_frames(config, args.smoke)
    attach_embedding_rows(frames, embeddings_dir / "manifest.parquet")
    references = load_reference_models(config)
    embeddings = load_embeddings(embeddings_dir, frames)
    features = feature_matrices(frames, embeddings, references)
    print("Inputs ready.", flush=True)

    outputs = {
        target: reference_outputs(target, frames, features, references)
        for target in TARGETS
    }
    if not args.smoke:
        check_reproduced(outputs, frames["calibration"], config)
        print("M3 and M4 reproduce their published calibration results.", flush=True)

    fit, calibration = frames["fit"], frames["calibration"]
    classes = np.unique(
        fit.loc[eligible_mask(fit, "T1", "complete"), "T1"].astype(str).to_numpy()
    )
    draws = config["representations"]["bootstrap_draws"]
    if args.smoke:
        draws = SMOKE_BOOTSTRAP_DRAWS
    results: dict[str, Any] = {}
    models: dict[str, Any] = {}
    predictions: dict[str, pd.DataFrame] = {}
    for target in TARGETS:
        trained, fitted = trained_outputs(target, frames, features, config)
        models.update(fitted)
        target_outputs = {**outputs[target], **trained}
        truth = calibration_truth(calibration, target)
        rows = calibration.loc[eligible_mask(calibration, target, "complete")]
        check = None
        if target == "T1":
            target_outputs, check = apply_m5_check(target_outputs, truth)
        result = compare_target(
            target,
            target_outputs,
            truth,
            rows["week"].to_numpy(),
            classes,
            config,
            draws,
        )
        result["fit_rows"] = int(eligible_mask(fit, target, "complete").sum())
        if check is not None:
            result["m5_sample_check"] = check
        results[target] = result
        predictions[target] = prediction_frame(rows, truth, target_outputs)
        print(f"{target}: selected {result['selection']['selected']}", flush=True)

    print_summary(results)
    if args.smoke:
        return

    metadata = {
        "stage": "M5B",
        "dvc_data_hash": read_json(PROJECT_ROOT / paths["input_contract"])["dvc_md5"],
        "embeddings_dvc_hash": embeddings_hash,
        "tfidf_dvc_hash": load_dvc_hash(PROJECT_ROOT / f"{paths['tfidf_models']}.dvc"),
        "bge_sample_dvc_hash": load_dvc_hash(PROJECT_ROOT / paths["bge_artifacts_dvc"]),
        "split_version": config["evaluation"]["split_version"],
        "sklearn_version": sklearn_version,
    }
    save_outputs(staging_dir, output_dir, models, predictions, metadata)
    report = {
        **metadata,
        "view": "no_shared_text",
        "rule": {
            "order": list(REPRESENTATIONS),
            "minimum_relative_gain": config["representations"]["minimum_relative_gain"],
            "bootstrap_unit": "week",
            "bootstrap_draws": draws,
            "seed": config["experiment"]["seed"],
        },
        "results": results,
    }
    write_json(report_path, report)
    save_records(config, results)
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")
    print(f"Artifacts: {output_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
