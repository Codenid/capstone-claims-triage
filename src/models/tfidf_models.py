"""Train and evaluate the M4 TF-IDF models."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, hstack
from sklearn import __version__ as sklearn_version
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import OneHotEncoder

from src.data.apply_taxonomy import CANONICAL_PRODUCT_COLUMN
from src.data.build_targets import COMPLETE_COLUMNS
from src.data.normalize_text import HASH_COLUMN, NORMALIZED_COLUMN
from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    load_experiment_config,
    save_run_record,
    set_seed,
)
from src.models.frequency_baselines import choose_threshold, flat_mlflow_metrics

DATE_COLUMN = "Date received"
NO_SHARED_COLUMN = "no_shared_reference_text"
TARGETS = ("T1", "T2", "T3", "T4")
VIEWS = ("complete", "no_shared_text")


def load_rows(input_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    splits = config["evaluation"]["splits"]
    calibration_start = pd.Timestamp(splits["calibration"]["start"])
    validation_start = pd.Timestamp(splits["validation"]["start"])
    end = pd.Timestamp(splits["validation"]["end"])

    columns = [
        DATE_COLUMN,
        HASH_COLUMN,
        NO_SHARED_COLUMN,
        NORMALIZED_COLUMN,
        CANONICAL_PRODUCT_COLUMN,
        *TARGETS,
        *COMPLETE_COLUMNS.values(),
    ]
    table = pq.read_table(
        input_path,
        columns=columns,
        filters=[
            (DATE_COLUMN, ">=", pd.Timestamp(splits["fit"]["start"]).date()),
            (DATE_COLUMN, "<", end.date()),
        ],
    )
    frame = table.to_pandas(categories=[CANONICAL_PRODUCT_COLUMN, "T1"])
    dates = pd.to_datetime(frame[DATE_COLUMN])
    frame["evaluation_split"] = np.select(
        [dates < calibration_start, dates < validation_start],
        ["fit", "calibration"],
        default="validation",
    )

    fit_hashes = set(frame.loc[frame["evaluation_split"] == "fit", HASH_COLUMN])
    frame["no_shared_text"] = True
    calibration = frame["evaluation_split"] == "calibration"
    validation = frame["evaluation_split"] == "validation"
    frame.loc[calibration, "no_shared_text"] = ~frame.loc[
        calibration,
        HASH_COLUMN,
    ].isin(fit_hashes)
    frame.loc[validation, "no_shared_text"] = frame.loc[
        validation,
        NO_SHARED_COLUMN,
    ]
    return frame.drop(columns=[DATE_COLUMN, HASH_COLUMN, NO_SHARED_COLUMN])


def eligible_mask(frame: pd.DataFrame, target: str, view: str) -> np.ndarray:
    mask = frame[COMPLETE_COLUMNS[target]].to_numpy(dtype=bool, copy=True)
    if view == "no_shared_text":
        mask &= frame["no_shared_text"].to_numpy(dtype=bool)
    return mask


def split_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        split: frame.loc[frame["evaluation_split"] == split].reset_index(drop=True)
        for split in ("fit", "calibration", "validation")
    }


def validate_counts(
    frames: dict[str, pd.DataFrame],
    contract: dict[str, Any],
) -> None:
    for split, frame in frames.items():
        for view in VIEWS:
            for target in TARGETS:
                actual = int(eligible_mask(frame, target, view).sum())
                expected = contract["targets"][split][view][target]["eligible"]
                if actual != expected:
                    raise ValueError(
                        f"Eligibility mismatch for {split}/{view}/{target}: "
                        f"expected {expected}, found {actual}."
                    )


def build_features(
    frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[
    TfidfVectorizer,
    OneHotEncoder,
    dict[str, csr_matrix],
]:
    settings = config["tfidf"]
    vectorizer = TfidfVectorizer(
        ngram_range=tuple(settings["ngram_range"]),
        min_df=settings["min_df"],
        max_features=settings["max_features"],
        sublinear_tf=settings["sublinear_tf"],
        dtype=np.float32,
    )
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        dtype=np.float32,
    )

    fit = frames["fit"]
    text_fit = vectorizer.fit_transform(fit[NORMALIZED_COLUMN])
    product_fit = encoder.fit_transform(fit[[CANONICAL_PRODUCT_COLUMN]])
    matrices = {
        "fit": hstack([text_fit, product_fit], format="csr", dtype=np.float32),
    }

    for split in ("calibration", "validation"):
        current = frames[split]
        text = vectorizer.transform(current[NORMALIZED_COLUMN])
        product = encoder.transform(current[[CANONICAL_PRODUCT_COLUMN]])
        matrices[split] = hstack([text, product], format="csr", dtype=np.float32)

    return vectorizer, encoder, matrices


def top_three_correct(decisions: np.ndarray, truth_codes: np.ndarray) -> np.ndarray:
    positions = np.argpartition(decisions, -3, axis=1)[:, -3:]
    return np.any(positions == truth_codes[:, None], axis=1)


def t1_metrics_for_view(
    truth: np.ndarray,
    prediction: np.ndarray,
    top_three: np.ndarray,
    selected: np.ndarray,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    selected_truth = truth[selected]
    selected_prediction = prediction[selected]
    labels = sorted(np.unique(selected_truth))
    metrics = {
        "eligible": int(selected.sum()),
        "macro_f1": float(
            f1_score(
                selected_truth,
                selected_prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "top_3_accuracy": float(top_three[selected].mean()),
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        selected_truth,
        selected_prediction,
        labels=labels,
        zero_division=0,
    )
    class_rows = [
        {
            "class": label,
            "support": int(label_support),
            "precision": float(label_precision),
            "recall": float(label_recall),
            "f1": float(label_f1),
        }
        for label, label_precision, label_recall, label_f1, label_support in zip(
            labels,
            precision,
            recall,
            f1,
            support,
        )
    ]
    return metrics, class_rows


def evaluate_t1(
    model: SGDClassifier,
    frames: dict[str, pd.DataFrame],
    matrices: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics: dict[str, Any] = {}
    all_class_rows: list[dict[str, Any]] = []

    for split in ("calibration", "validation"):
        frame = frames[split]
        complete = eligible_mask(frame, "T1", "complete")
        truth = frame.loc[complete, "T1"].astype(str).to_numpy()
        decisions = model.decision_function(matrices[split][complete])
        classes = np.asarray(getattr(model, "classes_"))
        truth_codes = pd.Categorical(truth, categories=classes).codes
        if np.any(truth_codes < 0):
            raise ValueError(f"{split} contains a T1 class unseen during fit.")
        prediction = classes[np.argmax(decisions, axis=1)]
        top_three = top_three_correct(decisions, truth_codes)

        metrics[split] = {}
        for view in VIEWS:
            if view == "complete":
                selected = np.ones(len(truth), dtype=bool)
            else:
                selected = frame.loc[complete, "no_shared_text"].to_numpy(dtype=bool)
            view_metrics, class_rows = t1_metrics_for_view(
                truth,
                prediction,
                top_three,
                selected,
            )
            metrics[split][view] = view_metrics
            for row in class_rows:
                row.update({"split": split, "view": view})
            all_class_rows.extend(class_rows)

    return metrics, all_class_rows


def binary_metrics_for_view(
    truth: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    selected: np.ndarray,
) -> dict[str, float | int]:
    selected_truth = truth[selected]
    selected_probabilities = probabilities[selected]
    prediction = selected_probabilities >= threshold
    return {
        "eligible": int(selected.sum()),
        "positive": int(selected_truth.sum()),
        "average_precision": float(
            average_precision_score(selected_truth, selected_probabilities)
        ),
        "precision": float(
            precision_score(selected_truth, prediction, zero_division=0)
        ),
        "recall": float(recall_score(selected_truth, prediction, zero_division=0)),
        "f1": float(f1_score(selected_truth, prediction, zero_division=0)),
        "brier_score": float(
            brier_score_loss(selected_truth, selected_probabilities)
        ),
    }


def evaluate_binary(
    model: CalibratedClassifierCV,
    target: str,
    frames: dict[str, pd.DataFrame],
    matrices: dict[str, Any],
) -> tuple[dict[str, Any], float, float]:
    calibration = frames["calibration"]
    calibration_complete = eligible_mask(calibration, target, "complete")
    calibration_truth = calibration.loc[calibration_complete, target].to_numpy(dtype=bool)
    calibration_probabilities = model.predict_proba(
        matrices["calibration"][calibration_complete]
    )[:, 1]
    calibration_no_shared = calibration.loc[
        calibration_complete,
        "no_shared_text",
    ].to_numpy(dtype=bool)
    threshold, calibration_f1 = choose_threshold(
        pd.Series(calibration_truth[calibration_no_shared]),
        pd.Series(calibration_probabilities[calibration_no_shared]),
    )

    metrics: dict[str, Any] = {}
    for split in ("calibration", "validation"):
        frame = frames[split]
        complete = eligible_mask(frame, target, "complete")
        truth = frame.loc[complete, target].to_numpy(dtype=bool)
        if split == "calibration":
            probabilities = calibration_probabilities
        else:
            probabilities = model.predict_proba(matrices[split][complete])[:, 1]

        metrics[split] = {}
        for view in VIEWS:
            if view == "complete":
                selected = np.ones(len(truth), dtype=bool)
            else:
                selected = frame.loc[complete, "no_shared_text"].to_numpy(dtype=bool)
            metrics[split][view] = binary_metrics_for_view(
                truth,
                probabilities,
                threshold,
                selected,
            )
    return metrics, threshold, calibration_f1


def new_classifier(
    config: dict[str, Any],
    target: str,
) -> SGDClassifier:
    settings = config["tfidf"]
    return SGDClassifier(
        loss="log_loss",
        alpha=settings["alpha"],
        max_iter=settings["max_iter"],
        tol=1e-3,
        class_weight=settings["class_weight"][target],
        average=True,
        random_state=config["experiment"]["seed"],
        n_jobs=-1,
    )


def train_models(
    config: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    matrices: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    results: dict[str, Any] = {}
    models: dict[str, Any] = {}

    fit = frames["fit"]
    t1_fit = eligible_mask(fit, "T1", "complete")
    t1_model = new_classifier(config, "T1")
    t1_model.fit(
        matrices["fit"][t1_fit],
        fit.loc[t1_fit, "T1"].astype(str),
    )
    t1_metrics, class_rows = evaluate_t1(t1_model, frames, matrices)
    results["T1"] = {"metrics": t1_metrics}
    models["T1"] = t1_model

    for target in ("T2", "T3", "T4"):
        fit_mask = eligible_mask(fit, target, "complete")
        base_model = new_classifier(config, target)
        base_model.fit(
            matrices["fit"][fit_mask],
            fit.loc[fit_mask, target].to_numpy(dtype=bool),
        )

        calibration = frames["calibration"]
        calibration_mask = eligible_mask(calibration, target, "no_shared_text")
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(base_model),
            method=config["tfidf"]["calibration_method"],
        )
        calibrated.fit(
            matrices["calibration"][calibration_mask],
            calibration.loc[calibration_mask, target].to_numpy(dtype=bool),
        )
        metrics, threshold, calibration_f1 = evaluate_binary(
            calibrated,
            target,
            frames,
            matrices,
        )
        results[target] = {
            "class_weight": config["tfidf"]["class_weight"][target],
            "threshold": threshold,
            "calibration_no_shared_f1_at_threshold": calibration_f1,
            "metrics": metrics,
        }
        models[target] = calibrated

    return results, models, class_rows


def add_baseline_comparison(
    results: dict[str, Any],
    baseline_report: dict[str, Any],
) -> None:
    for target, result in results.items():
        primary = "macro_f1" if target == "T1" else "average_precision"
        current = result["metrics"]["validation"]["no_shared_text"][primary]
        baseline = baseline_report["results"][target]["product_frequency"]["metrics"][
            "validation"
        ]["no_shared_text"][primary]
        result["product_baseline_primary_metric"] = baseline
        result["primary_metric_improvement"] = current - baseline


def save_models(
    output_dir: Path,
    vectorizer: TfidfVectorizer,
    encoder: OneHotEncoder,
    models: dict[str, Any],
    config: dict[str, Any],
    data_hash: str,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    joblib.dump(vectorizer, output_dir / "vectorizer.joblib", compress=3)
    joblib.dump(encoder, output_dir / "product_encoder.joblib", compress=3)
    for target, model in models.items():
        joblib.dump(model, output_dir / f"{target.lower()}_model.joblib", compress=3)

    metadata = {
        "stage": "M4",
        "dvc_data_hash": data_hash,
        "split_version": config["evaluation"]["split_version"],
        "features": config["features"]["initial"],
        "tfidf": config["tfidf"],
        "sklearn_version": sklearn_version,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_records(
    config: dict[str, Any],
    results: dict[str, Any],
) -> None:
    run_root = PROJECT_ROOT / config["paths"]["offline_runs"] / "m4"
    report_path = config["paths"]["tfidf_report"]
    class_path = config["paths"]["tfidf_t1_classes"]

    for target, result in results.items():
        parameters = {
            "model": "tfidf_product_sgd_logistic",
            "max_features": config["tfidf"]["max_features"],
            "ngram_range": str(tuple(config["tfidf"]["ngram_range"])),
            "min_df": config["tfidf"]["min_df"],
            "alpha": config["tfidf"]["alpha"],
            "max_iter": config["tfidf"]["max_iter"],
            "class_weight": str(config["tfidf"]["class_weight"][target]),
            "model_artifact": config["paths"]["tfidf_models"],
        }
        if target != "T1":
            parameters.update(
                {
                    "calibration_method": config["tfidf"]["calibration_method"],
                    "threshold": result["threshold"],
                    "threshold_rule": config["evaluation"]["binary_threshold"],
                }
            )
        artifacts = [report_path]
        if target == "T1":
            artifacts.append(class_path)

        record = build_run_record(
            config=config,
            run_name=f"m4-{target.lower()}-tfidf-product",
            stage="M4",
            target=target,
            split="calibration_validation",
            view="no_shared_text_primary",
            features=config["features"]["initial"],
            parameters=parameters,
            metrics=flat_mlflow_metrics(result["metrics"]),
            artifacts=artifacts,
        )
        save_run_record(record, run_root / target.lower() / "run.json")


def main() -> None:
    config = load_experiment_config()
    set_seed(config["experiment"]["seed"])
    input_path = PROJECT_ROOT / config["paths"]["input_data"]
    evaluation_path = PROJECT_ROOT / config["paths"]["evaluation_contract"]
    baseline_path = PROJECT_ROOT / config["paths"]["baseline_report"]
    report_path = PROJECT_ROOT / config["paths"]["tfidf_report"]
    class_path = PROJECT_ROOT / config["paths"]["tfidf_t1_classes"]
    model_path = PROJECT_ROOT / config["paths"]["tfidf_models"]

    evaluation_contract = json.loads(evaluation_path.read_text(encoding="utf-8"))
    baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
    frame = load_rows(input_path, config)
    frames = split_frames(frame)
    validate_counts(frames, evaluation_contract)
    vectorizer, encoder, matrices = build_features(frames, config)
    results, models, class_rows = train_models(config, frames, matrices)
    add_baseline_comparison(results, baseline_report)

    report = {
        "split_version": evaluation_contract["split_version"],
        "dvc_data_hash": evaluation_contract["dvc_data_hash"],
        "selection_view": evaluation_contract["selection_view"],
        "features": config["features"]["initial"],
        "tfidf": config["tfidf"],
        "results": results,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(class_rows).to_csv(class_path, index=False)
    save_models(
        model_path,
        vectorizer,
        encoder,
        models,
        config,
        evaluation_contract["dvc_data_hash"],
    )
    save_records(config, results)

    print("Validation without shared text:")
    for target, result in results.items():
        primary = "macro_f1" if target == "T1" else "average_precision"
        value = result["metrics"]["validation"]["no_shared_text"][primary]
        improvement = result["primary_metric_improvement"]
        print(f"{target}: {primary}={value:.6f}; vs product={improvement:+.6f}")
    print(f"Vocabulary: {len(vectorizer.vocabulary_):,}")
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")
    print(f"Models: {model_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
