"""Evaluate global and product-frequency baselines for T1-T4."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from src.data.apply_taxonomy import CANONICAL_PRODUCT_COLUMN
from src.data.build_targets import COMPLETE_COLUMNS
from src.data.normalize_text import HASH_COLUMN
from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    load_experiment_config,
    save_run_record,
)

DATE_COLUMN = "Date received"
NO_SHARED_COLUMN = "no_shared_reference_text"
TARGETS = ("T1", "T2", "T3", "T4")
VIEWS = ("complete", "no_shared_text")


def complete_top_three(values: Iterable[str], fallback: Iterable[str]) -> tuple[str, str, str]:
    ranking: list[str] = []
    for value in [*values, *fallback]:
        if value not in ranking:
            ranking.append(value)
        if len(ranking) == 3:
            return ranking[0], ranking[1], ranking[2]
    raise ValueError("At least three distinct T1 classes are required.")


def choose_threshold(y_true: pd.Series, scores: pd.Series) -> tuple[float, float]:
    truth = np.asarray(y_true, dtype=bool)
    probabilities = np.asarray(scores, dtype=float)
    precision, recall, thresholds = precision_recall_curve(truth, probabilities)
    if len(thresholds) == 0:
        raise ValueError("A threshold cannot be selected from empty scores.")

    numerator = 2 * precision[:-1] * recall[:-1]
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0,
    )
    best_f1 = float(f1_values.max())
    best_positions = np.flatnonzero(np.isclose(f1_values, best_f1))
    best_position = int(best_positions[-1])
    return float(thresholds[best_position]), best_f1


def load_evaluation_rows(input_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    split_config = config["evaluation"]["splits"]
    start = pd.Timestamp(split_config["fit"]["start"])
    calibration_start = pd.Timestamp(split_config["calibration"]["start"])
    validation_start = pd.Timestamp(split_config["validation"]["start"])
    end = pd.Timestamp(split_config["validation"]["end"])

    columns = [
        DATE_COLUMN,
        HASH_COLUMN,
        NO_SHARED_COLUMN,
        CANONICAL_PRODUCT_COLUMN,
        *TARGETS,
        *COMPLETE_COLUMNS.values(),
    ]
    table = pq.read_table(
        input_path,
        columns=columns,
        filters=[
            (DATE_COLUMN, ">=", start.date()),
            (DATE_COLUMN, "<", end.date()),
        ],
    )
    frame = table.to_pandas(categories=[CANONICAL_PRODUCT_COLUMN, "T1"])
    dates = pd.to_datetime(frame[DATE_COLUMN])
    frame["evaluation_split"] = np.select(
        [
            dates < calibration_start,
            dates < validation_start,
        ],
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


def target_mask(
    frame: pd.DataFrame,
    target: str,
    split: str,
    view: str,
) -> pd.Series:
    mask = (frame["evaluation_split"] == split) & frame[COMPLETE_COLUMNS[target]]
    if view == "no_shared_text":
        mask &= frame["no_shared_text"]
    return mask


def validate_counts(
    frame: pd.DataFrame,
    evaluation_contract: dict[str, Any],
) -> None:
    for split in ("fit", "calibration", "validation"):
        for view in VIEWS:
            for target in TARGETS:
                actual = int(target_mask(frame, target, split, view).sum())
                expected = evaluation_contract["targets"][split][view][target]["eligible"]
                if actual != expected:
                    raise ValueError(
                        f"Eligibility mismatch for {split}/{view}/{target}: "
                        f"expected {expected}, found {actual}."
                    )


def t1_rankings(frame: pd.DataFrame) -> tuple[list[str], dict[str, tuple[str, str, str]]]:
    fit = frame.loc[target_mask(frame, "T1", "fit", "complete")]
    global_ranking = fit["T1"].value_counts().index.astype(str).tolist()

    counts = (
        fit.groupby([CANONICAL_PRODUCT_COLUMN, "T1"], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(
            [CANONICAL_PRODUCT_COLUMN, "count", "T1"],
            ascending=[True, False, True],
        )
    )
    product_rankings = {}
    for product, rows in counts.groupby(CANONICAL_PRODUCT_COLUMN, observed=True):
        product_rankings[str(product)] = complete_top_three(
            rows["T1"].astype(str).tolist(),
            global_ranking,
        )
    return global_ranking, product_rankings


def evaluate_t1(
    frame: pd.DataFrame,
    model: str,
    global_ranking: list[str],
    product_rankings: dict[str, tuple[str, str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics: dict[str, Any] = {}
    class_rows: list[dict[str, Any]] = []
    global_top_three = complete_top_three(global_ranking, global_ranking)

    for split in ("calibration", "validation"):
        metrics[split] = {}
        for view in VIEWS:
            rows = frame.loc[target_mask(frame, "T1", split, view)]
            truth = rows["T1"].astype(str)
            if model == "global_frequency":
                first = pd.Series(global_top_three[0], index=rows.index)
                second = pd.Series(global_top_three[1], index=rows.index)
                third = pd.Series(global_top_three[2], index=rows.index)
            else:
                products = rows[CANONICAL_PRODUCT_COLUMN].astype(str)
                first = products.map({key: value[0] for key, value in product_rankings.items()})
                second = products.map({key: value[1] for key, value in product_rankings.items()})
                third = products.map({key: value[2] for key, value in product_rankings.items()})
                first = first.fillna(global_top_three[0])
                second = second.fillna(global_top_three[1])
                third = third.fillna(global_top_three[2])

            labels = sorted(truth.unique())
            macro_f1 = f1_score(
                truth,
                first,
                labels=labels,
                average="macro",
                zero_division=0,
            )
            top_three = ((truth == first) | (truth == second) | (truth == third)).mean()
            metrics[split][view] = {
                "eligible": int(len(rows)),
                "macro_f1": float(macro_f1),
                "top_3_accuracy": float(top_three),
            }

            precision, recall, f1, support = precision_recall_fscore_support(
                truth,
                first,
                labels=labels,
                zero_division=0,
            )
            for label, label_precision, label_recall, label_f1, label_support in zip(
                labels,
                precision,
                recall,
                f1,
                support,
            ):
                class_rows.append(
                    {
                        "model": model,
                        "split": split,
                        "view": view,
                        "class": label,
                        "support": int(label_support),
                        "precision": float(label_precision),
                        "recall": float(label_recall),
                        "f1": float(label_f1),
                    }
                )
    return metrics, class_rows


def binary_scores(
    frame: pd.DataFrame,
    target: str,
    model: str,
) -> tuple[pd.Series, float, dict[str, float]]:
    fit = frame.loc[target_mask(frame, target, "fit", "complete")]
    global_rate = float(fit[target].mean())
    product_rates = (
        fit.groupby(CANONICAL_PRODUCT_COLUMN, observed=True)[target]
        .mean()
        .astype(float)
        .to_dict()
    )

    if model == "global_frequency":
        scores = pd.Series(global_rate, index=frame.index, dtype=float)
    else:
        scores = (
            frame[CANONICAL_PRODUCT_COLUMN]
            .map(product_rates)
            .astype(float)
            .fillna(global_rate)
        )
    return scores, global_rate, {str(key): value for key, value in product_rates.items()}


def evaluate_binary(
    frame: pd.DataFrame,
    target: str,
    scores: pd.Series,
) -> tuple[dict[str, Any], float, float]:
    calibration_mask = target_mask(
        frame,
        target,
        "calibration",
        "no_shared_text",
    )
    threshold, calibration_f1 = choose_threshold(
        frame.loc[calibration_mask, target].astype(bool),
        scores.loc[calibration_mask].astype(float),
    )

    metrics: dict[str, Any] = {}
    for split in ("calibration", "validation"):
        metrics[split] = {}
        for view in VIEWS:
            mask = target_mask(frame, target, split, view)
            truth = frame.loc[mask, target].astype(bool)
            probabilities = scores.loc[mask].astype(float)
            prediction = probabilities >= threshold
            metrics[split][view] = {
                "eligible": int(mask.sum()),
                "positive": int(truth.sum()),
                "average_precision": float(average_precision_score(truth, probabilities)),
                "precision": float(precision_score(truth, prediction, zero_division=0)),
                "recall": float(recall_score(truth, prediction, zero_division=0)),
                "f1": float(f1_score(truth, prediction, zero_division=0)),
                "brier_score": float(brier_score_loss(truth, probabilities)),
            }
    return metrics, threshold, calibration_f1


def flat_mlflow_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        f"{split}_{view}_{metric}": float(value)
        for split, views in metrics.items()
        for view, values in views.items()
        for metric, value in values.items()
        if metric != "eligible" and metric != "positive"
    }


def save_run_records(
    config: dict[str, Any],
    results: dict[str, Any],
) -> None:
    report_path = config["paths"]["baseline_report"]
    class_path = config["paths"]["baseline_t1_classes"]
    run_root = PROJECT_ROOT / config["paths"]["offline_runs"] / "m3"

    for target, models in results["results"].items():
        for model, result in models.items():
            features = [] if model == "global_frequency" else [CANONICAL_PRODUCT_COLUMN]
            parameters = {"baseline": model}
            if target != "T1":
                parameters.update(
                    {
                        "fit_positive_rate": result["fit_positive_rate"],
                        "threshold": result["threshold"],
                        "threshold_rule": config["evaluation"]["binary_threshold"],
                    }
                )
            artifacts = [report_path]
            if target == "T1":
                artifacts.append(class_path)

            record = build_run_record(
                config=config,
                run_name=f"m3-{target.lower()}-{model}",
                stage="M3",
                target=target,
                split="calibration_validation",
                view="no_shared_text_primary",
                features=features,
                parameters=parameters,
                metrics=flat_mlflow_metrics(result["metrics"]),
                artifacts=artifacts,
            )
            save_run_record(record, run_root / f"{target.lower()}-{model}" / "run.json")


def main() -> None:
    config = load_experiment_config()
    input_path = PROJECT_ROOT / config["paths"]["input_data"]
    evaluation_path = PROJECT_ROOT / config["paths"]["evaluation_contract"]
    report_path = PROJECT_ROOT / config["paths"]["baseline_report"]
    class_path = PROJECT_ROOT / config["paths"]["baseline_t1_classes"]

    evaluation_contract = json.loads(evaluation_path.read_text(encoding="utf-8"))
    frame = load_evaluation_rows(input_path, config)
    validate_counts(frame, evaluation_contract)

    results: dict[str, Any] = {
        "split_version": evaluation_contract["split_version"],
        "dvc_data_hash": evaluation_contract["dvc_data_hash"],
        "selection_view": evaluation_contract["selection_view"],
        "results": {target: {} for target in TARGETS},
    }
    all_class_rows: list[dict[str, Any]] = []

    global_ranking, product_rankings = t1_rankings(frame)
    for model in ("global_frequency", "product_frequency"):
        metrics, class_rows = evaluate_t1(
            frame,
            model,
            global_ranking,
            product_rankings,
        )
        results["results"]["T1"][model] = {
            "global_top_3": global_ranking[:3],
            "product_top_3": product_rankings if model == "product_frequency" else None,
            "metrics": metrics,
        }
        all_class_rows.extend(class_rows)

    for target in ("T2", "T3", "T4"):
        for model in ("global_frequency", "product_frequency"):
            scores, global_rate, product_rates = binary_scores(frame, target, model)
            metrics, threshold, calibration_f1 = evaluate_binary(
                frame,
                target,
                scores,
            )
            results["results"][target][model] = {
                "fit_positive_rate": global_rate,
                "product_positive_rates": product_rates if model == "product_frequency" else None,
                "threshold": threshold,
                "calibration_no_shared_f1_at_threshold": calibration_f1,
                "metrics": metrics,
            }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(all_class_rows).to_csv(class_path, index=False)
    save_run_records(config, results)

    print("Validation without shared text:")
    for target, models in results["results"].items():
        for model, result in models.items():
            selected = result["metrics"]["validation"]["no_shared_text"]
            primary = "macro_f1" if target == "T1" else "average_precision"
            print(f"{target} / {model}: {primary}={selected[primary]:.6f}")
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")
    print(f"Offline records: {(PROJECT_ROOT / config['paths']['offline_runs'] / 'm3').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
