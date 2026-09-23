"""Audit and freeze temporal evaluation rules for M2."""

from __future__ import annotations

from collections import Counter
from datetime import date
import json
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data.build_targets import COMPLETE_COLUMNS, NO_SHARED_COLUMNS
from src.data.normalize_text import HASH_COLUMN
from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    load_experiment_config,
    save_run_record,
)

compute = cast(Any, pc)
DATE_COLUMN = "Date received"
NO_SHARED_COLUMN = "no_shared_reference_text"
TARGETS = ("T1", "T2", "T3", "T4")


def parse_splits(config: dict[str, Any]) -> dict[str, tuple[date, date]]:
    splits = {}
    previous_end = None
    for name, limits in config["evaluation"]["splits"].items():
        start = date.fromisoformat(str(limits["start"]))
        end = date.fromisoformat(str(limits["end"]))
        if start >= end:
            raise ValueError(f"Invalid date range for {name}.")
        if previous_end is not None and start != previous_end:
            raise ValueError("Evaluation splits must be contiguous and ordered.")
        splits[name] = (start, end)
        previous_end = end
    return splits


def date_mask(values: pa.Array, start: date, end: date) -> pa.Array:
    return compute.and_(
        compute.greater_equal(values, pa.scalar(start)),
        compute.less(values, pa.scalar(end)),
    )


def true_count(values: pa.Array) -> int:
    result = compute.sum(compute.cast(values, pa.int64())).as_py()
    return int(result or 0)


def collect_fit_hashes(
    source: pq.ParquetFile,
    fit_start: date,
    fit_end: date,
) -> set[str]:
    hashes: set[str] = set()
    for batch in source.iter_batches(
        batch_size=100_000,
        columns=[DATE_COLUMN, HASH_COLUMN],
    ):
        mask = date_mask(batch[DATE_COLUMN], fit_start, fit_end)
        hashes.update(compute.filter(batch[HASH_COLUMN], mask).to_pylist())
    return hashes


def empty_counts() -> dict[str, Any]:
    return {
        split: {
            view: {
                target: Counter() if target == "T1" else {"eligible": 0, "positive": 0}
                for target in TARGETS
            }
            for view in ("complete", "no_shared_text")
        }
        for split in ("fit", "calibration", "validation")
    }


def audit_evaluation(
    input_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    splits = parse_splits(config)
    source = pq.ParquetFile(input_path)
    fit_hashes = collect_fit_hashes(source, *splits["fit"])
    fit_hash_values = pa.array(list(fit_hashes), type=pa.string())

    counts = empty_counts()
    row_counts = {
        split: {"complete": 0, "no_shared_text": 0}
        for split in splits
    }
    columns = [
        DATE_COLUMN,
        HASH_COLUMN,
        NO_SHARED_COLUMN,
        *TARGETS,
        *COMPLETE_COLUMNS.values(),
        *NO_SHARED_COLUMNS.values(),
    ]

    for batch in source.iter_batches(batch_size=100_000, columns=columns):
        dates = batch[DATE_COLUMN]
        hashes = batch[HASH_COLUMN]
        repeated_from_fit = compute.is_in(hashes, value_set=fit_hash_values)

        for split, (start, end) in splits.items():
            split_rows = date_mask(dates, start, end)
            row_counts[split]["complete"] += true_count(split_rows)

            if split == "fit":
                no_shared_rows = split_rows
            elif split == "calibration":
                no_shared_rows = compute.and_(
                    split_rows,
                    compute.invert(repeated_from_fit),
                )
            else:
                no_shared_rows = compute.and_(
                    split_rows,
                    batch[NO_SHARED_COLUMN],
                )
            row_counts[split]["no_shared_text"] += true_count(no_shared_rows)

            for view in ("complete", "no_shared_text"):
                for target in TARGETS:
                    eligible_column = COMPLETE_COLUMNS[target]
                    if view == "no_shared_text" and split == "validation":
                        eligible_column = NO_SHARED_COLUMNS[target]

                    eligible = compute.and_(split_rows, batch[eligible_column])
                    if view == "no_shared_text" and split == "calibration":
                        eligible = compute.and_(eligible, compute.invert(repeated_from_fit))

                    values = compute.filter(batch[target], eligible)
                    if target == "T1":
                        counts[split][view][target].update(values.to_pylist())
                    else:
                        counts[split][view][target]["eligible"] += len(values)
                        counts[split][view][target]["positive"] += true_count(values)

    return finalize_contract(config, row_counts, counts, len(fit_hashes))


def finalize_contract(
    config: dict[str, Any],
    row_counts: dict[str, Any],
    counts: dict[str, Any],
    fit_unique_texts: int,
) -> dict[str, Any]:
    target_summary: dict[str, Any] = {}
    fit_labels = {
        view: set(counts["fit"][view]["T1"])
        for view in config["evaluation"]["views"]
    }

    for split in config["evaluation"]["splits"]:
        target_summary[split] = {}
        for view in config["evaluation"]["views"]:
            target_summary[split][view] = {}
            for target in TARGETS:
                values = counts[split][view][target]
                if target == "T1":
                    class_counts = dict(sorted(values.items()))
                    target_summary[split][view][target] = {
                        "eligible": sum(values.values()),
                        "classes": len(values),
                        "minimum_class_count": min(values.values()) if values else 0,
                        "unseen_classes_vs_fit": sorted(set(values) - fit_labels[view]),
                        "class_counts": class_counts,
                    }
                else:
                    eligible = values["eligible"]
                    positive = values["positive"]
                    target_summary[split][view][target] = {
                        "eligible": eligible,
                        "positive": positive,
                        "positive_rate": positive / eligible if eligible else None,
                    }

    input_contract_path = PROJECT_ROOT / config["paths"]["input_contract"]
    input_contract = json.loads(input_contract_path.read_text(encoding="utf-8"))
    train_rows = row_counts["fit"]["complete"] + row_counts["calibration"]["complete"]
    if train_rows != input_contract["period_rows"]["train_2023_2024"]:
        raise ValueError("Fit and calibration do not cover all 2023-2024 rows.")
    if row_counts["validation"]["complete"] != input_contract["period_rows"]["validation_2025_h1"]:
        raise ValueError("Validation does not cover all 2025-H1 rows.")

    for split in row_counts:
        if row_counts[split]["no_shared_text"] > row_counts[split]["complete"]:
            raise ValueError(f"Invalid no-shared-text row count for {split}.")
        for target in TARGETS:
            no_shared = target_summary[split]["no_shared_text"][target]["eligible"]
            complete = target_summary[split]["complete"][target]["eligible"]
            if no_shared > complete:
                raise ValueError(f"Invalid no-shared-text count for {split}/{target}.")

    return {
        "split_version": config["evaluation"]["split_version"],
        "dvc_data_hash": input_contract["dvc_md5"],
        "splits": {
            split: {
                "start": str(limits["start"]),
                "end": str(limits["end"]),
            }
            for split, limits in config["evaluation"]["splits"].items()
        },
        "blocked_periods": config["evaluation"]["blocked_periods"],
        "views": config["evaluation"]["views"],
        "selection_view": config["evaluation"]["selection_view"],
        "metrics": config["evaluation"]["metrics"],
        "binary_threshold": config["evaluation"]["binary_threshold"],
        "features": config["features"],
        "fit_unique_normalized_texts": fit_unique_texts,
        "row_counts": row_counts,
        "targets": target_summary,
    }


def main() -> None:
    config = load_experiment_config()
    input_path = PROJECT_ROOT / config["paths"]["input_data"]
    output_path = PROJECT_ROOT / config["paths"]["evaluation_contract"]
    contract = audit_evaluation(input_path, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    row_counts = contract["row_counts"]
    record = build_run_record(
        config=config,
        run_name="m2-evaluation-contract",
        stage="M2",
        target="evaluation_contract",
        split="fit_calibration_validation",
        view=contract["selection_view"],
        features=config["features"]["initial"],
        parameters={
            "split_version": contract["split_version"],
            "binary_threshold": contract["binary_threshold"],
        },
        metrics={
            "fit_rows": row_counts["fit"]["complete"],
            "calibration_rows": row_counts["calibration"]["complete"],
            "calibration_no_shared_rows": row_counts["calibration"]["no_shared_text"],
            "validation_rows": row_counts["validation"]["complete"],
            "validation_no_shared_rows": row_counts["validation"]["no_shared_text"],
        },
        artifacts=[config["paths"]["evaluation_contract"]],
    )
    record_path = (
        PROJECT_ROOT
        / config["paths"]["offline_runs"]
        / "m2-evaluation-contract"
        / "run.json"
    )
    save_run_record(record, record_path)

    print(f"Split version: {contract['split_version']}")
    for split, views in contract["row_counts"].items():
        print(
            f"{split}: {views['complete']:,} complete; "
            f"{views['no_shared_text']:,} without shared text"
        )
    print(f"Selection view: {contract['selection_view']}")
    print(f"Report: {output_path.relative_to(PROJECT_ROOT)}")
    print(f"Offline record: {record_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
