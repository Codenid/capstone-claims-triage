"""Build T1-T4, temporal periods and evaluation eligibility masks."""

from datetime import date
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.apply_taxonomy import (
    CANONICAL_ISSUE_COLUMN,
    KNOWN_ISSUE_COLUMN,
    KNOWN_PAIR_COLUMN,
)
from src.data.normalize_text import HASH_COLUMN

TARGETS_VERSION = "targets_periods_v1"

PERIOD_COLUMN = "period"
T1_COLUMN = "T1"
T2_COLUMN = "T2"
T3_COLUMN = "T3"
T4_COLUMN = "T4"
KNOWN_T1_COLUMN = "known_T1"
KNOWN_T2_COLUMN = "known_T2"
KNOWN_T3_COLUMN = "known_T3"
KNOWN_T4_COLUMN = "known_T4"
REVIEWED_ID_COLUMN = "previously_reviewed_id"
REVIEWED_GROUP_COLUMN = "excluded_by_reviewed_text_group"
HOLDOUT_STATUS_COLUMN = "holdout_status"
NO_SHARED_TEXT_COLUMN = "no_shared_reference_text"

CONTEXT = "context_2015_2022"
TRAIN = "train_2023_2024"
VALIDATION = "validation_2025_h1"
HOLDOUT = "holdout_2025_h2"
OOD = "ood_2026_partial"

COMPLETE_COLUMNS = {
    T1_COLUMN: "eligible_T1_complete",
    T2_COLUMN: "eligible_T2_complete",
    T3_COLUMN: "eligible_T3_complete",
    T4_COLUMN: "eligible_T4_complete",
}
NO_SHARED_COLUMNS = {
    T1_COLUMN: "eligible_T1_no_shared_text",
    T2_COLUMN: "eligible_T2_no_shared_text",
    T3_COLUMN: "eligible_T3_no_shared_text",
    T4_COLUMN: "eligible_T4_no_shared_text",
}

RELIEF_POSITIVE = {
    "Closed with monetary relief",
    "Closed with non-monetary relief",
}
EXPLANATION = "Closed with explanation"
MONETARY_RELIEF = "Closed with monetary relief"
NON_MONETARY_RELIEF = "Closed with non-monetary relief"


def period_for_date(received: date) -> str:
    """Assign one and only one approved temporal period."""
    if received < date(2023, 1, 1):
        return CONTEXT
    if received < date(2025, 1, 1):
        return TRAIN
    if received < date(2025, 7, 1):
        return VALIDATION
    if received < date(2026, 1, 1):
        return HOLDOUT
    return OOD


def relief_target(response: str | None) -> bool | None:
    if response in RELIEF_POSITIVE:
        return True
    if response == EXPLANATION:
        return False
    return None


def monetary_target(response: str | None) -> bool | None:
    if response == MONETARY_RELIEF:
        return True
    if response in {NON_MONETARY_RELIEF, EXPLANATION}:
        return False
    return None


def timely_target(value: str | None) -> bool | None:
    if value == "No":
        return True
    if value == "Yes":
        return False
    return None


def collect_reference_hashes(source: pq.ParquetFile) -> tuple[set[str], set[str]]:
    """Collect text groups from training and validation only."""
    train_hashes = set()
    validation_hashes = set()
    for batch in source.iter_batches(
        batch_size=100_000,
        columns=["Date received", HASH_COLUMN],
    ):
        dates = batch.column(0).to_pylist()
        hashes = batch.column(1).to_pylist()
        for received, text_hash in zip(dates, hashes):
            period = period_for_date(received)
            if period == TRAIN:
                train_hashes.add(text_hash)
            elif period == VALIDATION:
                validation_hashes.add(text_hash)
    return train_hashes, validation_hashes


def build_targets(
    table: pa.Table,
    train_hashes: set[str],
    validation_hashes: set[str],
    final_holdout_allowed: bool,
) -> pa.Table:
    """Append targets, periods and eligibility masks to a table."""
    dates = table["Date received"].combine_chunks().to_pylist()
    issues = table[CANONICAL_ISSUE_COLUMN].combine_chunks().to_pylist()
    known_issues = table[KNOWN_ISSUE_COLUMN].combine_chunks().to_pylist()
    known_pairs = table[KNOWN_PAIR_COLUMN].combine_chunks().to_pylist()
    responses = table["Company response to consumer"].combine_chunks().to_pylist()
    timely_values = table["Timely response?"].combine_chunks().to_pylist()
    hashes = table[HASH_COLUMN].combine_chunks().to_pylist()

    periods = []
    t1_values = []
    t2_values = []
    t3_values = []
    t4_values = []
    known_t1_values = []
    known_t2_values = []
    known_t3_values = []
    known_t4_values = []
    reviewed_ids = []
    reviewed_groups = []
    holdout_statuses = []
    no_shared_values = []
    complete_values = {target: [] for target in COMPLETE_COLUMNS}
    no_shared_eligible_values = {target: [] for target in NO_SHARED_COLUMNS}

    for received, issue, known_issue, known_pair, response, timely, text_hash in zip(
        dates,
        issues,
        known_issues,
        known_pairs,
        responses,
        timely_values,
        hashes,
    ):
        if received is None:
            raise ValueError("Date received cannot be null.")

        period = period_for_date(received)
        known_t1 = bool(known_issue and known_pair and issue is not None)
        t1 = issue if known_t1 else None
        t2 = relief_target(response)
        t3 = monetary_target(response)
        t4 = timely_target(timely)
        known_values = {
            T1_COLUMN: known_t1,
            T2_COLUMN: t2 is not None,
            T3_COLUMN: t3 is not None,
            T4_COLUMN: t4 is not None,
        }

        if period == VALIDATION:
            no_shared = text_hash not in train_hashes
        elif period in {HOLDOUT, OOD}:
            no_shared = text_hash not in train_hashes and text_hash not in validation_hashes
        else:
            no_shared = True

        holdout_is_usable = period != HOLDOUT or final_holdout_allowed
        results_are_mature = period != OOD
        eligibility = {
            T1_COLUMN: known_values[T1_COLUMN] and holdout_is_usable,
            T2_COLUMN: known_values[T2_COLUMN] and holdout_is_usable and results_are_mature,
            T3_COLUMN: known_values[T3_COLUMN] and holdout_is_usable and results_are_mature,
            T4_COLUMN: known_values[T4_COLUMN] and holdout_is_usable and results_are_mature,
        }

        periods.append(period)
        t1_values.append(t1)
        t2_values.append(t2)
        t3_values.append(t3)
        t4_values.append(t4)
        known_t1_values.append(known_values[T1_COLUMN])
        known_t2_values.append(known_values[T2_COLUMN])
        known_t3_values.append(known_values[T3_COLUMN])
        known_t4_values.append(known_values[T4_COLUMN])
        reviewed_ids.append(None if period == HOLDOUT and not final_holdout_allowed else False)
        reviewed_groups.append(None if period == HOLDOUT and not final_holdout_allowed else False)
        holdout_statuses.append(
            "contaminated_ids_unavailable" if period == HOLDOUT and not final_holdout_allowed else "not_blocked"
        )
        no_shared_values.append(no_shared)
        for target in COMPLETE_COLUMNS:
            complete_values[target].append(eligibility[target])
            no_shared_eligible_values[target].append(eligibility[target] and no_shared)

    result = table.append_column(PERIOD_COLUMN, pa.array(periods, type=pa.string()))
    result = result.append_column(T1_COLUMN, pa.array(t1_values, type=pa.large_string()))
    result = result.append_column(T2_COLUMN, pa.array(t2_values, type=pa.bool_()))
    result = result.append_column(T3_COLUMN, pa.array(t3_values, type=pa.bool_()))
    result = result.append_column(T4_COLUMN, pa.array(t4_values, type=pa.bool_()))
    result = result.append_column(KNOWN_T1_COLUMN, pa.array(known_t1_values))
    result = result.append_column(KNOWN_T2_COLUMN, pa.array(known_t2_values))
    result = result.append_column(KNOWN_T3_COLUMN, pa.array(known_t3_values))
    result = result.append_column(KNOWN_T4_COLUMN, pa.array(known_t4_values))
    result = result.append_column(REVIEWED_ID_COLUMN, pa.array(reviewed_ids, type=pa.bool_()))
    result = result.append_column(REVIEWED_GROUP_COLUMN, pa.array(reviewed_groups, type=pa.bool_()))
    result = result.append_column(HOLDOUT_STATUS_COLUMN, pa.array(holdout_statuses, type=pa.string()))
    result = result.append_column(NO_SHARED_TEXT_COLUMN, pa.array(no_shared_values))
    for target, column_name in COMPLETE_COLUMNS.items():
        result = result.append_column(column_name, pa.array(complete_values[target]))
    for target, column_name in NO_SHARED_COLUMNS.items():
        result = result.append_column(column_name, pa.array(no_shared_eligible_values[target]))

    metadata = dict(result.schema.metadata or {})
    metadata[b"targets_periods_version"] = TARGETS_VERSION.encode("utf-8")
    metadata[b"holdout_final_evaluation_allowed"] = str(final_holdout_allowed).lower().encode("utf-8")
    return result.replace_schema_metadata(metadata)


def prepare(input_path: Path, output_path: Path, holdout_status_path: Path) -> None:
    """Build targets and periods in batches and write a new Parquet."""
    holdout_status = json.loads(holdout_status_path.read_text(encoding="utf-8"))
    final_holdout_allowed = holdout_status["final_evaluation_allowed"]
    if holdout_status["status"] != "ids_not_recovered" or final_holdout_allowed:
        raise ValueError("The current holdout contract must record unavailable reviewed IDs.")

    source = pq.ParquetFile(input_path)
    train_hashes, validation_hashes = collect_reference_hashes(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    rows = 0
    try:
        for batch in source.iter_batches(batch_size=50_000):
            prepared = build_targets(
                pa.Table.from_batches([batch]),
                train_hashes,
                validation_hashes,
                final_holdout_allowed,
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    prepared.schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            writer.write_table(prepared, row_group_size=50_000)
            rows += prepared.num_rows
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError("The input file is empty.")
    if rows != source.metadata.num_rows:
        raise ValueError("Target construction changed the number of rows.")

    print(f"Rows: {rows:,}")
    print(f"Columns: {len(source.schema.names) + 21}")
    print(f"Version: {TARGETS_VERSION}")
    print("2025-H2 final evaluation allowed: false")
    print(f"Output: {output_path}")


def main() -> None:
    prepare(
        Path("data/interim/taxonomy.parquet"),
        Path("data/interim/targets_periods.parquet"),
        Path("configs/holdout_review_status.json"),
    )


if __name__ == "__main__":
    main()
