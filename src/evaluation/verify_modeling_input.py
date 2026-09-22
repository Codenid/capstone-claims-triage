"""Verify the prepared table before starting modeling."""

import json
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data.build_targets import (
    COMPLETE_COLUMNS,
    HOLDOUT,
    NO_SHARED_COLUMNS,
    OOD,
    PERIOD_COLUMN,
    T2_COLUMN,
    T3_COLUMN,
    T4_COLUMN,
)
from src.data.finalize_prepared import (
    EXPECTED_METADATA,
    validate_prepared_source,
)

compute = cast(Any, pc)

INPUT_PATH = Path("data/processed/prepared.parquet")
REPORT_PATH = Path("reports/modeling/input_contract.json")


def true_count(table: pa.Table, column: str, period: str) -> int:
    """Count true values for one period."""
    period_mask = compute.equal(table[PERIOD_COLUMN], pa.scalar(period))
    values = compute.filter(table[column], period_mask)
    if values.null_count:
        raise ValueError(f"{column} contains null values.")
    return int(compute.sum(compute.cast(values, pa.int64())).as_py())


def row_count(table: pa.Table, period: str) -> int:
    period_mask = compute.equal(table[PERIOD_COLUMN], pa.scalar(period))
    return int(compute.sum(compute.cast(period_mask, pa.int64())).as_py())


def build_contract(input_path: Path) -> dict[str, object]:
    """Validate the prepared Parquet and return its modeling contract."""
    validate_prepared_source(input_path)
    source = pq.ParquetFile(input_path)
    columns = [
        PERIOD_COLUMN,
        *COMPLETE_COLUMNS.values(),
        *NO_SHARED_COLUMNS.values(),
    ]
    audit = pq.read_table(input_path, columns=columns)
    periods = sorted(set(compute.unique(audit[PERIOD_COLUMN]).to_pylist()))

    complete = {
        period: {
            target: true_count(audit, column, period)
            for target, column in COMPLETE_COLUMNS.items()
        }
        for period in periods
    }
    no_shared = {
        period: {
            target: true_count(audit, column, period)
            for target, column in NO_SHARED_COLUMNS.items()
        }
        for period in periods
    }

    if any(complete[HOLDOUT].values()) or any(no_shared[HOLDOUT].values()):
        raise ValueError("2025-H2 must remain blocked for every target.")
    for target in [T2_COLUMN, T3_COLUMN, T4_COLUMN]:
        if complete[OOD][target] or no_shared[OOD][target]:
            raise ValueError(f"2026 must remain blocked for {target}.")

    metadata = source.schema_arrow.metadata or {}
    return {
        "input": input_path.as_posix(),
        "rows": source.metadata.num_rows,
        "columns": source.metadata.num_columns,
        "size_bytes": input_path.stat().st_size,
        "versions": {
            key.decode("utf-8"): metadata[key].decode("utf-8")
            for key in EXPECTED_METADATA
        },
        "period_rows": {period: row_count(audit, period) for period in periods},
        "eligible_complete": complete,
        "eligible_no_shared_text": no_shared,
    }


def main() -> None:
    contract = build_contract(INPUT_PATH)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Rows: {contract['rows']:,}")
    print(f"Columns: {contract['columns']}")
    print("2025-H2 evaluation: blocked")
    print("2026 T2-T4 evaluation: blocked")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
