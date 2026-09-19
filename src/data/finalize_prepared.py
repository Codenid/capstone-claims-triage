"""Validate and publish the prepared modeling table without changing it."""

from pathlib import Path
from shutil import copyfile
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data.apply_taxonomy import (
    CANONICAL_ISSUE_COLUMN,
    CANONICAL_PRODUCT_COLUMN,
    FREEZE_DATE,
    KNOWN_ISSUE_COLUMN,
    KNOWN_PAIR_COLUMN,
    KNOWN_PRODUCT_COLUMN,
    TAXONOMY_STATUS,
    TAXONOMY_VERSION,
)
from src.data.build_targets import (
    COMPLETE_COLUMNS,
    CONTEXT,
    HOLDOUT,
    HOLDOUT_STATUS_COLUMN,
    KNOWN_T1_COLUMN,
    KNOWN_T2_COLUMN,
    KNOWN_T3_COLUMN,
    KNOWN_T4_COLUMN,
    NO_SHARED_COLUMNS,
    NO_SHARED_TEXT_COLUMN,
    OOD,
    PERIOD_COLUMN,
    REVIEWED_GROUP_COLUMN,
    REVIEWED_ID_COLUMN,
    T1_COLUMN,
    T2_COLUMN,
    T3_COLUMN,
    T4_COLUMN,
    TARGETS_VERSION,
    TRAIN,
    VALIDATION,
)
from src.data.normalize_text import (
    HASH_COLUMN,
    NORMALIZED_COLUMN,
    NORMALIZER_VERSION,
)
from src.data.type_data import DATE_COLUMNS, EXPECTED_COLUMNS

compute = cast(Any, pc)

EXPECTED_ROW_COUNT = 3_837_184
EXPECTED_PERIODS = {CONTEXT, TRAIN, VALIDATION, HOLDOUT, OOD}
EXPECTED_PREPARED_COLUMNS = [
    *EXPECTED_COLUMNS,
    NORMALIZED_COLUMN,
    HASH_COLUMN,
    CANONICAL_PRODUCT_COLUMN,
    CANONICAL_ISSUE_COLUMN,
    KNOWN_PRODUCT_COLUMN,
    KNOWN_ISSUE_COLUMN,
    KNOWN_PAIR_COLUMN,
    PERIOD_COLUMN,
    T1_COLUMN,
    T2_COLUMN,
    T3_COLUMN,
    T4_COLUMN,
    KNOWN_T1_COLUMN,
    KNOWN_T2_COLUMN,
    KNOWN_T3_COLUMN,
    KNOWN_T4_COLUMN,
    REVIEWED_ID_COLUMN,
    REVIEWED_GROUP_COLUMN,
    HOLDOUT_STATUS_COLUMN,
    NO_SHARED_TEXT_COLUMN,
    *COMPLETE_COLUMNS.values(),
    *NO_SHARED_COLUMNS.values(),
]
EXPECTED_METADATA = {
    b"text_normalizer": NORMALIZER_VERSION.encode("utf-8"),
    b"taxonomy_version": TAXONOMY_VERSION.encode("utf-8"),
    b"taxonomy_status": TAXONOMY_STATUS.encode("utf-8"),
    b"taxonomy_freeze_date": FREEZE_DATE.isoformat().encode("utf-8"),
    b"targets_periods_version": TARGETS_VERSION.encode("utf-8"),
    b"holdout_final_evaluation_allowed": b"false",
}


def validate_prepared_source(
    input_path: Path,
    expected_rows: int = EXPECTED_ROW_COUNT,
) -> None:
    """Validate the fixed contract of the table delivered to modeling."""
    source = pq.ParquetFile(input_path)

    if source.metadata.num_rows != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} rows, found {source.metadata.num_rows:,}."
        )
    if source.schema_arrow.names != EXPECTED_PREPARED_COLUMNS:
        raise ValueError("The prepared table has unexpected or reordered columns.")

    for column_name in DATE_COLUMNS:
        if not pa.types.is_date32(source.schema_arrow.field(column_name).type):
            raise ValueError(f"{column_name} must use the date32 type.")

    metadata = source.schema_arrow.metadata or {}
    for key, expected_value in EXPECTED_METADATA.items():
        if metadata.get(key) != expected_value:
            raise ValueError(f"Missing or invalid metadata: {key.decode('utf-8')}.")

    identifiers = pq.read_table(input_path, columns=["Complaint ID"])["Complaint ID"]
    if identifiers.null_count:
        raise ValueError("Complaint ID cannot contain null values.")
    if compute.count_distinct(identifiers).as_py() != expected_rows:
        raise ValueError("Complaint ID must be unique.")

    periods = pq.read_table(input_path, columns=[PERIOD_COLUMN])[PERIOD_COLUMN]
    if periods.null_count:
        raise ValueError("Period cannot contain null values.")
    observed_periods = set(compute.unique(periods).to_pylist())
    if observed_periods != EXPECTED_PERIODS:
        raise ValueError(
            f"Unexpected periods. Expected {sorted(EXPECTED_PERIODS)}, "
            f"found {sorted(observed_periods)}."
        )


def finalize_prepared(
    input_path: Path,
    output_path: Path,
    expected_rows: int = EXPECTED_ROW_COUNT,
) -> None:
    """Validate and copy the approved table byte for byte."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")

    validate_prepared_source(input_path, expected_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    copyfile(input_path, output_path)

    if output_path.stat().st_size != input_path.stat().st_size:
        raise ValueError("The copied file size does not match the source.")

    print(f"Rows: {expected_rows:,}")
    print(f"Columns: {len(EXPECTED_PREPARED_COLUMNS)}")
    print(f"Periods: {len(EXPECTED_PERIODS)}")
    print(f"Output: {output_path}")


def main() -> None:
    finalize_prepared(
        Path("data/interim/targets_periods.parquet"),
        Path("data/processed/prepared.parquet"),
    )


if __name__ == "__main__":
    main()
