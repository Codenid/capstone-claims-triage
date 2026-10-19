"""Input contract of the pipeline (models_plan.md §29).

A new table needs four columns: `Complaint ID`, `Date received`, `Product`
and `Consumer complaint narrative`. The other twelve CFPB columns are
optional and are filled with nulls. The table then goes through the frozen
preparation of P1-P3: schema and dates, text normalization and the proposed
taxonomy. Rows dated on or before the taxonomy freeze date must bring their
`Issue`, as P3 requires; later rows may leave it empty.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa

from src.data.apply_taxonomy import (
    CANONICAL_PRODUCT_COLUMN,
    SETTINGS as TAXONOMY_SETTINGS,
    apply_taxonomy,
    load_mapping,
    load_pairs,
)
from src.data.normalize_text import NORMALIZED_COLUMN, add_normalized_columns
from src.data.type_data import DATE_COLUMNS, DATE_FORMAT, EXPECTED_COLUMNS, type_table
from src.evaluation.experiment import PROJECT_ROOT

ID_COLUMN = "Complaint ID"
DATE_COLUMN = "Date received"
PRODUCT_COLUMN = "Product"
NARRATIVE_COLUMN = "Consumer complaint narrative"
REQUIRED_COLUMNS = (ID_COLUMN, DATE_COLUMN, PRODUCT_COLUMN, NARRATIVE_COLUMN)
OUTPUT_COLUMNS = (
    ID_COLUMN,
    DATE_COLUMN,
    PRODUCT_COLUMN,
    "Issue",
    "Company",
    "State",
    NARRATIVE_COLUMN,
    NORMALIZED_COLUMN,
    CANONICAL_PRODUCT_COLUMN,
    "Issue canonical",
    "known_product",
)


def validate_input(frame: pd.DataFrame) -> pd.DataFrame:
    """The table with the sixteen CFPB columns in P1's order and clean keys."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"The input table lacks required columns: {missing}")
    result = frame.copy()
    for column in REQUIRED_COLUMNS:
        if bool(result[column].isna().any()):
            raise ValueError(f"The input table has nulls in {column}.")
    result[ID_COLUMN] = result[ID_COLUMN].astype(str)
    if bool(result[ID_COLUMN].duplicated().any()):
        raise ValueError("The input table repeats a Complaint ID.")
    for column in EXPECTED_COLUMNS:
        if column not in result.columns:
            result[column] = None
    for column in DATE_COLUMNS:
        values = result[column]
        has_dates = bool(values.map(lambda v: hasattr(v, "strftime")).any())
        if pd.api.types.is_datetime64_any_dtype(values) or has_dates:
            parsed = pd.to_datetime(values, errors="coerce")
            result[column] = parsed.dt.strftime(DATE_FORMAT).where(parsed.notna(), None)
        else:
            result[column] = values.where(values.notna(), None)
    return pd.DataFrame(result[list(EXPECTED_COLUMNS)]).reset_index(drop=True)


def taxonomy_inputs() -> tuple[dict[str, str], dict[str, str], set[tuple[str, str]]]:
    """The frozen product and issue maps and the valid pairs of P3."""
    root = PROJECT_ROOT
    product_map = load_mapping(
        root / TAXONOMY_SETTINGS["productos"], "raw_product", "canonical_product"
    )
    issue_map = load_mapping(
        root / TAXONOMY_SETTINGS["motivos"], "raw_issue", "canonical_issue"
    )
    pairs = load_pairs(root / TAXONOMY_SETTINGS["pares"])
    return product_map, issue_map, pairs


def prepare_table(frame: pd.DataFrame) -> pd.DataFrame:
    """P1-P3 on a validated table: typed dates, normalized text, taxonomy."""
    validated = validate_input(frame)
    # Every CFPB column travels as text; an all-null column must not be typed null.
    schema = pa.schema([(column, pa.string()) for column in EXPECTED_COLUMNS])
    table = pa.Table.from_pandas(
        validated.astype(object), schema=schema, preserve_index=False
    )
    table = type_table(table)
    table = add_normalized_columns(table)
    product_map, issue_map, pairs = taxonomy_inputs()
    table = apply_taxonomy(table, product_map, issue_map, pairs)
    prepared = table.to_pandas(date_as_object=False)
    prepared[DATE_COLUMN] = pd.to_datetime(prepared[DATE_COLUMN])
    return prepared


def read_table(path: Path) -> pd.DataFrame:
    """A parquet or CSV table, as the user hands it in."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={ID_COLUMN: str})
    raise ValueError(f"Unsupported input format: {path.suffix}")
