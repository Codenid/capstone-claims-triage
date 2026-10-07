"""Validate the raw schema and convert date columns to real dates."""

from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data.params import preparation_settings

compute = cast(Any, pc)

# Decisions declared in params.yaml (preparacion.type_data); dvc.lock records them.
SETTINGS = preparation_settings("type_data")
EXPECTED_COLUMNS: list[str] = list(SETTINGS["columnas_esperadas"])
DATE_COLUMNS: list[str] = list(SETTINGS["columnas_fecha"])
DATE_FORMAT: str = SETTINGS["formato_fecha"]


def type_table(table: pa.Table) -> pa.Table:
    """Return the same table with validated columns and typed dates."""
    if table.column_names != EXPECTED_COLUMNS:
        missing = sorted(set(EXPECTED_COLUMNS) - set(table.column_names))
        extra = sorted(set(table.column_names) - set(EXPECTED_COLUMNS))
        raise ValueError(f"Unexpected schema. Missing: {missing}. Extra: {extra}.")

    typed = table
    for column_name in DATE_COLUMNS:
        original = typed[column_name]
        parsed = compute.strptime(
            original,
            format=DATE_FORMAT,
            unit="s",
            error_is_null=True,
        )
        invalid = compute.sum(
            compute.and_(
                compute.is_valid(original),
                compute.is_null(parsed),
            )
        ).as_py()
        if invalid:
            raise ValueError(f"{column_name} contains {invalid} invalid dates.")

        column_index = typed.schema.get_field_index(column_name)
        typed = typed.set_column(
            column_index,
            column_name,
            compute.cast(parsed, pa.date32()),
        )

    return typed


def prepare(input_path: Path, output_path: Path) -> None:
    """Read the raw Parquet, type it and write a new Parquet."""
    raw = pq.read_table(input_path)
    typed = type_table(raw)

    if typed.num_rows != raw.num_rows:
        raise ValueError("Typing changed the number of rows.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        typed,
        output_path,
        compression="zstd",
        use_dictionary=True,
        row_group_size=100_000,
    )

    print(f"Rows: {typed.num_rows:,}")
    print(f"Columns: {typed.num_columns}")
    print(f"Output: {output_path}")


def main() -> None:
    prepare(Path(SETTINGS["entrada"]), Path(SETTINGS["salida"]))


if __name__ == "__main__":
    main()
