import tempfile
import unittest
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.build_targets import (
    CONTEXT,
    HOLDOUT,
    OOD,
    PERIOD_COLUMN,
    TRAIN,
    VALIDATION,
)
from src.data.finalize_prepared import (
    EXPECTED_METADATA,
    EXPECTED_PREPARED_COLUMNS,
    finalize_prepared,
    validate_prepared_source,
)
from src.data.type_data import DATE_COLUMNS


PERIODS = [CONTEXT, TRAIN, VALIDATION, HOLDOUT, OOD]


def sample_table(ids=None, include_metadata=True):
    ids = ids or ["1", "2", "3", "4", "5"]
    rows = len(ids)
    values = {
        column: pa.array([None] * rows, type=pa.string())
        for column in EXPECTED_PREPARED_COLUMNS
    }
    for column in DATE_COLUMNS:
        values[column] = pa.array([date(2025, 1, 1)] * rows, type=pa.date32())
    values["Complaint ID"] = pa.array(ids, type=pa.string())
    values[PERIOD_COLUMN] = pa.array(PERIODS[:rows], type=pa.string())

    table = pa.table(values)
    if include_metadata:
        table = table.replace_schema_metadata(EXPECTED_METADATA)
    return table


class FinalizePreparedTests(unittest.TestCase):
    def test_copies_valid_table_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.parquet"
            output = Path(directory) / "prepared.parquet"
            pq.write_table(sample_table(), source)

            finalize_prepared(source, output, expected_rows=5)

            self.assertEqual(output.read_bytes(), source.read_bytes())

    def test_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.parquet"
            pq.write_table(sample_table(ids=["1", "2", "3", "4", "4"]), source)

            with self.assertRaisesRegex(ValueError, "Complaint ID must be unique"):
                validate_prepared_source(source, expected_rows=5)

    def test_rejects_missing_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.parquet"
            pq.write_table(sample_table(include_metadata=False), source)

            with self.assertRaisesRegex(ValueError, "Missing or invalid metadata"):
                validate_prepared_source(source, expected_rows=5)


if __name__ == "__main__":
    unittest.main()
