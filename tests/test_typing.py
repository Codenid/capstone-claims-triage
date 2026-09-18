import unittest

import pyarrow as pa

from src.data.type_data import DATE_COLUMNS, EXPECTED_COLUMNS, type_table


def sample_table(date_received="2025-01-15"):
    values = {column: ["value"] for column in EXPECTED_COLUMNS}
    values["Date received"] = [date_received]
    values["Date sent to company"] = ["2025-01-16"]
    values["Complaint ID"] = ["123"]
    return pa.table(values)


class TypingTests(unittest.TestCase):
    def test_types_dates_without_changing_rows_or_columns(self):
        raw = sample_table()

        typed = type_table(raw)

        self.assertEqual(typed.num_rows, raw.num_rows)
        self.assertEqual(typed.column_names, EXPECTED_COLUMNS)
        for column in DATE_COLUMNS:
            self.assertEqual(typed.schema.field(column).type, pa.date32())
        self.assertEqual(typed["Complaint ID"].to_pylist(), ["123"])

    def test_rejects_invalid_dates(self):
        with self.assertRaisesRegex(ValueError, "1 invalid dates"):
            type_table(sample_table(date_received="not-a-date"))

    def test_rejects_unexpected_schema(self):
        raw = sample_table().drop(["Tags"])

        with self.assertRaisesRegex(ValueError, r"Missing: \['Tags'\]"):
            type_table(raw)


if __name__ == "__main__":
    unittest.main()
