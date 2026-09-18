import unittest

import pyarrow as pa

from src.data.normalize_text import (
    HASH_COLUMN,
    NARRATIVE_COLUMN,
    NORMALIZED_COLUMN,
    NORMALIZER_VERSION,
    add_normalized_columns,
    hash_text,
    normalize_text,
)


class NormalizeTextTests(unittest.TestCase):
    def test_applies_rules_in_the_approved_order(self):
        text = "  ＡBC\tXX \nStraße  "

        self.assertEqual(normalize_text(text), "abc <redacted> strasse")

    def test_does_not_redact_x_inside_a_word(self):
        self.assertEqual(normalize_text("Exxon XX"), "exxon <redacted>")

    def test_is_idempotent(self):
        once = normalize_text("  Claim\tXXXX\nTEXT  ")

        self.assertEqual(normalize_text(once), once)

    def test_hashes_utf8_text_with_sha256(self):
        self.assertEqual(
            hash_text("abc"),
            "ba7816bf8f01cfea414140de5dae2223"
            "b00361a396177a9cb410ff61f20015ad",
        )

    def test_adds_columns_without_changing_original_data(self):
        table = pa.table(
            {
                "Complaint ID": ["1", "2"],
                NARRATIVE_COLUMN: ["  XX  ", None],
            }
        )

        result = add_normalized_columns(table)

        self.assertEqual(result.num_rows, table.num_rows)
        self.assertEqual(result.column_names, table.column_names + [NORMALIZED_COLUMN, HASH_COLUMN])
        self.assertEqual(result[NARRATIVE_COLUMN].to_pylist(), ["  XX  ", None])
        self.assertEqual(result[NORMALIZED_COLUMN].to_pylist(), ["<redacted>", None])
        self.assertEqual(result[HASH_COLUMN].to_pylist(), [hash_text("<redacted>"), None])
        self.assertEqual(
            result.schema.metadata[b"text_normalizer"].decode("utf-8"),
            NORMALIZER_VERSION,
        )


if __name__ == "__main__":
    unittest.main()
