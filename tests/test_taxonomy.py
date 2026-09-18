import tempfile
import unittest
from datetime import date
from pathlib import Path

import pyarrow as pa

from src.data.apply_taxonomy import (
    CANONICAL_ISSUE_COLUMN,
    CANONICAL_PRODUCT_COLUMN,
    KNOWN_ISSUE_COLUMN,
    KNOWN_PAIR_COLUMN,
    KNOWN_PRODUCT_COLUMN,
    TAXONOMY_VERSION,
    UNKNOWN_CATEGORY,
    apply_taxonomy,
    load_mapping,
)


PRODUCT_MAP = {"Product A": "Canonical product"}
ISSUE_MAP = {"Issue A": "Canonical issue A", "Issue B": "Canonical issue B"}
VALID_PAIRS = {("Product A", "Issue A")}


def sample_table(received, product, issue):
    return pa.table(
        {
            "Date received": pa.array([received], type=pa.date32()),
            "Product": [product],
            "Issue": [issue],
            "Complaint ID": ["1"],
        }
    )


class TaxonomyTests(unittest.TestCase):
    def test_maps_known_categories_without_changing_original_data(self):
        raw = sample_table(date(2025, 6, 30), "Product A", "Issue A")

        result = apply_taxonomy(raw, PRODUCT_MAP, ISSUE_MAP, VALID_PAIRS)

        self.assertEqual(result.num_rows, raw.num_rows)
        self.assertEqual(result["Product"].to_pylist(), ["Product A"])
        self.assertEqual(result["Issue"].to_pylist(), ["Issue A"])
        self.assertEqual(result[CANONICAL_PRODUCT_COLUMN].to_pylist(), ["Canonical product"])
        self.assertEqual(result[CANONICAL_ISSUE_COLUMN].to_pylist(), ["Canonical issue A"])
        self.assertEqual(result[KNOWN_PRODUCT_COLUMN].to_pylist(), [True])
        self.assertEqual(result[KNOWN_ISSUE_COLUMN].to_pylist(), [True])
        self.assertEqual(result[KNOWN_PAIR_COLUMN].to_pylist(), [True])
        self.assertEqual(
            result.schema.metadata[b"taxonomy_version"].decode("utf-8"),
            TAXONOMY_VERSION,
        )

    def test_rejects_unmapped_development_category(self):
        raw = sample_table(date(2025, 6, 30), "New product", "Issue A")

        with self.assertRaisesRegex(ValueError, "before the taxonomy freeze date"):
            apply_taxonomy(raw, PRODUCT_MAP, ISSUE_MAP, VALID_PAIRS)

    def test_rejects_unseen_development_pair(self):
        raw = sample_table(date(2025, 6, 30), "Product A", "Issue B")

        with self.assertRaisesRegex(ValueError, "invalid Product-Issue pair"):
            apply_taxonomy(raw, PRODUCT_MAP, ISSUE_MAP, VALID_PAIRS)

    def test_marks_reserved_unknowns_without_updating_the_map(self):
        raw = sample_table(date(2025, 7, 1), "New product", "New issue")

        result = apply_taxonomy(raw, PRODUCT_MAP, ISSUE_MAP, VALID_PAIRS)

        self.assertEqual(result[CANONICAL_PRODUCT_COLUMN].to_pylist(), [UNKNOWN_CATEGORY])
        self.assertEqual(result[CANONICAL_ISSUE_COLUMN].to_pylist(), [None])
        self.assertEqual(result[KNOWN_PRODUCT_COLUMN].to_pylist(), [False])
        self.assertEqual(result[KNOWN_ISSUE_COLUMN].to_pylist(), [False])
        self.assertEqual(result[KNOWN_PAIR_COLUMN].to_pylist(), [False])

    def test_marks_unseen_reserved_pair(self):
        raw = sample_table(date(2025, 7, 1), "Product A", "Issue B")

        result = apply_taxonomy(raw, PRODUCT_MAP, ISSUE_MAP, VALID_PAIRS)

        self.assertEqual(result[CANONICAL_PRODUCT_COLUMN].to_pylist(), ["Canonical product"])
        self.assertEqual(result[CANONICAL_ISSUE_COLUMN].to_pylist(), ["Canonical issue B"])
        self.assertEqual(result[KNOWN_PRODUCT_COLUMN].to_pylist(), [True])
        self.assertEqual(result[KNOWN_ISSUE_COLUMN].to_pylist(), [True])
        self.assertEqual(result[KNOWN_PAIR_COLUMN].to_pylist(), [False])

    def test_rejects_duplicate_mapping_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.csv"
            path.write_text(
                "raw,canonical\nA,One\nA,Two\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate mapping"):
                load_mapping(path, "raw", "canonical")


if __name__ == "__main__":
    unittest.main()
