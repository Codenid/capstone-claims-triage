import unittest
from datetime import date

import pyarrow as pa

from src.data.apply_taxonomy import (
    CANONICAL_ISSUE_COLUMN,
    KNOWN_ISSUE_COLUMN,
    KNOWN_PAIR_COLUMN,
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
    TRAIN,
    VALIDATION,
    build_targets,
    period_for_date,
)
from src.data.normalize_text import HASH_COLUMN


def sample_table(received, response, timely="Yes", known_issue=True, known_pair=True, text_hash="new"):
    return pa.table(
        {
            "Date received": pa.array([received], type=pa.date32()),
            CANONICAL_ISSUE_COLUMN: ["Canonical issue" if known_issue else None],
            KNOWN_ISSUE_COLUMN: [known_issue],
            KNOWN_PAIR_COLUMN: [known_pair],
            "Company response to consumer": [response],
            "Timely response?": [timely],
            HASH_COLUMN: [text_hash],
        }
    )


class TargetTests(unittest.TestCase):
    def test_period_boundaries(self):
        self.assertEqual(period_for_date(date(2022, 12, 31)), CONTEXT)
        self.assertEqual(period_for_date(date(2023, 1, 1)), TRAIN)
        self.assertEqual(period_for_date(date(2025, 1, 1)), VALIDATION)
        self.assertEqual(period_for_date(date(2025, 7, 1)), HOLDOUT)
        self.assertEqual(period_for_date(date(2026, 1, 1)), OOD)

    def test_builds_all_known_targets(self):
        raw = sample_table(date(2025, 1, 1), "Closed with monetary relief", timely="No")

        result = build_targets(raw, set(), set(), final_holdout_allowed=False)

        self.assertEqual(result[T1_COLUMN].to_pylist(), ["Canonical issue"])
        self.assertEqual(result[T2_COLUMN].to_pylist(), [True])
        self.assertEqual(result[T3_COLUMN].to_pylist(), [True])
        self.assertEqual(result[T4_COLUMN].to_pylist(), [True])
        for column in [KNOWN_T1_COLUMN, KNOWN_T2_COLUMN, KNOWN_T3_COLUMN, KNOWN_T4_COLUMN]:
            self.assertEqual(result[column].to_pylist(), [True])

    def test_non_monetary_relief_is_positive_only_for_t2(self):
        raw = sample_table(date(2024, 1, 1), "Closed with non-monetary relief")

        result = build_targets(raw, set(), set(), final_holdout_allowed=False)

        self.assertEqual(result[T2_COLUMN].to_pylist(), [True])
        self.assertEqual(result[T3_COLUMN].to_pylist(), [False])

    def test_marks_unknown_response_targets(self):
        raw = sample_table(date(2024, 1, 1), "Closed")

        result = build_targets(raw, set(), set(), final_holdout_allowed=False)

        self.assertEqual(result[T2_COLUMN].to_pylist(), [None])
        self.assertEqual(result[T3_COLUMN].to_pylist(), [None])
        self.assertEqual(result[KNOWN_T2_COLUMN].to_pylist(), [False])
        self.assertEqual(result[KNOWN_T3_COLUMN].to_pylist(), [False])

    def test_unknown_pair_makes_t1_unknown(self):
        raw = sample_table(date(2025, 7, 1), "Closed with explanation", known_pair=False)

        result = build_targets(raw, set(), set(), final_holdout_allowed=False)

        self.assertEqual(result[T1_COLUMN].to_pylist(), [None])
        self.assertEqual(result[KNOWN_T1_COLUMN].to_pylist(), [False])

    def test_blocks_contaminated_holdout(self):
        raw = sample_table(date(2025, 7, 1), "Closed with explanation")

        result = build_targets(raw, set(), set(), final_holdout_allowed=False)

        self.assertEqual(result[PERIOD_COLUMN].to_pylist(), [HOLDOUT])
        self.assertEqual(result[REVIEWED_ID_COLUMN].to_pylist(), [None])
        self.assertEqual(result[REVIEWED_GROUP_COLUMN].to_pylist(), [None])
        self.assertEqual(result[HOLDOUT_STATUS_COLUMN].to_pylist(), ["contaminated_ids_unavailable"])
        for column in COMPLETE_COLUMNS.values():
            self.assertEqual(result[column].to_pylist(), [False])
        for column in NO_SHARED_COLUMNS.values():
            self.assertEqual(result[column].to_pylist(), [False])

    def test_blocks_2026_response_targets_but_keeps_t1(self):
        raw = sample_table(date(2026, 1, 1), "Closed with monetary relief", timely="No")

        result = build_targets(raw, set(), set(), final_holdout_allowed=False)

        self.assertEqual(result[COMPLETE_COLUMNS[T1_COLUMN]].to_pylist(), [True])
        for target in [T2_COLUMN, T3_COLUMN, T4_COLUMN]:
            self.assertEqual(result[COMPLETE_COLUMNS[target]].to_pylist(), [False])

    def test_excludes_shared_validation_text(self):
        raw = sample_table(date(2025, 1, 1), "Closed with explanation", text_hash="seen")

        result = build_targets(raw, {"seen"}, set(), final_holdout_allowed=False)

        self.assertEqual(result[NO_SHARED_TEXT_COLUMN].to_pylist(), [False])
        for column in NO_SHARED_COLUMNS.values():
            self.assertEqual(result[column].to_pylist(), [False])


if __name__ == "__main__":
    unittest.main()
