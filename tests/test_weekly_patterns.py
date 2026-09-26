from datetime import date
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.normalize_text import HASH_COLUMN
from src.models.semantic_space import ID_COLUMN
from src.models.tfidf_models import DATE_COLUMN
from src.models.weekly_patterns import (
    apply_novelty,
    config_fingerprint,
    assign_clusters,
    class_tfidf_terms,
    complete_weekly_counts,
    fit_final_model,
    load_state,
    novelty_thresholds,
    split_week_index,
    transform_split,
)


class WeeklyPatternsTests(unittest.TestCase):
    def setUp(self):
        self.periods = {
            "fit": {"start": "2023-01-01", "end": "2023-01-17"},
            "calibration": {"start": "2023-02-01", "end": "2023-02-15"},
            "validation": {"start": "2023-03-01", "end": "2023-03-15"},
        }
        self.assignments = pd.DataFrame(
            {
                ID_COLUMN: ["f1", "f2", "f3", "c1", "v1"],
                DATE_COLUMN: [
                    "2023-01-02",
                    "2023-01-02",
                    "2023-01-16",
                    "2023-02-06",
                    "2023-03-06",
                ],
                HASH_COLUMN: ["same", "same", "other", "cal", "val"],
                "split": [
                    "fit",
                    "fit",
                    "fit",
                    "calibration",
                    "validation",
                ],
                "cluster_id": [0, 0, 1, 1, 0],
                "is_novel": [False, True, False, True, False],
            }
        )

    def test_fingerprint_accepts_dates_loaded_from_yaml(self):
        config = {
            "weekly_patterns": {"clusters": 40},
            "evaluation": {
                "splits": {
                    "fit": {"start": date(2023, 1, 1), "end": date(2023, 2, 1)}
                }
            },
        }

        fingerprint = config_fingerprint(config)

        self.assertEqual(len(fingerprint), 64)

    def test_builds_weeks_from_frozen_split_boundaries(self):
        weeks = split_week_index(self.periods["fit"])

        self.assertEqual(weeks[0], pd.Timestamp("2022-12-26"))
        self.assertEqual(weeks[-1], pd.Timestamp("2023-01-16"))

    def test_completes_missing_week_cluster_pairs_with_zero(self):
        weekly = complete_weekly_counts(self.assignments, self.periods, clusters=2)
        missing = weekly.loc[
            (weekly["split"] == "fit")
            & (weekly["week"] == pd.Timestamp("2023-01-09"))
            & (weekly["cluster_id"] == 0)
        ].iloc[0]

        self.assertEqual(len(weekly), 20)
        self.assertEqual(missing["complaint_count"], 0)
        self.assertEqual(missing["unique_text_count"], 0)
        self.assertEqual(missing["novel_count"], 0)
        self.assertEqual(missing["weekly_total"], 0)
        self.assertEqual(missing["proportion"], 0.0)

    def test_completes_an_entirely_empty_split_with_zero(self):
        without_validation = self.assignments.loc[
            self.assignments["split"] != "validation"
        ]

        weekly = complete_weekly_counts(without_validation, self.periods, clusters=2)
        validation = weekly.loc[weekly["split"] == "validation"]

        self.assertFalse(validation.empty)
        self.assertEqual(validation["complaint_count"].sum(), 0)
        self.assertEqual(validation["weekly_total"].sum(), 0)

    def test_counts_complaints_unique_texts_and_novel_cases(self):
        weekly = complete_weekly_counts(self.assignments, self.periods, clusters=2)
        row = weekly.loc[
            (weekly["split"] == "fit")
            & (weekly["week"] == pd.Timestamp("2023-01-02"))
            & (weekly["cluster_id"] == 0)
        ].iloc[0]

        self.assertEqual(row["complaint_count"], 2)
        self.assertEqual(row["unique_text_count"], 1)
        self.assertEqual(row["novel_count"], 1)
        self.assertEqual(row["weekly_total"], 2)
        self.assertEqual(row["proportion"], 1.0)

    def test_marks_boundary_weeks_as_partial(self):
        weekly = complete_weekly_counts(self.assignments, self.periods, clusters=2)
        flags = weekly.loc[
            (weekly["split"] == "fit") & (weekly["cluster_id"] == 0),
            ["week", "is_complete_week"],
        ].set_index("week")["is_complete_week"]

        self.assertFalse(bool(flags.loc[pd.Timestamp("2022-12-26")]))
        self.assertTrue(bool(flags.loc[pd.Timestamp("2023-01-02")]))
        self.assertTrue(bool(flags.loc[pd.Timestamp("2023-01-09")]))
        self.assertFalse(bool(flags.loc[pd.Timestamp("2023-01-16")]))

    def test_positive_week_proportions_sum_to_one(self):
        weekly = complete_weekly_counts(self.assignments, self.periods, clusters=2)
        positive = weekly.loc[weekly["weekly_total"] > 0]
        sums = positive.groupby(["split", "week"])["proportion"].sum()

        np.testing.assert_allclose(sums.to_numpy(), 1.0)

    def test_rejects_dates_outside_their_frozen_split(self):
        invalid = self.assignments.copy()
        invalid.loc[0, DATE_COLUMN] = "2023-01-17"

        with self.assertRaisesRegex(ValueError, "outside the fit period"):
            complete_weekly_counts(invalid, self.periods, clusters=2)

    def test_calculates_and_applies_cluster_novelty_thresholds(self):
        labels = np.repeat(np.arange(40, dtype=np.int32), 2)
        distances = np.tile(np.array([0.0, 2.0], dtype=np.float32), 40)

        thresholds = novelty_thresholds(labels, distances, 40, quantile=0.5)
        novel = apply_novelty(
            np.array([0, 1], dtype=np.int32),
            np.array([1.1, 1.0], dtype=np.float32),
            thresholds,
        )

        self.assertEqual(len(thresholds), 40)
        np.testing.assert_allclose(thresholds, 1.0)
        np.testing.assert_array_equal(novel, [True, False])

    def test_future_assignment_does_not_change_fit_only_model(self):
        fit = np.array(
            [[0.0, 0.0], [0.0, 0.1], [10.0, 10.0], [10.0, 10.1]],
            dtype=np.float32,
        )
        model = fit_final_model(
            fit,
            clusters=2,
            n_init=5,
            batch_size=4,
            max_iter=100,
            seed=42,
        )
        centers = model.cluster_centers_.copy()

        labels, distances = assign_clusters(
            model,
            np.array([[100.0, 100.0]], dtype=np.float32),
            batch_rows=1,
        )

        self.assertEqual(len(np.unique(model.labels_)), 2)
        self.assertEqual(labels.shape, (1,))
        self.assertEqual(distances.shape, (1,))
        np.testing.assert_array_equal(model.cluster_centers_, centers)

    def test_extracts_cluster_specific_class_tfidf_terms(self):
        terms = class_tfidf_terms(
            [
                "mortgage escrow mortgage",
                "escrow payment mortgage",
                "credit card fee",
                "card fee credit",
            ],
            np.array([0, 0, 1, 1], dtype=np.int32),
            top_terms=5,
            max_features=100,
        )

        self.assertIn("mortgage", terms[0])
        self.assertIn("credit", terms[1])

    def test_checkpoint_tracks_all_transforms(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            state = load_state(path, "source-hash", "config-hash")

        self.assertEqual(
            state["transform_progress"],
            {"fit": 0, "calibration": 0, "validation": 0},
        )
        self.assertEqual(state["resume_count"], 0)

    def test_rejects_checkpoint_when_transform_array_is_missing(self):
        state = {
            "transform_progress": {"fit": 1, "calibration": 0, "validation": 0}
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "missing fit array"):
                transform_split(
                    "fit",
                    np.zeros((2, 2), dtype=np.float32),
                    object(),
                    object(),
                    root / "fit.npy",
                    state,
                    root / "checkpoint.json",
                    batch_rows=1,
                    output_dimensions=2,
                )


if __name__ == "__main__":
    unittest.main()
