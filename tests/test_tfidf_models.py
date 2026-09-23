import unittest

import numpy as np
import pandas as pd

from src.models.tfidf_models import (
    binary_metrics_for_view,
    build_features,
    eligible_mask,
    top_three_correct,
    train_models,
)


def sample_frame(rows: int, split: str) -> pd.DataFrame:
    labels = np.array(["alpha issue", "bravo issue", "charlie issue"])
    positions = np.arange(rows)
    return pd.DataFrame(
        {
            "Consumer complaint narrative normalized": [
                f"{labels[index % 3]} complaint example {index % 7}"
                for index in positions
            ],
            "Product canonical": [f"product {index % 2}" for index in positions],
            "T1": labels[positions % 3],
            "T2": positions % 2 == 0,
            "T3": positions % 3 == 0,
            "T4": positions % 5 == 0,
            "eligible_T1_complete": True,
            "eligible_T2_complete": True,
            "eligible_T3_complete": True,
            "eligible_T4_complete": True,
            "no_shared_text": positions % 4 != 0,
            "evaluation_split": split,
        }
    )


class TfidfModelTests(unittest.TestCase):
    def test_excludes_shared_text_only_from_second_view(self):
        frame = pd.DataFrame(
            {
                "eligible_T1_complete": [True, True, False],
                "no_shared_text": [True, False, True],
            }
        )

        self.assertEqual(
            eligible_mask(frame, "T1", "complete").tolist(),
            [True, True, False],
        )
        self.assertEqual(
            eligible_mask(frame, "T1", "no_shared_text").tolist(),
            [True, False, False],
        )

    def test_checks_whether_truth_is_in_top_three_scores(self):
        decisions = np.array(
            [
                [0.9, 0.1, 0.8, 0.7],
                [0.9, 0.8, 0.7, 0.1],
            ]
        )
        truth_codes = np.array([3, 3])

        self.assertEqual(
            top_three_correct(decisions, truth_codes).tolist(),
            [True, False],
        )

    def test_trains_small_end_to_end_models(self):
        config = {
            "experiment": {"seed": 42},
            "tfidf": {
                "ngram_range": [1, 2],
                "min_df": 1,
                "max_features": 100,
                "sublinear_tf": True,
                "alpha": 0.001,
                "max_iter": 200,
                "class_weight": {
                    "T1": None,
                    "T2": None,
                    "T3": "balanced",
                    "T4": "balanced",
                },
                "calibration_method": "sigmoid",
            },
        }
        frames = {
            "fit": sample_frame(90, "fit"),
            "calibration": sample_frame(45, "calibration"),
            "validation": sample_frame(45, "validation"),
        }

        _, _, matrices = build_features(frames, config)
        results, models, class_rows = train_models(config, frames, matrices)

        self.assertEqual(set(results), {"T1", "T2", "T3", "T4"})
        self.assertEqual(set(models), {"T1", "T2", "T3", "T4"})
        self.assertTrue(class_rows)
        self.assertEqual(matrices["fit"].shape[0], 90)

    def test_binary_metrics_use_only_selected_rows(self):
        truth = np.array([False, True, True])
        probabilities = np.array([0.1, 0.8, 0.4])
        selected = np.array([True, True, False])

        metrics = binary_metrics_for_view(
            truth,
            probabilities,
            threshold=0.5,
            selected=selected,
        )

        self.assertEqual(metrics["eligible"], 2)
        self.assertEqual(metrics["positive"], 1)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
