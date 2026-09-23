import unittest

import numpy as np
import pandas as pd

from src.models.bge_sample import add_comparison, sample_frames


class BgeSampleTests(unittest.TestCase):
    def test_samples_each_period_deterministically(self):
        frame = pd.DataFrame(
            {
                "Complaint ID": [str(value) for value in range(30)],
                "evaluation_split": ["fit"] * 10 + ["calibration"] * 10 + ["validation"] * 10,
                "T1": ["issue a", "issue b"] * 15,
                "eligible_T1_complete": True,
            }
        )
        sizes = {"fit": 4, "calibration": 3, "validation": 2}

        first = sample_frames(frame, sizes, seed=42)
        second = sample_frames(frame, sizes, seed=42)

        for split, expected in sizes.items():
            self.assertEqual(len(first[split]), expected)
            self.assertEqual(
                first[split]["Complaint ID"].tolist(),
                second[split]["Complaint ID"].tolist(),
            )

    def test_compares_primary_metric_on_same_sample(self):
        def result(value: float, metric: str) -> dict:
            return {
                "metrics": {
                    "validation": {
                        "no_shared_text": {metric: value},
                    }
                }
            }

        tfidf = {
            "T1": result(0.2, "macro_f1"),
            "T2": result(0.3, "average_precision"),
            "T3": result(0.1, "average_precision"),
            "T4": result(0.05, "average_precision"),
        }
        bge = {
            "T1": result(0.25, "macro_f1"),
            "T2": result(0.32, "average_precision"),
            "T3": result(0.12, "average_precision"),
            "T4": result(0.04, "average_precision"),
        }

        comparison = add_comparison(tfidf, bge)

        self.assertTrue(np.isclose(comparison["T1"]["bge_improvement"], 0.05))
        self.assertTrue(np.isclose(comparison["T4"]["bge_improvement"], -0.01))


if __name__ == "__main__":
    unittest.main()
