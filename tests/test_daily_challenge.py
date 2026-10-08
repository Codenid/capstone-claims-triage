import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.models.daily_counts.challenge import (
    attach_candidate,
    candidate_row,
    choose,
    pairwise,
    validation_report,
)

QUANTILES = ("p025", "p10", "p25", "p50", "p75", "p90", "p975")
CLUSTERS = 3


def predictions(seed: int, noise: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for split, start, days in (
        ("fit", "2024-01-01", 10),
        ("calibration", "2024-01-11", 6),
        ("validation", "2024-01-17", 4),
    ):
        for day in pd.date_range(start, periods=days, freq="D"):
            for cluster in range(CLUSTERS):
                observed = int(rng.poisson(20))
                row = {
                    "split": split,
                    "day": day.strftime("%Y-%m-%d"),
                    "cluster_id": cluster,
                    "complaint_count": observed,
                    "daily_total": 60,
                }
                for prefix, spread in (
                    ("model", noise),
                    ("baseline", 4.0),
                    ("b1_rolling_7", 3.0),
                ):
                    center = observed + rng.normal(0, spread)
                    for quantile, offset in zip(QUANTILES, (-3, -2, -1, 0, 1, 2, 3)):
                        row[f"{prefix}_{quantile}"] = center + offset * spread
                    row[f"{prefix}_impossible_probability"] = 0.0
                rows.append(row)
    return pd.DataFrame(rows)


def results(model_id: str, wis: float, status: str = "accepted") -> dict:
    return {
        "run_mode": "full",
        "model_id": model_id,
        "run_key": f"{model_id}-key",
        "candidate_status": status,
        "rejection_reason": "" if status == "accepted" else "coverage_95",
        "backtest": {
            "calibration": {
                "model": {
                    "wis": wis,
                    "wape": 0.3,
                    "mae": 4.0,
                    "coverage_80": 0.8,
                    "coverage_95": 0.95,
                }
            }
        },
        "comparison": {
            "best_baseline": "b1_rolling_7",
            "metrics": {"b1_rolling_7": {"wis": 9.0}, "model": {"wis": wis}},
            "wis_gain": 1 - wis / 9.0,
            "difference": {"wis": {"estimate": wis - 9.0, "p025": -3.0, "p975": -1.0}},
        },
        "diagnostics": {"rhat_max": 1.003, "divergences": 0},
    }


class DailyChallengeTests(unittest.TestCase):
    def test_lowest_wis_among_accepted_wins(self):
        rows = [
            candidate_row(results("a", 7.0), Path("runs/a")),
            candidate_row(results("b", 6.5, "rejected"), Path("runs/b")),
            candidate_row(results("c", 6.9), Path("runs/c")),
        ]

        winner = choose(rows)

        self.assertEqual(winner["model_id"], "c")
        self.assertEqual(rows[0]["best_baseline_wis"], 9.0)
        self.assertAlmostEqual(rows[0]["wis_gain"], 1 - 7.0 / 9.0)
        with self.assertRaises(ValueError):
            choose([rows[1]])

    def test_pairwise_bootstrap_uses_the_same_days(self):
        winner = predictions(1, 1.0)
        other = predictions(1, 3.0)

        merged = attach_candidate(winner, other, "other")
        comparison = pairwise(winner, {"other": other}, 7, "calibration")

        self.assertIn("other_p50", merged.columns)
        self.assertEqual(len(merged), len(winner))
        difference = comparison["other"]["difference"]["wis"]
        self.assertLess(difference["estimate"], 0)
        self.assertLessEqual(difference["p025"], difference["p975"])
        shifted = other.copy()
        shifted["complaint_count"] += 1
        with self.assertRaises(ValueError):
            attach_candidate(winner, shifted, "other")

    def test_validation_report_scores_2025_h1_only(self):
        winner = predictions(2, 1.0)

        report = validation_report(winner, {"other": predictions(2, 2.0)}, 7)

        self.assertTrue(report["already_consulted"])
        self.assertEqual(report["comparison"]["split"], "validation")
        self.assertEqual(report["comparison"]["weeks"], 4)
        self.assertIn("b1_rolling_7", report["metrics"])
        self.assertIn("other", report["pairwise"])

    def test_candidate_row_reads_a_results_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "results.json"
            path.write_text(json.dumps(results("a", 7.0)), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))
        row = candidate_row(loaded, Path(folder))
        self.assertEqual(row["divergences"], 0)
        self.assertEqual(row["status"], "accepted")


if __name__ == "__main__":
    unittest.main()
