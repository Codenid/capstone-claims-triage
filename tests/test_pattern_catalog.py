from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.models.pattern_catalog import (
    alert_summary,
    choose_examples,
    plot_series,
    snippet,
    weekly_summary,
)


def counts_frame() -> pd.DataFrame:
    rows = []
    for index, week in enumerate(pd.date_range("2024-10-07", periods=4, freq="7D")):
        split = "calibration" if index < 3 else "validation"
        for cluster in (0, 1):
            rows.append(
                {
                    "split": split,
                    "week": str(week.date()),
                    "cluster_id": cluster,
                    "complaint_count": 10 * (cluster + 1) + index,
                    "novel_count": index % 2,
                    "is_complete_week": index != 3,
                    "weekly_total": 100,
                    "proportion": (10 * (cluster + 1) + index) / 100,
                }
            )
    return pd.DataFrame(rows)


class PatternCatalogTests(unittest.TestCase):
    def test_weekly_summary_uses_complete_weeks_only(self):
        summary = weekly_summary(counts_frame())

        self.assertEqual(summary[0]["calibration"]["weeks"], 3)
        self.assertNotIn("validation", summary[0])
        self.assertAlmostEqual(summary[1]["calibration"]["mean_count"], 21.0)
        self.assertAlmostEqual(summary[0]["calibration"]["novel_fraction"], 1 / 33)

    def test_alert_summary_keeps_cusum_weeks_per_split(self):
        alerts = pd.DataFrame(
            {
                "split": ["calibration", "calibration", "validation"],
                "week": ["2024-10-14", "2024-10-21", "2025-01-13"],
                "cluster_id": [1, 1, 1],
                "cusum_alarm": [True, False, True],
            }
        )

        summary = alert_summary(alerts)

        self.assertEqual(summary[1]["calibration"], ["2024-10-14"])
        self.assertEqual(summary[1]["validation"], ["2025-01-13"])

    def test_examples_are_closest_and_textually_distinct(self):
        assignments = pd.DataFrame(
            {
                "Complaint ID": ["a", "b", "c", "d"],
                "cluster_id": [3, 3, 3, 4],
                "distance": [0.5, 0.2, 0.3, 0.1],
                "Consumer complaint narrative SHA-256": ["h1", "h1", "h2", "h3"],
            }
        )

        self.assertEqual(choose_examples(assignments, 3, 3), ["b", "c"])
        self.assertEqual(snippet("  a  b   c ", 20), "a b c")
        self.assertEqual(snippet("abcdefghij", 5), "abcd…")

    def test_series_plot_is_written(self):
        counts = counts_frame()
        predictions = pd.DataFrame(
            {
                "week": counts.loc[counts["cluster_id"] == 0, "week"],
                "cluster_id": 0,
                "expected_p025": 5.0,
                "expected_p10": 7.0,
                "expected_p25": 9.0,
                "expected_p50": 10.0,
                "expected_p75": 11.0,
                "expected_p90": 13.0,
                "expected_p975": 15.0,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "series" / "cluster_00.png"
            plot_series(0, counts, predictions, {"calibration": ["2024-10-14"]}, "t", path)

            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
