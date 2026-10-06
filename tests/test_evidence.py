from datetime import date
import unittest

import numpy as np
import pandas as pd

from src.evidence.deadline import deadlines, due_date
from src.evidence.facts import choose_complaints, top_issues

HOLIDAYS = [date(2026, 10, 8), date(2026, 12, 8), date(2026, 12, 25)]


class DeadlineTests(unittest.TestCase):
    def test_counts_business_days_after_registration(self):
        # Tuesday 2026-10-06; Thursday 2026-10-08 is a holiday.
        self.assertEqual(due_date(date(2026, 10, 6), 1, HOLIDAYS), date(2026, 10, 7))
        self.assertEqual(due_date(date(2026, 10, 6), 2, HOLIDAYS), date(2026, 10, 9))
        self.assertEqual(due_date(date(2026, 10, 6), 3, HOLIDAYS), date(2026, 10, 12))

    def test_weekend_registration_makes_monday_the_first_day(self):
        self.assertEqual(due_date(date(2026, 10, 10), 1, HOLIDAYS), date(2026, 10, 12))

    def test_refuses_a_year_without_holidays(self):
        with self.assertRaisesRegex(ValueError, "2027"):
            due_date(date(2026, 12, 28), 15, HOLIDAYS)

    def test_deadlines_give_both_due_dates(self):
        result = deadlines(date(2026, 10, 6), HOLIDAYS)

        self.assertEqual(result["standard_due"], "2026-10-28")
        self.assertEqual(result["extended_days"], 45)


class SelectionTests(unittest.TestCase):
    def test_rule_covers_alert_novelty_and_products(self):
        week = pd.Timestamp("2024-11-11")
        rows = pd.DataFrame(
            {
                "week": [week] * 8,
                "cluster_id": [14, 14, 3, 3, 5, 5, 6, 6],
                "is_novel": [False, False, True, False, False, False, False, False],
                "Product canonical": ["A", "A", "B", "B", "B", "C", "C", "D"],
            }
        )
        alerts = pd.DataFrame({"week": [week], "cluster_id": [14]})

        chosen = choose_complaints(rows, alerts, {"products": 2}, seed=0)

        self.assertEqual(len(chosen), 4)
        self.assertEqual(len(set(chosen.index)), 4)
        self.assertEqual(chosen.loc[0, "cluster_id"], 14)
        self.assertTrue(chosen.loc[1, "is_novel"])
        self.assertEqual(set(chosen.loc[2:, "Product canonical"]), {"A", "B"})

    def test_top_issues_orders_by_probability(self):
        classes = np.array(["a", "b", "c", "d"])
        probability = np.array([[0.1, 0.5, 0.3, 0.1]])

        issues = top_issues(probability, classes)

        self.assertEqual([item["issue"] for item in issues[0]], ["b", "c", "a"])


if __name__ == "__main__":
    unittest.main()
