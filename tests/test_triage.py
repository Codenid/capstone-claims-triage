from datetime import date
import json
import unittest

import numpy as np
import pandas as pd

from src.data.normalize_text import NORMALIZED_COLUMN
from src.models.semantic_space import ID_COLUMN
from src.triage import render
from src.triage.actions import ACTIONS, situation
from src.triage.complaints import (
    complaint_card,
    similar_complaints,
    top_issues,
    with_dates,
)
from src.triage.deadline import business_days_left, due_date, load_holidays
from src.triage.patterns import (
    active_alerts,
    last_closed_week,
    level_changes,
    level_percentiles,
    records,
    weekly_panel,
)

WEEKS = pd.date_range("2024-01-01", periods=6, freq="7D")


class DeadlineTests(unittest.TestCase):
    holidays = load_holidays()

    def test_registration_day_does_not_count_and_holidays_are_skipped(self):
        # Thursday 2026-10-08 is a national holiday.
        registered = date(2026, 10, 1)
        self.assertEqual(due_date(registered, 15, self.holidays), date(2026, 10, 23))
        self.assertEqual(due_date(registered, 45, self.holidays), date(2026, 12, 4))

    def test_weekend_or_holiday_registration_counts_from_next_business_day(self):
        saturday, holiday = date(2026, 10, 3), date(2026, 10, 8)
        self.assertEqual(due_date(saturday, 15, self.holidays), date(2026, 10, 26))
        self.assertEqual(due_date(holiday, 15, self.holidays), date(2026, 10, 29))

    def test_year_without_holiday_list_is_not_computed(self):
        with self.assertRaises(ValueError):
            due_date(date(2025, 2, 10), 15, self.holidays)
        with self.assertRaises(ValueError):
            due_date(date(2026, 12, 1), 45, self.holidays)

    def test_business_days_left(self):
        due = date(2026, 10, 23)
        self.assertEqual(business_days_left(date(2026, 10, 1), due, self.holidays), 15)
        self.assertEqual(business_days_left(date(2026, 10, 26), due, self.holidays), -1)


class ActionTests(unittest.TestCase):
    def test_rules_apply_in_the_preregistered_order(self):
        self.assertEqual(situation(True, True, True), "novel")
        self.assertEqual(situation(False, True, True), "template_alert")
        self.assertEqual(situation(False, True, False), "alert")
        self.assertEqual(situation(False, False, True), "normal")
        self.assertEqual(set(ACTIONS), {"novel", "template_alert", "alert", "normal"})


class PatternTests(unittest.TestCase):
    def test_level_change_compares_with_13_weeks_earlier(self):
        counts = pd.DataFrame({0: [100] * 20 + [200] * 10, 1: [900] * 30})
        changes = level_changes(counts)
        self.assertTrue(changes.iloc[:16].isna().all().all())
        self.assertAlmostEqual(changes.loc[16, 0], 0.0)
        self.assertAlmostEqual(changes.loc[23, 0], (800 / 4400) / (400 / 4000) - 1)

    def test_level_percentile_counts_reference_changes_at_or_below(self):
        reference = pd.DataFrame({0: [-0.1, 0.0, 0.1, 0.2]})
        current = pd.DataFrame({0: [0.1, np.nan, 0.5]})
        percentiles = level_percentiles(current, reference)
        np.testing.assert_allclose(percentiles[0], [0.75, np.nan, 1.0])

    def test_alert_stays_active_for_four_closed_weeks(self):
        rows = pd.DataFrame(
            {
                "week": WEEKS,
                "cluster_id": 0,
                "cusum_alarm": [False, True, False, False, False, False],
            }
        )
        active = active_alerts(rows, weeks=4)
        np.testing.assert_array_equal(
            active["active_alert"], [False, True, True, True, True, False]
        )

    def test_panel_puts_active_alerts_first_as_plain_json(self):
        history = pd.DataFrame(
            {
                "week": [WEEKS[2]] * 2,
                "cluster_id": [0, 1],
                "active_alert": [False, True],
                "cusum": [1.0, 0.5],
                "level_change": [np.nan, 0.2],
            }
        )
        panel = weekly_panel(history, WEEKS[2])
        self.assertEqual([item["cluster_id"] for item in panel], [1, 0])
        self.assertEqual(panel[0]["week"], "2024-01-15")
        self.assertIsNone(panel[1]["level_change"])
        with self.assertRaises(ValueError):
            weekly_panel(history, WEEKS[0])

    def test_complaints_see_the_last_closed_week(self):
        history = pd.DataFrame({"week": [WEEKS[0], WEEKS[2]]})
        self.assertEqual(last_closed_week(history, WEEKS[3]), WEEKS[2])
        with self.assertRaises(ValueError):
            last_closed_week(history, WEEKS[0])

    def test_records_turn_dates_into_iso_text(self):
        frame = pd.DataFrame({"week": [WEEKS[0]], "value": [np.int64(3)]})
        self.assertEqual(records(frame), [{"week": "2024-01-01", "value": 3}])


class FakeIndex:
    def search(self, values, neighbors):
        return np.array([[0.99, 0.80]]), np.array([[1, 0]])


class ComplaintTests(unittest.TestCase):
    def test_top_issues_are_ranked_by_score(self):
        scores = np.array([[0.1, 0.9, 0.5, 0.2]])
        issues = top_issues(scores, np.array(["a", "b", "c", "d"]))
        self.assertEqual([item["issue"] for item in issues[0]], ["b", "c", "d"])
        self.assertAlmostEqual(issues[0][0]["score"], 0.9)

    def test_similar_complaints_come_with_their_outcomes(self):
        fit = pd.DataFrame(
            {
                ID_COLUMN: ["10", "11"],
                "Date received": pd.to_datetime(["2024-01-02", "2024-03-04"]),
                "Product canonical": ["Card", "Loan"],
                "T1": ["Fraud", "Payment"],
                "T2": [True, np.nan],
                "T3": [False, np.nan],
            }
        )
        similar = similar_complaints(
            FakeIndex(), np.array(["10", "11"]), np.zeros((1, 3)), fit, 2
        )
        first, second = similar[0]
        self.assertEqual(first["complaint_id"], "11")
        self.assertEqual(first["received"], "2024-03-04")
        self.assertIsNone(first["relief"])
        self.assertEqual((second["issue"], second["relief"]), ("Fraud", True))


class CardDataTests(unittest.TestCase):
    def test_dates_follow_each_row_and_missing_ones_fail(self):
        rows = pd.DataFrame({ID_COLUMN: ["2", "1"]})
        received = pd.to_datetime(["2025-02-10", "2025-02-11"])
        dates = pd.Series(received, index=["1", "2"])
        dated = with_dates(rows, dates)
        self.assertEqual(list(dated["Date received"].dt.day), [11, 10])
        with self.assertRaises(ValueError):
            with_dates(pd.DataFrame({ID_COLUMN: ["3"]}), dates)

    def test_card_is_plain_json(self):
        row = pd.Series(
            {
                ID_COLUMN: "123",
                "Date received": pd.Timestamp("2025-02-10"),
                "Product canonical": "Credit reporting",
                NORMALIZED_COLUMN: "my credit report is wrong",
                "cluster_id": np.int64(7),
                "distance": np.float32(1.5),
                "novelty_threshold": np.float32(3.0),
                "is_novel": np.bool_(False),
            }
        )
        card = complaint_card(
            row, {"t1": []}, [], {"week": "2025-02-03"}, {}, {"situation": "normal"}, 10
        )
        self.assertEqual(card["text"], "my credit ")
        self.assertEqual(card["pattern"]["cluster_id"], 7)
        json.dumps(card)


class RenderTests(unittest.TestCase):
    def card(self):
        status = {
            "week": "2025-02-03",
            "representative_terms": "credit, report",
            "main_product": "Credit reporting",
            "template_dominated": False,
            "observed": 120,
            "expected": 100.0,
            "cusum": 4.2,
            "active_alert": True,
            "level_change": 0.2,
            "level_percentile": 0.9,
            "share": 0.012,
            "expected_share": 0.01,
            "share_low": 0.005,
            "share_high": 0.02,
        }
        outcome = {"probability": 0.6, "threshold": 0.4, "likely": True}
        return {
            "complaint_id": "123",
            "received": "2025-02-10",
            "product": "Credit reporting",
            "text": "<script>alert(1)</script> my report is wrong",
            "t1": [{"issue": "Incorrect information", "score": 2.5}],
            "t2": outcome,
            "t3": {**outcome, "likely": False},
            "pattern": {
                "cluster_id": 7,
                "distance": 1.5,
                "novelty_threshold": 3.0,
                "is_novel": False,
            },
            "similar": [],
            "pattern_status": status,
            "deadline": {
                "registered": "2026-10-01",
                "due": "2026-10-23",
                "business_days": 15,
                "due_extended": "2026-12-04",
                "extended_business_days": 45,
            },
            "action": {"situation": "alert", "text": ACTIONS["alert"]},
        }

    def test_card_escapes_the_complaint_text(self):
        page = render.card_page(self.card(), chart="")
        self.assertIn("&lt;script&gt;", page)
        self.assertNotIn("<script>", page)
        self.assertIn(ACTIONS["alert"], page)
        self.assertIn("2026-10-23", page)

    def test_panel_counts_active_alerts(self):
        item = {**self.card()["pattern_status"], "cluster_id": 7}
        page = render.panel_page("2025-02-03", [item], charts={}, threshold=3.37)
        self.assertIn("<b>1</b> de 1 patrones", page)

    def test_missing_values_show_a_dash(self):
        self.assertEqual(render.percent(None), "—")
        self.assertEqual(render.percent(float("nan"), signed=True), "—")
        self.assertEqual(render.percent(0.2, signed=True), "+20%")


if __name__ == "__main__":
    unittest.main()
