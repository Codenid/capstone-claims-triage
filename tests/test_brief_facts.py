import unittest

import numpy as np
import pandas as pd

from src.models.persistent_change import cusum, excess_scores, recent_shares
from src.triage import echo
from src.triage.brief_facts import alarms_in, due_without_holidays, plain, repeated_share, verdict

WEEKS = pd.date_range("2025-01-06", periods=8, freq="7D")


def counts_table() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.integers(20, 200, size=(len(WEEKS), 4)), index=WEEKS, columns=[0, 1, 2, 3])


class EchoTests(unittest.TestCase):
    def test_nothing_dropped_is_m9_expected_count(self):
        counts = counts_table()
        expected = recent_shares(counts.to_numpy()) * counts.sum(axis=1).to_numpy()[:, None]

        result = echo.expected_without(counts, [])

        np.testing.assert_allclose(result.to_numpy(), expected)

    def test_dropping_a_pattern_shares_out_the_rest_of_the_total(self):
        counts = counts_table()
        counts.loc[WEEKS[3], 2] *= 20  # a burst

        result = echo.expected_without(counts, [2])

        self.assertEqual(list(result.columns), [0, 1, 3])
        valid = result.dropna()
        np.testing.assert_allclose(valid.sum(axis=1), counts.loc[valid.index, [0, 1, 3]].sum(axis=1))

    def test_frozen_expected_keeps_the_reference_shares(self):
        counts = counts_table()

        result = echo.frozen_expected(counts, [3], list(WEEKS[:4]))

        pooled = counts.loc[WEEKS[:4], [0, 1, 2]].sum()
        shares = (pooled + 1) / (pooled.sum() + 3)
        np.testing.assert_allclose(result.loc[WEEKS[6]], shares * counts.loc[WEEKS[6], [0, 1, 2]].sum())

    def test_scores_are_m11_scores_per_week(self):
        counts = counts_table()
        expected = echo.expected_without(counts, [])
        alpha = np.full((5, 4), 30.0)

        result = echo.scores(counts, expected, alpha)

        week = WEEKS[5]
        direct = excess_scores(counts.loc[week].to_numpy(), expected.loc[week].to_numpy(), alpha)
        np.testing.assert_allclose(result.loc[week], direct)
        self.assertEqual(len(result), len(WEEKS) - 4)

    def test_accumulate_from_zero_is_the_m11_cusum(self):
        z = pd.DataFrame([[2.0, 0.0], [2.5, 3.0], [0.2, 1.0]], index=WEEKS[:3], columns=[0, 1])

        statistic, alarms = echo.accumulate(z, pd.Series(0.0, index=[0, 1]), 0.5, 3.0)

        expected_statistic, expected_alarms = cusum(z.to_numpy(), 0.5, 3.0)
        np.testing.assert_allclose(statistic.to_numpy(), expected_statistic)
        np.testing.assert_array_equal(alarms.to_numpy(), expected_alarms)

    def test_accumulate_starts_from_the_given_state_and_restarts_after_an_alarm(self):
        z = pd.DataFrame([[1.0], [1.0], [1.0]], index=WEEKS[:3], columns=[0])

        statistic, alarms = echo.accumulate(z, pd.Series({0: 2.8}), 0.5, 3.0)

        self.assertEqual(statistic[0].tolist(), [3.3, 0.5, 1.0])
        self.assertEqual(alarms[0].tolist(), [True, False, False])

    def test_state_after_an_alarm_is_zero(self):
        cusum_table = pd.DataFrame({0: [3.5], 1: [1.2]}, index=WEEKS[:1])
        alarm = pd.DataFrame({0: [True], 1: [False]}, index=WEEKS[:1])

        state = echo.state_after(cusum_table, alarm, WEEKS[0])

        self.assertEqual(state.tolist(), [0.0, 1.2])

    def test_echo_factor(self):
        self.assertAlmostEqual(echo.echo_factor(0.125, 0.35), 0.875 / 0.65)


class BriefFactTests(unittest.TestCase):
    def test_verdict_rule(self):
        sensitivity = {"frozen_pre_burst": {"12": ["2025-01-27"]}}
        self.assertEqual(verdict(8, [], sensitivity), "incidente")
        self.assertEqual(verdict(3, ["2025-02-03"], sensitivity), "actuar")
        self.assertEqual(verdict(12, [], sensitivity), "vigilar")
        self.assertEqual(verdict(13, [], sensitivity), "eco")

    def test_alarms_in_lists_only_patterns_that_alarm(self):
        alarms = pd.DataFrame({3: [False, True], 4: [False, False]}, index=WEEKS[:2])

        self.assertEqual(alarms_in(alarms, list(WEEKS[:2])), {"3": ["2025-01-13"]})

    def test_due_dates_skip_weekends_and_not_the_registration_day(self):
        days = pd.Series(pd.to_datetime(["2025-01-13", "2025-01-19", "2025-02-10"]))

        due = due_without_holidays(days, 15)

        self.assertEqual([str(d) for d in due], ["2025-02-03", "2025-02-07", "2025-03-03"])

    def test_repeated_share(self):
        texts = pd.Series(["a"] * 10 + ["b"] * 2)

        self.assertAlmostEqual(repeated_share(texts), 10 / 12)

    def test_plain_values_are_json_ready(self):
        value = {1: [np.int64(2), np.float64(0.5), np.bool_(True), pd.Timestamp("2025-02-03")]}

        self.assertEqual(plain(value), {"1": [2, 0.5, True, "2025-02-03"]})


if __name__ == "__main__":
    unittest.main()
