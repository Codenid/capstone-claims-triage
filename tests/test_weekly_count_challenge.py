import unittest

import pandas as pd

from src.models.weekly_counts.challenge import attach_reference, decide

ACCEPTANCE = {
    "coverage_80_minimum": 0.70,
    "coverage_80_maximum": 0.90,
    "coverage_95_minimum": 0.88,
    "coverage_95_maximum": 0.99,
}


def predictions(observed: list[int], median: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": "calibration",
            "week": ["2024-10-07"] * len(observed),
            "cluster_id": range(len(observed)),
            "complaint_count": observed,
            "model_p50": median,
        }
    )


def comparison(gain: float, wis: tuple[float, float], other: float = -1.0) -> dict:
    return {
        "wis_gain": gain,
        "difference": {
            "wis": {"p025": wis[0], "p975": wis[1]},
            "wape": {"p025": other, "p975": other},
            "mae": {"p025": other, "p975": other},
        },
    }


class ChallengeTests(unittest.TestCase):
    def test_attaches_m9_quantiles_by_week_and_cluster(self):
        candidate = predictions([5, 7], [5.0, 6.0])
        reference = predictions([5, 7], [4.0, 9.0]).iloc[::-1]

        merged = attach_reference(candidate, reference)

        self.assertEqual(merged["m9_p50"].tolist(), [4.0, 9.0])
        self.assertEqual(merged["model_p50"].tolist(), [5.0, 6.0])

    def test_rejects_a_reference_with_other_counts(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            attach_reference(predictions([5, 7], [5, 6]), predictions([5, 8], [5, 6]))

    def test_needs_gain_interval_and_coverage(self):
        coverage = {"coverage_80": 0.8, "coverage_95": 0.95}

        wins = decide(comparison(0.08, (-3.0, -1.0)), coverage, ACCEPTANCE, 0.05)
        small = decide(comparison(0.03, (-3.0, -1.0)), coverage, ACCEPTANCE, 0.05)
        noisy = decide(comparison(0.08, (-3.0, 0.5)), coverage, ACCEPTANCE, 0.05)
        narrow = decide(
            comparison(0.08, (-3.0, -1.0)),
            {"coverage_80": 0.6, "coverage_95": 0.95},
            ACCEPTANCE,
            0.05,
        )

        self.assertTrue(wins["wins"])
        self.assertFalse(small["checks"]["wis_gain"])
        self.assertFalse(noisy["checks"]["wis_interval_below_zero"])
        self.assertFalse(narrow["checks"]["coverage_80"])
        self.assertFalse(any(result["wins"] for result in (small, noisy, narrow)))

    def test_a_clearly_worse_wape_rejects(self):
        coverage = {"coverage_80": 0.8, "coverage_95": 0.95}

        result = decide(
            comparison(0.08, (-3.0, -1.0), other=0.2), coverage, ACCEPTANCE, 0.05
        )

        self.assertFalse(result["checks"]["wape_not_worse"])
        self.assertFalse(result["wins"])


if __name__ == "__main__":
    unittest.main()
