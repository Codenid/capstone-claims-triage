from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import average_precision_score, f1_score

from src.models.representation_comparison import (
    M5_SAMPLE,
    REPRESENTATIONS,
    apply_m5_check,
    attach_embedding_rows,
    check_new_outputs,
    classifier_config,
    primary_metric,
    product_rule_probabilities,
    product_rule_t1,
    representation_metrics,
    select_representation,
    trained_outputs,
    week_resamples,
    weighted_macro_f1,
)

LABELS = np.array(["alpha", "bravo", "charlie"])


def target_frame(rows: int, split: str) -> pd.DataFrame:
    positions = np.arange(rows)
    return pd.DataFrame(
        {
            "Complaint ID": [f"{split}-{index}" for index in positions],
            "Product canonical": [f"product {index % 2}" for index in positions],
            "T1": LABELS[positions % 3],
            "T2": positions % 2 == 0,
            "T3": positions % 3 == 0,
            "T4": positions % 4 == 0,
            "eligible_T1_complete": True,
            "eligible_T2_complete": True,
            "eligible_T3_complete": True,
            "eligible_T4_complete": True,
            "no_shared_text": True,
            "evaluation_split": split,
        }
    )


def small_config() -> dict:
    return {
        "experiment": {"seed": 42},
        "tfidf": {
            "alpha": 0.001,
            "max_iter": 50,
            "class_weight": {"T1": None, "T2": None, "T3": "balanced", "T4": "balanced"},
            "calibration_method": "sigmoid",
        },
        "bge": {"linear_max_iter": 80},
    }


class WeightedMetricTests(unittest.TestCase):
    def test_weighted_macro_f1_matches_sklearn(self):
        rng = np.random.default_rng(3)
        truth = rng.integers(0, 4, size=200)
        # Class 4 is predicted but never true, so it must not enter the mean.
        prediction = np.where(rng.random(200) < 0.7, truth, 4)
        labels = sorted(np.unique(truth))
        expected = f1_score(truth, prediction, labels=labels, average="macro")

        ones = np.ones(len(truth))
        self.assertAlmostEqual(weighted_macro_f1(truth, prediction, ones, 5), expected)
        self.assertAlmostEqual(
            weighted_macro_f1(truth, prediction, 2 * ones, 5), expected
        )

        kept = np.arange(len(truth)) < 120
        subset = f1_score(
            truth[kept],
            prediction[kept],
            labels=sorted(np.unique(truth[kept])),
            average="macro",
        )
        self.assertAlmostEqual(
            weighted_macro_f1(truth, prediction, kept.astype(float), 5), subset
        )

    def test_primary_metric_with_unit_weights_matches_reported_metric(self):
        truth = np.array([True, False, True, False, True, False])
        probability = np.array([0.9, 0.2, 0.4, 0.6, 0.8, 0.1])
        metric = primary_metric("T2", {"probability": probability}, truth, LABELS)
        expected = float(average_precision_score(truth, probability))
        self.assertAlmostEqual(metric(np.ones(len(truth))), expected)

        labels = LABELS[[0, 1, 2, 0, 1, 2]]
        outputs = {
            "prediction": LABELS[[0, 1, 1, 0, 2, 2]],
            "top_three": np.ones(6, dtype=bool),
        }
        metric = primary_metric("T1", outputs, labels, LABELS)
        reported = representation_metrics("T1", outputs, labels)["macro_f1"]
        self.assertAlmostEqual(metric(np.ones(len(labels))), reported)

    def test_week_resamples_are_reproducible(self):
        weeks = np.array(["2024-10-07", "2024-09-30", "2024-10-07", "2024-10-14"])

        counts, positions = week_resamples(weeks, draws=50, seed=42)
        again, _ = week_resamples(weeks, draws=50, seed=42)

        self.assertEqual(counts.shape, (50, 3))
        self.assertTrue((counts.sum(axis=1) == 3).all())
        np.testing.assert_array_equal(counts, again)
        self.assertEqual(positions.tolist(), [1, 0, 1, 2])


class SelectionTests(unittest.TestCase):
    def test_keeps_the_simpler_representation_without_a_clear_gain(self):
        noise = np.linspace(-0.01, 0.01, 200)
        estimates = {
            "product_rule": 0.50,
            "tfidf_text": 0.51,  # Clear but below the 5% minimum gain.
            "tfidf_product": 0.60,
            "bge_text": 0.66,  # Large gain, but the interval crosses zero.
            "bge_product": 0.70,
        }
        draws = {name: value + noise for name, value in estimates.items()}
        draws["bge_text"] = 0.66 + np.linspace(-0.2, 0.2, 200)

        selection = select_representation(estimates, draws, minimum_gain=0.05)

        self.assertEqual(selection["selected"], "bge_product")
        self.assertEqual(
            [step["current"] for step in selection["steps"]],
            ["product_rule", "product_rule", "tfidf_product", "tfidf_product"],
        )
        self.assertEqual(
            [step["replaces"] for step in selection["steps"]],
            [False, True, False, True],
        )

    def test_keeps_the_m5_sample_when_full_fit_is_not_better(self):
        labels = LABELS[[0, 1, 2, 0]]
        worse = {"prediction": LABELS[[1, 1, 1, 1]], "top_three": np.ones(4, dtype=bool)}
        better = {"prediction": labels, "top_three": np.ones(4, dtype=bool)}

        outputs, check = apply_m5_check({"bge_product": worse, M5_SAMPLE: better}, labels)

        self.assertFalse(check["passed"])
        self.assertIs(outputs["bge_product"], better)

        outputs, check = apply_m5_check({"bge_product": better, M5_SAMPLE: worse}, labels)

        self.assertTrue(check["passed"])
        self.assertIs(outputs["bge_product"], better)

    def test_orders_representations_from_simplest(self):
        self.assertEqual(REPRESENTATIONS[0], "product_rule")
        self.assertEqual(REPRESENTATIONS[-1], "bge_product")


class ProductRuleTests(unittest.TestCase):
    def test_uses_fit_rates_and_the_global_rate_for_unknown_products(self):
        fit = target_frame(8, "fit")
        calibration = pd.DataFrame({"Product canonical": ["product 0", "product 9"]})

        probability = product_rule_probabilities(fit, calibration, "T2")

        # product 0 rows are all even, so every one of them is T2-positive.
        self.assertEqual(probability.tolist(), [1.0, 0.5])

    def test_predicts_the_most_frequent_fit_class_of_each_product(self):
        fit = target_frame(12, "fit")
        fit["Product canonical"] = ["product 0"] * 6 + ["product 1"] * 6
        fit["T1"] = ["alpha"] * 6 + ["delta"] * 3 + ["bravo"] * 2 + ["charlie"]
        calibration = pd.DataFrame({"Product canonical": ["product 0", "product 9"]})
        truth = np.array(["charlie", "bravo"])

        outputs = product_rule_t1(fit, calibration, truth)

        # product 0 ranks alpha, then the global delta and bravo; product 9
        # is unknown and falls back to the global alpha, delta and bravo.
        self.assertEqual(outputs["prediction"].tolist(), ["alpha", "alpha"])
        self.assertEqual(outputs["top_three"].tolist(), [False, True])


class TrainingTests(unittest.TestCase):
    def test_bge_gets_the_m5_iterations_without_changing_the_config(self):
        config = small_config()

        self.assertIs(classifier_config(config, "tfidf_text"), config)
        self.assertEqual(classifier_config(config, "bge_text")["tfidf"]["max_iter"], 80)
        self.assertEqual(config["tfidf"]["max_iter"], 50)

    def test_trains_every_new_representation(self):
        frames = {
            "fit": target_frame(90, "fit"),
            "calibration": target_frame(30, "calibration"),
        }
        rng = np.random.default_rng(0)
        features = {}
        for name in ("tfidf_text", "bge_text", "bge_product"):
            features[name] = {}
            for split, frame in frames.items():
                signal = np.eye(3)[frame.index % 3] + rng.normal(0, 0.1, (len(frame), 3))
                values = signal if name.startswith("bge") else csr_matrix(signal)
                features[name][split] = values

        for target in ("T1", "T2"):
            outputs, models = trained_outputs(target, frames, features, small_config())

            self.assertEqual(
                sorted(models),
                sorted(f"{name}_{target.lower()}" for name in features),
            )
            for values in outputs.values():
                for array in values.values():
                    self.assertEqual(len(array), 30)


class OutputTests(unittest.TestCase):
    def test_never_overwrites_existing_results(self):
        with TemporaryDirectory() as temporary_directory:
            existing = Path(temporary_directory) / "report.json"
            existing.write_text("{}\n", encoding="utf-8")

            check_new_outputs([Path(temporary_directory) / "missing.json"])
            with self.assertRaises(FileExistsError):
                check_new_outputs([existing])

    def test_aligns_embedding_rows_by_complaint_id(self):
        frames = {"calibration": pd.DataFrame({"Complaint ID": ["7", "3"]})}
        manifest = pd.DataFrame(
            {
                "Complaint ID": ["3", "5", "7"],
                "Date received": [date(2024, 10, 1), date(2024, 10, 2), date(2024, 10, 9)],
                "embedding_row": [0, 1, 2],
                "split": "calibration",
            }
        )
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "manifest.parquet"
            manifest.to_parquet(path, index=False)

            attach_embedding_rows(frames, path)

            frame = frames["calibration"]
            self.assertEqual(frame["embedding_row"].tolist(), [2, 0])
            self.assertEqual(
                pd.to_datetime(frame["week"]).dt.strftime("%Y-%m-%d").tolist(),
                ["2024-10-07", "2024-09-30"],
            )

            missing = {"calibration": pd.DataFrame({"Complaint ID": ["8"]})}
            with self.assertRaises(ValueError):
                attach_embedding_rows(missing, path)


if __name__ == "__main__":
    unittest.main()
