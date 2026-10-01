import json
import unittest

import numpy as np

from src.evaluation.experiment import PROJECT_ROOT, load_experiment_config
from src.evaluation.final_confirmation import (
    M10_ARTIFACT,
    M10_REPORT,
    M9_REPORT,
    confirm_weekly_composition,
    confirm_weekly_volume,
    flatten,
    frozen_selection,
    model_path,
    paired_difference,
    view_metrics,
)


class ConfirmationHelperTests(unittest.TestCase):
    def test_flatten_joins_nested_names(self):
        value = {"metrics": {"wis": 2.5, "coverage": {"80": 0.8}}, "passed": True}

        self.assertEqual(
            flatten("m9", value),
            {"m9_metrics_wis": 2.5, "m9_metrics_coverage_80": 0.8, "m9_passed": 1.0},
        )

    def test_paired_difference_uses_the_same_draws(self):
        model = np.array([3.0, 4.0, 5.0, 6.0])
        reference = np.array([1.0, 1.0, 1.0, 1.0])

        difference = paired_difference(model, reference, estimate=3.5)

        self.assertEqual(difference["estimate"], 3.5)
        self.assertGreater(difference["p025"], 2.0)
        self.assertLess(difference["p975"], 5.0)

    def test_model_paths_follow_the_stage_that_trained_each_model(self):
        config = load_experiment_config()

        m4_model = model_path("tfidf_product", "T4", config)

        self.assertEqual(
            m4_model.relative_to(PROJECT_ROOT).as_posix(),
            "artifacts/models/tfidf/t4_model.joblib",
        )
        self.assertEqual(
            model_path("tfidf_text", "T2", config).name, "tfidf_text_t2.joblib"
        )

    def test_binary_metrics_keep_the_calibration_threshold(self):
        truth = np.array([True, False, True, False])
        outputs = {"probability": np.array([0.9, 0.6, 0.4, 0.1])}
        rows = np.ones(4, dtype=bool)

        low = view_metrics("T2", outputs, truth, 0.3, rows)
        high = view_metrics("T2", outputs, truth, 0.8, rows)

        self.assertEqual(low["recall"], 1.0)
        self.assertEqual(high["recall"], 0.5)
        with self.assertRaises(ValueError):
            view_metrics("T2", outputs, truth, None, rows)

    def test_m5b_report_still_names_the_frozen_classifiers(self):
        results = frozen_selection(load_experiment_config())

        self.assertEqual(results["T1"]["selection"]["selected"], "bge_product")


class RehearsalTests(unittest.TestCase):
    def test_m9_reproduces_its_published_calibration_bootstrap(self):
        path = M9_REPORT / "bootstrap.json"
        published = json.loads(path.read_text(encoding="utf-8"))

        result = confirm_weekly_volume("calibration", seed=42)

        self.assertAlmostEqual(
            result["metrics"]["model"]["wis"], published["metrics"]["model"]["wis"]
        )
        # Reading the predictions back from CSV changes only the last digit.
        for metric, values in published["difference"].items():
            for label, value in values.items():
                self.assertAlmostEqual(result["difference"][metric][label], value)

    @unittest.skipUnless(
        (M10_ARTIFACT / "posterior.nc").exists(), "the DM-R4 posterior is in DVC"
    )
    def test_m10_reproduces_its_published_calibration_comparison(self):
        results = json.loads((M10_REPORT / "results.json").read_text(encoding="utf-8"))

        result = confirm_weekly_composition(load_experiment_config(), "calibration", 42)

        published = results["comparison"]["difference"]
        for label, value in published.items():
            self.assertAlmostEqual(result["difference"][label], value)


if __name__ == "__main__":
    unittest.main()
