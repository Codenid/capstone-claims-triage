import unittest

import numpy as np

from src.models.foundation_t1 import REFERENCE, compare, top_three


def outputs(truth: np.ndarray, flips: dict[str, float], seed: int = 0) -> dict:
    """Reference and candidates that miss a given fraction of the truth."""
    rng = np.random.default_rng(seed)
    result = {}
    for name, fraction in flips.items():
        prediction = truth.copy()
        wrong = rng.random(len(truth)) < fraction
        prediction[wrong] = rng.choice(np.unique(truth), size=wrong.sum())
        result[name] = {"prediction": prediction, "top_three": ~wrong}
    return result


class FoundationT1Tests(unittest.TestCase):
    def test_top_three_orders_classes_by_probability(self):
        classes = np.array(["a", "b", "c", "d"])
        probability = np.array([[0.1, 0.5, 0.3, 0.1], [0.7, 0.1, 0.05, 0.15]])

        prediction, three = top_three(probability, classes)

        self.assertEqual(prediction.tolist(), ["b", "a"])
        self.assertEqual(three.tolist(), [["b", "c", "a"], ["a", "d", "b"]])

    def test_a_clearly_better_model_wins_and_an_equal_one_does_not(self):
        truth = np.repeat(np.array(["a", "b", "c", "d"]), 300)
        weeks = np.tile(np.arange(12), 100)
        flips = {REFERENCE: 0.5, "better": 0.2, "equal": 0.5}

        result = compare(
            outputs(truth, flips), truth, weeks, np.unique(truth), 0.05, 200, 0
        )

        self.assertTrue(result["comparisons"]["better"]["wins"])
        self.assertFalse(result["comparisons"]["equal"]["wins"])
        self.assertEqual(result["weeks"], 12)
        self.assertNotIn(REFERENCE, result["comparisons"])

    def test_the_reference_can_be_any_candidate(self):
        truth = np.repeat(np.array(["a", "b", "c", "d"]), 300)
        weeks = np.tile(np.arange(12), 100)
        flips = {"tabpfn": 0.5, "tabpfn_bge1024": 0.2}

        result = compare(
            outputs(truth, flips),
            truth,
            weeks,
            np.unique(truth),
            0.05,
            200,
            0,
            reference="tabpfn",
        )

        self.assertEqual(list(result["comparisons"]), ["tabpfn_bge1024"])
        self.assertTrue(result["comparisons"]["tabpfn_bge1024"]["wins"])


if __name__ == "__main__":
    unittest.main()
