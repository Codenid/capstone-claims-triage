from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.decomposition import PCA

from src.evaluation.experiment import load_experiment_config
from src.models.cluster_comparison import candidate_definitions, semantic_metrics
from src.models.space_sensitivity import (
    build_space,
    cluster_tasks,
    clustering_settings,
    decide,
    neighbor_pairs,
    reproduction_check,
    start_output,
    weighted_neighbor_lift,
)


def grouped_neighbors(groups: np.ndarray, neighbors: int) -> np.ndarray:
    """Each row's neighbors are the next rows of its own group."""
    result = np.empty((len(groups), neighbors), dtype=np.int32)
    for row, group in enumerate(groups):
        members = np.flatnonzero(groups == group)
        others = members[members != row]
        result[row] = np.roll(others, -np.searchsorted(others, row))[:neighbors]
    return result


def candidate(identifier: str, labels: np.ndarray, neighbors: np.ndarray) -> dict:
    return {
        "id": identifier,
        "algorithm": "kmeans",
        "selected": True,
        "accepted": True,
        **semantic_metrics(labels, neighbors),
    }


class SpaceSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.config = load_experiment_config()
        self.settings = self.config["space_sensitivity"]

    def test_lists_every_space_and_algorithm(self):
        tasks = cluster_tasks(self.settings)

        self.assertEqual(len(self.settings["spaces"]), 9)
        self.assertEqual(len(tasks), 36)
        self.assertEqual(tasks[0], ("pca128", "kmeans"))
        self.assertIn(self.settings["reference_space"], self.settings["spaces"])

    def test_gmm_covariance_follows_the_space(self):
        pca = clustering_settings(self.config, {"kind": "pca"})
        umap = clustering_settings(self.config, {"kind": "umap"})

        self.assertEqual(pca["gmm"]["covariance"], "diag")
        self.assertEqual(umap["gmm"]["covariance"], "full")
        identifiers = [entry["id"] for entry in candidate_definitions(pca)]
        self.assertEqual(len(identifiers), 15)
        self.assertIn("gmm_k40_diag", identifiers)

    def test_unit_weights_reproduce_the_m7_lift(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(-1, 5, size=300).astype(np.int32)
        neighbors = rng.integers(0, 300, size=(300, 10)).astype(np.int32)
        same, valid = neighbor_pairs(labels, neighbors)

        lift = weighted_neighbor_lift(labels, same, valid, np.ones(300))
        doubled = weighted_neighbor_lift(labels, same, valid, np.full(300, 2.0))

        expected = semantic_metrics(labels, neighbors)["neighbor_lift"]
        self.assertAlmostEqual(lift, expected)
        self.assertAlmostEqual(doubled, expected)

    def test_slices_the_m6_pca_and_fits_a_larger_one(self):
        rng = np.random.default_rng(1)
        fit_bge = rng.normal(size=(60, 8)).astype(np.float32)
        calibration_bge = rng.normal(size=(20, 8)).astype(np.float32)
        m6 = PCA(n_components=4, random_state=0).fit(fit_bge)
        inputs = {
            "pca_model": m6,
            "fit_bge": fit_bge,
            "calibration_bge": calibration_bge,
            "fit_pca": m6.transform(fit_bge),
            "calibration_pca": m6.transform(calibration_bge),
        }

        fit, calibration, details = build_space(
            {"kind": "pca", "components": 2}, inputs, {}, seed=0
        )
        larger, _, larger_details = build_space(
            {"kind": "pca", "components": 6}, inputs, {}, seed=0
        )

        np.testing.assert_allclose(fit, inputs["fit_pca"][:, :2], rtol=1e-6)
        self.assertEqual(calibration.shape, (20, 2))
        self.assertAlmostEqual(
            details["explained_variance"], m6.explained_variance_ratio_[:2].sum()
        )
        self.assertEqual(larger.shape, (60, 6))
        self.assertGreater(
            larger_details["explained_variance"], details["explained_variance"]
        )

    def test_a_clearly_better_space_wins_and_a_copy_does_not(self):
        rng = np.random.default_rng(2)
        groups = np.repeat(np.arange(4), 50).astype(np.int32)
        neighbors = grouped_neighbors(groups, neighbors=5)
        noisy = groups.copy()
        flipped = rng.random(len(groups)) < 0.4
        noisy[flipped] = rng.integers(0, 4, size=flipped.sum())
        almost = noisy.copy()
        almost[0] = (almost[0] + 1) % 4
        results = {}
        for space, labels in (("ref", noisy), ("better", groups), ("copy", almost)):
            results[f"{space}__kmeans"] = {
                "space": space,
                "records": [candidate("k4", labels, neighbors)],
                "labels": {"k4": labels},
            }
        settings = {
            "spaces": {"ref": {}, "better": {}, "copy": {}},
            "reference_space": "ref",
            "reference_candidate": "k4",
            "minimum_relative_gain": 0.05,
            "change_ari": 0.6,
        }
        weeks = np.repeat(np.arange(20), 10)

        decision = decide(results, neighbors, weeks, settings, draws=200, seed=0)

        outcome = {
            entry["challenger"]: entry["wins"] for entry in decision["comparisons"]
        }
        self.assertEqual(outcome, {"better/k4": True, "copy/k4": False})
        self.assertEqual(decision["winner"], "better/k4")
        self.assertFalse(decision["space_change"]["copy"]["changes"])

    def test_never_reuses_an_unfinished_output(self):
        with tempfile.TemporaryDirectory() as directory:
            final = Path(directory) / "pca128"

            staging = start_output(final)
            self.assertIsNotNone(staging)
            with self.assertRaisesRegex(FileExistsError, "by hand"):
                start_output(final)
            staging.rename(final)
            self.assertIsNone(start_output(final))

    def test_stops_when_m7_is_not_reproduced(self):
        with tempfile.TemporaryDirectory() as directory:
            clustering_dir = Path(directory)
            (clustering_dir / "labels").mkdir()
            m7_labels = np.array([0, 0, 1, 1])
            np.save(clustering_dir / "labels" / "kmeans_k40.npy", m7_labels)
            result = {"labels": {"kmeans_k40": np.array([0, 1, 0, 1])}}

            with self.assertRaisesRegex(ValueError, "does not reproduce"):
                reproduction_check(result, clustering_dir, 0.99, "kmeans_k40")


if __name__ == "__main__":
    unittest.main()
