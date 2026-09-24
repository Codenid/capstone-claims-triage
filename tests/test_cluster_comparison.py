import unittest

import numpy as np
from sklearn.metrics import pairwise_distances

from src.models.cluster_comparison import (
    acceptance_failures,
    candidate_definitions,
    labels_from_cure_clusters,
    semantic_metrics,
    silhouette_summary,
    structure_metrics,
)


class ClusterComparisonTests(unittest.TestCase):
    def test_builds_the_frozen_candidate_grid(self):
        settings = {
            "kmeans": {"clusters": [20, 40, 80]},
            "hdbscan": {
                "min_cluster_size": [100, 250, 500],
                "min_samples": [10, 30],
            },
            "cure": {
                "candidates": [
                    {"clusters": 20, "representatives": 5, "compression": 0.5},
                    {"clusters": 40, "representatives": 5, "compression": 0.5},
                    {"clusters": 80, "representatives": 5, "compression": 0.5},
                ]
            },
        }

        candidates = candidate_definitions(settings)

        self.assertEqual(len(candidates), 12)
        self.assertEqual(
            {candidate["algorithm"] for candidate in candidates},
            {"kmeans", "hdbscan", "cure"},
        )

    def test_converts_cure_clusters_to_labels(self):
        labels = labels_from_cure_clusters([[0, 2], [1, 3]], rows=4)

        np.testing.assert_array_equal(labels, [0, 1, 0, 1])

    def test_rejects_incomplete_cure_assignments(self):
        with self.assertRaisesRegex(ValueError, "did not assign"):
            labels_from_cure_clusters([[0], [2]], rows=3)

    def test_rejects_duplicate_cure_assignments(self):
        with self.assertRaisesRegex(ValueError, "more than once"):
            labels_from_cure_clusters([[0, 1], [1, 2]], rows=3)

    def test_silhouette_excludes_hdbscan_noise(self):
        values = np.array(
            [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0], [20.0, 20.0]],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 1, 1, -1], dtype=np.int32)
        distances = pairwise_distances(values)

        summary, samples = silhouette_summary(
            distances,
            labels,
            np.arange(len(values)),
        )

        self.assertEqual(summary["silhouette_rows"], 4.0)
        self.assertEqual(len(samples), 4)
        self.assertGreater(summary["silhouette_mean"], 0.9)

    def test_detects_a_giant_cluster(self):
        labels = np.array([0] * 90 + [1] * 10, dtype=np.int32)
        metrics = structure_metrics(labels, small_cluster_rows=5)
        metrics.update(
            {
                "silhouette_mean": 0.2,
                "negative_silhouette_fraction": 0.1,
                "neighbor_lift": 0.2,
                "template_dominated_rows_fraction": 0.0,
            }
        )
        acceptance = {
            "minimum_clusters": 2,
            "maximum_clusters": 100,
            "maximum_largest_cluster_fraction": 0.5,
            "maximum_small_cluster_fraction": 0.1,
            "maximum_noise_fraction": 0.4,
            "minimum_silhouette": 0.0,
            "maximum_negative_silhouette_fraction": 0.35,
            "minimum_neighbor_lift": 0.1,
            "maximum_template_dominated_fraction": 0.25,
        }

        failures = acceptance_failures(metrics, acceptance)

        self.assertIn("largest_cluster", failures)

    def test_semantic_metrics_handle_all_noise(self):
        labels = np.full(4, -1, dtype=np.int32)
        neighbors = np.array(
            [[1, 2], [0, 2], [0, 1], [1, 2]],
            dtype=np.int32,
        )

        metrics = semantic_metrics(labels, neighbors)

        self.assertEqual(metrics["neighbor_lift"], 0.0)


if __name__ == "__main__":
    unittest.main()
