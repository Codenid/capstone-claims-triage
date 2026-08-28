from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data.cluster_evidence import _ctfidf_terms, metricas_etiqueta


class ClusterEvidenceTest(unittest.TestCase):
    def test_label_metrics_exclude_conflicts_and_noise(self) -> None:
        etiquetas = pd.Series(["Card", None, "Loan", "Loan"])
        clusters = np.array([0, 0, 1, -1], dtype=np.int32)
        metricas = metricas_etiqueta(etiquetas, clusters, "product")
        self.assertEqual(metricas["product_n"], 2)
        self.assertAlmostEqual(metricas["product_coverage"], 0.5)
        self.assertAlmostEqual(metricas["product_purity_weighted"], 1.0)

    def test_ctfidf_returns_cluster_specific_terms(self) -> None:
        textos = pd.Series(
            [
                "apple apple orchard fruit",
                "apple fruit tree",
                "mortgage loan house",
                "mortgage payment loan",
            ]
        )
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        topics = _ctfidf_terms(textos, labels, max_features=100, top_n=5)
        terminos_0 = {termino for termino, _ in topics[0]}
        terminos_1 = {termino for termino, _ in topics[1]}
        self.assertIn("apple", terminos_0)
        self.assertIn("mortgage", terminos_1)


if __name__ == "__main__":
    unittest.main()
