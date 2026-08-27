from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.baseline import mascara_vista, metricas_binarias


class BaselineTest(unittest.TestCase):
    def test_purged_view_excludes_train_hashes(self) -> None:
        indice = pd.DataFrame(
            {
                "split": ["train", "validation", "validation"],
                "hash_narrativa": ["a", "a", "b"],
            }
        )
        operational = mascara_vista(indice, "validation", "operational", {"a"})
        purged = mascara_vista(indice, "validation", "purged", {"a"})
        self.assertEqual(operational.tolist(), [False, True, True])
        self.assertEqual(purged.tolist(), [False, False, True])

    def test_binary_metrics_include_capacity_lift(self) -> None:
        y = np.array([1, 0, 1, 0, 0], dtype=np.int8)
        p = np.array([0.9, 0.1, 0.8, 0.2, 0.3])
        metricas = metricas_binarias(y, p)
        self.assertAlmostEqual(metricas["average_precision"], 1.0)
        self.assertGreater(metricas["lift_at_10pct"], 1.0)
        self.assertIn("ece_10", metricas)


if __name__ == "__main__":
    unittest.main()
