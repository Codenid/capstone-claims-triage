from __future__ import annotations

import unittest

import numpy as np

from src.evaluation.e14_visualization import seleccionar_muestra_estratificada


class E14VisualizationTest(unittest.TestCase):
    def test_sample_is_reproducible_and_preserves_all_strata(self) -> None:
        labels = np.repeat(np.array([-1, 0, 1, 2]), 20)
        primera = seleccionar_muestra_estratificada(labels, n=40, semilla=42)
        segunda = seleccionar_muestra_estratificada(labels, n=40, semilla=42)

        np.testing.assert_array_equal(primera, segunda)
        self.assertEqual(len(primera), 40)
        self.assertEqual(set(labels[primera]), {-1, 0, 1, 2})

    def test_returns_all_rows_when_limit_is_large(self) -> None:
        labels = np.array([-1, 0, 1])
        muestra = seleccionar_muestra_estratificada(labels, n=10, semilla=42)
        np.testing.assert_array_equal(muestra, np.arange(3))


if __name__ == "__main__":
    unittest.main()
