from __future__ import annotations

import unittest

import pandas as pd

from src.data.prepare_eda import asignar_split, derivar_targets, hash_narrativa


SPLITS = {
    "train": {"start": "2023-01-01", "end": "2024-12-31"},
    "validation": {"start": "2025-01-01", "end": "2025-06-30"},
    "test": {"start": "2025-07-01", "end": "2025-12-31"},
    "ood_2026": {"start": "2026-01-01", "end": "2026-12-31"},
}


class PrepareEdaTest(unittest.TestCase):
    def test_asignar_split_respects_boundaries(self) -> None:
        fechas = pd.Series(
            ["2023-01-01", "2024-12-31", "2025-01-01", "2025-07-01", "2026-01-01", "2022-12-31"]
        )
        resultado = asignar_split(fechas, SPLITS)
        self.assertEqual(
            resultado.tolist(),
            ["train", "train", "validation", "test", "ood_2026", pd.NA],
        )

    def test_hash_normalizes_case_and_whitespace(self) -> None:
        self.assertEqual(hash_narrativa("  My  CLAIM\nText "), hash_narrativa("my claim text"))
        self.assertNotEqual(hash_narrativa("my claim text"), hash_narrativa("another claim"))

    def test_targets_follow_hierarchical_semantics(self) -> None:
        respuestas = pd.Series(
            [
                "Closed with explanation",
                "Closed with non-monetary relief",
                "Closed with monetary relief",
                "In progress",
            ],
            dtype="string",
        )
        plazos = pd.Series(["Yes", "Yes", "No", pd.NA], dtype="string")
        targets = derivar_targets(respuestas, plazos)

        self.assertEqual(targets["target_t2_relief"].tolist(), [0, 1, 1, pd.NA])
        self.assertEqual(targets["target_t3_monetary"].tolist(), [0, 0, 1, pd.NA])
        self.assertEqual(targets["target_t4_late"].tolist(), [0, 0, 1, pd.NA])
        self.assertEqual(targets["t2_ambiguous"].tolist(), [False, False, False, False])


if __name__ == "__main__":
    unittest.main()
