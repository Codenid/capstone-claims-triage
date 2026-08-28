from __future__ import annotations

import unittest

import pandas as pd

from src.features.e14_prepare import construir_indice_train, enmascarar_empresa


class E14PrepareTest(unittest.TestCase):
    def test_company_mask_uses_legal_name_and_prefix(self) -> None:
        texto = "bank of america denied my claim with bank of america, n.a."
        masked, matches = enmascarar_empresa(
            texto, ("Bank of America, N.A.",), min_chars=5
        )
        self.assertNotIn("bank of america", masked)
        self.assertGreaterEqual(matches, 1)
        self.assertIn("company", masked)

    def test_index_keeps_label_conflicts_explicit(self) -> None:
        muestra = pd.DataFrame(
            {
                "split": ["train", "train", "validation"],
                "hash_narrativa": ["a", "a", "b"],
                "complaint_id": [1, 2, 3],
                "narrative": ["same", "same", "other"],
                "company": ["A Bank", "B Bank", "C Bank"],
                "product": ["Card", "Loan", "Loan"],
                "issue": ["Fee", "Fee", "Payment"],
            }
        )
        indice = construir_indice_train(muestra)
        self.assertEqual(len(indice), 1)
        self.assertEqual(indice.loc[0, "n_eventos_train"], 2)
        self.assertEqual(indice.loc[0, "n_products"], 2)
        self.assertIsNone(indice.loc[0, "product_unique"])
        self.assertEqual(indice.loc[0, "issue_unique"], "Fee")


if __name__ == "__main__":
    unittest.main()
