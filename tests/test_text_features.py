from __future__ import annotations

import unittest

import pandas as pd

from src.features.text import construir_r0


class TextFeaturesTest(unittest.TestCase):
    def test_vocabulary_is_fit_only_on_train(self) -> None:
        textos = pd.Series(
            ["common train alpha", "common train beta", "validationonly token"],
            dtype="string",
        )
        splits = pd.Series(["train", "train", "validation"], dtype="string")
        config = {
            "word": {
                "ngram_range": [1, 1],
                "min_df": 1,
                "max_df": 1.0,
                "max_features": 100,
                "sublinear_tf": True,
            },
            "char": {
                "analyzer": "char_wb",
                "ngram_range": [3, 3],
                "min_df": 1,
                "max_features": 100,
                "sublinear_tf": True,
            },
        }

        matriz, vectorizadores = construir_r0(textos, splits, config)

        self.assertEqual(matriz.shape[0], 3)
        self.assertNotIn("validationonly", vectorizadores["word"].vocabulary_)
        self.assertIn("alpha", vectorizadores["word"].vocabulary_)
        self.assertEqual(matriz.dtype.name, "float32")


if __name__ == "__main__":
    unittest.main()
