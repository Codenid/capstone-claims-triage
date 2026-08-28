from __future__ import annotations

import unittest

import pandas as pd

from src.features.embeddings import textos_unicos


class EmbeddingsTest(unittest.TestCase):
    def test_repeated_hash_is_encoded_once(self) -> None:
        muestra = pd.DataFrame(
            {
                "hash_narrativa": ["b", "a", "a"],
                "narrative": ["Other text", "  SAME  text", "same text"],
            }
        )
        unicos = textos_unicos(muestra)

        self.assertEqual(unicos["hash_narrativa"].tolist(), ["a", "b"])
        self.assertEqual(unicos["embedding_row"].tolist(), [0, 1])
        self.assertEqual(unicos["n_eventos"].tolist(), [2, 1])
        self.assertEqual(unicos.loc[0, "texto_normalizado"], "same text")


if __name__ == "__main__":
    unittest.main()
