from __future__ import annotations

import unittest

import numpy as np
import scipy.sparse as sp

from src.evaluation.compare_embeddings import memoria_mib


class CompareEmbeddingsTest(unittest.TestCase):
    def test_memory_reports_dense_and_sparse_storage(self) -> None:
        dense = np.ones((10, 4), dtype=np.float32)
        sparse = sp.csr_matrix(dense)
        self.assertAlmostEqual(memoria_mib(dense), dense.nbytes / 1024**2)
        esperado = (
            sparse.data.nbytes + sparse.indices.nbytes + sparse.indptr.nbytes
        ) / 1024**2
        self.assertAlmostEqual(memoria_mib(sparse), esperado)


if __name__ == "__main__":
    unittest.main()
