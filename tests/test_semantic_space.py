import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.models.semantic_space import (
    build_faiss_index,
    fit_pca,
    fixed_sample_positions,
    load_dvc_hash,
    transform_to_npy,
    validate_embeddings,
)


class SemanticSpaceTests(unittest.TestCase):
    def test_reads_source_dvc_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.dvc"
            path.write_text(
                "outs:\n- md5: expected.dir\n  path: sample\n",
                encoding="utf-8",
            )

            self.assertEqual(load_dvc_hash(path), "expected.dir")

    def test_validates_normalized_float32_embeddings(self):
        values = {
            split: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            for split in ("fit", "calibration", "validation")
        }
        rows = {split: 2 for split in values}

        result = validate_embeddings(values, rows, 2, tolerance=1e-5, batch_rows=1)

        self.assertEqual(result["fit"]["dimensions"], 2)
        self.assertEqual(result["validation"]["maximum_norm_error"], 0.0)

    def test_rejects_embeddings_with_wrong_dimensions(self):
        values = {
            split: np.ones((2, 3), dtype=np.float32)
            for split in ("fit", "calibration", "validation")
        }
        rows = {split: 2 for split in values}

        with self.assertRaisesRegex(ValueError, "Unexpected embedding shape"):
            validate_embeddings(values, rows, 2, tolerance=1e-5, batch_rows=1)

    def test_pca_is_fitted_only_with_fit_values(self):
        fit = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        calibration = np.full((2, 3), 100.0, dtype=np.float32)
        model = fit_pca(fit, components=1, seed=42)

        with tempfile.TemporaryDirectory() as directory:
            transformed = transform_to_npy(
                model,
                calibration,
                Path(directory) / "calibration.npy",
                batch_rows=1,
            )

            np.testing.assert_allclose(model.mean_, fit.mean(axis=0))
            self.assertEqual(transformed.shape, (2, 1))
            self.assertEqual(transformed.dtype, np.float32)

    def test_faiss_inner_product_matches_numpy(self):
        fit = np.array(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            dtype=np.float32,
        )
        queries = np.array([[0.9, 0.1], [-0.8, 0.2]], dtype=np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        index = build_faiss_index(fit, batch_rows=2, threads=1)

        scores, indices = index.search(queries, 2)
        expected_scores = np.sort(queries @ fit.T, axis=1)[:, -2:][:, ::-1]

        np.testing.assert_allclose(scores, expected_scores, atol=1e-6)
        self.assertEqual(indices[0, 0], 0)
        self.assertEqual(indices[1, 0], 2)

    def test_fixed_sample_is_deterministic(self):
        first = fixed_sample_positions(100, requested=10, seed=42)
        second = fixed_sample_positions(100, requested=10, seed=42)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 10)


if __name__ == "__main__":
    unittest.main()
