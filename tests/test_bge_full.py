import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from src.models.bge_full import (
    config_fingerprint,
    initial_state,
    load_or_create_state,
    merge_embedding_batch,
    validate_state,
)


class BgeFullTests(unittest.TestCase):
    def setUp(self):
        self.config: dict[str, Any] = {
            "bge": {
                "model": "test-model",
                "revision": "test-revision",
                "batch_size": 2,
            },
            "bge_full": {
                "contract_version": "test-v1",
                "expected_rows": {
                    "fit": 3,
                    "calibration": 2,
                    "validation": 1,
                },
            },
            "evaluation": {
                "splits": {
                    "fit": {"start": "2023-01-01", "end": "2024-10-01"},
                    "calibration": {
                        "start": "2024-10-01",
                        "end": "2025-01-01",
                    },
                    "validation": {
                        "start": "2025-01-01",
                        "end": "2025-07-01",
                    },
                }
            },
        }

    def test_merges_reused_and_generated_embeddings_in_original_order(self):
        sample = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        sample_index = {"known-a": 0, "known-b": 1}

        def encode(texts):
            return np.array(
                [[len(text), -len(text)] for text in texts],
                dtype=np.float32,
            )

        values, reused, _ = merge_embedding_batch(
            ["known-b", "new", "known-a"],
            ["old", "abcd", "old"],
            sample_index,
            sample,
            encode,
            dimensions=2,
        )

        np.testing.assert_array_equal(reused, [True, False, True])
        np.testing.assert_array_equal(
            values,
            [[0.0, 1.0], [4.0, -4.0], [1.0, 0.0]],
        )

    def test_rejects_empty_new_narrative(self):
        with self.assertRaisesRegex(ValueError, "empty eligible narrative"):
            merge_embedding_batch(
                ["new"],
                [""],
                {},
                np.empty((0, 2), dtype=np.float32),
                lambda texts: np.ones((len(texts), 2), dtype=np.float32),
                dimensions=2,
            )

    def test_fingerprint_changes_with_relevant_configuration(self):
        first = config_fingerprint(self.config)
        self.config["bge"]["batch_size"] = 4
        second = config_fingerprint(self.config)

        self.assertNotEqual(first, second)

    def test_checkpoint_tracks_each_split(self):
        state = initial_state(self.config, "input-hash", "sample-hash")

        self.assertEqual(
            state["progress"],
            {"fit": 0, "calibration": 0, "validation": 0},
        )
        self.assertEqual(state["model_revision"], "test-revision")

    def test_rejects_checkpoint_from_another_configuration(self):
        expected = initial_state(self.config, "input-hash", "sample-hash")
        actual = dict(expected)
        actual["model_revision"] = "different"

        with self.assertRaisesRegex(ValueError, "model_revision"):
            validate_state(actual, expected)

    def test_loading_checkpoint_increments_resume_count(self):
        expected = initial_state(self.config, "input-hash", "sample-hash")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            first = load_or_create_state(path, expected)
            second = load_or_create_state(path, expected)

        self.assertEqual(first["resume_count"], 0)
        self.assertEqual(second["resume_count"], 1)


if __name__ == "__main__":
    unittest.main()
