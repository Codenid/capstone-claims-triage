import tempfile
import unittest
from pathlib import Path

import pyarrow as pa

from src.evaluation.verify_modeling_input import (
    file_md5,
    prepared_data_hash,
    row_count,
    true_count,
)


class ModelingSetupTests(unittest.TestCase):
    def test_reads_prepared_hash_from_dvc_lock(self):
        content = """\
schema: '2.0'
stages:
  finalize_prepared:
    outs:
      - path: data/processed/prepared.parquet
        md5: expected-hash
"""
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "dvc.lock"
            lock_path.write_text(content, encoding="utf-8")

            self.assertEqual(prepared_data_hash(lock_path), "expected-hash")

    def test_calculates_file_md5_in_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.bin"
            path.write_bytes(b"abc")

            self.assertEqual(file_md5(path, chunk_size=2), "900150983cd24fb0d6963f7d28e17f72")

    def test_counts_rows_and_true_values_by_period(self):
        table = pa.table(
            {
                "period": ["train", "train", "validation"],
                "eligible": [True, False, True],
            }
        )

        self.assertEqual(row_count(table, "train"), 2)
        self.assertEqual(true_count(table, "eligible", "train"), 1)


if __name__ == "__main__":
    unittest.main()
