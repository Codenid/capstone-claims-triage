from datetime import date
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa

from src.evaluation.experiment import build_run_record, set_seed
from src.evaluation.freeze_evaluation import (
    date_mask,
    parse_splits,
    true_count as array_true_count,
)
from src.evaluation.publish_run import load_run_record
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

    def test_resets_python_and_numpy_seeds(self):
        set_seed(42)
        first = (random.random(), np.random.random())
        set_seed(42)
        second = (random.random(), np.random.random())

        self.assertEqual(first, second)

    def test_builds_required_run_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            contract_path = Path(directory) / "contract.json"
            contract_path.write_text(
                json.dumps({"dvc_md5": "data-hash"}),
                encoding="utf-8",
            )
            config = {
                "experiment": {
                    "name": "test-experiment",
                    "seed": 42,
                    "run_contract_version": "test-v1",
                },
                "paths": {
                    "input_data": "data.parquet",
                    "input_contract": str(contract_path),
                },
                "evaluation": {"split_version": "temporal-test-v1"},
            }

            with (
                patch("src.evaluation.experiment.git_commit", return_value="git-hash"),
                patch.dict(
                    os.environ,
                    {"CLAIMS_EXECUTION_HOST": "khipu", "SLURM_JOB_ID": "123"},
                    clear=True,
                ),
            ):
                record = build_run_record(
                    config=config,
                    run_name="test-run",
                    stage="M1",
                    target="T1",
                    split="fit",
                    view="complete",
                    features=["narrative"],
                )

        self.assertEqual(record["tags"]["git_commit"], "git-hash")
        self.assertEqual(record["tags"]["dvc_data_hash"], "data-hash")
        self.assertEqual(record["tags"]["execution_host"], "khipu")
        self.assertEqual(record["tags"]["evaluation_split_version"], "temporal-test-v1")
        self.assertEqual(record["tags"]["execution_mode"], "slurm")
        self.assertEqual(record["tags"]["slurm_job_id"], "123")
        self.assertEqual(record["parameters"]["seed"], 42)
        self.assertEqual(record["parameters"]["features"], '["narrative"]')

    def test_parses_contiguous_evaluation_splits(self):
        config = {
            "evaluation": {
                "splits": {
                    "fit": {"start": "2023-01-01", "end": "2024-10-01"},
                    "calibration": {"start": "2024-10-01", "end": "2025-01-01"},
                    "validation": {"start": "2025-01-01", "end": "2025-07-01"},
                }
            }
        }

        splits = parse_splits(config)

        self.assertEqual(splits["fit"], (date(2023, 1, 1), date(2024, 10, 1)))
        self.assertEqual(splits["calibration"][0], splits["fit"][1])
        self.assertEqual(splits["validation"][0], splits["calibration"][1])

    def test_date_mask_uses_start_but_not_end(self):
        values = pa.array(
            [date(2022, 12, 31), date(2023, 1, 1), date(2024, 9, 30), date(2024, 10, 1)]
        )

        mask = date_mask(values, date(2023, 1, 1), date(2024, 10, 1))

        self.assertEqual(mask.to_pylist(), [False, True, True, False])

    def test_counts_empty_boolean_array_as_zero(self):
        self.assertEqual(array_true_count(pa.array([], type=pa.bool_())), 0)

    def test_rejects_incomplete_offline_run(self):
        with tempfile.TemporaryDirectory() as directory:
            record_path = Path(directory) / "run.json"
            record_path.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Run record is missing"):
                load_run_record(record_path)


if __name__ == "__main__":
    unittest.main()
