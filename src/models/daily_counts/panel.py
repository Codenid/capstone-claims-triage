"""Build the daily panel once from the frozen M8B assignments (§28.1).

Writes reports/modeling/daily_counts.csv and prints its SHA-256, which every
daily config freezes as `daily_counts_sha256`. It refuses to overwrite.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import yaml

from src.evaluation.experiment import PROJECT_ROOT, git_commit, load_experiment_config
from src.models.bge_sample import ID_COLUMN
from src.models.daily_counts.contracts import DAY_COLUMN, SPLITS, TOTAL_COLUMN
from src.models.daily_counts.data import (
    DATE_COLUMN,
    panel_from_assignments,
    validate_daily_counts,
    write_panel,
)
from src.models.semantic_space import load_dvc_hash

REFERENCE_CONFIG = PROJECT_ROOT / "configs/daily_counts/nb_daily_hierarchical_v1.yaml"


def load_assignments(config: dict[str, Any]) -> pd.DataFrame:
    directory = PROJECT_ROOT / config["paths"]["weekly_patterns_artifacts"]
    parts = [
        pd.read_parquet(
            directory / f"{split}_assignments.parquet",
            columns=[ID_COLUMN, DATE_COLUMN, "cluster_id", "is_novel", "split"],
        )
        for split in SPLITS
    ]
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    config = load_experiment_config()
    settings = yaml.safe_load(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    path = PROJECT_ROOT / config["paths"]["daily_counts"]
    if path.exists():
        raise FileExistsError(f"The daily panel is frozen: {path}")
    source_dvc_hash = load_dvc_hash(
        PROJECT_ROOT / config["paths"]["weekly_patterns_artifacts_dvc"]
    )
    if source_dvc_hash != settings["source_dvc_hash"]:
        raise ValueError("The weekly patterns DVC hash is not the frozen one.")
    panel = panel_from_assignments(
        load_assignments(config), settings["clusters"], settings["periods"]
    )
    panel = validate_daily_counts(
        panel, settings["clusters"], settings["expected_days"]
    )
    sha256 = write_panel(panel, path)
    summary = {
        "git_commit": git_commit(),
        "source_dvc_hash": source_dvc_hash,
        "daily_counts_sha256": sha256,
        "rows": int(len(panel)),
        "days": {
            split: int(panel.loc[panel["split"] == split, DAY_COLUMN].nunique())
            for split in SPLITS
        },
        "mean_daily_total": {
            split: float(
                panel.loc[panel["split"] == split]
                .groupby(DAY_COLUMN)[TOTAL_COLUMN]
                .first()
                .mean()
            )
            for split in SPLITS
        },
    }
    path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
