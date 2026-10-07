"""The declared decisions of the preparation pipeline (params.yaml, PB-11).

Every business constant of the stages type_data to finalize_prepared lives in
`params.yaml` under `preparacion`; dvc.yaml declares the section each stage
reads, so dvc.lock records the values and `dvc params diff` shows a change.
The scripts load their section once at import time and keep their public
constants, so the rest of the code base keeps importing column names from them.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


@cache
def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def preparation_settings(stage: str) -> dict[str, Any]:
    """The `preparacion.<stage>` section of params.yaml."""
    settings = load_params()["preparacion"].get(stage)
    if not isinstance(settings, dict):
        raise KeyError(f"params.yaml has no preparacion.{stage} section.")
    return settings
