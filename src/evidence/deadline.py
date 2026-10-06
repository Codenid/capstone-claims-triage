"""Response deadline in Peruvian business days (docs/registro/models_plan.md §23).

Business days are Monday to Friday except the national holidays in
configs/peru_holidays.yaml. A deadline that reaches a year without a holiday
list is not computed.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import yaml

from src.evaluation.experiment import PROJECT_ROOT

HOLIDAYS_PATH = PROJECT_ROOT / "configs/peru_holidays.yaml"
STANDARD_DAYS = 15
EXTENDED_DAYS = 45


def load_holidays(path: Path = HOLIDAYS_PATH) -> list[date]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return sorted(config["holidays"])


def due_date(registered: date, business_days: int, holidays: list[date]) -> date:
    """The n-th business day after registration; the registration day does not count."""
    calendar = np.array(holidays, dtype="datetime64[D]")
    # Registered on a weekend or holiday: the next business day is day one.
    offset = np.busday_offset(
        np.datetime64(registered, "D"),
        business_days,
        roll="backward",
        holidays=calendar,
    )
    due = date.fromisoformat(str(offset))
    years = set(range(registered.year, due.year + 1))
    missing = years - {day.year for day in holidays}
    if missing:
        raise ValueError(f"No holiday list for {sorted(missing)}.")
    return due


def deadlines(registered: date, holidays: list[date]) -> dict[str, str]:
    """Standard and extended due dates as ISO strings."""
    return {
        "registered": registered.isoformat(),
        "standard_days": STANDARD_DAYS,
        "standard_due": due_date(registered, STANDARD_DAYS, holidays).isoformat(),
        "extended_days": EXTENDED_DAYS,
        "extended_due": due_date(registered, EXTENDED_DAYS, holidays).isoformat(),
    }
