"""Response deadlines in Peruvian business days (models_plan.md §23)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import yaml

from src.evaluation.experiment import PROJECT_ROOT

HOLIDAYS_PATH = PROJECT_ROOT / "configs/peru_holidays.yaml"


def load_holidays(path: Path = HOLIDAYS_PATH) -> list[date]:
    """National holidays from the versioned gob.pe list, sorted."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return sorted(config["holidays"])


def due_date(registered: date, business_days: int, holidays: list[date]) -> date:
    """The n-th business day after registration, which itself does not count.

    Business days are Monday to Friday except national holidays. A deadline
    that touches a year without a holiday list is not computed (§23).
    """
    # Rolling back makes a weekend or holiday registration count from the next
    # business day.
    offset = np.busday_offset(
        np.datetime64(registered, "D"),
        business_days,
        roll="backward",
        holidays=np.array(holidays, dtype="datetime64[D]"),
    )
    due = date.fromisoformat(str(offset))
    missing = set(range(registered.year, due.year + 1)) - {day.year for day in holidays}
    if missing:
        raise ValueError(f"No holiday list for {sorted(missing)}.")
    return due


def business_days_left(today: date, due: date, holidays: list[date]) -> int:
    """Business days after today up to the due date; negative once it passed."""
    calendar = np.busdaycalendar(holidays=np.array(holidays, dtype="datetime64[D]"))
    start, end = np.datetime64(today, "D"), np.datetime64(due, "D")
    if start > end:
        # Counting backwards, NumPy counts the business days in (due, today].
        return int(np.busday_count(start, end, busdaycal=calendar))
    day = np.timedelta64(1, "D")
    return int(np.busday_count(start + day, end + day, busdaycal=calendar))
