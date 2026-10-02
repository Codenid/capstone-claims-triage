"""Facts of the decision brief in reports/modeling/decision_brief (models_plan.md §26).

Two steps, so the slow part runs once on Khipu:

1. `--export DIR` writes complaint_rows.parquet (the prepared complaints without
   narratives, with their M8 pattern) and pattern_history.parquet (the M12 panel
   of every pattern and week).
2. `--rows FILE --history FILE` writes the brief's data/facts.json from those two
   files, M9's dispersion draws and the frozen reports. Nothing is rounded here:
   the brief rounds each number once, when it writes it.

The reading is the one of Monday 10 Feb 2025, with the week of 3 Feb closed.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm

from src.evaluation.experiment import PROJECT_ROOT, load_experiment_config
from src.evaluation.final_confirmation import M9_MODEL, M9_REPORT, load_settings
from src.models.persistent_change import M9_ARTIFACT, PIT_CLIP, WEEKS_PER_MONTH
from src.models.weekly_counts.sampling import subsample_posterior
from src.triage import echo
from src.triage.patterns import pattern_history

T = pd.Timestamp
READ = T("2025-02-03")
READING_DAY = np.datetime64("2025-02-10", "D")
WEEK_END = np.datetime64("2025-02-14", "D")
KNOWN = [T("2025-01-13"), T("2025-01-20"), T("2025-01-27"), READ]
EPISODE = (T("2025-01-27"), T("2025-02-17"))
COVERAGE_EPISODE = (T("2025-01-13"), T("2025-02-24"))
SHOWN = (T("2024-10-07"), T("2025-06-23"))
BURST, LETTER_PATTERN, CAPACITY_PATTERN = 8, 3, 14
ALERTS = [3, 14, 4, 11, 12, 13, 17, 19, 25, 36, 8]
LETTER_DAY = T("2025-01-22")
CREDIT = "Credit reporting or other personal consumer reports"
SUB_ISSUE = "Their investigation did not fix an error on your report"
ID, TEXT = "Complaint ID", "Consumer complaint narrative SHA-256"
EXPORT_COLUMNS = [
    ID, "Date received", "Company", "State", "Sub-issue", "Issue canonical", "Product canonical",
    "Company response to consumer", "Timely response?", "T2", "T3", TEXT, "Tags", "Submitted via",
]
BRIEF = PROJECT_ROOT / "reports/modeling/decision_brief"
REPORTS = PROJECT_ROOT / "reports/modeling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, help="Write the two exports to this directory and stop.")
    parser.add_argument("--rows", type=Path, help="complaint_rows.parquet from --export.")
    parser.add_argument("--history", type=Path, help="pattern_history.parquet from --export.")
    parser.add_argument("--posterior", type=Path, default=M9_ARTIFACT / "posterior.nc", help="M9 posterior.nc.")
    parser.add_argument("--output", type=Path, default=BRIEF / "data/facts.json")
    return parser.parse_args()


# --- Exports (Khipu) ---------------------------------------------------------------

def complaint_rows(config: dict[str, Any]) -> pd.DataFrame:
    """Every complaint of 2023 to 2025-H1 with its M8 pattern, without the narrative."""
    table = pq.read_table(
        PROJECT_ROOT / config["paths"]["input_data"],
        columns=EXPORT_COLUMNS,
        filters=[("Date received", ">=", date(2023, 1, 1)), ("Date received", "<", date(2025, 7, 1))],
    )
    rows = table.to_pandas().astype({ID: str})
    assigned = pd.concat(
        pd.read_parquet(PROJECT_ROOT / config["paths"]["weekly_patterns_artifacts"] / f"{split}_assignments.parquet",
                        columns=[ID, "split", "cluster_id", "distance", "is_novel"])
        for split in ("fit", "calibration", "validation")
    ).astype({ID: str})
    merged = assigned.merge(rows, on=ID, how="left", validate="one_to_one")
    merged["week"] = pd.to_datetime(merged["Date received"]).dt.to_period("W-SUN").dt.start_time
    return merged


def export(directory: Path, config: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    complaint_rows(config).to_parquet(directory / "complaint_rows.parquet", index=False)
    seed = config["experiment"]["seed"]
    pattern_history(config, seed).to_parquet(directory / "pattern_history.parquet", index=False)


# --- Inputs ----------------------------------------------------------------------------

def load_rows(path: Path) -> pd.DataFrame:
    columns = [ID, "week", "split", "cluster_id", "Date received", "Company", "Sub-issue", "Issue canonical",
               "Product canonical", "Timely response?", TEXT, "is_novel"]
    rows = pd.read_parquet(path, columns=columns).rename(
        columns={TEXT: "text", "Date received": "date", "Timely response?": "timely"})
    rows["date"] = pd.to_datetime(rows["date"])
    rows["credit"] = rows["Product canonical"] == CREDIT
    return rows


def alpha_draws(path: Path, seed: int) -> np.ndarray:
    """M9's dispersion draws, as its predictions and M11 use them: (draws, patterns)."""
    m9 = load_settings("weekly_counts", M9_MODEL)
    posterior = subsample_posterior(az.from_netcdf(path), m9["sampling"]["prediction_draws"], seed)
    return posterior["alpha"].transpose("sample", "cluster").to_numpy()


class Panel:
    """The frozen weekly panel as one table per measure, weeks by patterns."""

    def __init__(self, history: pd.DataFrame) -> None:
        def table(measure: str) -> pd.DataFrame:
            return history.pivot(index="week", columns="cluster_id", values=measure).sort_index()

        self.history = history
        self.observed, self.expected = table("observed"), table("expected")
        self.z, self.cusum = table("z"), table("cusum")
        self.alarm, self.active = table("cusum_alarm").astype(bool), table("active_alert").astype(bool)
        self.share, self.expected_share = table("share"), table("expected_share")
        self.share_low, self.share_high = table("share_low"), table("share_high")
        self.level_change, self.level_percentile = table("level_change"), table("level_percentile")
        self.total = self.observed.sum(axis=1)
        self.weeks = list(self.observed.index)
        split = history.groupby("week")["split"].first()
        self.split = {name: [w for w in self.weeks if split[w] == name] for name in ("fit", "calibration", "validation")}

    def between(self, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
        return [w for w in self.weeks if start <= w <= end]


def iso(day: Any) -> str:
    return pd.Timestamp(day).date().isoformat()


def plain(value: Any) -> Any:
    """JSON-ready values at full precision: plain numbers, ISO dates, nested lists and dicts."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return iso(value)
    return value


def read_report(name: str) -> dict[str, Any]:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


# --- Weekly totals and a normal panel -------------------------------------------------

def totals_facts(panel: Panel, rows: pd.DataFrame) -> dict[str, Any]:
    fit = rows.loc[rows["split"] == "fit"].groupby("week").size()
    # The weeks of 26 Dec 2022 and 30 Sep 2024 are partial.
    fit = fit.loc[(fit.index >= T("2023-01-02")) & (fit.index <= T("2024-09-23"))]
    total, cal = panel.total, panel.split["calibration"]
    return {
        "fit_median_total": fit.median(), "fit_p10_total": fit.quantile(0.1), "fit_p90_total": fit.quantile(0.9),
        "cal_median_total": total.loc[cal].median(),
        "weekly_totals": [{"week": iso(w), "total": total[w], "p8": panel.observed.at[w, BURST]}
                          for w in panel.between(*SHOWN)],
        "total_0106": total[T("2025-01-06")],
        "post_weekly_avg": total.loc[[w for w in panel.split["validation"] if w >= T("2025-02-24")]].mean(),
    }


def alarm_facts(panel: Panel) -> dict[str, Any]:
    fit, validation = panel.split["fit"], panel.split["validation"]
    per_week = panel.alarm.sum(axis=1)
    active = panel.active.loc[fit].sum(axis=1)
    episode = [w for w in validation if EPISODE[0] <= w <= EPISODE[1]]
    outside = [w for w in validation if w not in episode]
    return {
        "fit_weeks_panel": len(fit), "fit_zero_alarm_weeks": (per_week.loc[fit] == 0).sum(),
        "fit_alarms_p90": per_week.loc[fit].quantile(0.9), "fit_alarms_max": per_week.loc[fit].max(),
        "fit_active_median": active.median(), "fit_active_p90": active.quantile(0.9), "fit_active_max": active.max(),
        "alarms_by_week_2025": {iso(w): per_week[w] for w in validation},
        "val_alarms": panel.alarm.loc[validation].to_numpy().sum(),
        "episode_alarms": panel.alarm.loc[episode].to_numpy().sum(),
        "outside_alarms": panel.alarm.loc[outside].to_numpy().sum(), "outside_weeks": len(outside),
    }


# --- The burst ---------------------------------------------------------------------------

def burst_facts(panel: Panel, rows: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    peak, after = KNOWN[0], KNOWN[1]
    week = rows.loc[(rows["cluster_id"] == BURST) & (rows["week"] == peak)]
    companies = week["Company"].value_counts(normalize=True)
    product = week["Product canonical"].value_counts(normalize=True)
    issue = week["Issue canonical"].value_counts(normalize=True)
    fit = rows.loc[(rows["cluster_id"] == BURST) & rows["week"].isin(panel.split["fit"])]
    days = rows.loc[(rows["cluster_id"] == BURST) & rows["date"].between(peak, T("2025-01-26"))]
    m9 = predictions.set_index(["week", "cluster_id"])
    burst_week = predictions.loc[predictions["week"] == peak]
    return {
        "peak_total": panel.total[peak], "peak_p8": panel.observed.at[peak, BURST],
        "w20_p8": panel.observed.at[after, BURST],
        "p8_cal_mean": panel.observed.loc[panel.split["calibration"], BURST].mean(),
        "p8_m9_expected_0113": m9.at[(peak, BURST), "expected_mean"],
        "p8_m9_p975_0113": m9.at[(peak, BURST), "model_p975"],
        "below_range_0113": (burst_week["complaint_count"] < burst_week["model_p025"]).sum(),
        "p8_0113_block": companies.get("Block, Inc.", 0.0),
        "p8_0113_ews": companies.get("Early Warning Services, LLC", 0.0),
        "p8_0113_top_product": {"value": product.index[0], "share": product.iloc[0]},
        "p8_0113_top_issue": {"value": issue.index[0], "share": issue.iloc[0]},
        "p8_0113_unique_text": week["text"].nunique() / len(week),
        "p8_0113_top_text_count": week["text"].value_counts().iloc[0],
        "p8_fit_weekly_unique_median": fit.groupby("week")["text"].agg(lambda s: s.nunique() / len(s)).median(),
        "p8_by_day": {iso(d): n for d, n in days.groupby("date").size().items()},
    }


# --- The echo --------------------------------------------------------------------------------

def echo_facts(panel: Panel, rows: pd.DataFrame) -> dict[str, Any]:
    observed, expected, total = panel.observed, panel.expected, panel.total
    window = panel.weeks[panel.weeks.index(READ) - 4:panel.weeks.index(READ)]
    deviation = observed.loc[READ] - expected.loc[READ]
    credit = rows.groupby("week")["credit"].mean()
    credit_others = rows.loc[rows["cluster_id"] != BURST].groupby("week")["credit"].mean()
    fit, cal = panel.split["fit"], panel.split["calibration"]
    factors = {f"echo_factor_{iso(w)}": echo.echo_factor(observed.at[w, BURST] / total[w], expected.at[w, BURST] / total[w])
               for w in (KNOWN[2], READ)}
    return factors | {
        "p8_share_expected": {iso(w): expected.at[w, BURST] / total[w] for w in panel.between(T("2025-01-06"), T("2025-03-31"))},
        "p8_share_real": {iso(w): observed.at[w, BURST] / total[w] for w in panel.between(T("2025-01-06"), SHOWN[1])},
        "p8_r4_share_0203": (observed.loc[window, BURST].sum() + 1) / (total.loc[window].sum() + observed.shape[1]),
        "excess_0203": deviation.clip(lower=0).sum(),
        "p8_deficit_0203": -deviation[BURST],
        "credit_fit_p10": credit.loc[fit].quantile(0.1), "credit_fit_median": credit.loc[fit].median(),
        "credit_fit_p90": credit.loc[fit].quantile(0.9),
        "credit_no8_cal_min": credit_others.loc[cal].min(), "credit_no8_cal_max": credit_others.loc[cal].max(),
        "credit_no8": {iso(w): credit_others[w] for w in [T("2025-01-06")] + KNOWN},
    }


def alarms_in(alarms: pd.DataFrame, weeks: list[pd.Timestamp]) -> dict[str, list[str]]:
    """Patterns that alarm in some of `weeks`, with those weeks."""
    found = {c: [iso(w) for w in weeks if alarms.at[w, c]] for c in alarms.columns}
    return {str(c): w for c, w in found.items() if w}


def echo_free(panel: Panel, alpha: np.ndarray, k: float, h: float) -> dict[str, Any]:
    """The accumulator without the burst pattern, from the system state after 6 Jan, and its variants."""
    observed = panel.observed
    start = echo.state_after(panel.cusum, panel.alarm, T("2025-01-06"))
    later = [w for w in panel.weeks if w >= KNOWN[0]]
    expected = echo.expected_without(observed, [BURST])
    z = echo.scores(observed, expected, alpha)
    statistic, alarms = echo.accumulate(z.loc[later], start, k, h)
    _, from_start = echo.accumulate(z, pd.Series(0.0, index=z.columns), k, h)
    reference = panel.weeks[panel.weeks.index(KNOWN[0]) - 4:panel.weeks.index(KNOWN[0])]
    frozen = echo.frozen_expected(observed, [BURST], reference).loc[KNOWN]
    _, frozen_alarms = echo.accumulate(echo.scores(observed, frozen, alpha), start, k, h)
    second = echo.expected_without(observed, [BURST, CAPACITY_PATTERN]).loc[KNOWN]
    _, second_alarms = echo.accumulate(echo.scores(observed, second, alpha), start, k, h)
    return {
        "expected": expected, "z": z, "statistic": statistic, "alarms": alarms,
        "sensitivity": {"full_history": alarms_in(from_start, KNOWN), "frozen_pre_burst": alarms_in(frozen_alarms, KNOWN),
                        "without_8_and_14": alarms_in(second_alarms, KNOWN)},
    }


def verdict(c: int, free_alarms: list[str], sensitivity: dict[str, dict[str, list[str]]]) -> str:
    """Post hoc rule of the brief: the frozen accumulator without the burst pattern."""
    if c == BURST:
        return "incidente"
    if free_alarms:
        return "actuar"
    if any(str(c) in variant for variant in sensitivity.values()):
        return "vigilar"
    return "eco"


def alert_record(c: int, panel: Panel, predictions: pd.DataFrame, rows: pd.DataFrame,
                 free: dict[str, Any]) -> dict[str, Any]:
    m9 = predictions.set_index(["week", "cluster_id"])
    cal = panel.split["calibration"]
    record = {
        "observed_0203": panel.observed.at[READ, c], "m9_expected_0203": panel.expected.at[READ, c],
        "cusum_0203": panel.cusum.at[READ, c], "share_0203": panel.share.at[READ, c],
        "m10_expected_share_0203": panel.expected_share.at[READ, c],
        "m10_low_0203": panel.share_low.at[READ, c], "m10_high_0203": panel.share_high.at[READ, c],
        "level_0203": panel.level_change.at[READ, c], "level_percentile_0203": panel.level_percentile.at[READ, c],
        "system_alarms_known": [iso(w) for w in KNOWN if panel.alarm.at[w, c]],
        "cal_mean": panel.observed.loc[cal, c].mean(),
        "cal_share": panel.observed.loc[cal, c].sum() / panel.total.loc[cal].sum(),
        "credit_fraction_fit": rows.loc[(rows["cluster_id"] == c) & (rows["split"] == "fit"), "credit"].mean(),
        "path": [{"week": iso(w), "observed": panel.observed.at[w, c], "expected": panel.expected.at[w, c],
                  "p025": m9.at[(w, c), "model_p025"], "p975": m9.at[(w, c), "model_p975"],
                  "z": panel.z.at[w, c], "S": panel.cusum.at[w, c], "alarm": panel.alarm.at[w, c]}
                 for w in panel.between(*SHOWN)],
    }
    if c == BURST:
        return record
    expected, statistic, alarms = free["expected"], free["statistic"], free["alarms"]
    return record | {
        "free_expected": {iso(w): expected.at[w, c] for w in panel.between(*SHOWN)},
        "free_ratio": {iso(w): panel.observed.at[w, c] / expected.at[w, c] for w in KNOWN},
        "free_z": {iso(w): free["z"].at[w, c] for w in KNOWN},
        "free_alarms_known": [iso(w) for w in KNOWN if alarms.at[w, c]],
        "free_alarms_2025": [iso(w) for w in alarms.index if alarms.at[w, c]],
        "free_S_max": statistic.loc[KNOWN, c].max(), "free_S_0203": statistic.at[READ, c],
    }


def alert_facts(panel: Panel, predictions: pd.DataFrame, rows: pd.DataFrame, alpha: np.ndarray,
                thresholds: dict[str, float], k: float) -> dict[str, Any]:
    free = echo_free(panel, alpha, k, thresholds["cusum"])
    records = {str(c): alert_record(c, panel, predictions, rows, free) for c in ALERTS}
    alarms = free["alarms"]
    return {
        "alerts": records,
        "free_alarms_by_week": {iso(w): [c for c in alarms.columns if alarms.at[w, c]] for w in alarms.index},
        "sensitivity": free["sensitivity"],
        "verdicts": {c: verdict(int(c), r.get("free_alarms_known", []), free["sensitivity"]) for c, r in records.items()},
        "cusum_h": thresholds["cusum"],
    }


# --- The two real increases ------------------------------------------------------------------

def repeated_share(texts: pd.Series, at_least: int = 10) -> float:
    """Share of complaints whose exact text appears `at_least` times among them."""
    return texts.map(texts.value_counts()).ge(at_least).mean()


def segments(panel: Panel, c: int, parts: list[tuple[str, str]]) -> list[dict[str, Any]]:
    out = []
    for start, end in parts:
        values = panel.observed.loc[panel.between(T(start), T(end)), c]
        out.append({"from": start, "to": end, "mean": values.mean(), "min": values.min(), "max": values.max()})
    return out


def capacity_facts(panel: Panel, rows: pd.DataFrame, first_seen: pd.Series) -> dict[str, Any]:
    """«Disputas generales del reporte de crédito», the pattern whose capacity must change."""
    c = CAPACITY_PATTERN
    pattern = rows.loc[rows["cluster_id"] == c]
    half = rows["date"].dt.year.astype(str) + "H" + np.where(rows["date"].dt.month <= 6, "1", "2")
    sub_cal = pattern.loc[pattern["week"].isin(panel.split["calibration"]), "Sub-issue"] == SUB_ISSUE
    sub_known = pattern.loc[pattern["week"].isin(KNOWN), "Sub-issue"] == SUB_ISSUE
    late = pattern.loc[pattern["week"] >= T("2025-05-19"), "text"].value_counts()
    return {
        "p14_share_by_half": (rows["cluster_id"] == c).groupby(half).mean().to_dict(),
        "p14_sub": {"value": SUB_ISSUE, "cal_weekly": sub_cal.sum() / len(panel.split["calibration"]),
                    "known_weekly": sub_known.sum() / len(KNOWN), "cal_share": sub_cal.mean(),
                    "known_share": sub_known.mean()},
        "p14_rep10_known": repeated_share(pattern.loc[pattern["week"].isin(KNOWN), "text"]),
        "p14_segments": segments(panel, c, [("2025-02-10", "2025-05-12"), ("2025-05-19", "2025-06-23")]),
        "p14_late_top_text": {"count": late.iloc[0], "first_seen": iso(first_seen[late.index[0]])},
        "p14_0106": panel.observed.at[T("2025-01-06"), c],
    }


def letter_facts(panel: Panel, rows: pd.DataFrame, first_seen: pd.Series) -> dict[str, Any]:
    """The template letters of «Disputas legales variadas citando la FCRA», first seen on 22 Jan 2025."""
    c = LETTER_PATTERN
    pattern = rows.loc[rows["cluster_id"] == c]
    known = pattern.loc[pattern["date"] <= T("2025-02-09")]
    copies = known.loc[known["text"].map(first_seen) >= KNOWN[0], "text"].value_counts()
    letters = [t for t in copies.loc[copies >= 100].index if first_seen[t] == LETTER_DAY]
    weekly = pattern.loc[pattern["text"].isin(letters)].groupby("week").size()
    cal_mean = panel.observed.loc[panel.split["calibration"], c].mean()
    since = KNOWN[1:]
    increase = sum(panel.observed.at[w, c] - cal_mean for w in since)
    fit = pattern.loc[pattern["split"] == "fit", "text"]
    week = pattern.loc[pattern["week"] == READ, "text"]
    letter_rows = known.loc[known["text"].isin(letters)]
    due = due_without_holidays(letter_rows["date"], 15)
    return {
        "p3_letters": [{"first_seen": iso(first_seen[t]), "copies_by_0209": copies[t],
                        "sub_issue": pattern.loc[pattern["text"] == t, "Sub-issue"].mode().iloc[0]} for t in letters],
        "p3_letters_weekly": {iso(w): n for w, n in weekly.items()},
        "p3_letters_copies_by_0209": copies[letters].sum(),
        "p3_letters_in_increase": sum(weekly.get(w, 0) for w in since) / increase,
        "p3_letters_share_0203": weekly.get(READ, 0) / panel.observed.at[READ, c],
        "p3_top10": {iso(w): top_texts_share(pattern.loc[pattern["week"] == w, "text"])
                     for w in (T("2025-01-06"), READ)},
        "p3_dominant_text_fit": fit.value_counts().iloc[0] / len(fit),
        "p3_top_text_share_0203": week.value_counts().iloc[0] / len(week),
        "p3_segments": segments(panel, c, [("2025-02-10", "2025-05-12"), ("2025-05-19", "2025-06-23")]),
        "p3_share_no8": {iso(w): panel.observed.at[w, c] / (panel.total[w] - panel.observed.at[w, BURST])
                         for w in [T("2025-01-06")] + KNOWN},
        "letters_rows": len(letter_rows),
        "letters_due_this_week": ((due >= READING_DAY) & (due <= WEEK_END)).sum(),
        "letters_due_first": due.min(),
    }


def top_texts_share(texts: pd.Series, top: int = 10) -> float:
    return texts.value_counts().head(top).sum() / len(texts)


def level_facts(panel: Panel) -> dict[str, Any]:
    """The 3 Feb level window as M12 computes it: 13 panel rows back, and the panel has no split week of 30 Dec."""
    observed, total = panel.observed, panel.total
    reference = [panel.weeks[panel.weeks.index(w) - 13] for w in KNOWN]
    others = total - observed[BURST]

    def share_without_burst(c: int, weeks: list[pd.Timestamp]) -> float:
        return observed.loc[weeks, c].sum() / others.loc[weeks].sum()

    return {
        "level_windows": {"recent": [iso(w) for w in KNOWN], "reference": [iso(w) for w in reference],
                          "p8_recent": observed.loc[KNOWN, BURST].sum() / total.loc[KNOWN].sum(),
                          "p8_reference": observed.loc[reference, BURST].sum() / total.loc[reference].sum()},
        "level_no8": {str(c): share_without_burst(c, KNOWN) / share_without_burst(c, reference) - 1
                      for c in ALERTS if c != BURST},
    }


# --- Deadlines, timely responses and growth -----------------------------------------------------

def due_without_holidays(days: pd.Series, business_days: int) -> np.ndarray:
    """The rule of src/triage/deadline.py without holidays: the 2025 list is not loaded."""
    return np.busday_offset(days.to_numpy().astype("datetime64[D]"), business_days, roll="backward")


def deadline_facts(rows: pd.DataFrame) -> dict[str, Any]:
    burst = rows.loc[(rows["cluster_id"] == BURST) & rows["week"].isin(KNOWN[:2])]
    due = due_without_holidays(burst["date"], 15)
    card_day = pd.Series([pd.Timestamp(READING_DAY)])
    calibration = rows.loc[(rows["split"] == "calibration") & (rows["cluster_id"] == CAPACITY_PATTERN)]
    return {
        "burst_due_before_reading": (due < READING_DAY).sum(),
        "burst_due_this_week": ((due >= READING_DAY) & (due <= WEEK_END)).sum(),
        "card_due_2025": due_without_holidays(card_day, 15)[0],
        "card_due_2025_extended": due_without_holidays(card_day, 45)[0],
        "timely_cal_p14": (calibration["timely"] == "Yes").mean(),
        "timely_burst_p8": (burst["timely"] == "Yes").mean(),
    }


def growth_facts(panel: Panel) -> dict[str, Any]:
    """Echo patterns that alarmed later: growth from Oct-Dec 2024 to 10 Feb-7 Apr 2025, against the total."""
    cal, spring = panel.split["calibration"], panel.between(T("2025-02-10"), T("2025-04-07"))
    observed, total = panel.observed, panel.total
    return {
        "growth_total_spring": total.loc[spring].mean() / total.loc[cal].mean() - 1,
        "growth_spring": {str(c): observed.loc[spring, c].mean() / observed.loc[cal, c].mean() - 1
                          for c in (13, 25, 36, 11)},
    }


# --- Reliability ------------------------------------------------------------------------------------

def coverage(part: pd.DataFrame) -> dict[str, float]:
    y = part["complaint_count"]
    return {"in95": ((y >= part["model_p025"]) & (y <= part["model_p975"])).mean(),
            "in80": ((y >= part["model_p10"]) & (y <= part["model_p90"])).mean(),
            "above95": (y > part["model_p975"]).mean()}


def reliability_facts(panel: Panel, predictions: pd.DataFrame) -> dict[str, Any]:
    validation = predictions.loc[predictions["split"] == "validation"]
    episode = validation["week"].between(*COVERAGE_EPISODE)
    history = panel.history
    inside = (history["share"] >= history["share_low"]) & (history["share"] <= history["share_high"])
    in_episode = history["week"].between(*COVERAGE_EPISODE)
    m11, m11v = read_report("persistent_change/results.json"), read_report("persistent_change/validation_results.json")
    results = read_report("representation_results.json")["results"]
    final = read_report("final_confirmation.json")["classifiers"]
    faiss = read_report("semantic_space_results.json")["faiss"]["metrics"]["validation"]
    return {
        "m9_coverage": {"calibration": coverage(predictions.loc[predictions["split"] == "calibration"]),
                        "validation": coverage(validation), "validation_outside": coverage(validation.loc[~episode]),
                        "validation_episode": coverage(validation.loc[episode])},
        "m10_coverage": {"calibration": inside[history["split"] == "calibration"].mean(),
                         "validation": inside[history["split"] == "validation"].mean(),
                         "validation_outside": inside[(history["split"] == "validation") & ~in_episode].mean()},
        "m11_detection": m11["detection"],
        "m11_alarms_per_month": {"fit": m11["real_alarms"]["fit"]["cusum_per_month"],
                                 "calibration": m11["real_alarms"]["calibration"]["cusum_per_month"],
                                 "validation": m11v["real_alarms"]["validation"]["cusum_per_month"]},
        "weeks_per_month": WEEKS_PER_MONTH, "z_cap": norm.ppf(1 - PIT_CLIP), "weekly_rule_z": m11["thresholds"]["weekly"],
        "t1": t1_metrics(results, final), "t2": binary_metrics("T2", results, final),
        "t3": binary_metrics("T3", results, final),
        "faiss": {"nearest_similarity": {"mediana": faiss["top1_similarity_median"], "p5": faiss["top1_similarity_p05"],
                                         "p95": faiss["top1_similarity_p95"]},
                  "same_product": faiss["top1_same_product_rate"], "same_issue": faiss["top1_same_t1_rate"]},
    }


def t1_metrics(results: dict[str, Any], final: dict[str, Any]) -> dict[str, float]:
    calibration, validation = results["T1"]["metrics"], final["T1"]["metrics"]["no_shared_text"]
    return {"cal_top3": calibration["bge_product"]["top_3_accuracy"],
            "cal_rule_top3": calibration["product_rule"]["top_3_accuracy"],
            "val_top3": validation["model"]["top_3_accuracy"], "val_rule_top3": validation["product_rule"]["top_3_accuracy"],
            "val_macro_f1": validation["model"]["macro_f1"], "val_rule_macro_f1": validation["product_rule"]["macro_f1"]}


def binary_metrics(target: str, results: dict[str, Any], final: dict[str, Any]) -> dict[str, float]:
    calibration = results[target]["metrics"]["tfidf_text"]
    validation = final[target]["metrics"]["no_shared_text"]
    return {
        "threshold": calibration["threshold"], "cal_precision": calibration["precision"],
        "cal_recall": calibration["recall"], "cal_ap": calibration["average_precision"],
        "cal_base": calibration["positive"] / calibration["eligible"],
        "cal_rule_precision": results[target]["metrics"]["product_rule"]["precision"],
        "val_precision": validation["model"]["precision"], "val_recall": validation["model"]["recall"],
        "val_ap": validation["model"]["average_precision"],
        "val_base": validation["model"]["positive"] / validation["model"]["eligible"],
        "val_rule_ap": validation["product_rule"]["average_precision"],
    }


# --- Blind spots, families and examples ----------------------------------------------------------

def text_facts(panel: Panel, rows: pd.DataFrame, first_seen: pd.Series, families: list[dict[str, Any]]) -> dict[str, Any]:
    design = panel.history.loc[panel.history["split"].isin(["fit", "calibration"])]
    known = rows.loc[rows["date"] <= T("2025-02-09")]
    novel = rows.loc[(rows["split"] == "validation") & rows["is_novel"]]
    novel_known = novel.loc[novel["date"] <= T("2025-02-09")]
    family_of = {c: f["familia"] for f in families for c in f["patrones"]}
    fit = rows.loc[rows["split"] == "fit"]
    # The FCRA privacy letter is the most repeated text of pattern 2 in fit.
    letter = fit.loc[fit["cluster_id"] == 2, "text"].value_counts().index[0]
    letter_patterns = fit.loc[fit["text"] == letter, "cluster_id"].unique()
    return {
        "active_rows_design": design["active_alert"].sum(),
        "active_below_design": (design["active_alert"] & (design["observed"] < design["expected"])).sum(),
        "multi_pattern_text_share": patterns_per_text(rows).ge(2).mean(),
        "multi_pattern_share_known": patterns_per_text(known).ge(2).mean(),
        "novel_2025": len(novel), "novel_2025_seen_before": (novel["text"].map(first_seen) < novel["date"]).sum(),
        "novel_known": len(novel_known),
        "novel_known_seen_before": (novel_known["text"].map(first_seen) < novel_known["date"]).sum(),
        "privacy_patterns": len(letter_patterns), "privacy_families": len({family_of[c] for c in letter_patterns}),
        "fit_repeated_text_share": fit["text"].map(fit["text"].value_counts()).ge(2).mean(),
        "family_share": {s: rows.loc[rows["split"] == s, "cluster_id"].map(family_of).value_counts(normalize=True).to_dict()
                         for s in ("fit", "calibration", "validation")},
    }


def patterns_per_text(rows: pd.DataFrame) -> pd.Series:
    """For every complaint, the number of patterns that its exact text falls in."""
    return rows["text"].map(rows.groupby("text")["cluster_id"].nunique())


def accumulator_examples(panel: Panel, weekly_rule: float) -> list[dict[str, Any]]:
    """CUSUM alarms of fit and calibration reached in 3 or more weeks, none above the weekly rule."""
    examples = []
    design = set(panel.split["fit"] + panel.split["calibration"])
    for c in panel.observed.columns:
        for i, week in enumerate(panel.weeks):
            if week not in design or not panel.alarm.at[week, c]:
                continue
            j = i
            while j > 0 and panel.cusum.at[panel.weeks[j - 1], c] > 0:
                j -= 1
            run = panel.weeks[j:i + 1]
            if len(run) >= 3 and all(panel.z.at[w, c] < weekly_rule for w in run):
                split = "fit" if week in panel.split["fit"] else "calibration"
                examples.append({"cluster_id": c, "split": split, "alarm_week": iso(week),
                                 "run": [{"week": iso(w), "z": panel.z.at[w, c], "S": panel.cusum.at[w, c],
                                          "observed": panel.observed.at[w, c], "expected": panel.expected.at[w, c]}
                                         for w in run]})
    return examples


def card_facts(rows: pd.DataFrame) -> dict[str, Any]:
    """The «novel» demo card: how often its exact text had arrived before."""
    cards = read_report("triage_demo/cards.json")["cards"]
    card = next(c for c in cards if c["action"]["situation"] == "novel")
    text = rows.loc[rows[ID] == str(card["complaint_id"]), "text"].iloc[0]
    same = rows.loc[rows["text"] == text, "date"]
    received = pd.Timestamp(card["received"])
    return {"novel_card_text": {"copies_before": (same < received).sum(), "first_seen": iso(same.min())}}


# --- Assembly -------------------------------------------------------------------------------------

def build(rows: pd.DataFrame, history: pd.DataFrame, alpha: np.ndarray) -> dict[str, Any]:
    panel = Panel(history)
    predictions = pd.read_csv(M9_REPORT / "predictions.csv", parse_dates=["week"])
    m11 = read_report("persistent_change/results.json")
    families = json.loads((BRIEF / "data/pattern_families.json").read_text(encoding="utf-8"))
    first_seen = rows.groupby("text")["date"].min()
    facts = (
        totals_facts(panel, rows) | alarm_facts(panel) | burst_facts(panel, rows, predictions)
        | echo_facts(panel, rows)
        | alert_facts(panel, predictions, rows, alpha, m11["thresholds"], m11["settings"]["cusum_k"])
        | capacity_facts(panel, rows, first_seen) | letter_facts(panel, rows, first_seen) | level_facts(panel)
        | deadline_facts(rows) | growth_facts(panel) | reliability_facts(panel, predictions)
        | text_facts(panel, rows, first_seen, families)
        | {"accumulator_examples": accumulator_examples(panel, m11["thresholds"]["weekly"])} | card_facts(rows)
    )
    if sorted(c for c in panel.observed.columns if panel.active.at[READ, c]) != sorted(ALERTS):
        raise ValueError("The 3 Feb panel no longer has the 11 alerts of the brief.")
    return plain(facts)


def main() -> None:
    args = parse_args()
    config = load_experiment_config()
    if args.export:
        export(args.export, config)
        return
    if not (args.rows and args.history):
        raise SystemExit("Give --export DIR, or --rows and --history.")
    facts = build(load_rows(args.rows), pd.read_parquet(args.history),
                  alpha_draws(args.posterior, config["experiment"]["seed"]))
    args.output.write_text(json.dumps(facts, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {args.output} with {len(facts)} facts.")


if __name__ == "__main__":
    main()
