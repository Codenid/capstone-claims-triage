"""Catalogue of the 40 semantic patterns: what each one is about, and its series.

For every pattern of M7 it gathers what the frozen artifacts already know
(size, dominant product, representative terms, text reuse), measures its
weekly volume per period, counts its M11 alerts, picks three public CFPB
narratives closest to its centre as examples, and draws its weekly series
with the predictive band of the current M9 model. Nothing is refitted.

Outputs under reports/modeling/patterns: catalog.json, series/cluster_NN.png.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

from src.evaluation.experiment import PROJECT_ROOT, git_commit, load_experiment_config
from src.models.bge_sample import ID_COLUMN

SPLITS = ("fit", "calibration", "validation")
SPLIT_LABELS = {"fit": "ajuste", "calibration": "calibración", "validation": "2025-H1"}
NARRATIVE = "Consumer complaint narrative"
HASH = "Consumer complaint narrative SHA-256"
BANDS = (
    ("p025", "p975", 0.12, "95 %"),
    ("p10", "p90", 0.2, "80 %"),
    ("p25", "p75", 0.3, "50 %"),
)


def weekly_summary(counts: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """Mean weekly count and share per pattern and period, complete weeks only."""
    complete = counts.loc[counts["is_complete_week"]]
    summary: dict[int, dict[str, Any]] = {}
    for cluster, rows in complete.groupby("cluster_id"):
        entry: dict[str, Any] = {}
        for split in SPLITS:
            part = rows.loc[rows["split"] == split]
            if part.empty:
                continue
            entry[split] = {
                "weeks": int(len(part)),
                "mean_count": float(part["complaint_count"].mean()),
                "mean_share": float(part["proportion"].mean()),
                "novel_fraction": float(
                    part["novel_count"].sum() / max(part["complaint_count"].sum(), 1)
                ),
            }
        summary[int(cluster)] = entry
    return summary


def alert_summary(alerts: pd.DataFrame) -> dict[int, dict[str, list[str]]]:
    """Weeks where the chosen rule (CUSUM) fired, per pattern and period."""
    fired = alerts.loc[alerts["cusum_alarm"].astype(bool)]
    summary: dict[int, dict[str, list[str]]] = {}
    for (cluster, split), rows in fired.groupby(["cluster_id", "split"]):
        weeks = sorted(pd.to_datetime(rows["week"]).dt.date.astype(str))
        summary.setdefault(int(cluster), {})[str(split)] = weeks
    return summary


def choose_examples(
    assignments: pd.DataFrame,
    cluster: int,
    examples: int,
) -> list[str]:
    """IDs of the complaints closest to the centre, with distinct normalized text."""
    rows = assignments.loc[assignments["cluster_id"] == cluster]
    rows = rows.sort_values("distance").drop_duplicates(HASH)
    return rows[ID_COLUMN].astype(str).head(examples).tolist()


def snippet(text: str, characters: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= characters else text[: characters - 1] + "…"


def plot_series(
    cluster: int,
    counts: pd.DataFrame,
    predictions: pd.DataFrame,
    alert_weeks: dict[str, list[str]],
    title: str,
    path: Path,
) -> None:
    observed = counts.loc[counts["cluster_id"] == cluster].sort_values("week")
    expected = predictions.loc[predictions["cluster_id"] == cluster].sort_values("week")
    weeks = pd.to_datetime(observed["week"])
    figure, axis = plt.subplots(figsize=(11, 3.6), dpi=110)
    if not expected.empty:
        band_weeks = pd.to_datetime(expected["week"])
        for low, high, alpha, label in BANDS:
            axis.fill_between(
                band_weeks,
                expected[f"expected_{low}"],
                expected[f"expected_{high}"],
                color="#4C72B0",
                alpha=alpha,
                linewidth=0,
                label=f"M9 intervalo {label}",
            )
        axis.plot(band_weeks, expected["expected_p50"], color="#4C72B0", linewidth=1)
    axis.plot(
        weeks,
        observed["complaint_count"],
        color="#222222",
        linewidth=1.4,
        label="observado",
    )
    flagged = {week for weeks_ in alert_weeks.values() for week in weeks_}
    marks = observed.loc[observed["week"].astype(str).str[:10].isin(flagged)]
    if not marks.empty:
        axis.scatter(
            pd.to_datetime(marks["week"]),
            marks["complaint_count"],
            color="#C44E52",
            s=28,
            zorder=5,
            label="alerta CUSUM (M11)",
        )
    for split, color in (("calibration", "#999999"), ("validation", "#999999")):
        starts = pd.to_datetime(observed.loc[observed["split"] == split, "week"])
        if not starts.empty:
            axis.axvline(starts.min(), color=color, linewidth=0.8, linestyle="--")
    axis.set_title(title, fontsize=10, loc="left")
    axis.set_ylabel("reclamos por semana")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#e5e5e5", linewidth=0.6)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.legend(loc="upper left", fontsize=7, frameon=False, ncol=3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    change = config["persistent_change"]
    m9_dir = (
        PROJECT_ROOT
        / "reports/modeling/weekly_counts"
        / change["m9_model"]
        / change["m9_run_key"]
    )
    alert_dir = PROJECT_ROOT / change["report_dir"]
    alerts = pd.concat(
        [
            pd.read_csv(alert_dir / "alerts.csv"),
            pd.read_csv(alert_dir / "validation_alerts.csv"),
        ],
        ignore_index=True,
    )
    patterns_dir = PROJECT_ROOT / config["paths"]["weekly_patterns_artifacts"]
    assignments = pd.read_parquet(
        patterns_dir / "fit_assignments.parquet",
        columns=[ID_COLUMN, "cluster_id", "distance", HASH, "is_novel"],
    )
    return {
        "counts": pd.read_csv(PROJECT_ROOT / config["paths"]["weekly_counts"]),
        "predictions": pd.read_csv(m9_dir / "predictions.csv"),
        "alerts": alerts,
        "summary": pd.read_csv(PROJECT_ROOT / config["paths"]["cluster_summary"]),
        "assignments": assignments.loc[~assignments["is_novel"].astype(bool)],
        "m9_run": f"{change['m9_model']}/{change['m9_run_key']}",
    }


def narratives(ids: list[str]) -> pd.DataFrame:
    """Date, product and narrative of the chosen complaints, read once."""
    table = pq.read_table(
        PROJECT_ROOT / "data/processed/prepared.parquet",
        columns=[ID_COLUMN, "Date received", "Product canonical", NARRATIVE],
        filters=[(ID_COLUMN, "in", ids)],
    )
    return table.to_pandas().set_index(ID_COLUMN)


def build_catalog(
    inputs: dict[str, Any],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    weekly = weekly_summary(inputs["counts"])
    alerts = alert_summary(inputs["alerts"])
    chosen = {
        int(row.cluster_id): choose_examples(
            inputs["assignments"], int(row.cluster_id), settings["examples"]
        )
        for row in inputs["summary"].itertuples()
    }
    texts = narratives([item for ids in chosen.values() for item in ids])
    report_dir = PROJECT_ROOT / settings["report_dir"]
    catalog = []
    for row in inputs["summary"].sort_values("cluster_id").itertuples():
        cluster = int(row.cluster_id)
        examples = []
        for complaint in chosen[cluster]:
            if complaint not in texts.index:
                continue
            record = texts.loc[complaint]
            examples.append(
                {
                    "complaint_id": complaint,
                    "received": str(pd.Timestamp(record["Date received"]).date()),
                    "product": str(record["Product canonical"]),
                    "snippet": snippet(
                        record[NARRATIVE], settings["snippet_characters"]
                    ),
                }
            )
        title = (
            f"Patrón {cluster} · {row.main_product} "
            f"({row.main_product_fraction:.0%}) · {row.fit_rows:,} reclamos en ajuste"
        )
        image = Path("series") / f"cluster_{cluster:02d}.png"
        plot_series(
            cluster,
            inputs["counts"],
            inputs["predictions"],
            alerts.get(cluster, {}),
            title,
            report_dir / image,
        )
        catalog.append(
            {
                "cluster_id": cluster,
                "fit_rows": int(row.fit_rows),
                "main_product": str(row.main_product),
                "main_product_fraction": float(row.main_product_fraction),
                "unique_text_fraction": float(row.unique_text_fraction),
                "dominant_template_fraction": float(row.dominant_template_fraction),
                "p95_distance": float(row.p95_distance),
                "terms": [
                    term.strip() for term in str(row.representative_terms).split(",")
                ],
                "weekly": weekly.get(cluster, {}),
                "alert_weeks": alerts.get(cluster, {}),
                "examples": examples,
                "image": str(image).replace("\\", "/"),
            }
        )
    return catalog


def main() -> None:
    started = time.perf_counter()
    config = load_experiment_config()
    settings = config["pattern_catalog"]
    inputs = load_inputs(config)
    catalog = build_catalog(inputs, settings)
    report_dir = PROJECT_ROOT / settings["report_dir"]
    report = {
        "stage": "M8C",
        "git_commit": git_commit(),
        "m9_run": inputs["m9_run"],
        "examples_per_pattern": settings["examples"],
        "snippet_characters": settings["snippet_characters"],
        "patterns": catalog,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (report_dir / "catalog.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{len(catalog)} patterns -> {report_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
