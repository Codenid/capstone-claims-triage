"""M12 demo: the weekly panel and one complaint card per situation (§25).

Everything comes from frozen artifacts on Khipu. The pages are only written
after all of them are built, so a failure leaves no partial demo.
"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    load_experiment_config,
    save_run_record,
    set_seed,
)
from src.evaluation.final_confirmation import load_split
from src.models.persistent_change import DESIGN_FILES
from src.models.representation_comparison import (
    check_new_outputs,
    load_embeddings,
    read_json,
    write_json,
)
from src.models.semantic_space import ID_COLUMN
from src.triage import render
from src.triage.actions import ACTIONS, situation
from src.triage.complaints import (
    classifier_signals,
    complaint_card,
    neighbor_index,
    pattern_assignments,
    received_dates,
    similar_complaints,
    with_dates,
)
from src.triage.deadline import due_date, load_holidays
from src.triage.patterns import (
    last_closed_week,
    pattern_history,
    records,
    weekly_panel,
)


def deadline_info(settings: dict[str, Any]) -> dict[str, Any]:
    """Due dates as if the complaint were registered on the demo date (§25)."""
    holidays = load_holidays()
    registered = date.fromisoformat(settings["simulated_registration"])
    return {
        "registered": registered.isoformat(),
        "simulated": True,
        "business_days": settings["response_days"],
        "due": due_date(registered, settings["response_days"], holidays).isoformat(),
        "extended_business_days": settings["extended_days"],
        "due_extended": due_date(
            registered, settings["extended_days"], holidays
        ).isoformat(),
    }


def demo_complaints(rows: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    """The complaint with the smallest ID in each situation (§25)."""
    flags = status.set_index("cluster_id")
    situations = [
        situation(
            bool(novel),
            bool(flags.at[cluster, "active_alert"]),
            bool(flags.at[cluster, "template_dominated"]),
        )
        for novel, cluster in zip(rows["is_novel"], rows["cluster_id"])
    ]
    ordered = rows.assign(
        situation=situations, order=pd.to_numeric(rows[ID_COLUMN])
    ).sort_values("order")
    return ordered.groupby("situation", sort=False).head(1).drop(columns="order")


def level_chart_for(history: pd.DataFrame, cluster: int, week: pd.Timestamp) -> str:
    levels = history.loc[(history["cluster_id"] == cluster) & (history["week"] <= week)]
    latest = levels.iloc[-1]
    return render.level_chart(
        levels,
        float(latest["level_low"]),
        float(latest["level_high"]),
        f"Patrón {cluster}: nivel frente a hace 3 meses",
    )


def demo_record(
    config: dict[str, Any],
    panel: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    artifacts: list[Path],
) -> dict[str, Any]:
    settings = config["triage"]
    record = build_run_record(
        config=config,
        run_name="m12-triage-demo",
        stage="M12",
        target="triage",
        split="validation",
        view="demo",
        features=["narrative", "product", "weekly_counts"],
        parameters={
            "rule": "models_plan.md §25",
            "settings": json.dumps(settings, sort_keys=True),
            "situations": ",".join(card["action"]["situation"] for card in cards),
        },
        metrics={
            "patterns": float(len(panel)),
            "active_alerts": float(sum(item["active_alert"] for item in panel)),
            "cards": float(len(cards)),
        },
        artifacts=[path.relative_to(PROJECT_ROOT).as_posix() for path in artifacts],
    )
    record["tags"].update(
        {"run_mode": "full", "validation_used_for_selection": "false"}
    )
    return record


def main() -> None:
    config = load_experiment_config()
    seed = config["experiment"]["seed"]
    set_seed(seed)
    settings = config["triage"]
    output_dir = PROJECT_ROOT / config["paths"]["triage_reports"]
    record_path = PROJECT_ROOT / config["paths"]["offline_runs"] / "triage_demo"
    record_path = record_path / "run.json"
    check_new_outputs([output_dir, record_path])

    history = pattern_history(config, seed)
    week = pd.to_datetime(settings["demo_week"])
    panel = weekly_panel(history, week)
    design_dir = PROJECT_ROOT / config["paths"]["persistent_change_reports"]
    threshold = read_json(design_dir / DESIGN_FILES["results"])["thresholds"]["cusum"]
    charts = {
        item["cluster_id"]: level_chart_for(history, item["cluster_id"], week)
        for item in panel
        if item["active_alert"]
    }

    # Complaints of the next week see the status of this closed week.
    next_week = week + pd.Timedelta(weeks=1)
    if last_closed_week(history, next_week) != week:
        raise ValueError(f"{next_week:%Y-%m-%d} does not follow a closed week.")
    frames = load_split(config, "validation")
    validation = frames["validation"]
    received = validation.loc[validation["week"] == next_week].reset_index(drop=True)
    assigned = pattern_assignments(received, config, "validation")
    received = received.join(assigned.drop(columns=ID_COLUMN))
    status = history.loc[history["week"] == week]
    embeddings_dir = PROJECT_ROOT / config["paths"]["bge_full_artifacts"]
    manifest = embeddings_dir / "manifest.parquet"
    chosen = demo_complaints(received, status).reset_index(drop=True)
    chosen = with_dates(chosen, received_dates(manifest, "validation"))

    signals = classifier_signals(chosen, config)
    embeddings = load_embeddings(embeddings_dir, {"validation": chosen})["validation"]
    index, ids = neighbor_index(config)
    fit = with_dates(frames["fit"], received_dates(manifest, "fit"))
    similar = similar_complaints(index, ids, embeddings, fit, settings["neighbors"])
    deadline = deadline_info(settings)
    by_pattern = {item["cluster_id"]: item for item in records(status)}
    cards = [
        complaint_card(
            row,
            signal,
            near,
            by_pattern[int(row["cluster_id"])],
            deadline,
            {"situation": row["situation"], "text": ACTIONS[row["situation"]]},
            settings["text_characters"],
        )
        for (_, row), signal, near in zip(chosen.iterrows(), signals, similar)
    ]

    week_text = f"{week:%Y-%m-%d}"
    pages = {"panel.html": render.panel_page(week_text, panel, charts, threshold)}
    for card in cards:
        chart = level_chart_for(history, card["pattern"]["cluster_id"], week)
        name = f"ficha_{card['action']['situation']}.html"
        pages[name] = render.card_page(card, chart)
    pages["index.html"] = render.index_page(week_text, cards)
    documents = {
        "panel.json": {
            "week": week_text,
            "cusum_threshold": threshold,
            "patterns": panel,
        },
        "cards.json": {"week": f"{next_week:%Y-%m-%d}", "cards": cards},
    }
    paths = [output_dir / name for name in [*documents, *pages]]
    record = demo_record(config, panel, cards, paths)
    # Serialize before writing anything, so a failure leaves no partial demo.
    json.dumps([documents, record])

    output_dir.mkdir(parents=True)
    for name, document in documents.items():
        write_json(output_dir / name, document)
    for name, text in pages.items():
        (output_dir / name).write_text(text, encoding="utf-8")
    save_run_record(record, record_path)
    found = ", ".join(card["action"]["situation"] for card in cards)
    print(f"Panel {week_text}: {record['metrics']['active_alerts']:.0f} active alerts.")
    print(f"Cards for {next_week:%Y-%m-%d}: {found}. Output: {output_dir}")


if __name__ == "__main__":
    main()
