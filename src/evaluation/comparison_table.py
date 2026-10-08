"""The baseline-versus-candidates table the course asks for (PB-18).

Every row comes from a result JSON under reports/modeling: nothing is typed
by hand. Calibration rows are the selection evidence (2024-10 to 2024-12);
validation rows are the single 2025-H1 report of each winner, consulted once.
The table goes to etapa_4_5_modelado_evaluacion/resultados/
comparacion_modelos.csv, the one results table the course keeps in Git.

Columns: etapa, tarea, modelo, rol, conjunto, vista, metrica, valor,
referencia, ganancia_relativa, ic95_inferior, ic95_superior, decision, fuente.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.evaluation.experiment import PROJECT_ROOT

OUTPUT = Path("etapa_4_5_modelado_evaluacion/resultados/comparacion_modelos.csv")
COLUMNS = [
    "etapa",
    "tarea",
    "modelo",
    "rol",
    "conjunto",
    "vista",
    "metrica",
    "valor",
    "referencia",
    "ganancia_relativa",
    "ic95_inferior",
    "ic95_superior",
    "decision",
    "fuente",
]
REPRESENTATION = "reports/modeling/representation_results.json"
BASELINES = "reports/modeling/baseline_results.json"
FOUNDATION = "reports/modeling/foundation_t1/results.json"
FOUNDATION_VALIDATION = "reports/modeling/foundation_t1/validation_results.json"
ABLATION = "reports/modeling/foundation_t1/ablation_results.json"
TABPFN_LABEL = "TabPFN-3.5 (PCA 100 + producto)"
ABLATION_LABELS = {
    "tabpfn": TABPFN_LABEL,
    "tabpfn_pca256": "TabPFN-3.5 (PCA 256 + producto)",
    "tabpfn_bge1024": "TabPFN-3.5 (BGE 1,024 + producto)",
}
FINAL = "reports/modeling/final_confirmation.json"
COUNTS = "reports/modeling/weekly_counts"
COMPOSITION = "reports/modeling/weekly_composition"
CHALLENGES = "reports/modeling/weekly_counts/challenges"
PERSISTENT = "reports/modeling/persistent_change_c_a/results.json"
PERSISTENT_VALIDATION = "reports/modeling/persistent_change_c_a/validation_results.json"
SPACE = "reports/modeling/space_sensitivity/results.json"
DAILY = "reports/modeling/daily_counts"
DAILY_CHALLENGE = "reports/modeling/daily_counts/challenge"
DAILY_FAMILIES = {
    "nb_daily_hierarchical_v1": "D-A: binomial negativa diaria con día de semana",
    "nb_daily_no_dow_v1": "D-B2: binomial negativa diaria sin día de semana",
    "nb_daily_fourier_v1": "D-C: binomial negativa diaria, Fourier y tendencia",
    "zinb_daily_hierarchical_v1": "D-D: binomial negativa diaria, inflación de ceros",
    "nb_daily_state_space_v1": "D-B: espacio de estados diario",
    "dirichlet_multinomial_daily_v1": "D-E: Dirichlet-multinomial diaria",
}
VALIDATION = "validacion_2025_h1 (ya consultada)"
CALIBRATION = "calibracion_2024_q4"
# Models whose record names a stage; the rest of the catalogue is in modelos.md.
COUNT_FAMILIES = {
    "poisson_static_pymc_v1": "Poisson estática (PyMC)",
    "nb_static_global_v3": "Binomial negativa estática",
    "nb_rolling_4_global_v1": "Binomial negativa, participaciones de 4 semanas, global",
    "nb_rolling_4_hierarchical_v1": "Binomial negativa, 4 semanas, jerárquica v1",
    "nb_rolling_4_hierarchical_v2": "Binomial negativa, 4 semanas, jerárquica v2",
    "nb_rolling_4_hierarchical_v3": "Binomial negativa, 4 semanas, jerárquica v3",
    "nb_discounted_hierarchical_v1": (
        "Binomial negativa jerárquica con memoria que decae (C-A)"
    ),
    "nb_state_space_v1": "Espacio de estados, paseo aleatorio (C-B)",
    "chronos2_zero_shot_v1": "Chronos-2 zero-shot",
    "timesfm3_zero_shot_v1": "TimesFM 3.0 zero-shot",
}


def load(path: str | Path) -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))


def row(**values: Any) -> dict[str, Any]:
    record = {column: "" for column in COLUMNS}
    record.update(values)
    for key in ("valor", "ganancia_relativa", "ic95_inferior", "ic95_superior"):
        if record[key] != "":
            record[key] = f"{float(record[key]):.6f}"
    return record


def interval(difference: dict[str, Any], label: str, keep: bool) -> Any:
    """One bound of a bootstrap interval, or blank for a baseline row."""
    return difference[label] if keep else ""


def verdict(wins: bool | None) -> str:
    """The §25.2 outcome of a candidate; blank when the run only reports."""
    if wins is None:
        return ""
    return "reemplaza" if wins else "no reemplaza"


def classifier_rows() -> list[dict[str, Any]]:
    """T1–T4: baselines, representations, TabPFN and the 2025-H1 confirmation."""
    rows = []
    baselines = load(BASELINES)["results"]
    report = load(REPRESENTATION)
    for target, result in report["results"].items():
        metric = result["primary_metric"]
        for name in ("global_frequency",):
            value = baselines[target][name]["metrics"]["calibration"]["no_shared_text"]
            rows.append(
                row(
                    etapa="M3",
                    tarea=target,
                    modelo=f"baseline {name}",
                    rol="baseline",
                    conjunto=CALIBRATION,
                    vista="no_shared_text",
                    metrica=metric,
                    valor=value[metric],
                    fuente=BASELINES,
                )
            )
        steps = {step["challenger"]: step for step in result["selection"]["steps"]}
        selected = result["selection"]["selected"]
        for name, metrics in result["metrics"].items():
            step = steps.get(name, {})
            if name == "product_rule":
                role = "baseline"
            elif name == selected:
                role = "ganador M5B"
            else:
                role = "candidato"
            rows.append(
                row(
                    etapa="M5B",
                    tarea=target,
                    modelo=name,
                    rol=role,
                    conjunto=CALIBRATION,
                    vista="no_shared_text",
                    metrica=metric,
                    valor=metrics[metric],
                    referencia=step.get("current", ""),
                    ganancia_relativa=step.get("relative_gain", ""),
                    ic95_inferior=step.get("difference", {}).get("p025", ""),
                    ic95_superior=step.get("difference", {}).get("p975", ""),
                    decision="reemplaza" if step.get("replaces") else "",
                    fuente=REPRESENTATION,
                )
            )
    rows.extend(foundation_rows())
    final = load(FINAL)
    for target, result in final["classifiers"].items():
        metric = result["primary_metric"]
        for name in ("product_rule", "model"):
            is_model = name == "model"
            label = result["representation"] if is_model else name
            rows.append(
                row(
                    etapa="M5B §22",
                    tarea=target,
                    modelo=label,
                    rol="baseline" if name == "product_rule" else "ganador M5B",
                    conjunto=VALIDATION,
                    vista="no_shared_text",
                    metrica=metric,
                    valor=result["metrics"]["no_shared_text"][name][metric],
                    referencia="product_rule" if name == "model" else "",
                    ic95_inferior=interval(result["difference"], "p025", is_model),
                    ic95_superior=interval(result["difference"], "p975", is_model),
                    decision=(
                        ("confirmado" if result["confirmed"] else "no confirmado")
                        if name == "model"
                        else ""
                    ),
                    fuente=FINAL,
                )
            )
    return rows


def foundation_rows() -> list[dict[str, Any]]:
    """Block B: TabPFN-3.5 against the frozen T1, calibration and 2025-H1."""
    rows = []
    for path, split in ((FOUNDATION, CALIBRATION), (FOUNDATION_VALIDATION, VALIDATION)):
        report = load(path)
        reference = "bge_product"
        for name, metrics in report["metrics"].items():
            comparison = report["comparisons"].get(name, {})
            wins = comparison.get("wins")
            if name == reference:
                role = "ganador M5B (referencia)"
            elif wins:
                role = "ganador bloque B"
            else:
                role = "candidato"
            rows.append(
                row(
                    etapa="M5F bloque B",
                    tarea="T1",
                    modelo=TABPFN_LABEL if name == "tabpfn" else name,
                    rol=role,
                    conjunto=split,
                    vista="no_shared_text",
                    metrica="macro_f1",
                    valor=metrics["macro_f1"],
                    referencia=reference if name != reference else "",
                    ganancia_relativa=comparison.get("relative_gain", ""),
                    ic95_inferior=comparison.get("difference", {}).get("p025", ""),
                    ic95_superior=comparison.get("difference", {}).get("p975", ""),
                    decision=verdict(wins),
                    fuente=path,
                )
            )
    report = load(ABLATION)
    for name, metrics in report["metrics"].items():
        comparison = report["comparisons"].get(name, {})
        rows.append(
            row(
                etapa="M5F ablación §26",
                tarea="T1",
                modelo=ABLATION_LABELS[name],
                rol="referencia (bloque B)" if name == "tabpfn" else "candidato",
                conjunto=CALIBRATION,
                vista="no_shared_text",
                metrica="macro_f1",
                valor=metrics["macro_f1"],
                referencia=TABPFN_LABEL if name != "tabpfn" else "",
                ganancia_relativa=comparison.get("relative_gain", ""),
                ic95_inferior=comparison.get("difference", {}).get("p025", ""),
                ic95_superior=comparison.get("difference", {}).get("p975", ""),
                decision=verdict(comparison.get("wins")),
                fuente=ABLATION,
            )
        )
    return rows


def full_runs(directory: str) -> list[Path]:
    """The results.json of every full run, one per model, newest first."""
    runs = sorted((PROJECT_ROOT / directory).glob("*/*-full*/results.json"))
    latest: dict[str, Path] = {}
    for path in runs:
        latest[path.parents[1].name] = path
    return list(latest.values())


def count_rows() -> list[dict[str, Any]]:
    """M9: every full run against its best baseline, then the §25.5 challenges."""
    rows = []
    for path in full_runs(COUNTS):
        report = json.loads(path.read_text(encoding="utf-8"))
        comparison = report.get("comparison")
        if not comparison or "metrics" not in comparison:
            continue
        model = report["model_id"]
        wis = comparison["difference"]["wis"]
        for name, metrics in comparison["metrics"].items():
            is_model = name == "model"
            rows.append(
                row(
                    etapa="M9",
                    tarea="conteo semanal por patrón",
                    modelo=COUNT_FAMILIES.get(model, model) if is_model else name,
                    rol="candidato" if is_model else "baseline",
                    conjunto=CALIBRATION,
                    vista="40 patrones",
                    metrica="wis",
                    valor=metrics["wis"],
                    referencia=comparison["best_baseline"] if is_model else "",
                    ganancia_relativa=comparison["wis_gain"] if is_model else "",
                    ic95_inferior=interval(wis, "p025", is_model),
                    ic95_superior=interval(wis, "p975", is_model),
                    decision=report.get("candidate_status", "") if is_model else "",
                    fuente=str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                )
            )
    for path in sorted((PROJECT_ROOT / CHALLENGES).glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        comparison = report["comparison"]
        model = report["candidate"]
        validation = report.get("split") == "validation"
        wins = report.get("wins")
        rows.append(
            row(
                etapa="M9 §25.5",
                tarea="conteo semanal por patrón",
                modelo=COUNT_FAMILIES.get(model, model),
                rol="candidato",
                conjunto=VALIDATION if validation else CALIBRATION,
                vista="40 patrones",
                metrica="wis",
                valor=comparison["metrics"]["model"]["wis"],
                referencia="M9 vigente (nb_rolling_4_hierarchical_v3)",
                ganancia_relativa=comparison["wis_gain"],
                ic95_inferior=comparison["difference"]["wis"]["p025"],
                ic95_superior=comparison["difference"]["wis"]["p975"],
                decision=verdict(wins),
                fuente=str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            )
        )
    final = load(FINAL)["weekly_volume"]
    for name, metrics in final["metrics"].items():
        is_model = name == "model"
        rows.append(
            row(
                etapa="M9 §22",
                tarea="conteo semanal por patrón",
                modelo=COUNT_FAMILIES[final["model_id"]] if is_model else name,
                rol="ganador M9" if is_model else "baseline",
                conjunto=VALIDATION,
                vista="40 patrones",
                metrica="wis",
                valor=metrics["wis"],
                referencia="b1_rolling_4" if is_model else "",
                ic95_inferior=interval(final["difference"]["wis"], "p025", is_model),
                ic95_superior=interval(final["difference"]["wis"], "p975", is_model),
                decision="confirmado" if is_model else "",
                fuente=FINAL,
            )
        )
    return rows


def composition_rows() -> list[dict[str, Any]]:
    """M10: Dirichlet-multinomial runs against the multinomial baselines."""
    rows = []
    for path in full_runs(COMPOSITION):
        report = json.loads(path.read_text(encoding="utf-8"))
        comparison = report["comparison"]
        for name, score in comparison["log_score"].items():
            is_model = name == "model"
            rows.append(
                row(
                    etapa="M10",
                    tarea="composición semanal",
                    modelo=report["model_id"] if is_model else name,
                    rol="candidato" if is_model else "baseline",
                    conjunto=CALIBRATION,
                    vista="40 patrones",
                    metrica="log_score",
                    valor=score,
                    referencia=comparison["best_baseline"] if is_model else "",
                    ic95_inferior=interval(comparison["difference"], "p025", is_model),
                    ic95_superior=interval(comparison["difference"], "p975", is_model),
                    decision=report.get("candidate_status", "") if is_model else "",
                    fuente=str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                )
            )
    final = load(FINAL)["weekly_composition"]
    for name, score in final["log_score"].items():
        is_model = name == "model"
        rows.append(
            row(
                etapa="M10 §22",
                tarea="composición semanal",
                modelo=final["model_id"] if is_model else name,
                rol="ganador M10" if is_model else "baseline",
                conjunto=VALIDATION,
                vista="40 patrones",
                metrica="log_score",
                valor=score,
                referencia="b2_rolling_4" if is_model else "",
                decision="confirmado" if is_model else "",
                fuente=FINAL,
            )
        )
    return rows


def daily_rows() -> list[dict[str, Any]]:
    """M9D (§28.2): daily fulls against their baselines, then the daily winner."""
    rows = []
    for path in full_runs(DAILY):
        report = json.loads(path.read_text(encoding="utf-8"))
        comparison = report["comparison"]
        composition = "log_score" in comparison
        metric = "log_score" if composition else "wis"
        scores = comparison["log_score"] if composition else comparison["metrics"]
        difference = (
            comparison["difference"] if composition else comparison["difference"]["wis"]
        )
        for name, value in scores.items():
            is_model = name == "model"
            rows.append(
                row(
                    etapa="M9D",
                    tarea="composición diaria" if composition else "conteo diario",
                    modelo=DAILY_FAMILIES.get(report["model_id"], report["model_id"])
                    if is_model
                    else name,
                    rol="candidato" if is_model else "baseline",
                    conjunto=CALIBRATION,
                    vista="40 patrones, diario",
                    metrica=metric,
                    valor=value if composition else value["wis"],
                    referencia=comparison["best_baseline"] if is_model else "",
                    ganancia_relativa=(
                        comparison.get("wis_gain", "") if is_model else ""
                    ),
                    ic95_inferior=interval(difference, "p025", is_model),
                    ic95_superior=interval(difference, "p975", is_model),
                    decision=report.get("candidate_status", "") if is_model else "",
                    fuente=str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                )
            )
    for path in sorted((PROJECT_ROOT / DAILY_CHALLENGE).glob("*.json")):
        if path.name.endswith(".run.json"):
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        winner_id = report["winner"]["model_id"]
        winner = next(c for c in report["candidates"] if c["model_id"] == winner_id)
        validation = report["split"] == "validation"
        if validation:
            comparison = report["validation"]["comparison"]
            value = report["validation"]["metrics"]["model"]["wis"]
            difference = comparison["difference"]["wis"]
        else:
            value, difference = winner["wis"], {
                "p025": winner["wis_difference_p025"],
                "p975": winner["wis_difference_p975"],
            }
            comparison = {"wis_gain": winner["wis_gain"]}
        rows.append(
            row(
                etapa="M9D §28.2",
                tarea="conteo diario",
                modelo=DAILY_FAMILIES.get(winner["model_id"], winner["model_id"]),
                rol="ganador M9D",
                conjunto=VALIDATION if validation else CALIBRATION,
                vista="40 patrones, diario",
                metrica="wis",
                valor=value,
                referencia=winner["best_baseline"],
                ganancia_relativa=comparison["wis_gain"],
                ic95_inferior=difference["p025"],
                ic95_superior=difference["p975"],
                decision="" if validation else "elegido",
                fuente=str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            )
        )
    return rows


def alert_rows() -> list[dict[str, Any]]:
    """M11: the two alert rules on injected growth, and the real alert counts."""
    rows = []
    report = load(PERSISTENT)
    chosen = report["decision"]["rule"]
    for scenario, result in report["detection"].items():
        for rule in ("cusum", "weekly"):
            rows.append(
                row(
                    etapa="M11",
                    tarea=f"alertas, escenario {scenario}",
                    modelo=rule,
                    rol="ganador M11" if rule == chosen else "candidato",
                    conjunto="calibracion simulada",
                    vista="40 patrones",
                    metrica="tasa_de_deteccion",
                    valor=result[rule]["detected"],
                    decision="elegido" if rule == chosen else "",
                    fuente=PERSISTENT,
                )
            )
    for path, split in ((PERSISTENT, CALIBRATION), (PERSISTENT_VALIDATION, VALIDATION)):
        alarms = load(path)["real_alarms"]
        key = "calibration" if split == CALIBRATION else "validation"
        if key not in alarms:
            continue
        for rule in ("cusum", "weekly"):
            rows.append(
                row(
                    etapa="M11",
                    tarea="alertas reales",
                    modelo=rule,
                    rol="ganador M11" if rule == chosen else "candidato",
                    conjunto=split,
                    vista="40 patrones",
                    metrica="alertas_por_mes",
                    valor=alarms[key][f"{rule}_per_month"],
                    fuente=path,
                )
            )
    return rows


def space_rows() -> list[dict[str, Any]]:
    """Block A: every space and clustering candidate against M7 by neighbour lift."""
    report = load(SPACE)
    reference = report["reference"]
    rows = [
        row(
            etapa="M7S bloque A",
            tarea="patrones semánticos",
            modelo=reference["key"],
            rol="ganador M7 (referencia)",
            conjunto="muestra de ajuste",
            vista="20,000 filas",
            metrica="neighbour_lift",
            valor=reference["neighbor_lift"],
            fuente=SPACE,
        )
    ]
    for comparison in report["comparisons"]:
        rows.append(
            row(
                etapa="M7S bloque A",
                tarea="patrones semánticos",
                modelo=comparison["challenger"],
                rol="candidato",
                conjunto="muestra de ajuste",
                vista="20,000 filas",
                metrica="neighbour_lift",
                valor=reference["neighbor_lift"] + comparison["difference"]["estimate"],
                referencia=reference["key"],
                ganancia_relativa=comparison["relative_gain"],
                ic95_inferior=comparison["difference"]["p025"],
                ic95_superior=comparison["difference"]["p975"],
                decision="reemplaza" if comparison["wins"] else "no reemplaza",
                fuente=SPACE,
            )
        )
    return rows


def build() -> list[dict[str, Any]]:
    return (
        classifier_rows()
        + count_rows()
        + composition_rows()
        + daily_rows()
        + alert_rows()
        + space_rows()
    )


def main() -> None:
    rows = build()
    path = PROJECT_ROOT / OUTPUT
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {OUTPUT}")


if __name__ == "__main__":
    main()
