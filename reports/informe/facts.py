"""Facts of the report (models_plan.md §30): every number read from the files.

The Quarto document imports this module. Nothing here is typed by hand: each
value comes from `reports/modeling/*.json`, from the course comparison table
or from the pattern catalog. The helpers format numbers in Spanish style.
"""

from __future__ import annotations

import ast
import csv
import glob
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "modeling"
TABLE = ROOT / "etapa_4_5_modelado_evaluacion/resultados/comparacion_modelos.csv"

WEEKLY_LABELS = {
    "poisson_static_pymc_v1": "B1, Poisson con participaciones fijas",
    "b1_rolling_4_v1": "B1-R4, Poisson con participaciones de 4 semanas",
    "b1_rolling_13_v1": "B1-R13, ídem con 13 semanas",
    "nb_static_global_v3": "NB-V3, binomial negativa estática",
    "nb_rolling_4_global_v1": "NB-R4, 4 semanas, una dispersión",
    "nb_rolling_4_hierarchical_v3": "NB-R4-H v3, dispersión por patrón",
    "nb_discounted_hierarchical_v1": "C-A, memoria que decae (δ = 0.5)",
    "nb_state_space_v1": "C-B, paseo aleatorio t de Student",
    "chronos2_zero_shot_v1": "Chronos-2, sin entrenamiento",
    "timesfm3_zero_shot_v1": "TimesFM 3.0, sin entrenamiento",
}
DAILY_LABELS = {
    "nb_daily_hierarchical_v1": "D-A, binomial negativa con día de semana",
    "nb_daily_no_dow_v1": "D-B2, sin día de semana",
    "nb_daily_fourier_v1": "D-C, Fourier y tendencia local",
    "zinb_daily_hierarchical_v1": "D-D, D-A con inflación de ceros",
    "nb_daily_state_space_v1": "D-B, espacio de estados diario",
    "dirichlet_multinomial_daily_v1": "D-E, Dirichlet-multinomial diaria",
}
T1_LABELS = {
    "product_rule": "Regla por producto",
    "tfidf_text": "TF-IDF, solo texto",
    "tfidf_product": "TF-IDF + producto",
    "bge_text": "BGE, solo texto",
    "bge_product": "BGE + producto",
    "tabpfn": "TabPFN-3.5 (PCA 100 + producto)",
}
PERIOD_LABELS = {
    "context_2015_2022": "Contexto 2015–2022",
    "train_2023_2024": "Ajuste y calibración 2023–2024",
    "validation_2025_h1": "Validación 2025-H1",
    "holdout_2025_h2": "2025-H2 (bloqueado)",
    "ood_2026_partial": "2026 parcial (sellado)",
}


# ----------------------------------------------------------------- loading
def load_json(relative: str | Path) -> dict[str, Any]:
    path = Path(relative)
    if not path.is_absolute():
        path = ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def latest_full(directory: str) -> dict[str, dict[str, Any]]:
    """Latest full results.json per model folder under `directory`."""
    found: dict[str, dict[str, Any]] = {}
    pattern = str(REPORTS / directory / "*" / "*-full*" / "results.json")
    for path in sorted(glob.glob(pattern)):
        found[Path(path).parents[1].name] = load_json(path)
    return found


def latest_file(pattern: str) -> dict[str, Any]:
    paths = sorted(glob.glob(str(ROOT / pattern)))
    if not paths:
        raise FileNotFoundError(pattern)
    return load_json(paths[-1])


def parse_catalog_value(value: Any) -> Any:
    """The catalog stores every field as text; recover numbers, lists, dicts."""
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


# -------------------------------------------------------------- formatting
def n(value: float, decimals: int = 3) -> str:
    """Number with a fixed count of decimals, thousands separated by comma."""
    return f"{value:,.{decimals}f}"


def i(value: float) -> str:
    return f"{int(round(value)):,}"


def pct(value: float, decimals: int = 1) -> str:
    return f"{100 * value:.{decimals}f} %"


def signed_pct(value: float, decimals: int = 1) -> str:
    return f"{100 * value:+.{decimals}f} %"


def ic(difference: dict[str, float], decimals: int = 3, scale: float = 1.0) -> str:
    low, high = difference["p025"] * scale, difference["p975"] * scale
    return f"[{low:+,.{decimals}f}, {high:+,.{decimals}f}]"


def md_table(
    headers: list[str], rows: list[list[Any]], align: str | None = None
) -> str:
    """Markdown pipe table; `align` is a string of l/r/c per column."""
    align = align or "l" * len(headers)
    marks = {"l": ":---", "r": "---:", "c": ":---:"}
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(marks[a] for a in align) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------- facts
def data_facts() -> dict[str, Any]:
    contract = load_json("reports/modeling/input_contract.json")
    evaluation = load_json("reports/modeling/evaluation_contract.json")
    cleaning = load_json("reports/preparation/informe_limpieza.json")
    problems = {p["problema"]: p for p in cleaning["problemas"]}
    shared = next(
        p["cifra"] for p in cleaning["problemas"] if "reutilizaci" in p["problema"]
    )
    taxonomy = next(
        p["cifra"] for p in cleaning["problemas"] if "taxonom" in p["problema"]
    )
    positives = next(
        p["cifra"] for p in cleaning["problemas"] if "pocos positivos" in p["problema"]
    )
    drift = next(
        p["cifra"]
        for p in cleaning["problemas"]
        if "cambian con el tiempo" in p["problema"]
    )
    rare = next(p["cifra"] for p in cleaning["problemas"] if "raros" in p["problema"])
    return {
        "rows": contract["rows"],
        "columns": contract["columns"],
        "md5": contract["dvc_md5"],
        "size_gib": contract["size_bytes"] / 1024**3,
        "period_rows": contract["period_rows"],
        "eligible": contract["eligible_no_shared_text"],
        "splits": evaluation["splits"],
        "row_counts": evaluation["row_counts"],
        "fit_unique_texts": evaluation["fit_unique_normalized_texts"],
        "shared_pct": shared["pct_filas_que_comparten_texto"],
        "distinct_texts": shared["textos_normalizados_distintos"],
        "taxonomy": taxonomy,
        "positives": positives,
        "drift": drift,
        "t1_motives": rare["motivos_T1_en_entrenamiento"],
        "problems": cleaning["problemas"],
        "stages": cleaning["etapas"],
        "dictionary": cleaning["diccionario"],
        "versions": contract["versions"],
        "problem_index": problems,
    }


def classifier_facts() -> dict[str, Any]:
    representation = load_json("reports/modeling/representation_results.json")
    baselines = load_json("reports/modeling/baseline_results.json")
    final = load_json("reports/modeling/final_confirmation.json")
    foundation = load_json("reports/modeling/foundation_t1/results.json")
    foundation_validation = load_json(
        "reports/modeling/foundation_t1/validation_results.json"
    )
    ablation = load_json("reports/modeling/foundation_t1/ablation_results.json")
    tasks = {}
    for task, result in representation["results"].items():
        metric = result["primary_metric"]
        rule = baselines["results"][task]["product_frequency"]["metrics"]
        tasks[task] = {
            "metric": metric,
            "rule_calibration": rule["calibration"]["no_shared_text"][metric],
            "rule_validation": rule["validation"]["no_shared_text"][metric],
            "candidates": {
                name: values[metric]
                for name, values in result["metrics"].items()
                if name != "product_rule"
            },
            "calibration_rows": result["calibration_rows"],
            "weeks": result["weeks"],
            "selected": result["selection"]["selected"],
            "steps": result["selection"]["steps"],
            "thresholds": {
                name: values.get("threshold")
                for name, values in result["metrics"].items()
                if isinstance(values, dict)
            },
            "final": final["classifiers"][task],
            "final_model": final["classifiers"][task]["metrics"]["no_shared_text"][
                "model"
            ][metric],
            "final_rule": final["classifiers"][task]["metrics"]["no_shared_text"][
                "product_rule"
            ][metric],
        }
    return {
        "rule": representation["rule"],
        "tasks": tasks,
        "confirmed": final["confirmed"],
        "tabpfn": {
            "context_rows": foundation["context_rows"],
            "context_classes": foundation["context_classes"],
            "query_rows": foundation["query_rows"],
            "pca": foundation["pca_components"],
            "estimators": foundation["estimators"],
            "macro_f1": foundation["metrics"]["tabpfn"]["macro_f1"],
            "top_three": foundation["metrics"]["tabpfn"]["top_three"],
            "reference_f1": foundation["metrics"]["bge_product"]["macro_f1"],
            "gain": foundation["comparisons"]["tabpfn"]["relative_gain"],
            "difference": foundation["comparisons"]["tabpfn"]["difference"],
            "wins": foundation["comparisons"]["tabpfn"]["wins"],
            "validation_f1": foundation_validation["metrics"]["tabpfn"]["macro_f1"],
            "validation_reference_f1": foundation_validation["metrics"]["bge_product"][
                "macro_f1"
            ],
            "validation_top_three": foundation_validation["metrics"]["tabpfn"][
                "top_three"
            ],
            "minutes": foundation["elapsed_seconds"]["tabpfn"] / 60,
        },
        "ablation": {
            name: {
                "label": ablation["inputs"][name]["label"],
                "macro_f1": ablation["metrics"][name]["macro_f1"],
                "gain": ablation["comparisons"][name]["relative_gain"],
                "difference": ablation["comparisons"][name]["difference"],
            }
            for name in ablation["inputs"]
        },
    }


def semantic_facts() -> dict[str, Any]:
    space = load_json("reports/modeling/semantic_space_results.json")
    clustering = load_json("reports/modeling/clustering_results.json")
    patterns = load_json("reports/modeling/weekly_patterns_results.json")
    sensitivity = load_json("reports/modeling/space_sensitivity/results.json")
    kmeans = clustering["selected"]["kmeans"]
    faiss = space["faiss"]["metrics"]["calibration"]
    accepted = [c for c in sensitivity["selected"] if c["accepted"]]
    seeds = {
        key: value["ari_same_candidate"]
        for key, value in sensitivity["space_change"].items()
        if key.startswith("umap15")
    }
    return {
        "bge_model": space["source"]["bge_model"],
        "pca_components": space["pca"]["components"],
        "pca_variance": space["pca"]["explained_variance"],
        "pca_checkpoints": space["pca"]["variance_checkpoints"],
        "faiss_rows": space["faiss"]["index_rows"],
        "faiss": faiss,
        "umap_components": clustering["umap"]["n_components"],
        "candidate_count": clustering["candidate_count"],
        "kmeans": kmeans,
        "hdbscan": clustering["selected"]["hdbscan"],
        "cure": clustering["selected"]["cure"],
        "acceptance": clustering["acceptance"],
        "novelty_quantile": patterns["novelty_quantile"],
        "novelty": patterns["assignment"],
        "fit_clusters": patterns["fit_clusters"],
        "sensitivity_spaces": len(sensitivity["spaces"]),
        "sensitivity_candidates": len(sensitivity["selected"]),
        "sensitivity_accepted": len(accepted),
        "sensitivity_winner": sensitivity["winner"],
        "sensitivity_comparisons": sensitivity["comparisons"],
        "seed_ari": seeds,
    }


def weekly_calibration(report: dict[str, Any]) -> dict[str, Any] | None:
    """Calibration metrics of a weekly run, whatever runner wrote it."""
    if "backtest" in report:
        calibration = report["backtest"]["calibration"]
        return calibration.get("model", calibration)
    if "calibration" in report and "wis" in report["calibration"]:
        return report["calibration"]
    return None


def weekly_facts() -> dict[str, Any]:
    runs = latest_full("weekly_counts")
    final = load_json("reports/modeling/final_confirmation.json")
    challenges = {
        Path(path).stem: load_json(path)
        for path in sorted(glob.glob(str(REPORTS / "weekly_counts/challenges/*.json")))
    }
    selection = load_json(
        "reports/modeling/weekly_counts/nb_discounted_hierarchical_v1/"
        "share_selection.json"
    )
    candidates = []
    for model_id, report in runs.items():
        calibration = weekly_calibration(report)
        if calibration is None:
            continue
        comparison = report.get("comparison", {})
        challenge = challenges.get(model_id)
        if challenge is not None:
            status = "gana a v3" if challenge["wins"] else "no gana a v3"
            gain = challenge["comparison"]["wis_gain"]
            difference = challenge["comparison"]["difference"]["wis"]
            reference = "NB-R4-H v3"
        elif "metrics" in comparison:
            status = report["candidate_status"]
            gain = comparison["wis_gain"]
            difference = comparison["difference"]["wis"]
            reference = comparison["best_baseline"]
        else:
            status, gain, difference, reference = "referencia", None, None, ""
        candidates.append(
            {
                "model_id": model_id,
                "label": WEEKLY_LABELS.get(model_id, model_id),
                "wis": calibration["wis"],
                "coverage_80": calibration.get("coverage_80"),
                "coverage_95": calibration.get("coverage_95"),
                "status": status,
                "reference": reference,
                "wis_gain": gain,
                "difference": difference,
            }
        )
    candidates.sort(key=lambda c: c["wis"])
    reference = runs["b1_rolling_4_v1"]["calibration"]["wis"]
    c_a = challenges["nb_discounted_hierarchical_v1"]
    c_a_validation = challenges["nb_discounted_hierarchical_v1_validation"]
    return {
        "candidates": candidates,
        "reference_wis": reference,
        "v3": runs["nb_rolling_4_hierarchical_v3"]["backtest"]["calibration"]["model"],
        "c_a": weekly_calibration(runs["nb_discounted_hierarchical_v1"]),
        "c_a_vs_poisson": runs["nb_discounted_hierarchical_v1"]["comparison"],
        "c_a_challenge": c_a,
        "c_a_validation": c_a_validation,
        "challenges": challenges,
        "final": final["weekly_volume"],
        "share_selection": {
            **selection,
            "chosen_discount": selection["chosen"]["discount"],
        },
    }


def composition_facts() -> dict[str, Any]:
    runs = latest_full("weekly_composition")
    final = load_json("reports/modeling/final_confirmation.json")
    signal = load_json(
        "reports/modeling/weekly_composition/mixture_signal/results.json"
    )
    dm_r4 = runs["dirichlet_multinomial_rolling_4_v1"]
    kappa = None
    summary_path = sorted(
        glob.glob(
            str(
                REPORTS
                / "weekly_composition/dirichlet_multinomial_rolling_4_v1"
                / "*-full-*/posterior_summary.csv"
            )
        )
    )
    if summary_path:
        with open(summary_path[-1], encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("parameter") == "kappa":
                    kappa = float(row["mean"])
    return {
        "dm_r4": dm_r4["comparison"],
        "dm_r4_status": dm_r4["candidate_status"],
        "dm_v1": runs["dirichlet_multinomial_static_v1"]["comparison"],
        "dm_v1_status": runs["dirichlet_multinomial_static_v1"][
            "candidate_status"
        ],
        "kappa": kappa,
        "final": final["weekly_composition"],
        "signal": signal,
    }


def alert_facts() -> dict[str, Any]:
    design = load_json("reports/modeling/persistent_change_c_a/results.json")
    validation = load_json(
        "reports/modeling/persistent_change_c_a/validation_results.json"
    )
    previous = load_json("reports/modeling/persistent_change/results.json")
    return {
        "thresholds": design["thresholds"],
        "false_alarm_rate": design["false_alarm_rate"],
        "settings": design["settings"],
        "detection": design["detection"],
        "real": design["real_alarms"],
        "validation_real": validation["real_alarms"]["validation"],
        "decision": design["decision"],
        "previous_detection": previous["detection"],
        "previous_real": previous["real_alarms"],
    }


def daily_facts() -> dict[str, Any]:
    panel = load_json("reports/modeling/daily_counts.json")
    runs = latest_full("daily_counts")
    challenge = latest_file(
        "reports/modeling/daily_counts/challenge/*-calibration.json"
    )
    validation = latest_file(
        "reports/modeling/daily_counts/challenge/*-validation.json"
    )
    change = load_json("reports/modeling/daily_change/results.json")
    selection = load_json(
        "reports/modeling/daily_counts/nb_daily_hierarchical_v1/share_selection.json"
    )
    pilot = latest_file(
        "reports/modeling/daily_counts/nb_daily_state_space_v1/*-pilot/results.json"
    )
    counts = []
    composition = None
    for model_id, report in runs.items():
        comparison = report["comparison"]
        if "log_score" in comparison:
            composition = {
                "model_id": model_id,
                "label": DAILY_LABELS[model_id],
                "comparison": comparison,
                "posterior": report["posterior"],
                "signal": report["signal"],
                "status": report["candidate_status"],
            }
            continue
        calibration = report["backtest"]["calibration"]["model"]
        counts.append(
            {
                "model_id": model_id,
                "label": DAILY_LABELS.get(model_id, model_id),
                "wis": calibration["wis"],
                "coverage_80": calibration["coverage_80"],
                "coverage_95": calibration["coverage_95"],
                "status": report["candidate_status"],
                "wis_gain": comparison["wis_gain"],
                "difference": comparison["difference"]["wis"],
                "best_baseline": comparison["best_baseline"],
                "baseline_wis": comparison["metrics"][comparison["best_baseline"]][
                    "wis"
                ],
            }
        )
    counts.sort(key=lambda c: c["wis"])
    winner_id = challenge["winner"]["model_id"]
    return {
        "panel": panel,
        "counts": counts,
        "composition": composition,
        "winner": next(c for c in counts if c["model_id"] == winner_id),
        "winner_label": DAILY_LABELS[winner_id],
        "pairwise": challenge["pairwise"],
        "validation": validation["validation"],
        "change": change,
        "share_selection": selection,
        "state_space_pilot": pilot,
    }


def catalog_facts() -> list[dict[str, Any]]:
    catalog = load_json("reports/modeling/patterns/catalog.json")
    patterns = []
    for raw in catalog["patterns"]:
        pattern = {key: parse_catalog_value(value) for key, value in raw.items()}
        pattern["cluster_id"] = int(pattern["cluster_id"])
        patterns.append(pattern)
    return patterns


def table_rows() -> list[dict[str, str]]:
    with TABLE.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build() -> dict[str, Any]:
    return {
        "datos": data_facts(),
        "clasificadores": classifier_facts(),
        "semantico": semantic_facts(),
        "semanal": weekly_facts(),
        "composicion": composition_facts(),
        "alertas": alert_facts(),
        "diario": daily_facts(),
        "catalogo": catalog_facts(),
        "tabla": table_rows(),
    }
