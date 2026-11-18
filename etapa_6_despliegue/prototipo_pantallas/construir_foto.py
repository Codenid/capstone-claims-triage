"""Incrusta en la maqueta una foto fija de una salida real del pipeline.

Lee la carpeta que escribe ``python -m src.pipeline score`` (``weekly.csv``,
``daily.csv``, ``mixture.csv``, ``score.json`` y ``complaints.parquet``) junto
con el catálogo de patrones y su masa de frontera, y reemplaza el bloque de
datos de ``mesa-de-triaje.html``. La página no cambia: solo cambian sus datos.

Uso, desde la raíz del repositorio::

    uv run python etapa_6_despliegue/prototipo_pantallas/construir_foto.py \
        --score reports/pipeline/demo_2024-11-04
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAGE = HERE / "mesa-de-triaje.html"
CATALOG = ROOT / "reports" / "modeling" / "patterns" / "catalog.json"
BOUNDARY = ROOT / "reports" / "modeling" / "patterns" / "boundary_mass.json"
NUMERIC = re.compile(r"[\d ]+")  # términos del catálogo que son solo cifras
BLOCK = re.compile(r'(<script id="foto" type="application/json">)(.*?)(</script>)', flags=re.S)

PER_DAY = 4  # reclamos de muestra por día
EXTRA = 3  # reclamos extra por cada caso que la demo debe poder mostrar


def read_patterns() -> list[dict]:
    """Los 40 patrones del catálogo, con la masa de frontera medida en calibración."""
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))["patterns"]
    boundary = json.loads(BOUNDARY.read_text(encoding="utf-8"))["splits"]["calibration"]["patterns"]
    frontier = {int(p["cluster_id"]): float(p["boundary_fraction_distance"]) for p in boundary}
    patterns = []
    for raw in catalog:
        cluster = int(raw["cluster_id"])
        example = (raw.get("examples") or [{}])[0]
        patterns.append(
            {
                "id": cluster,
                "product": raw["main_product"],
                "product_share": round(float(raw["main_product_fraction"]), 3),
                "terms": [t for t in raw["terms"] if "redacted" not in t and not NUMERIC.fullmatch(t)][:6],
                "fit_rows": int(raw["fit_rows"]),
                "frontier": round(frontier.get(cluster, 0.0), 3),
                "example": {"id": str(example.get("complaint_id", "")), "text": example.get("snippet", "")},
            }
        )
    return sorted(patterns, key=lambda p: p["id"])


def read_counts(path: Path, with_alert: bool) -> tuple[list[str], dict[int, list[list]]]:
    """Por patrón, una fila por periodo: observado, esperado, p10, p90 y, si es semanal, z, CUSUM y alarma."""
    table = pd.read_csv(path).sort_values(["cluster_id", "period"])
    periods = sorted(table["period"].unique().tolist())
    series: dict[int, list[list]] = {}
    for row in table.to_dict("records"):
        item = [int(row["complaint_count"]), round(float(row["expected"]), 1), float(row["model_p10"]), float(row["model_p90"])]
        if with_alert:
            item += [round(float(row["z"]), 2), round(float(row["cusum"]), 2), bool(row["cusum_alarm"])]
        series.setdefault(int(row["cluster_id"]), []).append(item)
    return periods, series


def sample_complaints(table: pd.DataFrame, alarms: list[tuple[str, int]]) -> pd.DataFrame:
    """Muestra determinista: variedad de productos por día, más novedosos y reclamos de los patrones en alerta."""
    table = table.sort_values("Complaint ID")
    day = table["Date received"].dt.strftime("%Y-%m-%d")
    chosen: list[int] = []
    for _, group in table.groupby(day):
        by_product = group.drop_duplicates("Product canonical")
        chosen += by_product.index[:PER_DAY].tolist()
    chosen += table.index[table["is_novel"]][:EXTRA].tolist()
    for week, cluster in alarms:
        start = pd.Timestamp(week)
        in_week = (table["Date received"] >= start) & (table["Date received"] < start + pd.Timedelta(days=7))
        chosen += table.index[in_week & (table["cluster_id"] == cluster)][:EXTRA].tolist()
    return table.loc[list(dict.fromkeys(chosen))].sort_values(["Date received", "Complaint ID"])


def complaint_record(row) -> dict:
    return {
        "id": str(row["Complaint ID"]),
        "date": row["Date received"].strftime("%Y-%m-%d"),
        "product": row["Product canonical"],
        "issue": row["Issue canonical"],
        "pattern": int(row["cluster_id"]),
        "novel": bool(row["is_novel"]),
        "t2": round(float(row["t2_probability"]), 3),
        "t2_likely": bool(row["t2_likely"]),
        "t3": round(float(row["t3_probability"]), 3),
        "t3_likely": bool(row["t3_likely"]),
        "top": [[row[f"issue_top{k}"], round(float(row[f"issue_top{k}_probability"]), 3)] for k in (1, 2, 3)],
        "neighbors": json.loads(row["neighbors"]),
        "similarity": round(float(row["neighbor_similarity_top1"]), 3),
    }


def build(score_dir: Path) -> dict:
    score = json.loads((score_dir / "score.json").read_text(encoding="utf-8"))
    weeks, weekly = read_counts(score_dir / "weekly.csv", with_alert=True)
    days, daily = read_counts(score_dir / "daily.csv", with_alert=False)
    mixture = pd.read_csv(score_dir / "mixture.csv")
    complaints = pd.read_parquet(score_dir / "complaints.parquet")
    alarms = [(weeks[k], cluster) for cluster, rows in weekly.items() for k, row in enumerate(rows) if row[6]]
    sample = sample_complaints(complaints, alarms)
    signals = score["signals"]
    return {
        "meta": {
            "rows": int(score["rows"]),
            "first": score["dates"]["first"],
            "last": score["dates"]["last"],
            "commit": score["git_commit"][:7],
            "t1_model": score["t1_model"],
            "novel_fraction": round(float(signals["novelty"]["novel_fraction"]), 4),
            "coverage_80": round(float(signals["coverage"]["coverage_80"]), 4),
            "macro_f1": round(float(signals["macro_f1"]["macro_f1"]), 3),
            "retrain": bool(signals["retrain_recommended"]),
            "t2_threshold": round(float(complaints.loc[complaints["t2_likely"], "t2_probability"].min()), 2),
            "t3_threshold": round(float(complaints.loc[complaints["t3_likely"], "t3_probability"].min()), 2),
            "sample": int(len(sample)),
        },
        "patterns": read_patterns(),
        "weeks": weeks,
        "weekly": weekly,
        "days": days,
        "daily": daily,
        "mixture": [
            {"week": r["period"], "p_value": round(float(r["p_value"]), 3), "rare": bool(r["rare_mixture"])}
            for r in mixture.to_dict("records")
        ],
        "complaints": [complaint_record(row) for _, row in sample.iterrows()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--score", type=Path, required=True, help="carpeta de salida de `python -m src.pipeline score`")
    args = parser.parse_args()
    snapshot = json.dumps(build(args.score), ensure_ascii=False, separators=(",", ":"))
    snapshot = snapshot.replace("</", "<\\/")  # un texto con </script> no puede cerrar el bloque
    page = PAGE.read_text(encoding="utf-8")
    if not BLOCK.search(page):
        raise SystemExit(f"{PAGE.name} no tiene el bloque de datos <script id=\"foto\">")
    page = BLOCK.sub(lambda m: m.group(1) + snapshot + m.group(3), page, count=1)
    PAGE.write_text(page, encoding="utf-8", newline="\n")
    print(f"{PAGE.name}: foto de {args.score} incrustada ({len(snapshot) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
