"""Static HTML views of the panel and the complaint cards (models_plan.md §25).

They only present the JSON data built elsewhere; the future web app can render
the same data its own way.
"""

from __future__ import annotations

import base64
import html
import io
import math
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

from src.triage.actions import ACTIONS

STYLE = """
:root { --text: #1f2933; --muted: #52606d; --line: #d9e2ec; --alert: #b42318;
  --ok: #027a48; --soft: #f5f7fa; }
body { font-family: system-ui, sans-serif; color: var(--text); background: #fff;
  max-width: 980px; margin: 2rem auto; padding: 0 16px; line-height: 1.45; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.15rem; margin-top: 1.6rem; }
table { border-collapse: collapse; width: 100%; font-size: .88rem; }
th, td { border-bottom: 1px solid var(--line); padding: .35rem .45rem;
  text-align: left; vertical-align: top; }
th { background: var(--soft); }
.box { border: 1px solid var(--line); border-radius: 8px; padding: .8rem 1rem; }
.alert { color: var(--alert); font-weight: 600; }
.ok { color: var(--ok); }
.muted { color: var(--muted); font-size: .85rem; }
.text { background: var(--soft); border-radius: 6px; padding: .6rem .8rem; }
.wrap { overflow-x: auto; }
img { max-width: 100%; height: auto; }
"""
SITUATIONS = {
    "novel": "Reclamo novedoso",
    "template_alert": "Patrón con alerta, dominado por una plantilla",
    "alert": "Patrón con alerta",
    "normal": "Patrón estable",
}


def escape(value: Any) -> str:
    return html.escape(str(value))


def missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def percent(value: Any, signed: bool = False) -> str:
    if missing(value):
        return "—"
    return f"{value:+.0%}" if signed else f"{value:.1%}"


def yes_no(value: Any) -> str:
    if missing(value):
        return "—"
    return "Sí" if value else "No"


def page(title: str, body: str) -> str:
    return (
        "<!doctype html><html lang='es'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{escape(title)}</title><style>{STYLE}</style></head>"
        f"<body>{body}<p class='muted'>Modelos congelados del proyecto; datos de "
        "demostración de CFPB 2025-H1. La decisión final es de la persona.</p>"
        "</body></html>"
    )


def figure_html(figure: Any, alt: str) -> str:
    """A matplotlib figure as an inline PNG, so the page is a single file."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"<img alt='{escape(alt)}' src='data:image/png;base64,{encoded}'>"


def level_chart(levels: pd.DataFrame, low: float, high: float, title: str) -> str:
    """Δ of one pattern over time, with its usual band in fit (p5–p95)."""
    figure, axis = plt.subplots(figsize=(7, 2.6))
    axis.axhspan(low, high, color="tab:blue", alpha=0.15, label="Rango habitual")
    axis.plot(levels["week"], levels["level_change"], color="tab:blue", lw=1.5)
    last = levels.dropna(subset=["level_change"]).tail(1)
    axis.scatter(last["week"], last["level_change"], color="tab:red", zorder=3,
                 label="Semana actual")
    axis.axhline(0, color="black", lw=0.6)
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.set_title(title, fontsize=10)
    axis.legend(fontsize=8, loc="upper left")
    return figure_html(figure, title)


def pattern_rows(patterns: list[dict[str, Any]]) -> str:
    rows = []
    for item in patterns:
        alert = "<span class='alert'>Sí</span>" if item["active_alert"] else "No"
        share = (
            f"{percent(item['share'])} (esperada {percent(item['expected_share'])}; "
            f"95%: {percent(item['share_low'])}–{percent(item['share_high'])})"
        )
        level = (
            f"{percent(item['level_change'], signed=True)} "
            f"(percentil {percent(item['level_percentile'])})"
        )
        rows.append(
            f"<tr><td>{item['cluster_id']}</td><td>{escape(item['main_product'])}</td>"
            f"<td>{escape(item['representative_terms'])}</td>"
            f"<td>{item['observed']} ({item['expected']:.0f})</td>"
            f"<td>{item['cusum']:.2f}</td><td>{alert}</td>"
            f"<td>{level}</td><td>{share}</td></tr>"
        )
    return "".join(rows)


def panel_page(
    week: str,
    patterns: list[dict[str, Any]],
    charts: dict[int, str],
    threshold: float,
) -> str:
    alerts = sum(item["active_alert"] for item in patterns)
    body = (
        f"<h1>Panel semanal de patrones: semana del {escape(week)}</h1>"
        "<p>Cada fila es un patrón de reclamos. Hay <b>alerta</b> cuando el patrón "
        "lleva varias semanas recibiendo más reclamos de lo esperado. El "
        "<b>nivel</b> compara su participación con la de hace 3 meses, y la "
        "franja de los gráficos muestra cuánto suele variar.</p>"
        f"<p class='box'><b>{alerts}</b> de {len(patterns)} patrones tienen una "
        "alerta activa (CUSUM sobre el umbral en alguna de las últimas 4 "
        "semanas).</p>"
        "<div class='wrap'><table><tr><th>Patrón</th><th>Producto principal</th>"
        "<th>Palabras representativas</th><th>Reclamos (esperados)</th>"
        f"<th>CUSUM (umbral {threshold:.2f})</th><th>Alerta</th>"
        "<th>Nivel frente a hace 3 meses</th><th>Participación</th></tr>"
        f"{pattern_rows(patterns)}</table></div>"
    )
    if charts:
        body += "<h2>Nivel de los patrones con alerta</h2>" + "".join(charts.values())
    return page(f"Panel semanal {week}", body)


def issue_rows(issues: list[dict[str, Any]]) -> str:
    return "".join(
        f"<tr><td>{position}</td><td>{escape(item['issue'])}</td>"
        f"<td>{item['score']:.2f}</td></tr>"
        for position, item in enumerate(issues, start=1)
    )


def outcome_row(label: str, outcome: dict[str, Any]) -> str:
    verdict = "Probable" if outcome["likely"] else "Poco probable"
    return (
        f"<tr><td>{label}</td><td>{percent(outcome['probability'])}</td>"
        f"<td>{percent(outcome['threshold'])}</td><td>{verdict}</td></tr>"
    )


def similar_rows(similar: list[dict[str, Any]]) -> str:
    return "".join(
        f"<tr><td>{item['similarity']:.3f}</td><td>{escape(item['received'])}</td>"
        f"<td>{escape(item['product'])}</td><td>{escape(item['issue'])}</td>"
        f"<td>{yes_no(item['relief'])}</td><td>{yes_no(item['monetary'])}</td></tr>"
        for item in similar
    )


def index_page(week: str, cards: list[dict[str, Any]]) -> str:
    links = "".join(
        f"<li><a href='ficha_{escape(card['action']['situation'])}.html'>"
        f"{escape(SITUATIONS[card['action']['situation']])}</a>: reclamo "
        f"{escape(card['complaint_id'])}</li>"
        for card in cards
    )
    body = (
        "<h1>Demo de triaje (M12)</h1>"
        f"<p><a href='panel.html'>Panel semanal de patrones, semana del {escape(week)}"
        "</a></p><h2>Fichas de reclamos de la semana siguiente</h2>"
        f"<ul>{links}</ul>"
    )
    return page("Demo de triaje", body)


def card_page(card: dict[str, Any], chart: str) -> str:
    pattern, status = card["pattern"], card["pattern_status"]
    deadline, action = card["deadline"], card["action"]
    novel = "<span class='alert'>Sí</span>" if pattern["is_novel"] else "No"
    alert = "<span class='alert'>Sí</span>" if status["active_alert"] else "No"
    level = percent(status["level_change"], signed=True)
    body = (
        f"<h1>Ficha de triaje: reclamo {escape(card['complaint_id'])}</h1>"
        "<div class='box'><b>Acción sugerida "
        f"({escape(SITUATIONS[action['situation']])}):</b> "
        f"{escape(ACTIONS[action['situation']])}</div>"
        "<h2>Reclamo</h2>"
        f"<p>Recibido el {escape(card['received'])} · {escape(card['product'])}</p>"
        f"<p class='text'>{escape(card['text'])}…</p>"
        "<h2>Motivo probable (T1)</h2><table><tr><th>#</th><th>Motivo</th>"
        f"<th>Puntaje</th></tr>{issue_rows(card['t1'])}</table>"
        "<h2>Resultado probable</h2><table><tr><th></th><th>Probabilidad</th>"
        "<th>Umbral</th><th>Lectura</th></tr>"
        f"{outcome_row('Alguna solución (T2)', card['t2'])}"
        f"{outcome_row('Compensación monetaria (T3)', card['t3'])}</table>"
        f"<h2>Patrón {pattern['cluster_id']}</h2>"
        f"<p>{escape(status['representative_terms'])} · producto principal: "
        f"{escape(status['main_product'])}</p>"
        f"<p>¿Novedoso? {novel} (distancia {pattern['distance']:.2f}; umbral "
        f"{pattern['novelty_threshold']:.2f}) · ¿Dominado por una plantilla? "
        f"{yes_no(status['template_dominated'])}</p>"
        "<h2>Estado del patrón al cierre de la semana del "
        f"{escape(status['week'])}</h2>"
        f"<p>{status['observed']} reclamos (esperados {status['expected']:.0f}) · "
        f"CUSUM {status['cusum']:.2f} · alerta activa: {alert}</p>"
        f"<p>Nivel frente a hace 3 meses: {level}"
        f" (percentil {percent(status['level_percentile'])}) · participación "
        f"{percent(status['share'])}, esperada {percent(status['expected_share'])} "
        f"(95%: {percent(status['share_low'])}–{percent(status['share_high'])})</p>"
        f"{chart}"
        "<h2>Reclamos parecidos (FAISS)</h2><div class='wrap'><table><tr>"
        "<th>Similitud</th><th>Recibido</th><th>Producto</th><th>Motivo</th>"
        "<th>¿Alguna solución?</th><th>¿Compensación?</th></tr>"
        f"{similar_rows(card['similar'])}</table></div>"
        "<h2>Plazo de respuesta (simulado)</h2>"
        f"<p>Si se registrara el {escape(deadline['registered'])}: vence el "
        f"<b>{escape(deadline['due'])}</b> "
        f"({deadline['business_days']} días hábiles) "
        f"o el {escape(deadline['due_extended'])} con extensión "
        f"({deadline['extended_business_days']} días hábiles), sin contar feriados "
        "nacionales.</p>"
    )
    return page(f"Ficha {card['complaint_id']}", body)
