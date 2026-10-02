"""Data, formatting and charts for the decision brief once_alertas.qmd.

Every number comes from data/facts.json, recomputed at full precision from the
frozen panel of M11 and M12, the M9 predictions and an export of complaints
without narratives (see README.md). Numbers are rounded once, half up, when
they are written. Pattern names and families are AI-assisted proposals.
"""

from __future__ import annotations

import html
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def load(name: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


F = load("facts.json")
FAISS = F["faiss"]
CARDS = json.loads((HERE.parent / "triage_demo/cards.json").read_text(encoding="utf-8"))["cards"]
NAMES = {p["cluster_id"]: p for p in load("pattern_names.json")}
FAMILIES = load("pattern_families.json")
# Family B shared its name with pattern 14; it gets a distinct name in the brief.
FAMILY_NAME = {f["familia"]: f["nombre"] for f in FAMILIES} | {"B": "Errores en el reporte de crédito"}
FAMILY_OF = {c: f["familia"] for f in FAMILIES for c in f["patrones"]}

A = {int(c): v for c, v in F["alerts"].items()}
PATH = {c: {r["week"]: r for r in a["path"]} for c, a in A.items()}
H = F["cusum_h"]
READ = "2025-02-03"
KNOWN = ["2025-01-13", "2025-01-20", "2025-01-27", READ]
VERDICTS = {int(c): v for c, v in F["verdicts"].items()}
EXPECTED_VERDICTS = {3: "actuar", 14: "actuar", 12: "vigilar", 4: "eco", 11: "eco", 13: "eco", 17: "eco",
                     19: "eco", 25: "eco", 36: "eco", 8: "incidente"}
assert VERDICTS == EXPECTED_VERDICTS, VERDICTS
VERDICT_ORDER = ["actuar", "vigilar", "eco", "incidente"]
VERDICT_LABEL = {"actuar": "Actuar", "vigilar": "Vigilar", "eco": "No escalar: eco", "incidente": "Incidente"}
VERDICT_CHIP = {"actuar": "act", "vigilar": "watch", "eco": "echo", "incidente": "incident"}


def by_verdict(v: str) -> list[int]:
    return sorted((c for c in VERDICTS if VERDICTS[c] == v), key=lambda c: -A[c]["observed_0203"])


def system_peak(c: int) -> float:
    return max(PATH[c][w]["S"] for w in KNOWN)


# --- Formatting: one rounding, half up -----------------------------------------------

def rounded(value: float, digits: int = 0) -> Decimal:
    return Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)


def minus(text: str) -> str:
    """A typographic minus sign instead of the hyphen."""
    return "−" + text[1:] if text.startswith("-") else text


def n(value: float, digits: int = 0) -> str:
    """Thousands with comma and decimals with point."""
    return minus(f"{rounded(value, digits):,.{digits}f}")


def pct(share: float | None, digits: int = 0) -> str:
    return "—" if share is None else minus(f"{rounded(share * 100, digits):.{digits}f}%")


def per100(share: float) -> str:
    return f"{rounded(share * 100, 0):.0f}"


def times(value: float, digits: int = 2) -> str:
    return minus(f"{rounded(value, digits):.{digits}f}")


def esc(value) -> str:
    return html.escape(str(value))


def name(cluster: int) -> str:
    return NAMES[cluster]["nombre"]


def label(cluster: int) -> str:
    """Name first; the pattern number stays small and gray."""
    return f"{esc(name(cluster))} <span class='pid'>patrón {cluster}</span>"


def quoted(cluster: int) -> str:
    return f"«{esc(name(cluster))}»"


MONTHS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
MONTHS_LONG = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
               "noviembre", "diciembre"]
WEEKDAYS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def day(text: str) -> str:
    stamp = pd.Timestamp(text)
    return f"{stamp.day} {MONTHS[int(stamp.month) - 1]}"


def long_day(text: str) -> str:
    stamp = pd.Timestamp(text)
    return f"{stamp.day} {MONTHS[int(stamp.month) - 1]} {stamp.year}"


def spanish_date(text: str, weekday: bool = False, year: bool = True) -> str:
    stamp = pd.Timestamp(text)
    head = f"{WEEKDAYS[stamp.weekday()]} " if weekday else ""
    tail = f" de {stamp.year}" if year else ""
    return f"{head}{stamp.day} de {MONTHS_LONG[int(stamp.month) - 1]}{tail}"


# --- SVG charts --------------------------------------------------------------------

def nice_max(value: float, step: float) -> float:
    return step * -(-value // step)


def ticks(top: float, step: float) -> list[float]:
    return [step * i for i in range(int(round(top / step)) + 1)]


def tip(*lines: str) -> str:
    return esc("|".join(lines))


def svg(width: int, height: int, body: str, title: str, cls: str = "") -> str:
    return (
        f"<svg class='chart {cls}' viewBox='0 0 {width} {height}' role='img' "
        f"aria-label='{esc(title)}' style='min-width:{min(width, 560)}px'>"
        f"<title>{esc(title)}</title>{body}"
        "<line class='cross' x1='0' x2='0' y1='0' y2='0' hidden></line></svg>"
    )


def column_path(x: float, top: float, width: float, height: float, radius: float = 4) -> str:
    """A column with a rounded data end and a square baseline."""
    if height <= 0:
        return ""
    r = min(radius, width / 2, height)
    bottom = top + height
    return (
        f"M{x:.1f},{bottom:.1f} V{top + r:.1f} Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} "
        f"H{x + width - r:.1f} Q{x + width:.1f},{top:.1f} {x + width:.1f},{top + r:.1f} "
        f"V{bottom:.1f} Z"
    )


class Frame:
    """One plot area with a band x scale over weeks and a linear y scale."""

    def __init__(self, weeks, left, right, top, bottom, ymax):
        self.weeks = list(weeks)
        self.left, self.right, self.top, self.bottom = left, right, top, bottom
        self.ymax = ymax
        self.band = (right - left) / len(self.weeks)

    def x(self, index: int) -> float:
        return self.left + (index + 0.5) * self.band

    def y(self, value: float) -> float:
        return self.bottom - (value / self.ymax) * (self.bottom - self.top)

    def grid(self, values, fmt) -> str:
        parts = [
            f"<line class='grid' x1='{self.left}' x2='{self.right}' y1='{self.y(v):.1f}' y2='{self.y(v):.1f}'/>"
            f"<text class='tick' x='{self.left - 8}' y='{self.y(v) + 4:.1f}' text-anchor='end'>{fmt(v)}</text>"
            for v in values
        ]
        parts.append(f"<line class='axis' x1='{self.left}' x2='{self.right}' y1='{self.bottom}' y2='{self.bottom}'/>")
        return "".join(parts)

    def month_labels(self, y: float) -> str:
        parts, seen = [], set()
        for index, week in enumerate(self.weeks):
            stamp = pd.Timestamp(week)
            key = (stamp.year, stamp.month)
            if key in seen:
                continue
            seen.add(key)
            text = MONTHS[int(stamp.month) - 1] + (f" {stamp.year}" if stamp.month == 1 or index == 0 else "")
            parts.append(f"<text class='tick' x='{self.x(index):.1f}' y='{y}' text-anchor='middle'>{text}</text>")
        return "".join(parts)

    def reading_line(self, top: float, bottom: float, text: bool = True) -> str:
        x = self.x(self.weeks.index(READ)) + self.band / 2
        label_text = (f"<text class='note' x='{x + 4:.1f}' y='{top + 10:.1f}'>lectura: lunes 10 feb</text>"
                      if text else "")
        return f"<line class='reading' x1='{x:.1f}' x2='{x:.1f}' y1='{top}' y2='{bottom}'/>{label_text}"

    def hits(self, tips: list[str], top: float, bottom: float) -> str:
        return "".join(
            f"<rect class='hit' x='{self.x(i) - self.band / 2:.1f}' y='{top}' width='{self.band:.1f}' "
            f"height='{bottom - top:.1f}' data-cx='{self.x(i):.1f}' data-cy1='{top}' "
            f"data-cy2='{bottom}' data-tip='{t}'/>"
            for i, t in enumerate(tips)
        )


def later(week: str) -> bool:
    """Weeks after the closed week of the reading are drawn faded."""
    return pd.Timestamp(week) > pd.Timestamp(READ)


def week_cell(week: str) -> str:
    """Data-table weeks after the reading carry a «después» tag."""
    tag = " <span class='later-tag'>después</span>" if later(week) else ""
    return long_day(week) + tag


def polyline(points, cls: str) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f"<polyline class='{cls}' points='{coords}'/>"


def table(head: list[str], rows: list[list[str]], cls: str = "") -> str:
    """Each cell carries its column name, so narrow screens can show the rows as cards."""
    th = "".join(f"<th>{h}</th>" for h in head)
    labels = [esc(re.sub(r"<[^>]+>", "", h)) for h in head]
    body = "".join("<tr>" + "".join(f"<td data-label='{lab}'>{c}</td>" for lab, c in zip(labels, r)) + "</tr>"
                   for r in rows)
    return f"<table class='{cls}'><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def v1_weekly_columns() -> tuple[str, str]:
    """V1: weekly complaints, the burst pattern under the rest."""
    series = [s for s in F["weekly_totals"] if s["week"] <= "2025-03-31"]
    width, height = 720, 330
    frame = Frame([s["week"] for s in series], 64, 704, 28, 280, 70000)
    body = [frame.grid(ticks(70000, 10000), lambda v: n(v))]
    low, high = frame.y(F["fit_p10_total"]), frame.y(F["fit_p90_total"])
    # The band is named in the legend; a label inside the plot would collide with the columns.
    body.append(f"<rect class='band-ref' x='{frame.left}' y='{high:.1f}' width='{frame.right - frame.left}' "
                f"height='{low - high:.1f}'/>")
    column = min(24.0, frame.band * 0.62)
    tips = []
    for index, s in enumerate(series):
        x = frame.x(index) - column / 2
        faded = " faded" if later(s["week"]) else ""
        y8 = frame.y(s["p8"])
        body.append(f"<rect class='s-burst{faded}' x='{x:.1f}' y='{y8:.1f}' width='{column:.1f}' "
                    f"height='{frame.bottom - y8:.1f}'/>")
        top = frame.y(s["total"])
        body.append(f"<path class='s-rest{faded}' d='{column_path(x, top, column, y8 - top - 2)}'/>")
        tips.append(tip(f"Semana del {long_day(s['week'])}", f"Total: {n(s['total'])}",
                        f"{name(8)}: {n(s['p8'])}", f"Resto: {n(s['total'] - s['p8'])}"))
    peak = next(i for i, s in enumerate(series) if s["week"] == "2025-01-13")
    body.append(f"<text class='val' x='{frame.x(peak) + column / 2 + 6:.1f}' y='{frame.y(series[peak]['total']) + 12:.1f}'>"
                f"13 ene: {n(series[peak]['total'])}</text>")
    body.append(frame.reading_line(24, frame.bottom))
    body.append(frame.month_labels(frame.bottom + 22))
    body.append(frame.hits(tips, frame.top, frame.bottom))
    chart = svg(width, height, "".join(body), "Reclamos por semana, de octubre de 2024 a marzo de 2025")
    data = table(["Semana", "Total", esc(name(8)), "Resto"],
                 [[week_cell(s["week"]), n(s["total"]), n(s["p8"]), n(s["total"] - s["p8"])] for s in series])
    return chart, data


def v2_anatomy() -> tuple[str, str]:
    """V2: «Disputas legales FCRA», what arrived against what was normal, and the accumulator."""
    rows = [r for r in A[3]["path"] if r["week"] <= "2025-04-28"]
    weeks = [r["week"] for r in rows]
    width, height = 720, 470
    top_panel = Frame(weeks, 64, 704, 24, 250, 4000)
    low_panel = Frame(weeks, 64, 704, 300, 430, 5)
    body = [top_panel.grid(ticks(4000, 1000), lambda v: n(v))]
    band = [(top_panel.x(i), top_panel.y(r["p975"])) for i, r in enumerate(rows)]
    band += [(top_panel.x(i), top_panel.y(r["p025"])) for i, r in reversed(list(enumerate(rows)))]
    body.append(f"<polygon class='band' points='{' '.join(f'{x:.1f},{y:.1f}' for x, y in band)}'/>")
    body.append(polyline([(top_panel.x(i), top_panel.y(r["expected"])) for i, r in enumerate(rows)], "line-expected"))
    body.append(polyline([(top_panel.x(i), top_panel.y(r["observed"])) for i, r in enumerate(rows)], "line-b"))
    for i, r in enumerate(rows):
        faded = " faded" if later(r["week"]) else ""
        body.append(f"<circle class='dot-b{faded}' cx='{top_panel.x(i):.1f}' cy='{top_panel.y(r['observed']):.1f}' r='4'/>")
    focus = weeks.index(READ)
    fr = rows[focus]
    body.append(
        f"<text class='val' x='{top_panel.x(focus) - 8:.1f}' y='{top_panel.y(fr['observed']) - 12:.1f}' text-anchor='end'>"
        f"3 feb: llegaron {n(fr['observed'])}</text>"
        f"<text class='note' x='{top_panel.x(focus) - 8:.1f}' y='{top_panel.y(fr['p025']) + 16:.1f}' text-anchor='end'>"
        f"esperados {n(fr['expected'])} (normal {n(fr['p025'])} a {n(fr['p975'])})</text>"
    )
    body.append(f"<text class='panel-title' x='{top_panel.left}' y='14'>Reclamos por semana: llegaron frente a lo normal</text>")
    body.append(low_panel.grid([0, 1, 2, 3, 4, 5], lambda v: f"{v:.0f}"))
    yh = low_panel.y(H)
    body.append(f"<line class='threshold' x1='{low_panel.left}' x2='{low_panel.right}' y1='{yh:.1f}' y2='{yh:.1f}'/>"
                f"<text class='note' x='{low_panel.right - 4}' y='{yh - 6:.1f}' text-anchor='end'>umbral {times(H)}</text>")
    body.append(polyline([(low_panel.x(i), low_panel.y(min(r["S"], 5))) for i, r in enumerate(rows)], "line-b"))
    for i, r in enumerate(rows):
        cls = "dot-alarm" if r["alarm"] else "dot-b"
        faded = " faded" if later(r["week"]) else ""
        body.append(f"<circle class='{cls}{faded}' cx='{low_panel.x(i):.1f}' cy='{low_panel.y(min(r['S'], 5)):.1f}' r='4'/>")
    body.append(f"<text class='panel-title' x='{low_panel.left}' y='{low_panel.top - 12}'>"
                "Acumulador: suena al pasar el umbral y vuelve a 0</text>")
    body.append(top_panel.reading_line(20, low_panel.bottom))
    body.append(low_panel.month_labels(low_panel.bottom + 22))
    tips = [tip(f"Semana del {long_day(r['week'])}", f"Llegaron: {n(r['observed'])}",
                f"Esperados: {n(r['expected'])} (normal {n(r['p025'])} a {n(r['p975'])})",
                f"Rareza: {times(r['z'])}", f"Acumulador: {times(r['S'])}" + (" · alarma" if r["alarm"] else ""))
            for r in rows]
    body.append(top_panel.hits(tips, top_panel.top, low_panel.bottom))
    chart = svg(width, height, "".join(body), f"{name(3)}: llegaron, rango normal y acumulador")
    data = table(["Semana", "Llegaron", "Esperados", "Rango normal (95%)", "Rareza", "Acumulador", "Alarma"],
                 [[week_cell(r["week"]), n(r["observed"]), n(r["expected"]), f"{n(r['p025'])} a {n(r['p975'])}",
                   times(r["z"]), times(r["S"]), "Sí" if r["alarm"] else "No"] for r in rows])
    return chart, data


def v3_memory() -> tuple[str, str]:
    """V3: the burst pattern, the share the system kept expecting and the real one."""
    expected = F["p8_share_expected"]
    weeks = [w for w in expected if w <= "2025-03-31"]
    real = F["p8_share_real"]
    counts = {s["week"]: s for s in F["weekly_totals"]}
    width, height = 720, 300
    frame = Frame(weeks, 56, 704, 24, 252, 0.6)
    body = [frame.grid([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], lambda v: pct(v))]
    gap = [i for i, w in enumerate(weeks) if w >= "2025-01-20" and expected[w] > real[w]]
    upper = [(frame.x(i), frame.y(expected[weeks[i]])) for i in gap]
    lower = [(frame.x(i), frame.y(real[weeks[i]])) for i in reversed(gap)]
    body.append(f"<polygon class='gap' points='{' '.join(f'{x:.1f},{y:.1f}' for x, y in upper + lower)}'/>")
    body.append(polyline([(frame.x(i), frame.y(expected[w])) for i, w in enumerate(weeks)], "line-expected"))
    body.append(polyline([(frame.x(i), frame.y(real[w])) for i, w in enumerate(weeks)], "line-burst"))
    for i, w in enumerate(weeks):
        faded = " faded" if later(w) else ""
        body.append(f"<circle class='dot-burst{faded}' cx='{frame.x(i):.1f}' cy='{frame.y(real[w]):.1f}' r='4'/>")
        body.append(f"<circle class='dot-expected{faded}' cx='{frame.x(i):.1f}' cy='{frame.y(expected[w]):.1f}' r='4'/>")
    focus = weeks.index(READ)
    body.append(
        f"<text class='val' x='{frame.x(focus) + 8:.1f}' y='{frame.y(expected[READ]) - 10:.1f}'>esperado {pct(expected[READ], 1)}</text>"
        f"<text class='val' x='{frame.x(focus) + 8:.1f}' y='{frame.y(real[READ]) + 20:.1f}'>real {pct(real[READ], 1)}</text>"
    )
    peak = weeks.index("2025-01-13")
    body.append(f"<text class='note' x='{frame.x(peak) + 8:.1f}' y='{frame.y(real['2025-01-13']) + 4:.1f}'>"
                f"ráfaga {pct(real['2025-01-13'])}</text>")
    body.append(frame.reading_line(20, frame.bottom))
    body.append(frame.month_labels(frame.bottom + 22))
    tips = [tip(f"Semana del {long_day(w)}", f"Porción real: {pct(real[w], 1)}",
                f"Porción que esperaba el sistema: {pct(expected[w], 1)}") for w in weeks]
    body.append(frame.hits(tips, frame.top, frame.bottom))
    chart = svg(width, height, "".join(body), f"Porción de {name(8)}: la real y la que esperaba el sistema")
    data = table(["Semana", "Reclamos", "Porción real", "Porción esperada por el sistema"],
                 [[week_cell(w), n(counts[w]["p8"]), pct(real[w], 1), pct(expected[w], 1)] for w in weeks])
    return chart, data


def v4_accumulators() -> tuple[str, str]:
    """V4: the highest accumulator of each alert from 13 Jan to 3 Feb, with and without the echo.

    An HTML dumbbell, so names and dots stay legible at any width: the name sits beside the
    track on wide screens and above it on narrow ones.
    """
    groups = [(v, sorted(by_verdict(v), key=lambda c: -A[c]["free_S_max"])) for v in ("actuar", "vigilar", "eco")]
    hi = 7.0

    def left(value: float) -> str:
        return f"{min(max(value, 0.0), hi) / hi * 100:.2f}%"

    threshold = f"<span class='db-threshold' style='left:{left(H)}'></span>"
    parts, rows = ["<p class='db-title'>Acumulador más alto del 13 de enero al 3 de febrero</p>"], []
    for v, clusters in groups:
        parts.append(f"<p class='db-group'>{esc(VERDICT_LABEL[v])}</p>")
        for c in clusters:
            with_echo, without = system_peak(c), A[c]["free_S_max"]
            low, high = sorted((with_echo, without))
            parts.append(
                f"<div class='db-row' tabindex='0' data-tip='{tip(name(c), f'Con el eco (sistema): {times(with_echo)}', f'Sin el eco: {times(without)}', f'Umbral: {times(H)}')}'>"
                f"<span class='db-name'>{esc(name(c))}</span><span class='db-track'>{threshold}"
                f"<span class='db-link' style='left:{left(low)};width:calc({left(high)} - {left(low)})'></span>"
                f"<span class='db-ring' style='left:{left(with_echo)}'></span>"
                f"<span class='db-dot' style='left:{left(without)}'></span></span></div>"
            )
            rows.append([esc(name(c)), esc(VERDICT_LABEL[v]), times(with_echo), times(without)])
    ticks_html = "".join(f"<span class='db-tick' style='left:{left(t)}'>{t}</span>" for t in range(0, 8))
    parts.append(f"<div class='db-row db-scale-row' aria-hidden='true'><span class='db-name'></span><span class='db-scale'>"
                 f"{ticks_html}<span class='db-tick db-umbral' style='left:{left(H)}'>umbral {times(H)}</span></span></div>")
    chart = (f"<div class='dumbbell' role='img' aria-label='Acumulador de cada alerta con y sin el eco, frente al umbral "
             f"de {times(H)}'>{''.join(parts)}</div>")
    data = table(["Alerta", "Decisión", "Con el eco (sistema)", "Sin el eco"], rows)
    return chart, data


def v6_real_changes() -> tuple[str, str]:
    """V6: the two real increases against what was expected, with and without the echo."""
    panels, data_rows = [], []
    letters = F["p3_letters_weekly"]
    notes = {14: [("2025-01-13", "13 ene: el sistema lo vio por debajo")],
             3: [("2025-01-20", "22 ene: 5 cartas nuevas"), ("2025-05-19", "19 may: dejan de llegar")]}
    for cluster, cls in ((14, "a"), (3, "b")):
        rows = A[cluster]["path"]
        weeks = [r["week"] for r in rows]
        free = A[cluster]["free_expected"]
        top = max(max(r["observed"] for r in rows), max(r["expected"] for r in rows), max(free[w] for w in weeks))
        step = 2000 if top > 6000 else 1000
        ymax = nice_max(top * 1.08, step)
        width, height = 720, 270
        frame = Frame(weeks, 60, 704, 34, 226, ymax)
        body = [frame.grid(ticks(ymax, step), lambda v: f"{n(v / 1000)} mil" if v else "0")]
        if cluster == 3:
            area = [(frame.x(i), frame.y(letters.get(w, 0))) for i, w in enumerate(weeks)]
            area = [(frame.x(0), frame.bottom)] + area + [(frame.x(len(weeks) - 1), frame.bottom)]
            body.append(f"<polygon class='area-letters' points='{' '.join(f'{x:.1f},{y:.1f}' for x, y in area)}'/>")
        before = A[cluster]["cal_mean"]
        # The average is named in the legend; a label here would sit on the data line.
        body.append(f"<line class='before' x1='{frame.left}' x2='{frame.right}' y1='{frame.y(before):.1f}' "
                    f"y2='{frame.y(before):.1f}'/>")
        body.append(polyline([(frame.x(i), frame.y(r["expected"])) for i, r in enumerate(rows)], "line-expected"))
        body.append(polyline([(frame.x(i), frame.y(free[w])) for i, w in enumerate(weeks)], "line-noecho"))
        body.append(polyline([(frame.x(i), frame.y(r["observed"])) for i, r in enumerate(rows)], f"line-{cls}"))
        for i, r in enumerate(rows):
            faded = " faded" if later(r["week"]) else ""
            body.append(f"<circle class='dot-{cls}{faded}' cx='{frame.x(i):.1f}' cy='{frame.y(r['observed']):.1f}' r='3.5'/>")
        for week, text in notes[cluster]:
            i = weeks.index(week)
            anchor = "end" if i > len(weeks) / 2 else "start"
            dx = -6 if anchor == "end" else 6
            body.append(f"<text class='note' x='{frame.x(i) + dx:.1f}' y='{frame.top - 8:.1f}' text-anchor='{anchor}'>{text}</text>"
                        f"<line class='reading' x1='{frame.x(i):.1f}' x2='{frame.x(i):.1f}' y1='{frame.top - 4}' y2='{frame.bottom}'/>")
        body.append(f"<text class='panel-title' x='{frame.left}' y='14'>{esc(name(cluster))}</text>")
        body.append(frame.reading_line(frame.top, frame.bottom, text=False))
        body.append(frame.month_labels(frame.bottom + 20))
        tips = []
        for r in rows:
            w = r["week"]
            lines = [f"Semana del {long_day(w)}", f"Llegaron: {n(r['observed'])}", f"Esperado por el sistema: {n(r['expected'])}",
                     f"Esperado sin el eco: {n(free[w])}"]
            if cluster == 3:
                lines.append(f"Copias de las 5 cartas nuevas: {n(letters.get(w, 0))}")
            tips.append(tip(*lines))
            data_rows.append([esc(name(cluster)), week_cell(w), n(r["observed"]), n(r["expected"]), n(free[w]),
                              n(letters.get(w, 0)) if cluster == 3 else "—"])
        body.append(frame.hits(tips, frame.top, frame.bottom))
        panels.append(svg(width, height, "".join(body), f"{name(cluster)}: llegaron frente a lo esperado"))
    data = table(["Patrón", "Semana", "Llegaron", "Esperado por el sistema", "Esperado sin el eco", "Copias de las cartas nuevas"],
                 data_rows)
    return "".join(panels), data


def range_bar(low: float, mid: float, high: float, observed: float, fmt, cls: str = "b") -> str:
    """One reading rule for both ranges: a dot outside the bar is unusual."""
    width, height = 300, 56
    lo, hi = min(low, observed) * 0.9, max(high, observed) * 1.06

    def x(v):
        return 12 + (v - lo) / (hi - lo) * (width - 24)

    return (
        f"<svg class='rangebar' viewBox='0 0 {width} {height}' role='img' "
        f"aria-label='Rango normal de {fmt(low)} a {fmt(high)}; esperado {fmt(mid)}; llegó {fmt(observed)}'>"
        f"<text class='tick' x='{x(mid):.1f}' y='11' text-anchor='middle'>esperado {fmt(mid)}</text>"
        f"<line class='rb-track' x1='{x(low):.1f}' x2='{x(high):.1f}' y1='26' y2='26'/>"
        f"<line class='rb-mid' x1='{x(mid):.1f}' x2='{x(mid):.1f}' y1='19' y2='33'/>"
        f"<circle class='dot-{cls}' cx='{x(observed):.1f}' cy='26' r='5'/>"
        f"<text class='tick' x='{x(low):.1f}' y='50' text-anchor='middle'>{fmt(low)}</text>"
        f"<text class='tick' x='{x(high):.1f}' y='50' text-anchor='middle'>{fmt(high)}</text>"
        f"<text class='val' x='{x(observed):.1f}' y='50' text-anchor='middle'>llegó {fmt(observed)}</text>"
        "</svg>"
    )
