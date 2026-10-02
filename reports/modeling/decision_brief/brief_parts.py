"""HTML parts of once_alertas.qmd that are not running text: tables, cards and boxes.

The prose lives in the .qmd. These functions return HTML strings that the .qmd
prints inside raw HTML blocks with `show()`.
"""

from __future__ import annotations

from IPython.display import HTML, Markdown  # noqa: F401 (HTML is used by the .qmd)

from brief_lib import (
    A, CARDS, F, FAISS, FAMILIES, FAMILY_NAME, H, NAMES, PATH, READ, VERDICT_CHIP, VERDICT_LABEL, VERDICTS, by_verdict,
    day, esc, label, long_day, n, name, pct, per100, quoted, range_bar, rounded, spanish_date, table, times,
)

# --- Facts used across the document -------------------------------------------

CARD = next(c for c in CARDS if c["action"]["situation"] == "alert")
NORMAL_CARD = next(c for c in CARDS if c["action"]["situation"] == "normal")
NOVEL_CARD = next(c for c in CARDS if c["action"]["situation"] == "novel")
CONF = {level: sum(1 for p in NAMES.values() if p["confianza"] == level) for level in ("alta", "media", "baja")}
TOTALS = {s["week"]: s for s in F["weekly_totals"]}
P8_REAL, P8_EXP = F["p8_share_real"], F["p8_share_expected"]
DET = F["m11_detection"]
PER_MONTH = F["m11_alarms_per_month"]
OUTSIDE_PER_MONTH = F["outside_alarms"] / F["outside_weeks"] * F["weeks_per_month"]
EXAMPLE = next(e for e in F["accumulator_examples"] if e["cluster_id"] == 29 and e["split"] == "calibration")
LETTERS = F["p3_letters"]
EPISODE = ["2025-01-27", "2025-02-03", "2025-02-10", "2025-02-17"]
EPISODE_FREE = sum(len(F["free_alarms_by_week"][w]) for w in EPISODE)
T1, T2, T3 = F["t1"], F["t2"], F["t3"]
M9C, M10C = F["m9_coverage"], F["m10_coverage"]
a3, a8, a12, a14 = A[3], A[8], A[12], A[14]
WATCH, ECHO = by_verdict("vigilar"), by_verdict("eco")
assert len(WATCH) == 1 and len(ECHO) == 7
# Echo patterns that alarmed without the echo in the two weeks after the reading.
ECHO_LATER = sorted(((c, A[c]["free_alarms_2025"][0]) for c in ECHO
                     if A[c]["free_alarms_2025"] and A[c]["free_alarms_2025"][0] <= "2025-02-17"), key=lambda x: x[1])
peak_ratio_cal = F["peak_total"] / F["cal_median_total"]
peak_ratio_fit = F["peak_total"] / F["fit_median_total"]
p8_share_peak = F["peak_p8"] / F["peak_total"]
p8_vs_cal = a8["share_0203"] / a8["cal_share"]
echo_share = F["p8_deficit_0203"] / F["excess_0203"]
# Capacity range for pattern 14: from the week of 6 Jan, before the burst, to the last closed week, in hundreds.
PLAN = [float(rounded(F["p14_0106"] / 100) * 100), float(rounded(PATH[14][READ]["observed"] / 100) * 100)]
PLAN_RATIO = [F["p14_0106"] / a14["cal_mean"], PATH[14][READ]["observed"] / a14["cal_mean"]]

ES = {
    "Incorrect information on your report": "Información incorrecta en su reporte",
    "Improper use of your report": "Uso indebido de su reporte",
    "Problem with a company's investigation into an existing problem": "Problema con la investigación de la empresa",
    "Other transaction problem": "Otro problema con una transacción",
    "Information belongs to someone else": "la información pertenece a otra persona",
    "Their investigation did not fix an error on your report": "su investigación no corrigió el error de su reporte",
    "Money transfer, virtual currency, or money service": "transferencias de dinero, moneda virtual o servicios de dinero",
}


def es(text: str) -> str:
    """The Spanish label, with the original CFPB text on hover."""
    return f"<span title='{esc(text)}'>{esc(ES.get(text, text))}</span>"


def qname(c: int) -> str:
    """A pattern name in quotes for running text; Quarto escapes inline output itself."""
    return f"«{name(c)}»"


def md_label(c: int) -> Markdown:
    """Name first and the pattern number small and gray, as Markdown for running text."""
    return Markdown(f"{name(c)} [patrón {c}]{{.pid}}")


def show(html_text: str) -> None:
    """Print HTML as a raw block, so Pandoc passes it through untouched."""
    print("```{=html}\n" + html_text + "\n```")


def outside_plan(weeks: list[str]) -> list[str]:
    """Second of two consecutive weeks with pattern 14 outside the capacity range."""
    flags = [not PLAN[0] <= PATH[14][w]["observed"] <= PLAN[1] for w in weeks]
    return [weeks[i] for i in range(1, len(weeks)) if flags[i] and flags[i - 1]]


AFTER = [r["week"] for r in a14["path"] if r["week"] > READ]
REPLAN = outside_plan(AFTER)
REPLAN_DOWN = next(w for w in REPLAN if PATH[14][w]["observed"] < PLAN[0])
REPLAN_UP = next(w for w in REPLAN if PATH[14][w]["observed"] > PLAN[1])


# --- Small blocks --------------------------------------------------------------

def chip(kind: str, text: str) -> str:
    return f"<span class='chip chip-{kind}'><span class='chip-dot' aria-hidden='true'></span>{text}</span>"


def verdict_chip(c: int) -> str:
    return chip(VERDICT_CHIP[VERDICTS[c]], VERDICT_LABEL[VERDICTS[c]])


def legend(*items: tuple[str, str]) -> str:
    return "<p class='legend'>" + "".join(
        f"<span class='legend-item'><span class='key key-{cls}' aria-hidden='true'></span>{text}</span>"
        for cls, text in items) + "</p>"


def figure(chart: str, data: str, message: str, note: str = "") -> str:
    note_html = f" <span class='muted'>{note}</span>" if note else ""
    return (
        f"<figure class='fig'><figcaption><strong>{message}</strong>{note_html}</figcaption>"
        f"<div class='scroll'>{chart}</div>"
        "<details class='data'><summary>Ver los datos del gráfico</summary>"
        f"<div class='scroll'>{data}</div></details></figure>"
    )


def scroll_table(head: list[str], rows: list[list[str]], cls: str = "") -> str:
    return f"<div class='scroll'>{table(head, rows, cls)}</div>"


def names_list(clusters) -> str:
    return ", ".join(quoted(c) for c in clusters)


def and_list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1]


# --- First screen ----------------------------------------------------------------

def tiles() -> str:
    return (
        "<div class='tiles'>"
        f"<div class='tile'><p class='tile-value'>{n(F['peak_total'])}</p><p class='tile-label'>reclamos la semana del "
        f"13 de enero, {times(peak_ratio_cal, 1)} veces una semana normal</p></div>"
        f"<div class='tile'><p class='tile-value'>{pct(p8_share_peak)}</p><p class='tile-label'>de esa semana fue de "
        f"un solo patrón, {quoted(8)}</p></div>"
        f"<div class='tile'><p class='tile-value'>{F['alarms_by_week_2025'][READ]}</p><p class='tile-label'>alarmas la "
        f"semana del 3 de febrero; el máximo anterior era {F['fit_alarms_max']}</p></div>"
        "<div class='tile'><p class='tile-value'>2 de 11</p><p class='tile-label'>patrones con alerta suenan también sin "
        "el eco</p></div></div>"
    )


DECISION_HEAD = ["Decisión", "Qué hacer", "Quién y cuándo (propuesto)", "Cómo sabremos si funcionó"]


def decisions_this_week() -> str:
    rows = [
        [chip("act", "Actuar"),
         f"Ajustar la capacidad para {quoted(14)}: planificar entre {n(PLAN[0])} y {n(PLAN[1])} reclamos por semana, de "
         f"{times(PLAN_RATIO[0], 1)} a {times(PLAN_RATIO[1], 1)} veces su promedio de fin de 2024, y replanificar si el "
         "volumen sale de ese rango dos lunes seguidos.",
         "Gerencia de atención al cliente · plan antes del viernes 14 de febrero; revisión cada lunes",
         f"Las respuestas a tiempo no bajan del {pct(F['timely_cal_p14'], 1)} de octubre a diciembre de 2024."],
        [chip("act", "Actuar"),
         "Agrupar las copias de cinco cartas modelo nuevas (textos idénticos que muchos consumidores copian) de "
         f"{quoted(3)} y responder cada una con un texto estándar revisado por Legal, verificando la identidad del "
         f"cliente, porque dicen que la información es de otra persona. Van {n(F['letters_rows'])} copias; las primeras "
         f"{n(F['letters_due_this_week'])} vencen esta semana.",
         "Liderazgo de triaje con Legal y Cumplimiento · esta semana",
         "Cada copia se responde dentro del plazo, y las copias nuevas se cuentan cada lunes."],
        [chip("watch", "Vigilar"),
         f"No escalar todavía {quoted(12)}: sin el eco no suena, pero su acumulador está cerca del umbral desde antes de "
         f"la ráfaga ({times(a12['free_S_0203'])} de {times(H)} el 3 de febrero). Si lo pasa, escalar ese mismo lunes a "
         "la Jefatura de reclamos.",
         "Analítica recalcula cada lunes y Triaje revisa 30 reclamos si suena · hasta la lectura del lunes 3 de marzo",
         "Si suena, la revisión dice antes del viernes si es un aumento real."],
        [chip("echo", "No escalar: eco"),
         "No abrir investigaciones por las otras 7 alertas. Avisar a las áreas dueñas de esos productos que son eco, y "
         "rehacer cada lunes la cuenta sin la ráfaga: la que suene se trata como «Vigilar».",
         "Gerencia de riesgo avisa y Analítica recalcula · hasta la lectura del lunes 3 de marzo, la primera sin las "
         "semanas del 13 y 20 de enero en su referencia",
         "Una muestra de 30 reclamos de cada patrón no muestra un tema nuevo."],
        [chip("incident", "Incidente"),
         f"Abrir un incidente por la ráfaga de {quoted(8)}. Con la regla de 15 días hábiles, sus "
         f"{n(F['burst_due_before_reading'])} reclamos de la semana del 13 de enero vencieron entre el 3 y el 7 de "
         f"febrero, y los {n(F['burst_due_this_week'])} de la semana del 20 vencen esta semana. Averiguar la causa. Su "
         "alerta del panel no pide acción: se apaga sola el lunes 17 de febrero si no vuelve a sonar.",
         "Riesgo operacional con el dueño del producto de transferencias · hoy; debió abrirse el 16 o 17 de enero",
         "Todos respondidos o con extensión notificada, y la causa identificada antes del viernes 14 de febrero."],
        [chip("fix", "Dotación"),
         f"Ajustar la dotación total: la semana del 6 de enero, antes de la ráfaga, ya llegaron {n(F['total_0106'])} "
         f"reclamos, un {pct(F['total_0106'] / F['cal_median_total'] - 1)} más que una semana normal de fin de 2024, y el "
         "panel no avisa por esto.",
         "Planeamiento de personal · antes del viernes 14 de febrero; luego cada mes",
         "La dotación sigue al total semanal y a su rango histórico."],
    ]
    return scroll_table(DECISION_HEAD, rows, "decisions")


def decisions_before_pilot() -> str:
    rows = [
        [chip("fix", "Sistema"),
         "Mostrar la cuenta sin la ráfaga como aviso de eco, contar las cartas modelo nuevas cada semana y comparar con "
         "una base fija (<a href='#sec-decisiones'>Decisiones y cambios</a>).",
         "Analítica y el dueño del proyecto",
         "Cada cambio se deja por escrito antes de probarlo, se prueba con ráfagas simuladas y se evalúa con datos "
         "nuevos."],
        [chip("fix", "Gerencia"),
         "Decidir cuántas alarmas aceptar: unas 2 a 4 al mes, o subir el umbral y detectar menos "
         "(<a href='#sec-decisiones'>Decisiones y cambios</a>).",
         "Gerencia de riesgo con atención al cliente", "La decisión queda escrita, con lo que se deja de detectar."],
    ]
    return scroll_table(DECISION_HEAD, rows, "decisions")


def confidence_cards() -> str:
    cards = [
        ("Seguro", "Lo que pasó.", "La ráfaga, su tamaño, sus empresas y las alarmas son conteos directos."),
        ("Alto, revisado cada lunes", "Que 7 alertas son eco.",
         f"Lo que el sistema le reservó de más a la ráfaga es el {pct(echo_share)} de lo que pareció sobrar en los demás. "
         "Pero el criterio se definió después de ver este episodio, el único, y un eco puede tapar un aumento que empieza "
         "después."),
        ("Alto", "Que 2 son aumentos reales.", "Suenan sin el eco en todas las formas de quitarlo que probamos. Uno "
         "crecía desde 2024; el otro tiene cartas nuevas identificables."),
        ("Desconocido", "La causa de la ráfaga.", "Los datos muestran quién, qué y cuándo, no por qué."),
    ]
    body = "".join(f"<div class='conf'><p class='conf-level'>{level}</p><p><b>{what}</b> {why}</p></div>"
                   for level, what, why in cards)
    return ("<div class='confidence' aria-label='Qué tan seguros estamos'><p class='conf-title'>Qué tan seguros "
            f"estamos</p>{body}</div>")


# --- 1. Pieces --------------------------------------------------------------------

def lanes() -> str:
    return (
        "<div class='lanes' role='img' aria-label='Dos vistas del sistema: la ficha de cada reclamo y el panel de cada "
        "semana, unidas por el patrón'>"
        "<div class='lane'><p class='lane-title'>Ficha · el agente · cada reclamo</p><div class='flow'>"
        "<span class='node'>Texto del reclamo</span><span class='arrow' aria-hidden='true'>→</span>"
        "<span class='node-group'><span class='node'>Motivo probable</span><span class='node'>¿Alguna solución?</span>"
        "<span class='node'>¿Compensación en dinero?</span><span class='node'>Patrón y borde del grupo</span>"
        "<span class='node'>Reclamos parecidos</span><span class='node'>Plazo</span></span>"
        "<span class='arrow' aria-hidden='true'>→</span><span class='node strong'>Acción sugerida</span></div></div>"
        "<p class='lane-link'>La ficha toma la alerta del patrón al que pertenece el reclamo</p>"
        "<div class='lane'><p class='lane-title'>Panel · quien decide · cada semana</p><div class='flow'>"
        "<span class='node'>Reclamos por patrón</span><span class='arrow' aria-hidden='true'>→</span>"
        "<span class='node'>Esperados y rango</span><span class='arrow' aria-hidden='true'>→</span>"
        "<span class='node'>Rareza</span><span class='arrow' aria-hidden='true'>→</span>"
        "<span class='node'>Acumulador</span><span class='arrow' aria-hidden='true'>→</span>"
        "<span class='node strong'>Alarma y alerta</span></div>"
        "<p class='lane-context'>Contexto, sin alarmas: porción esperada y nivel frente a hace 3 meses</p></div></div>"
    )


def pieces_table() -> str:
    p3 = PATH[3][READ]
    rows = [
        ["Ficha", "Motivo probable <span class='tag'>T1</span>", "¿De qué se queja?", "3 motivos en orden, sin puntaje",
         es(CARD["t1"][0]["issue"])],
        ["Ficha", "¿Alguna solución? <span class='tag'>T2</span>", "¿Es probable que la empresa dé algún remedio, en "
         "dinero o no?", "«probable» o «no probable»", "Probable" if CARD["t2"]["likely"] else "No probable"],
        ["Ficha", "¿Compensación en dinero? <span class='tag'>T3</span>", "¿Es probable que termine con un pago?",
         "«probable» o «no probable»", "Probable" if CARD["t3"]["likely"] else "No probable"],
        ["Ficha", "Patrón y borde del grupo <span class='tag'>M8</span>", "¿A qué grupo pertenece y es típico de él?",
         "el nombre del patrón y una marca si está en el borde", "típico de su grupo"],
        ["Ficha", "Reclamos parecidos <span class='tag'>FAISS</span>", "¿Cómo terminaron casos casi iguales?",
         "5 reclamos ya resueltos, con su similitud", "5 casi copias de 2024"],
        ["Ficha", "Plazo", "¿Para cuándo hay que responder?", "una fecha a 15 días hábiles, o 45 con extensión",
         "regla fija, no un modelo"],
        ["Panel", "Esperados y rango normal <span class='tag'>M9 · Binomial Negativa</span>", "Con el total de la "
         "semana, ¿cuántos eran normales?", "un número esperado y un rango del 95%",
         f"{n(p3['expected'])} ({n(p3['p025'])} a {n(p3['p975'])})"],
        ["Panel", "Rareza", "¿Qué tan raro fue?", "un puntaje: 0 es lo típico y 2 queda cerca del límite del rango",
         f"{times(p3['z'])}, el tope"],
        ["Panel", "Acumulador <span class='tag'>M11 · CUSUM</span>", "¿Se viene sumando?",
         f"una suma semanal que suena al pasar {times(H)}", f"{times(p3['S'])}: alarma"],
        ["Panel", "Alerta activa", "¿Hubo una alarma hace poco?", "sí, si sonó en las últimas 4 semanas cerradas", "sí"],
        ["Panel", "Porción esperada <span class='tag'>M10 · Dirichlet-Multinomial</span>", "¿Qué parte del total le "
         "tocaba?", "un porcentaje esperado y su rango",
         f"{pct(a3['m10_expected_share_0203'], 1)} ({pct(a3['m10_low_0203'], 1)} a {pct(a3['m10_high_0203'], 1)})"],
        ["Panel", "Nivel frente a hace 3 meses <span class='tag'>Δ</span>", "¿Está más alto que hace un trimestre?",
         "un cambio y su percentil",
         f"{pct(a3['level_0203'], 1)}; aquí engaña (ver <a href='#sec-alerta'>Cómo se lee una alerta</a>)"],
    ]
    return scroll_table(["Vista", "Pieza", "Pregunta que responde", "Qué entrega", "En el ejemplo de febrero"], rows)


# --- 2. A normal week ------------------------------------------------------------

def families_table() -> str:
    share = F["family_share"]
    rows = [
        [f"<b>{esc(FAMILY_NAME[f['familia']])}</b>", str(len(f["patrones"])),
         f"<span class='bar' style='--w:{share['fit'][f['familia']] * 100:.1f}%'></span>{pct(share['fit'][f['familia']], 1)}",
         pct(share["calibration"][f["familia"]], 1),
         f"<span class='later-cell'>{pct(share['validation'][f['familia']], 1)}</span>"]
        for f in sorted(FAMILIES, key=lambda f: -share["fit"][f["familia"]])
    ]
    return scroll_table(["Familia", "Patrones", "Porción, ene 2023 a sep 2024", "Oct a dic 2024",
                         "<span class='later-cell'>Ene a jun 2025 (después)</span>"], rows)


# --- 4. Anatomy of one alert --------------------------------------------------------

def piece_card(title: str, tag: str, rows: list[tuple[str, str]]) -> str:
    items = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
    return f"<article class='piece'><h3>{title} <span class='tag'>{tag}</span></h3><dl>{items}</dl></article>"


def piece_cards() -> str:
    p, r = PATH[3], PATH[3][READ]
    ratio = r["observed"] / r["expected"]
    total = TOTALS[READ]["total"]
    g10, g20, none = DET["growth_10"], DET["growth_20"], DET["none"]
    run = EXAMPLE["run"]
    count_bar = range_bar(r["p025"], r["expected"], r["p975"], r["observed"], lambda v: n(v))
    share_bar = range_bar(a3["m10_low_0203"], a3["m10_expected_share_0203"], a3["m10_high_0203"], a3["share_0203"],
                          lambda v: pct(v, 1))
    cards = [
        piece_card("1 · Reclamos esperados y rango normal", "M9 · Binomial Negativa", [
            ("Pregunta", "Con el total que llegó esta semana, ¿cuántos reclamos de este patrón eran normales?"),
            ("Qué entrega", "El esperado es el total de la semana por la porción que tuvo el patrón en las 4 semanas "
             "anteriores. El rango del 95% sale de una Binomial Negativa, la forma estándar de describir conteos que "
             "varían más que el azar puro. En calibración, el rango contuvo el valor real en el "
             f"{pct(M9C['calibration']['in95'], 1)} de los casos."),
            ("En el ejemplo", f"{n(total)} × {pct(r['expected'] / total, 2)} ≈ {n(r['expected'])}, con un rango de "
             f"{n(r['p025'])} a {n(r['p975'])}. Llegaron {n(r['observed'])}: {times(ratio, 1)} veces lo esperado y "
             "fuera del rango." + count_bar),
            ("Cómo leerlo", "Si el punto queda fuera de la barra, la semana fue rara."),
            ("Qué no dice", "No avisa si sube el total, porque lo toma como dado. Y tiene memoria de 4 semanas: si esas "
             "semanas venían distorsionadas, lo esperado también (ver <a href='#sec-eco'>El eco</a>)."),
        ]),
        piece_card("2 · Rareza", "z", [
            ("Pregunta", "¿Qué tan improbable era un conteo así de alto?"),
            ("Qué entrega", "Un puntaje de qué tan improbable era el conteo según la Binomial Negativa: 0 es lo típico, "
             f"2 queda cerca del límite del rango normal y el tope es {times(F['z_cap'])} (<a href='#anexos'>anexo "
             "A4</a>)."),
            ("En el ejemplo", f"{times(r['z'])}, el tope. Varios patrones llegan al mismo tope."),
            ("Cómo leerlo", f"Más alto es más raro. Junto a ella conviene mostrar «veces lo esperado» ({times(ratio, 1)}) "
             "y el rango."),
            ("Qué no dice", "No ordena las alertas que llegan al tope."),
        ]),
        piece_card("3 · Acumulador", "M11 · CUSUM", [
            ("Pregunta", "¿Es una semana rara suelta o algo que se viene acumulando?"),
            ("Qué entrega", f"Cada semana suma la rareza menos 0.5, nunca baja de 0 y suena una alarma al pasar el umbral "
             f"de {times(H)}; después vuelve a 0."),
            ("En el ejemplo", f"Sonó el 27 de enero ({times(p['2025-01-27']['S'])}), volvió a 0 y sonó otra vez el 3 de "
             f"febrero ({times(r['S'])}), con la rareza en el tope."),
            ("Cómo leerlo", f"Sirve para cambios que se acumulan. En calibración, {quoted(29)} tuvo 4 semanas seguidas "
             "por encima de lo esperado, ninguna lo bastante rara para la regla que mira una semana a la vez (máximo "
             f"{times(max(x['z'] for x in run))} frente a {times(F['weekly_rule_z'])}). El acumulador sumó "
             f"{and_list([times(x['S']) for x in run])}, y sonó el 9 de diciembre. En pruebas con aumentos artificiales "
             f"detectó, en 8 semanas, {per100(g10['cusum']['detected'])} de cada 100 patrones que crecían 10% por semana "
             f"(2.1 veces en 8 semanas) y {per100(g20['cusum']['detected'])} de cada 100 que crecían 20%; mirando una "
             f"semana a la vez se detectaban {per100(g10['weekly']['detected'])} y {per100(g20['weekly']['detected'])}."),
            ("Qué no dice", "No ve caídas. Un aumento que se queda deja de sonar en unas 4 semanas, porque lo esperado lo "
             f"absorbe. Y sin ningún aumento suena en el {pct(none['cusum']['detected'], 1)} de las ventanas de 8 "
             "semanas; con un modelo perfecto sería cerca del 4.6%."),
        ]),
        piece_card("4 · Porción esperada", "M10 · Dirichlet-Multinomial", [
            ("Pregunta", "¿Qué parte del total le tocaba?"),
            ("Qué entrega", "El porcentaje del total que le toca, con su rango del 95%. La Dirichlet-Multinomial reparte "
             "el total entre los 40 patrones a la vez, con la misma referencia de 4 semanas."),
            ("En el ejemplo", f"Tuvo el {pct(a3['share_0203'], 1)} de los reclamos de la semana; le tocaba el "
             f"{pct(a3['m10_expected_share_0203'], 1)}, con un rango de {pct(a3['m10_low_0203'], 1)} a "
             f"{pct(a3['m10_high_0203'], 1)}." + share_bar),
            ("Cómo leerlo", "Es la pieza 1 dicha en porcentaje, más fácil de comunicar. Su rango contuvo el valor real en "
             f"el {pct(M10C['calibration'], 1)} de los casos de calibración."),
            ("Qué no dice", "No hace sonar alarmas ni dice nada del total. Usa la misma referencia de 4 semanas, así que "
             "también tiene eco."),
        ]),
    ]
    return "<div class='pieces'>" + "".join(cards) + "</div>"


# --- 5. The echo ---------------------------------------------------------------------

def why_line(c: int) -> str:
    a = A[c]
    if c == 8:
        return (f"Es la ráfaga. Su alarma del 13 de enero la mantiene activa. El 3 de febrero recibió "
                f"{n(a['observed_0203'])}, menos que los {n(a['m9_expected_0203'])} que esperaba el sistema, pero su "
                f"porción aún era {times(p8_vs_cal)} veces la de fin de 2024.")
    ratio = a["free_ratio"]
    if c == 14:
        return (f"Sin el eco sonó el 13 de enero ({times(ratio['2025-01-13'])} veces lo esperado). Esa semana el sistema "
                "lo vio por debajo, porque la ráfaga infló el total; para el 3 de febrero la referencia ya lo había "
                "absorbido.")
    if c == 3:
        return (f"Sin el eco sonó el 3 de febrero ({times(ratio[READ])} veces lo esperado). Cinco cartas modelo nuevas "
                "desde el 22 de enero.")
    if c == 12:
        return (f"Sin el eco no suena: estaba en {times(a['free_S_0203'])} el 3 de febrero. Su máximo, "
                f"{times(a['free_S_max'])} el 13 de enero, venía de antes de la ráfaga ({times(PATH[12]['2025-01-06']['S'])} "
                "el 6 de enero). Suena en 2 de las 3 variantes del <a href='#anexos'>anexo A4</a>.")
    if c == 13:
        return (f"El sistema le dio la alarma más fuerte ({times(a['cusum_0203'])}), pero su porción es la misma que a fin "
                f"de 2024 ({pct(a['share_0203'], 2)} frente a {pct(a['cal_share'], 2)}).")
    if a["observed_0203"] < 1000:
        return (f"Sin el eco, {times(ratio[READ])} veces lo esperado; para un patrón de {n(a['observed_0203'])} "
                f"reclamos esa diferencia es normal (rareza {times(a['free_z'][READ])}).")
    return f"Sin el eco, {times(ratio[READ])} veces lo esperado: dentro de lo normal."


def alerts_table() -> str:
    rows = []
    for c in by_verdict("actuar") + WATCH + ECHO + [8]:
        a = A[c]
        if c == 8:
            ratios, free_s = "—", "—"
        else:
            ratios = f"{times(a['observed_0203'] / a['m9_expected_0203'])} → {times(a['free_ratio'][READ])}"
            free_s = times(a["free_S_max"]) + (" · suena" if a["free_alarms_known"] else "")
        rows.append([label(c), verdict_chip(c), f"<span class='num'>{n(a['observed_0203'])}</span>",
                     f"<span class='num'>{ratios}</span>", f"<span class='num'>{free_s}</span>", why_line(c)])
    return scroll_table(["Alerta", "Decisión", "Llegaron el 3 feb", "Veces lo esperado: sistema → sin eco",
                         f"Acumulador sin eco, máx. 13 ene–3 feb (umbral {times(H)})", "Por qué"], rows, "alerts")


def echo_later_text() -> str:
    dates = sorted({w for _, w in ECHO_LATER})
    parts = [and_list([qname(c) for c, w in ECHO_LATER if w == d]) + f" el {spanish_date(d, year=False)}" for d in dates]
    return ", y ".join(parts)


def echo_later_growth() -> str:
    growth = F["growth_spring"]
    shown = [f"{qname(c)}, {pct(growth[str(c)])}" for c, _ in ECHO_LATER]
    return and_list(shown)


# --- 8. The complaint card --------------------------------------------------------------

def mock_card() -> str:
    t2, t3 = CARD["t2"], CARD["t3"]
    sims = [s["similarity"] for s in CARD["similar"]]
    years = sorted({s["received"][:4] for s in CARD["similar"]})
    rows = [
        ("Motivo probable", " · ".join(f"{i + 1}.º {es(t['issue'])}" for i, t in enumerate(CARD["t1"]))),
        ("¿Alguna solución?", "Probable" if t2["likely"] else "No probable"),
        ("¿Compensación en dinero?", "Probable" if t3["likely"] else "No probable"),
        ("Patrón", f"{esc(name(3))} · típico de su grupo · patrón con alerta"),
        ("Reclamos parecidos", f"5 casos de {', '.join(years)}, similitud {min(sims):.3f} a {max(sims):.3f}"),
        ("Plazo", f"vence el {long_day(F['card_due_2025'])} · con extensión, {long_day(F['card_due_2025_extended'])} "
         "(a confirmar con los feriados de 2025)"),
    ]
    items = "".join(f"<div class='mock-row'><span class='num-badge'>{i + 1}</span><span class='mock-k'>{k}</span>"
                    f"<span>{v}</span></div>" for i, (k, v) in enumerate(rows))
    action = (f"<div class='mock-row action'><span class='num-badge'>{len(rows) + 1}</span><span class='mock-k'>Acción "
              f"sugerida</span><span>{esc(CARD['action']['text'])}</span></div>")
    return (f"<div class='mock' aria-label='Maqueta de la ficha del reclamo'><p class='mock-head'>Reclamo "
            f"{esc(CARD['complaint_id'])} · recibido el {esc(long_day(CARD['received']))} · reportes de crédito</p>"
            f"{items}{action}</div>")


# --- 9. Decisions ----------------------------------------------------------------------------

def decision_card(title: str, what: str, rows: list[tuple[str, str]], later: str) -> str:
    items = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
    return (f"<article class='decision'><h3>{title}</h3><p>{what}</p><dl>{items}"
            f"<dt class='later-cell'>Mirando después</dt><dd class='later-cell'>{later}</dd></dl></article>")


def decision_cards() -> str:
    seg14 = F["p14_segments"]
    g10, g20 = DET["growth_10"], DET["growth_20"]
    replan_down, replan_up = PATH[14][REPLAN_DOWN], PATH[14][REPLAN_UP]
    cards = [
        decision_card(
            "A1 · Cerrar el eco con una regla, y vigilar uno",
            f"No abrir investigaciones por {names_list(ECHO)}. Avisar a las áreas dueñas de esos productos que no son "
            f"incidentes nuevos. Cada lunes, rehacer la cuenta sin la ráfaga. Si un patrón suena, Analítica avisa ese mismo "
            f"lunes a la Jefatura de reclamos y Triaje revisa 30 de sus reclamos antes del viernes. {quoted(12)} queda en "
            "vigilancia: sin el eco no suena, pero está cerca del umbral desde antes de la ráfaga.",
            [("Responsable propuesto", "Gerencia de riesgo avisa; Analítica recalcula; Triaje revisa"),
             ("Cuándo", "Desde hoy hasta la lectura del lunes 3 de marzo, la primera cuya referencia ya no incluye las "
              "semanas del 13 y 20 de enero"),
             ("Por qué (hasta el 3 feb)", f"Sin el eco ninguno de los 7 suena, y lo reservado de más para la ráfaga es el "
              f"{pct(echo_share)} de lo que pareció sobrar en los demás."),
             ("Cómo sabremos si funcionó", "Una muestra de 30 reclamos de cada patrón no muestra un tema nuevo; si "
              "aparece, se escala."),
             ("Se puede hacer", "Hoy, a mano con Analítica; automático con P1"),
             ("Confianza", "Alta en el mecanismo; el criterio es posterior al episodio")],
            f"En las dos semanas siguientes sonaron sin el eco {echo_later_text()}. De febrero a principios de abril "
            f"crecieron más que el total ({echo_later_growth()}, frente a {pct(F['growth_total_spring'])}): la revisión "
            f"del lunes los habría escalado. {quoted(12)} no volvió a sonar."),
        decision_card(
            f"A2 · Capacidad para {quoted(14)}",
            f"Planificar entre {n(PLAN[0])} y {n(PLAN[1])} reclamos por semana, de {times(PLAN_RATIO[0], 1)} a "
            f"{times(PLAN_RATIO[1], 1)} veces su promedio de fin de 2024 ({n(a14['cal_mean'])}), y replanificar si el "
            f"volumen sale de ese rango dos lunes seguidos. Revisar con las centrales de riesgo por qué crece "
            f"«{es(F['p14_sub']['value'])}».",
            [("Responsable propuesto", "Gerencia de atención al cliente, con quien lleva la relación con las centrales"),
             ("Cuándo", "Plan antes del viernes 14 de febrero; revisión cada lunes"),
             ("Por qué (hasta el 3 feb)", f"Sonó sin el eco el 13 de enero, y la semana del 6 de enero, antes de la "
              f"ráfaga, ya recibió {n(F['p14_0106'])} reclamos. El rango va de esa semana a la última."),
             ("Cómo sabremos si funcionó", f"Las respuestas a tiempo no bajan del {pct(F['timely_cal_p14'], 1)} de octubre "
              "a diciembre de 2024."),
             ("Se puede hacer", "Hoy"),
             ("Confianza", "Media: los conteos son seguros, el rango es una proyección")],
            f"Del 10 de febrero al 12 de mayo recibió entre {n(seg14[0]['min'])} y {n(seg14[0]['max'])} por semana, y desde "
            f"el 19 de mayo entre {n(seg14[1]['min'])} y {n(seg14[1]['max'])}. La regla de los dos lunes habría "
            f"replanificado a la baja con la semana del {spanish_date(REPLAN_DOWN, year=False)} "
            f"({n(replan_down['observed'])}) y al alza con la del {spanish_date(REPLAN_UP, year=False)} "
            f"({n(replan_up['observed'])})."),
        decision_card(
            f"A3 · Cartas nuevas de {quoted(3)}",
            "Agrupar las copias de las cinco cartas nuevas y responder cada una con un texto estándar revisado por Legal, "
            "verificando la identidad del cliente, porque dicen que «la información pertenece a otra persona». Pregunta "
            "abierta: ¿vienen de una misma fuente? Los datos no lo dicen.",
            [("Responsable propuesto", "Liderazgo de triaje, con Legal y Cumplimiento"),
             ("Cuándo", f"Esta semana: las primeras {n(F['letters_due_this_week'])} copias vencen desde el "
              f"{spanish_date(F['letters_due_first'], weekday=True, year=False)}"),
             ("Por qué (hasta el 3 feb)", f"Cinco textos idénticos desde el 22 de enero, {n(F['letters_rows'])} copias al 9 "
              "de febrero."),
             ("Cómo sabremos si funcionó", "Cada copia se responde dentro del plazo con el texto estándar, y las copias "
              "nuevas se cuentan cada lunes."),
             ("Se puede hacer", "Hoy con una consulta; automático con P2"),
             ("Confianza", "Alta en el aumento y las cartas; la causa se desconoce")],
            "Las cartas llegaron hasta mediados de mayo. El sistema volvió a sonar el 10 de febrero y el 10 de marzo."),
        decision_card(
            "A4 · La ráfaga como incidente",
            f"Abrir un incidente por {quoted(8)}. Con la regla de 15 días hábiles, sus {n(F['burst_due_before_reading'])} "
            f"reclamos de la semana del 13 de enero vencieron entre el 3 y el 7 de febrero, y los "
            f"{n(F['burst_due_this_week'])} de la semana del 20 vencen del 10 al 14 de febrero. Averiguar la causa y seguir "
            f"el residuo: el 3 de febrero su porción aún era {times(p8_vs_cal)} veces la de fin de 2024. Su alerta del "
            "panel no pide acción: solo arrastra la alarma del 13 de enero y se apaga sola el lunes 17 de febrero si no "
            "vuelve a sonar. Hacia adelante, aplicar la regla propuesta en <a href='#sec-rafaga'>La ráfaga de enero</a>.",
            [("Responsable propuesto", "Riesgo operacional y el dueño del producto de transferencias"),
             ("Cuándo", "Hoy; debió abrirse el 16 o 17 de enero"),
             ("Por qué (hasta el 3 feb)", f"2 empresas, 4 días y un texto repetido {n(F['p8_0113_top_text_count'])} veces."),
             ("Cómo sabremos si funcionó", "Todos respondidos o con extensión notificada, y la causa identificada antes "
              f"del viernes 14 de febrero; después, su porción vuelve a la de fin de 2024 ({pct(a8['cal_share'], 1)})."),
             ("Se puede hacer", "Hoy"),
             ("Confianza", "Media: los datos no dicen la causa")],
            f"En los datos de la CFPB, el {pct(F['timely_burst_p8'], 1)} de esos reclamos tuvo respuesta a tiempo de la "
            "empresa. Su porción volvió a cerca del 8% desde la semana del 3 de marzo."),
        decision_card(
            "A5 · Dotación total",
            "El panel no alerta por el crecimiento del total: ajustar la dotación con el total semanal y su rango "
            "histórico.",
            [("Responsable propuesto", "Planeamiento de personal"),
             ("Cuándo", "Antes del viernes 14 de febrero; luego cada mes"),
             ("Por qué (hasta el 3 feb)", f"La semana del 6 de enero llegaron {n(F['total_0106'])} reclamos, frente a "
              f"{n(F['cal_median_total'])} en una semana normal de fin de 2024."),
             ("Cómo sabremos si funcionó", "La dotación sigue al total semanal y a su rango histórico."),
             ("Se puede hacer", "Hoy"),
             ("Confianza", "Alta")],
            f"Unos {n(F['post_weekly_avg'])} por semana de fines de febrero a junio."),
        decision_card(
            "A6 · Cuántas alarmas aceptar",
            "El umbral se fijó para 1 falsa alarma al mes en total, si el modelo estuviera bien calibrado. Hubo "
            f"{times(PER_MONTH['fit'], 1)} alarmas al mes de enero de 2023 a septiembre de 2024 y "
            f"{times(PER_MONTH['calibration'], 1)} de octubre a diciembre de 2024. Opciones: aceptar unas 2 a 4 al mes, o "
            "subir el umbral a 6.11 para acercarse a 1 al mes, que detectaría 27 y 53 de cada 100 crecimientos del 10% y "
            f"20% semanal en lugar de {per100(g10['cusum']['detected'])} y {per100(g20['cusum']['detected'])}. Con la regla "
            "de revisar 30 reclamos por alarma, 2 a 4 alarmas al mes son 60 a 120 reclamos revisados.",
            [("Responsable propuesto", "Gerencia de riesgo con atención al cliente y Analítica"),
             ("Cuándo", "Antes del piloto"),
             ("Por qué (hasta el 3 feb)", "Las alarmas de más vienen de semanas extremas (de 2023 a septiembre de 2024, el "
              "1.8% de las semanas superó un nivel que el modelo da con probabilidad 0.6%) y de rachas (la correlación "
              "entre semanas seguidas es 0.29 y el modelo supone 0). Parte de esas rachas pueden ser cambios reales."),
             ("Cómo sabremos si funcionó", "La decisión queda escrita, con lo que se deja de detectar."),
             ("Se puede hacer", "Decisión de gerencia"),
             ("Confianza", "—")],
            f"En el primer semestre de 2025 hubo {times(PER_MONTH['validation'], 1)} al mes, y "
            f"{times(OUTSIDE_PER_MONTH, 1)} fuera de las 4 semanas del 27 de enero al 17 de febrero."),
    ]
    return "".join(cards)


CHANGES = [
    ("P1", "Alta", "Aviso de posible eco: si un patrón ocupó una porción anómala de las 4 semanas de referencia, mostrar al "
     f"lado la cuenta sin ese patrón y el factor de eco (aquí, {times(F['echo_factor_2025-02-03'])}). Dejarlo por escrito "
     "antes de probarlo, diseñarlo con ráfagas simuladas sobre 2023 y 2024 y confirmarlo con datos nuevos; probar también "
     "una referencia más robusta."),
    ("P2", "Alta", "Cartas modelo por semana: detectar textos idénticos nuevos que crecen, aunque caigan dentro de un grupo "
     "o se repartan entre varios patrones."),
    ("P3", "Alta", "Nivel frente a una base fija, por ejemplo el trimestre anterior, para que un aumento que se queda no "
     "desaparezca a las 4 semanas."),
    ("P4", "Media", "Mostrar las caídas fuertes como aviso informativo y distinguir una alerta nueva de una arrastrada que "
     "ya está bajando."),
    ("P5", "Alta", "Que la acción sugerida de la ficha dependa de la decisión (eco o real), no solo de la alerta."),
    ("P6", "Alta", "Nombres y familias en todas las vistas, y validarlos con el negocio, empezando por los "
     f"{CONF['baja']} de confianza baja."),
    ("P7", "Media", "Indicador del total semanal con su rango histórico."),
    ("P8", "Media", f"Sacar el nivel (Δ) de la vista principal: con o sin la ráfaga, aquí no mostraba el aumento de "
     f"{quoted(3)}."),
    ("P9", "Baja", "Mostrar «veces lo esperado» junto al rango y la rareza, nunca solo: en un patrón pequeño, 1.3 veces lo "
     "esperado puede ser normal."),
    ("P10", "Media", "En la ficha: el motivo como 1-2-3 sin puntaje; T2 y T3 como marca con su fiabilidad; y «novedoso» "
     "cambiado por «en el borde de su grupo», con aviso si el mismo texto ya llegó antes."),
    ("P11", "Media", "Feriados de 2025 y de otros años, días no laborables y feriados regionales."),
    ("P12", "Alta", "Investigar antes del piloto por qué textos idénticos caen en patrones distintos "
     f"({pct(F['multi_pattern_share_known'], 1)} de los reclamos hasta el 9 de febrero), porque afecta los conteos."),
]


def change_list() -> str:
    return "<ul class='changes'>" + "".join(
        f"<li><span class='code'>{c}</span> <span class='prio prio-{p.lower()}'>{p}</span> {t}</li>" for c, p, t in CHANGES
    ) + "</ul>"


# --- 11. Confidence ---------------------------------------------------------------------------

def confidence_table() -> str:
    g10, g20 = DET["growth_10"], DET["growth_20"]
    novel = F["novel_2025_seen_before"] / F["novel_2025"]
    rows = [
        ["<b>Motivo probable</b>", f"Entre los 3 primeros en {per100(T1['val_top3'])} de cada 100 reclamos; la regla, en "
         f"{per100(T1['val_rule_top3'])}.", "Como respuesta única acierta mucho menos (<a href='#anexos'>anexo A7</a>).",
         "Confirmado"],
        ["<b>Alguna solución</b>", f"{n(T2['val_precision'] * 10)} de cada 10 marcados la obtienen y encuentra "
         f"{n(T2['val_recall'] * 10)} de cada 10; sin modelo, {per100(T2['val_base'])} de cada 100.",
         "Es una prioridad, no una promesa.", "Confirmado"],
        ["<b>Compensación en dinero</b>", f"{n(T3['val_precision'] * 10)} de cada 10 marcados la obtienen y encuentra "
         f"{n(T3['val_recall'] * 10)} de cada 10; sin modelo, 1 de cada {n(1 / T3['val_base'])}.",
         "Su precisión bajó porque el evento se volvió más raro.", "Confirmado con límites"],
        ["<b>Plazo</b>", "Regla fija de 15 o 45 días hábiles.", "Solo están cargados los feriados de 2026.",
         "Regla; el modelo de plazo se retiró"],
        ["<b>Esperados y rango</b>", f"Fuera del episodio, el rango del 95% contuvo el "
         f"{pct(M9C['validation_outside']['in95'], 1)} de los casos.",
         f"Del 13 de enero al 24 de febrero, el {pct(M9C['validation_episode']['in95'], 1)}: la ráfaga y su eco.",
         "Confirmado con límites"],
        ["<b>Porción esperada</b>", f"Fuera del episodio, el {pct(M10C['validation_outside'], 1)} quedó dentro del rango.",
         "No hace sonar alarmas y tiene el mismo eco.", "Confirmado"],
        ["<b>Acumulador</b>", f"Detecta {per100(g10['cusum']['detected'])} de cada 100 crecimientos del 10% semanal y "
         f"{per100(g20['cusum']['detected'])} del 20% en 8 semanas.",
         f"Más alarmas que las presupuestadas: {times(PER_MONTH['fit'], 1)} al mes de enero de 2023 a septiembre de 2024 y "
         f"{times(PER_MONTH['validation'], 1)} en el primer semestre de 2025 ({times(OUTSIDE_PER_MONTH, 1)} fuera de las 4 "
         "semanas del episodio), frente a 1.", "Confirmado con límites"],
        ["<b>Reclamos parecidos</b>", "Precedentes de casos casi iguales.",
         f"Comparten el motivo solo en {per100(FAISS['same_issue'])} de cada 100: no predicen.", "Herramienta de consulta"],
        ["<b>En el borde de su grupo</b>", "Marca reclamos atípicos.", f"El {pct(novel)} repite un texto ya visto.",
         "Herramienta de consulta"],
        ["<b>Patrones</b>", f"{CONF['alta']} nombres con confianza alta, {CONF['media']} media y {CONF['baja']} baja.",
         f"Son una partición de trabajo, no categorías naturales; el {pct(F['multi_pattern_text_share'], 1)} de los "
         "reclamos tiene textos repartidos entre patrones.", "A validar con el negocio"],
        ["<b>Separación del eco</b>", "Mismo cálculo y mismo umbral del sistema; todas las formas de quitar el eco "
         "coinciden en los 2 aumentos.", "Criterio definido después de ver el episodio; un patrón queda cerca del umbral.",
         "Diagnóstico"],
    ]
    return scroll_table(["Pieza", "Qué tan bien funciona", "Dónde falla", "Estado"], rows)


# --- Annexes -----------------------------------------------------------------------------------

def tech_table() -> str:
    rows = []
    for c in by_verdict("actuar") + WATCH + ECHO + [8]:
        a, r = A[c], PATH[c][READ]
        no8 = "—" if c == 8 else times(a["free_ratio"][READ])
        rows.append([
            label(c), esc(VERDICT_LABEL[VERDICTS[c]]), n(a["observed_0203"]),
            f"{n(r['expected'])} ({n(r['p025'])}–{n(r['p975'])})", times(r["z"]), times(r["S"]),
            ", ".join(day(w) for w in a["system_alarms_known"]) or "—",
            f"{pct(a['share_0203'], 2)} ({pct(a['m10_low_0203'], 2)}–{pct(a['m10_high_0203'], 2)})",
            f"{pct(a['level_0203'], 1)} (p{per100(a['level_percentile_0203'])})",
            "—" if c == 8 else pct(F["level_no8"][str(c)], 1),
            f"{times(a['observed_0203'] / a['m9_expected_0203'])} → {no8}",
            "—" if c == 8 else times(a["free_S_max"]),
            "—" if c == 8 else (", ".join(day(w) for w in a["free_alarms_known"]) or "—"),
        ])
    return scroll_table(["Alerta", "Decisión", "Llegaron", "Esperados M9 (95%)", "Rareza", "Acumulador",
                         "Alarmas 13 ene–3 feb", "Porción observada (rango M10)", "Δ (percentil)", "Δ sin la ráfaga",
                         "Veces lo esperado: sistema → sin eco", "Acumulador sin eco (máx.)", "Alarmas sin eco"],
                        rows, "tech")


def families_patterns() -> str:
    share = F["family_share"]
    parts = []
    for f in FAMILIES:
        items = "".join(
            f"<li><b>{esc(NAMES[c]['nombre'])}</b> <span class='pid'>patrón {c}</span> "
            f"<span class='conf-tag'>confianza {esc(NAMES[c]['confianza'])}</span> {esc(NAMES[c]['descripcion'])}</li>"
            for c in f["patrones"])
        parts.append(f"<h4>{esc(FAMILY_NAME[f['familia']])} <span class='muted'>· {pct(share['fit'][f['familia']], 1)} / "
                     f"{pct(share['calibration'][f['familia']], 1)} / {pct(share['validation'][f['familia']], 1)}</span></h4>"
                     f"<ul class='patterns'>{items}</ul>")
    return "".join(parts)


def sensitivity_table() -> str:
    sens = F["sensitivity"]
    rows = []
    for c in by_verdict("actuar") + WATCH + ECHO:
        cells = [label(c), esc(VERDICT_LABEL[VERDICTS[c]]), ", ".join(day(w) for w in A[c]["free_alarms_known"]) or "—"]
        for key in ("full_history", "frozen_pre_burst", "without_8_and_14"):
            if key == "without_8_and_14" and c == 14:
                cells.append("no aplica")
            else:
                cells.append(", ".join(day(w) for w in sens[key].get(str(c), [])) or "—")
        rows.append(cells)
    return scroll_table(["Alerta", "Decisión", "Cuenta principal, desde el estado del 6 ene",
                         "Variante 1: corrida desde 2023", "Variante 2: referencia congelada del 9 dic al 6 ene",
                         f"Variante 3: sin la ráfaga ni {quoted(14)}"], rows, "tech")


def level_list() -> str:
    return and_list([f"{esc(name(c))} {pct(F['level_no8'][str(c)], 1)}" for c in by_verdict("actuar") + WATCH + ECHO])


def glossary() -> str:
    terms = [
        ("Acumulador (CUSUM)", f"Suma semana a semana la rareza menos 0.5, sin bajar de 0. Suena una alarma al pasar el "
         f"umbral de {times(H)} y vuelve a 0. Solo mira subidas."),
        ("Alarma", "La semana en que el acumulador de un patrón pasa el umbral."),
        ("Alerta activa", "El patrón tuvo una alarma en alguna de las últimas 4 semanas cerradas, aunque ya esté bajando."),
        ("Alguna solución (T2)", "Marca «probable» cuando el puntaje alcanza el umbral de 0.30, fijado en calibración."),
        ("Binomial Negativa", "Forma estadística estándar de describir conteos que varían más que el azar puro. De ella "
         "sale el ancho del rango normal."),
        ("Calibración", "Octubre a diciembre de 2024: el periodo en que se fijaron los umbrales de los clasificadores y se "
         "revisaron los rangos."),
        ("Carta modelo", "Texto idéntico que muchos consumidores copian. Se detecta comparando el texto exacto, así que las "
         "variantes casi iguales no cuentan."),
        ("Centrales de riesgo", "Equifax, TransUnion y Experian, las empresas que elaboran los reportes de crédito en "
         "EE. UU."),
        ("Compensación en dinero (T3)", "Marca «probable» cuando el puntaje alcanza el umbral de 0.18."),
        ("Cuenta sin el eco", "La misma cuenta del sistema, con el mismo umbral, rehecha sin el patrón de la ráfaga. Es un "
         "diagnóstico del análisis, no un modelo validado."),
        ("Dirichlet-Multinomial", "Forma estadística de repartir un total entre varias partes, con más variación que el "
         "azar puro. De ella sale el rango de la porción esperada."),
        ("Eco", "Alarmas que aparecen porque las 4 semanas de referencia contienen la ráfaga de otro patrón, no porque el "
         "patrón alertado haya crecido."),
        ("En el borde de su grupo", "Reclamo más lejos del centro de su patrón que el 99% de los reclamos de la historia "
         "de aprendizaje. Hoy se llama «novedoso», pero no significa «nunca visto»."),
        ("Factor de eco", "Cuántas veces mayor habría sido lo esperado de los demás patrones sin la ráfaga; el 3 de "
         f"febrero, {times(F['echo_factor_2025-02-03'])}."),
        ("Familia", "Uno de los 6 grupos de negocio que reúnen los 40 patrones."),
        ("FCRA", "La ley de EE. UU. sobre reportes de crédito (15 U.S.C. § 1681 y siguientes). Sus secciones 609 y 611 "
         "aparecen en muchas cartas modelo; su significado legal debe validarlo un especialista."),
        ("Mirando después", "Lo que se supo después de la lectura del lunes 10 de febrero de 2025 y que no estaba "
         "disponible para decidir ese día."),
        ("Motivo probable (T1)", "Los 3 motivos de reclamo más probables, en orden. Los puntajes no son probabilidades."),
        ("Nivel frente a hace 3 meses (Δ)", "La porción de las últimas 4 semanas comparada con la de 4 semanas de unos 3 "
         "meses antes. Se lee por su percentil."),
        ("Normal estándar", "La curva de campana con media 0 y desviación 1: la escala de la rareza."),
        ("Patrón", "Grupo de reclamos con textos parecidos; hay 40. Su nombre en español es una propuesta del análisis "
         "hecha con IA, que deben validar expertos."),
        ("Percentil", "El lugar de un valor en la historia: percentil 0 es el más bajo visto y 100, el más alto."),
        ("Piloto", "La prueba de la app con datos del banco. Aún no tiene fecha."),
        ("Plazo", "Fecha límite de respuesta: 15 días hábiles después del registro, o 45 con extensión, sin fines de "
         "semana ni feriados nacionales."),
        ("Porción esperada (M10)", "El porcentaje del total que se esperaba para un patrón, con su rango. No hace sonar "
         "alarmas."),
        ("Ráfaga", "Muchísimos reclamos en pocos días, concentrados en pocas empresas y a menudo con un texto repetido."),
        ("Rango normal (95%)", "Los valores entre los que el conteo debería caer 95 de cada 100 veces."),
        ("Rareza (z)", "La probabilidad, según la Binomial Negativa, de ver un conteo así de alto, en la escala de una "
         f"normal estándar. Tiene un tope en {times(F['z_cap'])}."),
        ("Reclamos esperados (M9)", "Total real de la semana por la porción que tuvo el patrón en las 4 semanas "
         "anteriores, con un rango normal."),
        ("Reclamos parecidos (FAISS)", "Los 5 reclamos de una muestra de la historia de aprendizaje con el texto de "
         "significado más cercano. La similitud va hasta 1, que es prácticamente el mismo texto."),
        ("Umbral", f"El valor del acumulador a partir del cual suena una alarma: {times(H)}."),
        ("Veces lo esperado", "Lo que llegó dividido por lo esperado."),
    ]
    return "<dl class='glossary'>" + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms) + "</dl>"


def annex(title: str, anchor: str, body: str) -> str:
    return f"<details class='annex' id='{anchor}'><summary>{title}</summary>{body}</details>"
