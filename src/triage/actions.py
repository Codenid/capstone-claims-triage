"""Suggested action for a complaint (models_plan.md §25, docs/modelos.md)."""

from __future__ import annotations

ACTIONS = {
    "novel": "Revisión manual: reclamo aislado y novedoso; no elevarlo "
    "automáticamente.",
    "template_alert": "El patrón tiene una alerta, pero lo domina una plantilla: "
    "no alertar automáticamente hasta validar su importancia.",
    "alert": "Patrón con crecimiento anormal: agrupar los casos y alertar al "
    "equipo especializado.",
    "normal": "Triaje normal.",
}


def situation(is_novel: bool, active_alert: bool, template_dominated: bool) -> str:
    """The first rule that applies, in the order of §25."""
    if is_novel:
        return "novel"
    if active_alert and template_dominated:
        return "template_alert"
    if active_alert:
        return "alert"
    return "normal"
