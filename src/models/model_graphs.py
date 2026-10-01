"""Draw the structure of the main M9 and M10 PyMC models as PNG files.

pm.model_to_graphviz needs the Graphviz program, which is not installed, so this
uses pm.model_to_networkx and matplotlib. Each model is built with the frozen
fit data, so the dimensions in the drawing are the real ones.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pymc as pm
import yaml

from src.evaluation.experiment import PROJECT_ROOT, load_experiment_config
from src.models.weekly_composition import registry as composition_registry
from src.models.weekly_composition.data import composition_panel, split_weeks
from src.models.weekly_composition.run import model_panel
from src.models.weekly_counts import registry as count_registry
from src.models.weekly_counts.data import fit_rows
from src.models.weekly_counts.run import (
    MODELING_CONFIG_PATH,
    load_frozen_frame,
    model_frame,
)

OUTPUT_DIR = PROJECT_ROOT / "reports/modeling/model_graphs"
# Model ID -> short name used in models_plan.md.
COUNT_MODELS = {
    "poisson_static_pymc_v1": "B1 PyMC",
    "nb_static_global_v3": "NB-V3",
    "nb_rolling_4_global_v1": "NB-R4",
    "nb_rolling_4_hierarchical_v3": "NB-R4-H v3",
}
COMPOSITION_MODELS = {
    "dirichlet_multinomial_static_v1": "DM-V1",
    "dirichlet_multinomial_rolling_4_v1": "DM-R4",
}
# Node style by role: (box style, face color).
STYLES = {
    "data": ("round,pad=0.4", "#e8e8e8"),
    "free": ("round,pad=0.4", "#ffffff"),
    "deterministic": ("square,pad=0.4", "#ffffff"),
    "observed": ("round,pad=0.4", "#b9cde5"),
}


def load_settings(stage_dir: str, model_id: str) -> dict:
    path = PROJECT_ROOT / "configs" / stage_dir / f"{model_id}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def count_model(model_id: str, config: dict) -> pm.Model:
    settings = load_settings("weekly_counts", model_id)
    module = count_registry.get_model(model_id)
    panel, _, _, _ = load_frozen_frame(config, settings)
    return module.build_model(fit_rows(model_frame(module, panel, settings)), settings)


def composition_model(model_id: str, config: dict) -> pm.Model:
    settings = load_settings("weekly_composition", model_id)
    module = composition_registry.get_model(model_id)
    frame, _, _, _ = load_frozen_frame(config, settings)
    panel = composition_panel(frame, settings["clusters"])
    return module.build_model(split_weeks(model_panel(module, panel), "fit"), settings)


def node_role(model: pm.Model, name: str) -> str:
    if name in {variable.name for variable in model.observed_RVs}:
        return "observed"
    if name in {variable.name for variable in model.free_RVs}:
        return "free"
    if name in {variable.name for variable in model.deterministics}:
        return "deterministic"
    return "data"


def node_label(name: str, attributes: dict) -> str:
    """Name, distribution and dimensions, e.g. "observed / ~ NB / week (87)"."""
    distribution = attributes["label"].split("\n")[-1]
    lines = [name, f"~ {distribution}"]
    if attributes.get("cluster"):
        lines.append(attributes["cluster"].removeprefix("cluster"))
    return "\n".join(lines)


def neighbor_center(graph: nx.DiGraph, node: str, positions: dict) -> float:
    columns = [positions[n][0] for n in nx.all_neighbors(graph, node) if n in positions]
    return sum(columns) / len(columns) if columns else 0.0


def layered_positions(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    """Rows from the priors down to the observed data.

    A node's row is its distance to the bottom, so each data input sits right
    above the variable it feeds. A few sweeps place nodes under their neighbors
    to reduce edge crossings.
    """
    height: dict[str, int] = {}
    for node in reversed(list(nx.topological_sort(graph))):
        children = [height[child] + 1 for child in graph.successors(node)]
        height[node] = max(children, default=0)
    for node in graph:
        if graph.out_degree(node) == 0 and graph.nodes[node]["shape"] == "box":
            # Summaries such as rho sit right below what they are computed from.
            parents = [height[parent] for parent in graph.predecessors(node)]
            height[node] = min(parents) - 1
    top = max(height.values())
    rows = [
        sorted(n for n in graph if height[n] == top - row) for row in range(top + 1)
    ]
    positions: dict[str, tuple[float, float]] = {}
    for _ in range(3):
        for row, nodes in enumerate(rows):
            nodes.sort(key=lambda node: neighbor_center(graph, node, positions))
            for column, node in enumerate(nodes):
                positions[node] = (column - (len(nodes) - 1) / 2, -row)
    return positions


def draw_model(model: pm.Model, title: str, path: Path) -> None:
    graph = pm.model_to_networkx(model)
    positions = layered_positions(graph)
    row_sizes = Counter(row for _, row in positions.values())
    size = (max(8.0, 2.8 * max(row_sizes.values())), 1.9 * len(row_sizes) + 1.2)

    figure, axis = plt.subplots(figsize=size)
    boxes = {}
    for node, (x, y) in positions.items():
        box_style, color = STYLES[node_role(model, node)]
        boxes[node] = axis.text(
            x,
            y,
            node_label(node, graph.nodes[node]),
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": box_style, "facecolor": color, "edgecolor": "black"},
        )
    for source, target in graph.edges():
        # patchA and patchB cut each arrow at the border of its boxes.
        axis.annotate(
            "",
            xy=positions[target],
            xytext=positions[source],
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#555555",
                "patchA": boxes[source].get_bbox_patch(),
                "patchB": boxes[target].get_bbox_patch(),
                "shrinkA": 2,
                "shrinkB": 2,
                "connectionstyle": "arc3,rad=0.08",
            },
        )
    columns = [x for x, _ in positions.values()]
    rows = [y for _, y in positions.values()]
    axis.set_xlim(min(columns) - 0.7, max(columns) + 0.7)
    axis.set_ylim(min(rows) - 0.6, max(rows) + 0.6)
    axis.set_title(title)
    axis.text(
        0.0,
        -0.04,
        "Gris: datos · blanco redondeado: parámetro estimado · "
        "blanco cuadrado: cálculo determinista · azul: dato observado",
        transform=axis.transAxes,
        fontsize=8,
    )
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    config = load_experiment_config(MODELING_CONFIG_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    builders = [
        *((model_id, name, count_model) for model_id, name in COUNT_MODELS.items()),
        *(
            (model_id, name, composition_model)
            for model_id, name in COMPOSITION_MODELS.items()
        ),
    ]
    for model_id, name, build in builders:
        path = OUTPUT_DIR / f"{model_id}.png"
        draw_model(build(model_id, config), f"{name} ({model_id})", path)
        print(f"Saved {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
