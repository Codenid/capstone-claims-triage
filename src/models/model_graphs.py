"""Draw the structure of the main M9 and M10 PyMC models with Graphviz.

Each model is built with the frozen fit data, so the boxes around the variables
show the real dimensions. Rendering needs the Graphviz program `dot`.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil

import graphviz
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
# winget installs Graphviz here without adding it to PATH.
WINDOWS_GRAPHVIZ = Path("C:/Program Files/Graphviz/bin")
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


def model_graph(model: pm.Model, title: str) -> graphviz.Digraph:
    """PyMC's own drawing of the model, with a title on top."""
    return pm.model_to_graphviz(
        model,
        dpi=200,
        graph_attr={"label": title, "labelloc": "t", "fontsize": "18"},
    )


def main() -> None:
    if shutil.which("dot") is None and WINDOWS_GRAPHVIZ.exists():
        os.environ["PATH"] += os.pathsep + str(WINDOWS_GRAPHVIZ)
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
        graph = model_graph(build(model_id, config), f"{name} ({model_id})")
        path = Path(graph.render(OUTPUT_DIR / model_id, format="png", cleanup=True))
        print(f"Saved {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
