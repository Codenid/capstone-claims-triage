"""Registry of versioned weekly composition models."""

from types import ModuleType

from .models import dirichlet_multinomial_rolling_4_v1, dirichlet_multinomial_static_v1

MODELS = {
    dirichlet_multinomial_static_v1.MODEL_ID: dirichlet_multinomial_static_v1,
    dirichlet_multinomial_rolling_4_v1.MODEL_ID: dirichlet_multinomial_rolling_4_v1,
}


def get_model(model_id: str) -> ModuleType:
    return MODELS[model_id]
