"""Registry of versioned daily count models (models_plan.md §28.2)."""

from .models import (
    nb_daily_fourier_v1,
    nb_daily_hierarchical_v1,
    nb_daily_no_dow_v1,
    zinb_daily_hierarchical_v1,
)

MODELS = {
    nb_daily_hierarchical_v1.MODEL_ID: nb_daily_hierarchical_v1,
    nb_daily_no_dow_v1.MODEL_ID: nb_daily_no_dow_v1,
    nb_daily_fourier_v1.MODEL_ID: nb_daily_fourier_v1,
    zinb_daily_hierarchical_v1.MODEL_ID: zinb_daily_hierarchical_v1,
}


def get_model(model_id: str):
    return MODELS[model_id]
