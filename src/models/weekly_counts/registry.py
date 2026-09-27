"""Registry of versioned weekly count models."""

from .models import (
    nb_independent_linear_v1,
    nb_rolling_4_global_v1,
    nb_rolling_4_hierarchical_v1,
    nb_rolling_4_hierarchical_v2,
    nb_rolling_4_hierarchical_v3,
    nb_softmax_linear_v2,
    nb_static_global_v3,
    poisson_static_pymc_v1,
)

MODELS = {
    nb_independent_linear_v1.MODEL_ID: nb_independent_linear_v1,
    nb_rolling_4_global_v1.MODEL_ID: nb_rolling_4_global_v1,
    nb_rolling_4_hierarchical_v1.MODEL_ID: nb_rolling_4_hierarchical_v1,
    nb_rolling_4_hierarchical_v2.MODEL_ID: nb_rolling_4_hierarchical_v2,
    nb_rolling_4_hierarchical_v3.MODEL_ID: nb_rolling_4_hierarchical_v3,
    nb_softmax_linear_v2.MODEL_ID: nb_softmax_linear_v2,
    nb_static_global_v3.MODEL_ID: nb_static_global_v3,
    poisson_static_pymc_v1.MODEL_ID: poisson_static_pymc_v1,
}
DEFAULT_MODEL_ID = nb_softmax_linear_v2.MODEL_ID


def get_model(model_id):
    return MODELS[model_id]
