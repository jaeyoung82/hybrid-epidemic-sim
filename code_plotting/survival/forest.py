"""Forest plot and model comparison visualization functions.

This module re-exports the full-featured implementations from
``code_survival.forest_plot`` to avoid code duplication.
"""
from code_survival.forest_plot import (
    create_forest_plot_hazard_ratios,
    create_discrimination_performance_plot,
    create_model_parsimony_plot,
)

__all__ = [
    'create_forest_plot_hazard_ratios',
    'create_discrimination_performance_plot',
    'create_model_parsimony_plot',
]
