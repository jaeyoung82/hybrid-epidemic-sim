"""Plotting package for epidemic simulation visualizations.

This package consolidates all plotting and data-I/O utilities previously
spread across the legacy ``plotting_utils.py`` module.  Refactored
submodules take precedence over the legacy implementations where names
overlap.
"""
from .results import (
    # Data I/O and analysis functions (formerly plotting_utils.py)
    save_realizations,
    save_event_recruitment_records,
    save_full_results,
    analyze_and_save_peak_time,
    save_district_active_cases_per_run,
    save_arrival_time_records,
    save_arrival_time_records_I,
    save_invasion_edge_records,
    save_invasion_tree_from_records,
    analyze_and_plot_arrival_times,
    update_summary_csv,
    save_arrival_time_statistics,
    analyze_arrival_time_sequence,
    plot_active_cases_comparison,
    plot_peak_summary_rectangles,
    plot_arrival_time_comparison,
    plot_mean_arrival_time_by_district_and_scenario_clustered,
    plot_cumulative_cases_time_to_threshold_comparison,
    plot_cumulative_active_cases_boxplot_at_day,
    plot_initial_exposed_infected_distribution_boxplot,
    plot_infection_arrival_interval_distribution,
    plot_cumulative_active_cases_hist_at_day,
)

from .seir.curves import (
    plot_seir_evolution,
    plot_exposed_infected_zoom,
    plot_peak_time_distribution,
)
from .spatial.maps import (
    plot_spacetime_heatmap,
    plot_spatial_spread_snapshots,
)
from .survival.forest import (
    create_forest_plot_hazard_ratios,
    create_discrimination_performance_plot,
    create_model_parsimony_plot,
)

__all__ = [
    # Results / data I/O (from results.py)
    'save_realizations',
    'save_event_recruitment_records',
    'save_full_results',
    'analyze_and_save_peak_time',
    'save_district_active_cases_per_run',
    'save_arrival_time_records',
    'save_arrival_time_records_I',
    'save_invasion_edge_records',
    'save_invasion_tree_from_records',
    'analyze_and_plot_arrival_times',
    'update_summary_csv',
    'save_arrival_time_statistics',
    'analyze_arrival_time_sequence',
    'plot_active_cases_comparison',
    'plot_peak_summary_rectangles',
    'plot_arrival_time_comparison',
    'plot_mean_arrival_time_by_district_and_scenario_clustered',
    'plot_cumulative_cases_time_to_threshold_comparison',
    'plot_cumulative_active_cases_boxplot_at_day',
    'plot_initial_exposed_infected_distribution_boxplot',
    'plot_infection_arrival_interval_distribution',
    'plot_cumulative_active_cases_hist_at_day',
    # SEIR curves
    'plot_seir_evolution',
    'plot_exposed_infected_zoom',
    'plot_peak_time_distribution',
    # Spatial maps
    'plot_spacetime_heatmap',
    'plot_spatial_spread_snapshots',
    # Survival analysis
    'create_forest_plot_hazard_ratios',
    'create_discrimination_performance_plot',
    'create_model_parsimony_plot',
]
