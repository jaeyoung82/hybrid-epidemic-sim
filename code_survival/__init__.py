"""
code_survival: A modular framework for epidemic invasion timing analysis.

Provides three analysis modes:
- Mode A (scenario): Compare gathering scenarios with fixed epidemiological parameters
- Mode B (parameter): Analyze parameter sensitivity across R0/beta/I_ss combinations
- Mode C (seed): Seed-location dependence of epidemic arrival and predictability

Primary analysis (always run by default):
    - Cluster-robust Cox proportional hazards models (Huber-White sandwich SEs)

Supplementary analyses (opt-in via --supplementary flag):
    - Shared frailty model (mixed-effects Cox)
    - Bootstrap uncertainty estimation
    - Model diagnostics (future)

Mode C structure:
    C1: Seed-location dependence (main analysis)
        C1.1 Seed × Destination mean arrival-time matrix
        C1.3 Effective distance vs arrival-time correlations
        C1.4 Representative scatter plots
        C1.5 Per-seed cluster-robust Cox models
    C2-A: Sensitivity to transmission conditions (robustness)
    Supplementary: Network centrality, trajectory similarity (exploratory)
"""

from .discovery import (
    discover_simulations,
    get_scenarios,
    get_param_combinations,
    create_param_combo_label,
    get_no_event_baseline,
    discover_seed_directories,
    discover_seed_param_combinations,
    get_c2a_transmission_conditions,
)

from .dataset_builder import (
    build_survival_dataset,
    load_mobility_matrix,
    load_population_data,
    load_daily_active_cases,
    load_Z_seed,
    load_arrival_times,
    load_E_by_run,
    compute_invasion_intensity,
    compute_pre_invasion_pressure,
    _build_records_for_config,
    _resolve_daily_cases_path,
)

from .cox_models import (
    run_scenario_cox_models,
    run_parameter_cox_models,
    run_seed_cox_models,
    create_causal_model_comparison_table,
    validate_survival_data,
    fit_naive_cox_model,
    fit_clustered_cox_model,
    extract_model_summary,
    compare_naive_vs_clustered,
    create_model_summary_df
)

from .km_analysis import (
    compute_km_curves,
    run_pairwise_logrank_tests,
    print_at_risk_table,
    compute_median_survival_times
)

from .bootstrap import bootstrap_cox_models

from .calibration import (
    assess_model_calibration,
    create_model_validation_summary
)

from .forest_plot import (
    create_forest_plot_hazard_ratios,
    generate_clustered_forest_plot,
    create_discrimination_performance_plot,
    create_model_parsimony_plot,
    create_parameter_sensitivity_heatmap,
    create_parameter_hazard_heatmap,
    create_parameter_forest_plot,
    create_parameter_forest_plot_by_beta,
    extract_param_combo_hrs
)

from .mediation import (
    compute_mediation_metrics,
    generate_mediation_interpretation,
    create_mediation_visualization
)

from .frailty_analysis import (
    fit_frailty_model,
    fit_frailty_M0,
    fit_frailty_M1,
    fit_frailty_M2,
    run_frailty_models,
    run_shared_frailty_analysis,
    save_frailty_outputs,
    generate_frailty_interpretation,
    create_frailty_comparison_table,
    create_frailty_variance_table,
    extract_frailty_summary_dict,
    print_frailty_summary,
    FrailtyResult
)

from .reporting import (
    run_analysis,
    create_survival_figures,
    run_primary_survival_analysis,
    run_kaplan_meier,
    run_clustered_cox_models,
    compare_models,
    save_primary_results,
    generate_primary_figures,
    generate_primary_interpretation,
)

from .seed_distance import (
    compute_effective_distance_matrix,
    compute_shortest_effective_path,
    load_or_compute_effective_distance,
    get_effective_distances_from_seed,
    get_effective_distance_csv_path,
)

from .network_centrality import (
    build_transportation_graph,
    compute_all_centralities,
    load_or_compute_centralities,
)

from .mode_c_dataset import (
    load_mode_c_arrival_times,
    compute_mean_arrival,
    compute_mean_arrival_multi,
    build_mode_c_survival_dataset,
    build_mode_c_dataset_for_params,
    get_effective_distances_from_seed,
)

from .mode_c_analysis import (
    run_mode_c_analysis,
    run_effective_distance_correlations,
    run_c2a_sensitivity,
    compute_c1_seed_destination_analysis,
    compute_seed_destination_arrival_matrix,
    validate_c1_data,
    compute_trajectory_similarity_exploratory,
    compute_network_predictability_supplementary,
    discover_seed_locations,
    plot_seed_scatter,
    plot_regression_comparison,
    plot_seed_destination_arrival_heatmap,
    plot_c2a_sensitivity_heatmap,
    generate_c1_summary,
    REPRESENTATIVE_SEEDS,
    EXPECTED_DISTRICTS,
)

from .supplementary import (
    SUPPLEMENTARY_ANALYSES,
    SUPPORTED_SUPPLEMENTARY,
    resolve_supplementary_requests,
    run_supplementary_frailty,
    save_frailty_results,
    generate_frailty_figures,
    interpret_frailty_models,
    run_supplementary_bootstrap,
    run_supplementary_diagnostics,
)

from .seed_location_robustness import (
    run_seed_location_robustness,
    SELECTED_SEED_DISTRICTS,
    M0_FORMULA,
    M1_FORMULA,
    DEFAULT_R0,
    DEFAULT_BETA,
    DEFAULT_ISS,
    PENALIZER,
    discover_all_seed_districts,
    build_full_seed_dataset,
    fit_per_seed_models,
    extract_seed_row,
    compute_c_index_se,
    run_consistency_checks,
    create_robustness_figure,
    print_robustness_summary,
)

__all__ = [
    'discover_simulations',
    'get_scenarios',
    'get_param_combinations',
    'create_param_combo_label',
    'get_no_event_baseline',
    'discover_seed_param_combinations',
    'get_c2a_transmission_conditions',
    'build_survival_dataset',
    'load_mobility_matrix',
    'load_population_data',
    'load_daily_active_cases',
    'load_Z_seed',
    'load_arrival_times',
    'load_E_by_run',
    'compute_invasion_intensity',
    'compute_pre_invasion_pressure',
    '_build_records_for_config',
    '_resolve_daily_cases_path',
    'run_scenario_cox_models',
    'run_parameter_cox_models',
    'run_seed_cox_models',
    'create_causal_model_comparison_table',
    'validate_survival_data',
    'fit_naive_cox_model',
    'fit_clustered_cox_model',
    'extract_model_summary',
    'compare_naive_vs_clustered',
    'create_model_summary_df',
    'compute_km_curves',
    'run_pairwise_logrank_tests',
    'print_at_risk_table',
    'compute_median_survival_times',
    'bootstrap_cox_models',
    'assess_model_calibration',
    'create_model_validation_summary',
    'create_forest_plot_hazard_ratios',
    'generate_clustered_forest_plot',
    'create_discrimination_performance_plot',
    'create_model_parsimony_plot',
    'create_parameter_sensitivity_heatmap',
    'create_parameter_hazard_heatmap',
    'create_parameter_forest_plot',
    'create_parameter_forest_plot_by_beta',
    'extract_param_combo_hrs',
    'compute_mediation_metrics',
    'generate_mediation_interpretation',
    'create_mediation_visualization',
    'fit_frailty_model',
    'fit_frailty_M0',
    'fit_frailty_M1',
    'fit_frailty_M2',
    'run_frailty_models',
    'run_shared_frailty_analysis',
    'save_frailty_outputs',
    'generate_frailty_interpretation',
    'create_frailty_comparison_table',
    'create_frailty_variance_table',
    'extract_frailty_summary_dict',
    'print_frailty_summary',
    'FrailtyResult',
    'run_analysis',
    'create_survival_figures',
    'run_primary_survival_analysis',
    'run_kaplan_meier',
    'run_clustered_cox_models',
    'compare_models',
    'save_primary_results',
    'generate_primary_figures',
    'generate_primary_interpretation',
    'SUPPLEMENTARY_ANALYSES',
    'SUPPORTED_SUPPLEMENTARY',
    'resolve_supplementary_requests',
    'run_supplementary_frailty',
    'save_frailty_results',
    'generate_frailty_figures',
    'interpret_frailty_models',
    'run_supplementary_bootstrap',
    'run_supplementary_diagnostics',
    'compute_effective_distance_matrix',
    'compute_shortest_effective_path',
    'load_or_compute_effective_distance',
    'get_effective_distances_from_seed',
    'get_effective_distance_csv_path',
    'build_transportation_graph',
    'compute_all_centralities',
    'load_or_compute_centralities',
    'discover_seed_directories',
    'load_mode_c_arrival_times',
    'compute_mean_arrival',
    'compute_mean_arrival_multi',
    'build_mode_c_survival_dataset',
    'build_mode_c_dataset_for_params',
    'get_effective_distances_from_seed',
    'run_mode_c_analysis',
    'run_effective_distance_correlations',
    'run_c2a_sensitivity',
    'compute_c1_seed_destination_analysis',
    'compute_seed_destination_arrival_matrix',
    'validate_c1_data',
    'compute_trajectory_similarity_exploratory',
    'compute_network_predictability_supplementary',
    'discover_seed_locations',
    'plot_seed_scatter',
    'plot_regression_comparison',
    'plot_seed_destination_arrival_heatmap',
    'plot_c2a_sensitivity_heatmap',
    'generate_c1_summary',
    'REPRESENTATIVE_SEEDS',
    'EXPECTED_DISTRICTS',
    'run_seed_location_robustness',
    'SELECTED_SEED_DISTRICTS',
    'discover_all_seed_districts',
    'build_full_seed_dataset',
    'fit_per_seed_models',
    'extract_seed_row',
    'compute_c_index_se',
    'run_consistency_checks',
    'create_robustness_figure',
    'print_robustness_summary',
    'M0_FORMULA',
    'M1_FORMULA',
    'DEFAULT_R0',
    'DEFAULT_BETA',
    'DEFAULT_ISS',
    'PENALIZER',
]
