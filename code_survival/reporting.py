"""
Primary survival analysis: reporting and driver functions.

The cluster-robust Cox proportional hazards model is the **primary** analysis.
The shared frailty (mixed-effects Cox) model is a separate **supplementary**
analysis — see ``code_survival.supplementary``.

Primary analysis pipeline
-------------------------
1. Kaplan-Meier survival curves + log-rank tests
2. Cluster-robust Cox proportional hazards models (Huber-White sandwich SEs)
3. Sequential model comparison (AIC, delta-AIC, likelihood-ratio tests)
4. Model performance evaluation (concordance index)
5. Publication-quality figures (KM curves, forest plots, heatmaps)
6. Interpretation text

Outputs are written directly to ``<output_dir>`` with subdirectory:
    naive/            - naive (model-based SE) model summaries, forest plots
Supplementary outputs go to ``<output_dir>/supplementary/``.
"""

import time
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any

# Ensure Unicode output works on Windows (✓, …, etc.)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .discovery import discover_simulations, get_param_combinations, create_param_combo_label
from .dataset_builder import build_survival_dataset
from .cox_models import (
    run_scenario_cox_models,
    run_parameter_cox_models,
    run_seed_cox_models,
    create_causal_model_comparison_table,
    validate_survival_data,
    extract_model_summary,
    compare_naive_vs_clustered,
    create_model_summary_df,
)
from .km_analysis import compute_km_curves, print_at_risk_table, compute_median_survival_times
from .calibration import assess_model_calibration, create_model_validation_summary
from .forest_plot import (
    create_forest_plot_hazard_ratios,
    generate_clustered_forest_plot,
    create_discrimination_performance_plot,
    create_model_parsimony_plot,
    create_parameter_sensitivity_heatmap,
    create_parameter_hazard_heatmap,
    create_parameter_forest_plot,
    create_parameter_forest_plot_by_beta,
)
from .mediation import generate_mediation_interpretation, compute_mediation_metrics



# ============================================================================
# Directory-structure helpers
# ============================================================================

def _setup_primary_directories(output_dir: str) -> Dict[str, Path]:
    """Create and return the primary analysis directory layout.

    Primary output files are saved directly under ``<output_dir>`` so users
    can inspect them without navigating subfolders.  Only ``naive/`` keeps
    a subdirectory (for the naive-SE comparison results).
    """
    base_dir = Path(output_dir)
    naive_dir = base_dir / "naive"
    dirs = {
        "base": base_dir,
        "km": base_dir,
        "clustered_cox": base_dir,
        "naive": naive_dir,
        "model_comparison": base_dir,
        "figures": base_dir,
        "tables": base_dir,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _setup_supplementary_directory(output_dir: str) -> Path:
    """Create and return the supplementary analysis directory.

    The ``supplementary/`` directory is created under *output_dir* and serves
    as the root for all supplementary analysis subdirectories (e.g.
    ``supplementary/frailty/``).
    """
    supp_dir = Path(output_dir) / "supplementary"
    supp_dir.mkdir(parents=True, exist_ok=True)
    return supp_dir


# ============================================================================
# Console-output helpers
# ============================================================================

def _print_primary_header(analysis_mode: str, r0: float, beta: float,
                           iss: int, fixed_scenario: Optional[str]):
    """Print the primary-analysis header block."""
    print("=" * 60)
    print("Primary Survival Analysis")
    print("=" * 60)
    print()
    if analysis_mode == "scenario":
        print("  Mode A (scenario comparison)")
        print(f"  Parameters: R0={r0}, beta={beta}, Iss={iss}")
    elif analysis_mode == "seed":
        print("  Mode C (seed location analysis)")
        print(f"  Parameters: R0={r0}, beta={beta}, Iss={iss}")
        print(f"  Cluster: simulation_run_id (prefixed with seed_location)")
    else:
        print("  Mode B (parameter sensitivity)")
        print(f"  Scenario: {fixed_scenario}")
    print()
    print("  Primary analysis: Cluster-robust Cox proportional hazards models")
    print("  (Huber-White sandwich standard errors clustered on simulation_run_id)")
    print()


def _print_primary_summary(steps: List[str]):
    """Print the checkmark summary for completed primary-analysis steps."""
    print()
    print("=" * 60)
    print("Primary Survival Analysis")
    print("=" * 60)
    print()
    for step in steps:
        print(f"  \u2713 {step}")
    print()


def _print_supplementary_status(supplementary_requested: Optional[List[str]]):
    """Print the supplementary-analysis status table."""
    from .supplementary import SUPPORTED_SUPPLEMENTARY, SUPPLEMENTARY_ANALYSES

    requested_set = set(supplementary_requested or [])

    print("=" * 60)
    print("Supplementary Analyses")
    print("=" * 60)
    print()

    for key in SUPPORTED_SUPPLEMENTARY:
        meta = SUPPLEMENTARY_ANALYSES[key]
        if "all" in requested_set or key in requested_set:
            status = "running"
        else:
            status = "skipped"
        label = meta["name"]
        width = 32
        dots = "." * max(1, width - len(label))
        print(f"  {label} {dots} {status}")

    print()
    print("-" * 50)
    print()


# ============================================================================
# Step 1a: Kaplan-Meier analysis (primary)
# ============================================================================

def run_kaplan_meier(
    survival_df: pd.DataFrame,
    km_dir: Path,
    grouping_col: str,
    group_values: List,
    analysis_name: str = "by_scenario",
) -> Dict:
    """Run Kaplan-Meier survival curves, log-rank tests, and at-risk tables.

    Saves CSV data and PNG/PDF figure to *km_dir*.

    Returns a dict with keys: ``curve_data``, ``km_data``,
    ``logrank_pvalue``, ``global_test``, ``pairwise_results``.
    """
    curve_data, km_data, logrank_pvalue, global_test, pairwise_results = compute_km_curves(
        survival_df, grouping_col, group_values
    )

    km_data.to_csv(km_dir / "km_curve_data.csv", index=False)

    print("\n  Kaplan-Meier Analysis")
    print("  " + "=" * 50)

    print("\n  Global log-rank test")
    print("  " + "-" * 30)
    print(f"  Chi-square = {global_test['chi_square']:.4f}")
    print(f"  df = {global_test['df']}")
    print(f"  p = {global_test['p_value']:.6f}")

    print("\n  Pairwise comparisons (Holm corrected)")
    print("  " + "-" * 40)
    if len(pairwise_results) > 0:
        for _, row in pairwise_results.iterrows():
            print(f"  {row['Comparison']:<30} p = {row['Raw p-value']:.6f}")

        all_significant = pairwise_results["Significant"].all()
        print(f"\n  All comparisons significant after Holm correction: {'Yes' if all_significant else 'No'}")

        pairwise_results.to_csv(km_dir / "km_pairwise_logrank_results.csv", index=False)
    else:
        print("  No pairwise comparisons could be computed (insufficient data)")

    print_at_risk_table(km_data, group_values)

    _plot_km_curves(curve_data, group_values, grouping_col, km_dir, analysis_name)

    return {
        "curve_data": curve_data,
        "km_data": km_data,
        "logrank_pvalue": logrank_pvalue,
        "global_test": global_test,
        "pairwise_results": pairwise_results,
    }


def _plot_km_curves(curve_data: pd.DataFrame, group_values: List,
                    grouping_col: str, km_dir: Path,
                    analysis_name: str = "by_scenario"):
    """Generate and save the Kaplan-Meier survival curve figure."""
    import matplotlib.pyplot as plt

    colors = {
        "baseline": "#1f77b4",
        "no_event": "#1f77b4",
        "AMS_dance": "#ff7f0e",
        "AMS_football": "#2ca02c",
        "Leipzig_1": "#d62728",
        "Leipzig_2": "#9467bd",
        "Leipzig_3": "#8c564b",
    }

    markers = {
        "baseline": "o",
        "no_event": "o",
        "AMS_dance": "s",
        "AMS_football": "^",
        "Leipzig_1": "D",
        "Leipzig_2": "v",
        "Leipzig_3": "p",
    }

    fig, ax = plt.subplots(figsize=(6, 4))

    for group in curve_data["group"].unique():
        group_data = curve_data[curve_data["group"] == group].sort_values("time")

        ax.step(group_data["time"], group_data["survival_probability"],
                label=group, color=colors.get(group, "gray"), linewidth=2, where="post")

        ax.fill_between(group_data["time"],
                        group_data["lower_confidence_interval"],
                        group_data["upper_confidence_interval"],
                        alpha=0.15, color=colors.get(group, "gray"))

        times = group_data["time"].values
        survivals = group_data["survival_probability"].values
        max_time = int(times.max())
        for day in range(0, max_time + 1, 10):
            idx = np.searchsorted(times, day, side="right")
            if idx > 0:
                surv_at_day = survivals[idx - 1]
            elif idx < len(times):
                surv_at_day = survivals[idx]
            else:
                continue
            ax.plot(day, surv_at_day,
                    marker=markers.get(group, "o"), markersize=6,
                    markerfacecolor="white", markeredgewidth=1.5,
                    color=colors.get(group, "gray"))

    if grouping_col == "scenario":
        ax.set_xlabel("Time (days)", fontsize=12)
        ax.set_ylabel("Survival Probability", fontsize=12)
        ax.set_title("Kaplan-Meier Survival Curves: Epidemic Arrival", fontsize=14)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="#1f77b4", linestyle="-",
                   markerfacecolor="white", markeredgecolor="#1f77b4",
                   markeredgewidth=1.5, markersize=8, label="baseline/no_event"),
            Line2D([0], [0], marker="s", color="#ff7f0e", linestyle="-",
                   markerfacecolor="white", markeredgecolor="#ff7f0e",
                   markeredgewidth=1.5, markersize=8, label="AMS_dance"),
            Line2D([0], [0], marker="^", color="#2ca02c", linestyle="-",
                   markerfacecolor="white", markeredgecolor="#2ca02c",
                   markeredgewidth=1.5, markersize=8, label="AMS_football"),
            Line2D([0], [0], marker="D", color="#d62728", linestyle="-",
                   markerfacecolor="white", markeredgecolor="#d62728",
                   markeredgewidth=1.5, markersize=8, label="Leipzig_1"),
            Line2D([0], [0], marker="v", color="#9467bd", linestyle="-",
                   markerfacecolor="white", markeredgecolor="#9467bd",
                   markeredgewidth=1.5, markersize=8, label="Leipzig_2"),
            Line2D([0], [0], marker="p", color="#8c564b", linestyle="-",
                   markerfacecolor="white", markeredgecolor="#8c564b",
                   markeredgewidth=1.5, markersize=8, label="Leipzig_3"),
        ]
        ax.legend(handles=legend_elements, loc="upper center",
                  ncol=3, fontsize=9, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(str(km_dir / f"km_survival_{analysis_name}.png"), dpi=600, bbox_inches="tight")
        plt.savefig(str(km_dir / f"km_survival_{analysis_name}.pdf"), bbox_inches="tight")
        plt.close()
    else:
        plt.close()


# ============================================================================
# Step 1b: Cluster-robust Cox models (primary)
# ============================================================================

def run_clustered_cox_models(
    survival_df: pd.DataFrame,
    dirs: Dict[str, Path],
    grouping_col: str,
    analysis_mode: str,
    cluster_col: str = "simulation_run_id",
    penalizer: float = 0.1,
) -> Dict:
    """Fit cluster-robust Cox proportional hazards models.

    Fits both naive (model-based SEs) and cluster-robust (Huber-White
    sandwich) Cox models for each covariate set in the model hierarchy.
    Saves summaries to ``<output_dir>/`` (robust) and ``naive/``.

    Returns a dict keyed by model name, each value a result dict with both
    naive and clustered summaries.
    """
    print("\n  Fitting Cox models...")
    print("  * Cluster-robust (Huber-White sandwich) standard errors clustered on simulation_run_id")

    if analysis_mode == "parameter":
        model_results = run_parameter_cox_models(
            survival_df, factorial=False, cluster_col=cluster_col
        )
    elif analysis_mode == "seed":
        model_results = run_seed_cox_models(
            survival_df, cluster_col=cluster_col
        )
    else:
        model_results = run_scenario_cox_models(survival_df, cluster_col=cluster_col)

    naive_dir = dirs["naive"]
    clustered_dir = dirs["clustered_cox"]

    for model_name, results in model_results.items():
        # Clustered (robust) summary
        clustered_summary_df = create_model_summary_df(results, version="clustered")
        clustered_summary_df.to_csv(
            clustered_dir / f"{model_name}_summary.csv", index=False
        )

        # Naive summary
        naive_summary_df = create_model_summary_df(results, version="naive")
        naive_summary_df.to_csv(
            naive_dir / f"{model_name}_summary.csv", index=False
        )

        # Print results
        print(f"\n  {model_name}:")
        se_type = "Robust clustered" if results.get("robust_se_used", False) else "Naive"
        print(f"    SE type: {se_type} (cluster=simulation_run_id)")
        print(f"    Concordance: {results['concordance']:.4f}")
        print(f"    AIC: {results['aic']:.2f}")
        print(f"    Partial log-likelihood: {results['log_likelihood']:.4f}")
        print(f"    N_obs: {results.get('n_obs', 'N/A')}, N_clusters: {results.get('n_clusters', 'N/A')}")
        for var, coef in results["coefficients"].items():
            hr = results["hazard_ratios"][var]
            p_idx = list(results["coefficients"].keys()).index(var)
            p_val = results["p_values"][p_idx] if p_idx < len(results["p_values"]) else np.nan
            se_val = results.get("se", [np.nan])[p_idx] if p_idx < len(results.get("se", [np.nan])) else np.nan
            ci_lo = results.get("ci_lower", [np.nan])[p_idx] if p_idx < len(results.get("ci_lower", [np.nan])) else np.nan
            ci_hi = results.get("ci_upper", [np.nan])[p_idx] if p_idx < len(results.get("ci_upper", [np.nan])) else np.nan
            print(f"    {var}: coef={coef:.4f}, HR={hr:.3f}, robust_SE={se_val:.4f}, "
                  f"95% CI=[{ci_lo:.3f}, {ci_hi:.3f}], p={p_val:.4f}")

    n_saved = len(model_results)
    print(f"\n  Saved {n_saved} model summaries to each of naive/ and clustered/")
    return model_results


# ============================================================================
# Step 2b: Model comparison (primary)
# ============================================================================

def compare_models(
    model_results: Dict,
    dirs: Dict[str, Path],
) -> Dict:
    """Create and save model comparison tables.

    Generates:
    - ``model_comparison.csv`` -- sequential model comparison (AIC, LRT, etc.)
    - ``model_comparison_naive_vs_clustered.csv`` -- SE comparison

    Saves to ``<output_dir>/`` (base) and ``naive/``.
    Returns a dict with ``comparison`` and ``naive_vs_clustered`` DataFrames.
    """
    print("\n  Model comparison...")
    comparison = create_causal_model_comparison_table(model_results)
    comparison.to_csv(
        dirs["model_comparison"] / "model_comparison.csv", index=False
    )
    comparison.to_csv(
        dirs["naive"] / "model_comparison.csv", index=False
    )
    print(comparison.to_string(index=False))

    naive_vs_robust = compare_naive_vs_clustered(model_results)
    naive_vs_robust.to_csv(
        dirs["model_comparison"] / "model_comparison_naive_vs_clustered.csv",
        index=False,
    )
    naive_vs_robust.to_csv(
        dirs["naive"] / "model_comparison_naive_vs_clustered.csv",
        index=False,
    )
    print(f"\n  Naive vs. Cluster-robust comparison ({len(naive_vs_robust)} rows):")
    print(naive_vs_robust.to_string(index=False))

    return {"comparison": comparison, "naive_vs_clustered": naive_vs_robust}


# ============================================================================
# Step 3: Save primary results (data tables)
# ============================================================================

def save_primary_results(
    survival_df: pd.DataFrame,
    dirs: Dict[str, Path],
    analysis_name: str,
) -> None:
    """Save the survival dataset CSV to ``tables/``."""
    dataset_path = dirs["tables"] / f"{analysis_name}.csv"
    survival_df.to_csv(dataset_path, index=False)
    print(f"  Saved survival dataset to {dataset_path}")


# ============================================================================
# Step 4: Generate primary figures
# ============================================================================

def generate_primary_figures(
    model_results: Dict,
    survival_df: pd.DataFrame,
    dirs: Dict[str, Path],
    analysis_name: str,
    analysis_mode: str,
) -> None:
    """Generate publication-quality figures.

    Saves all figures to ``figures/`` and forest plots to
    ``clustered_cox/naive/`` and ``clustered_cox/clustered/``.
    """
    print("\n  Generating figures...")
    try:
        figures_dir = dirs["figures"]
        base = str(figures_dir / analysis_name)

        # Short analysis name without mode prefix for cleaner figure filenames
        short_name = analysis_name
        for _prefix in ('scenario_analysis_', 'parameter_analysis_', 'seed_analysis_'):
            if short_name.startswith(_prefix):
                short_name = short_name[len(_prefix):]
                break

        if analysis_mode == "parameter":
            create_parameter_hazard_heatmap(
                model_results, survival_df, f"{base}_heatmap"
            )
            create_parameter_forest_plot(
                model_results, survival_df, f"{base}_forest_R0", use_robust=True
            )
            create_parameter_forest_plot_by_beta(
                model_results, survival_df, f"{base}_forest_beta", use_robust=True
            )
            create_discrimination_performance_plot(
                model_results, str(figures_dir / f"discrimination_performance_{short_name}")
            )
            create_model_parsimony_plot(
                model_results, str(figures_dir / f"model_parsimony_{short_name}")
            )
        else:
            # Naive (model-based CIs) -- goes to naive/ dir
            seed_grouping = "seed_location" if analysis_mode == "seed" else "scenario"
            create_forest_plot_hazard_ratios(
                model_results,
                str(dirs["naive"] / "forest_plot"),
                grouping_col=seed_grouping,
                use_robust=False,
            )
            # Clustered (robust CIs) -- saved at figures/ level with analysis name suffix
            generate_clustered_forest_plot(
                model_results,
                str(figures_dir / f"forest_plot_{short_name}"),
                grouping_col=seed_grouping,
            )
            create_discrimination_performance_plot(
                model_results, str(figures_dir / f"discrimination_performance_{short_name}")
            )
            create_model_parsimony_plot(
                model_results, str(figures_dir / f"model_parsimony_{short_name}")
            )
    except Exception as e:
        print(f"  Warning: Could not generate figures: {e}")


# ============================================================================
# Step 5: Primary interpretation
# ============================================================================

def generate_primary_interpretation(
    model_results: Dict,
    analysis_mode: str,
) -> str:
    """Generate interpretation text for the primary analysis."""
    return generate_mediation_interpretation(
        model_results,
        analysis_mode=analysis_mode,
        verbose=True,
    )


# ============================================================================
# Primary analysis pipeline
# ============================================================================

def run_primary_survival_analysis(
    survival_df: pd.DataFrame,
    analysis_mode: str,
    r0: float,
    beta: float,
    iss: int,
    fixed_scenario: Optional[str],
    output_dir: str,
    cluster_col: str = "simulation_run_id",
    expected_districts: int = 21,
    seed_locations: Optional[List[int]] = None,
) -> Dict:
    """
    Run the complete primary survival analysis pipeline.

    Executes:
    1. Kaplan-Meier analysis with log-rank tests
    2. Cluster-robust Cox proportional hazards models
    3. Sequential model comparison
    4. Model performance evaluation (concordance index)
    5. Publication-quality figures
    6. Interpretation text

    All outputs are saved directly under ``<output_dir>`` with
    subdirectories: ``km/``, ``clustered_cox/``, ``model_comparison/``,
    ``figures/``, ``tables/``.

    Args:
        seed_locations: Seed district IDs being analyzed.  When provided,
            the seed IDs are incorporated into ``analysis_name`` to avoid
            file collisions between runs with different seed selections.

    Returns a dict with: ``survival_df``, ``model_results``,
    ``comparison``, ``naive_vs_clustered``, ``mediation_metrics``,
    ``interpretation``, ``timing_log``, ``dirs``.
    """
    timing_log: List[str] = []
    total_start = time.time()
    primary_steps: List[str] = []

    # --- Compute analysis name (used for file prefixes) ---
    if analysis_mode == "parameter" and fixed_scenario:
        analysis_name = f"{analysis_mode}_analysis_{fixed_scenario}"
    else:
        analysis_name = (
            f"{analysis_mode}_analysis_R{int(r0 * 100)}"
            f"_beta{int(beta * 100)}_Iss{iss}"
        )

    # Incorporate seed info to avoid file collisions when seeds are present
    if seed_locations:
        seed_str = "seeds" + "_".join(str(s) for s in sorted(seed_locations))
        analysis_name = analysis_name + f"_{seed_str}"

    # Short analysis name without mode prefix for figure filenames
    short_name = analysis_name
    for _prefix in ('scenario_analysis_', 'parameter_analysis_', 'seed_analysis_'):
        if short_name.startswith(_prefix):
            short_name = short_name[len(_prefix):]
            break

    # --- Create directory structure ---
    dirs = _setup_primary_directories(output_dir)
    print(f"  Primary output directory: {dirs['base']}")

    # --- Data validation ---
    print("\n  Validating data for cluster-robust analysis...")
    try:
        validate_survival_data(
            survival_df, cluster_col=cluster_col,
            expected_districts=expected_districts,
        )
    except Exception as e:
        warnings.warn(f"Data validation warning: {e}")

    # --- Step 1: Kaplan-Meier analysis ---
    step_start = time.time()
    print("\n  Step 1: Kaplan-Meier analysis...")
    if analysis_mode == "parameter":
        grouping_col = "param_combo"
    elif analysis_mode == "seed":
        grouping_col = "seed_location"
    else:
        grouping_col = "scenario"
    group_values = survival_df[grouping_col].unique().tolist()

    run_kaplan_meier(survival_df, dirs["km"], grouping_col, group_values, analysis_name=short_name)

    step_time = time.time() - step_start
    timing_log.append(f"Step 1 (Kaplan-Meier analysis): {step_time:.2f} seconds")
    print(f"  Completed in {step_time:.2f} seconds")
    primary_steps.extend(["Kaplan-Meier analysis", "Log-rank tests"])

    # --- Step 2: Cluster-robust Cox models ---
    step_start = time.time()
    print("\n  Step 2: Fitting Cox models...")
    model_results = run_clustered_cox_models(
        survival_df, dirs, grouping_col, analysis_mode, cluster_col=cluster_col
    )

    step_time = time.time() - step_start
    timing_log.append(
        f"Step 2 (cluster-robust Cox models): {step_time:.2f} seconds"
    )
    print(f"  Completed in {step_time:.2f} seconds")
    primary_steps.extend(["Cluster-robust Cox models", "Performance evaluation"])

    # --- Step 3: Model comparison ---
    step_start = time.time()
    comparison_results = compare_models(model_results, dirs)

    step_time = time.time() - step_start
    timing_log.append(f"Step 3 (model comparison): {step_time:.2f} seconds")
    print(f"  Completed in {step_time:.2f} seconds")
    primary_steps.append("Model comparison")

    # --- Step 3b: Save primary results (data tables) ---
    step_start = time.time()
    print("\n  Step 3b: Saving primary results...")
    save_primary_results(survival_df, dirs, analysis_name)
    step_time = time.time() - step_start
    timing_log.append(f"Step 3b (save results): {step_time:.2f} seconds")

    # --- Step 5: Mediation metrics ---
    step_start = time.time()
    mediation_metrics = compute_mediation_metrics(model_results)
    step_time = time.time() - step_start
    timing_log.append(f"Step 4 (mediation metrics): {step_time:.2f} seconds")

    # --- Step 6: Interpretation ---
    step_start = time.time()
    print("\n  Step 5: Interpretation...")
    interpretation = generate_primary_interpretation(
        model_results,
        analysis_mode,
    )

    interp_file = dirs["base"] / f"{analysis_name}_interpretation.txt"
    with open(interp_file, "w") as f:
        f.write(interpretation)
    step_time = time.time() - step_start
    timing_log.append(f"Step 5 (interpretation): {step_time:.2f} seconds")

    # --- Step 7: Figures ---
    step_start = time.time()
    print("\n  Step 6: Generating figures...")
    generate_primary_figures(
        model_results, survival_df, dirs, analysis_name, analysis_mode
    )
    step_time = time.time() - step_start
    timing_log.append(f"Step 6 (figures): {step_time:.2f} seconds")
    print(f"  Completed in {step_time:.2f} seconds")
    primary_steps.append("Figures")

    # --- Write timing log ---
    total_time = time.time() - total_start
    timing_log.append(f"Total execution time: {total_time:.2f} seconds")
    timing_path = dirs["base"] / f"{analysis_name}_timing_log.txt"
    with open(timing_path, "w") as f:
        f.write("\n".join(timing_log))

    # --- Print primary summary ---
    _print_primary_summary(primary_steps)

    return {
        "survival_df": survival_df,
        "model_results": model_results,
        "comparison": comparison_results["comparison"],
        "naive_vs_clustered": comparison_results["naive_vs_clustered"],
        "mediation_metrics": mediation_metrics,
        "interpretation": interpretation,
        "timing_log": timing_log,
        "dirs": dirs,
        "analysis_name": analysis_name,
    }


# ============================================================================
# Backward-compatible wrapper
# ============================================================================

def create_survival_figures(output_base: str,
                            survival_df: pd.DataFrame,
                            model_results: Dict,
                            grouping_col: str = "scenario",
                            analysis_mode: str = "scenario"):
    """Backward-compatible wrapper for Kaplan-Meier figure generation.

    Now delegates to :func:`run_kaplan_meier` which saves to
    ``km/`` under the parent of *output_base*.
    """
    output_dir = Path(output_base).parent
    dirs = _setup_primary_directories(str(output_dir))
    group_values = survival_df[grouping_col].unique().tolist()
    run_kaplan_meier(survival_df, dirs["km"], grouping_col, group_values)


# ============================================================================
# Main entry point
# ============================================================================

def run_analysis(
    analysis_mode: str = "scenario",
    results_dir: str = "results_hybrid_sim",
    r0: float = 1.5,
    beta: float = 0.5,
    iss: int = 1,
    fixed_scenario: Optional[str] = None,
    skip_bootstrap: bool = False,
    n_bootstrap: int = 1000,
    output_dir: str = "survival_analysis_results",
    supplementary: Optional[List[str]] = None,
    seed_locations: Optional[List[int]] = None,
    per_seed: bool = False,
) -> Dict:
    """
    Run the complete survival analysis pipeline.

    **Primary analysis** (always run):
        - Kaplan-Meier analysis with log-rank tests
        - Cluster-robust Cox proportional hazards models
        - Sequential model comparison (AIC, LRT)
        - Model performance evaluation (concordance index)
        - Publication-quality figures and interpretation

    **Supplementary analyses** (only when requested via *supplementary*):
        - ``frailty``     -- shared frailty (mixed-effects Cox) model
        - ``bootstrap``   -- non-parametric bootstrap for HR uncertainty
        - ``diagnostics`` -- model diagnostics (reserved/future)
        - ``all``         -- run every registered supplementary analysis

    Args:
        analysis_mode: ``"scenario"`` (Mode A) or ``"parameter"`` (Mode B).
        results_dir: Path to simulation results directory.
        r0, beta, iss: Default parameters for Mode A.
        fixed_scenario: Scenario name for Mode B (e.g. ``"AMS_dance"``).
        skip_bootstrap: Deprecated — kept for backward compatibility. Bootstrap is now
            a supplementary analysis controlled by *supplementary*.
         n_bootstrap: Number of bootstrap samples (used when ``bootstrap`` is in
             *supplementary*). Default 1000.
         output_dir: Top-level output directory.
         supplementary: List of supplementary analysis names to run,
             e.g. ``["frailty"]`` or ``["all"]``.  ``None`` (default) runs
             no supplementary analyses.
        seed_locations: Seed district IDs to analyze.  When ``None``,
            auto-discovers all ``seed_*`` subdirectories under *results_dir*.
            When an empty list ``[]`` is given, forces flat-directory mode.
        per_seed: When ``True`` and seed locations are available, runs the
            full analysis pipeline independently for each seed, writing
            outputs to ``<output_dir>/seed_{X}/``.  When ``False`` (default),
            all seeds are combined into a single analysis.

     Returns:
        Dictionary with ``survival_df``, ``model_results``, ``comparison``,
        ``naive_vs_clustered``, ``frailty_results``, ``mediation_metrics``,
        ``interpretation``, and ``supplementary_results``.
    """
    _print_primary_header(analysis_mode, r0, beta, iss, fixed_scenario)

    # --- Per-seed mode: run analysis independently for each seed ---
    if per_seed:
        from .discovery import discover_seed_directories

        effective_seeds = seed_locations
        if effective_seeds is None:
            effective_seeds = discover_seed_directories(results_dir)

        if not effective_seeds:
            print("  ERROR: No seed directories discovered.")
            return {}

        print(f"  Running per-seed analysis for seeds: {sorted(effective_seeds)}")
        per_seed_results: Dict[int, Dict] = {}
        for seed in sorted(effective_seeds):
            seed_output_dir = str(Path(output_dir) / f"seed_{seed}")
            print(f"\n>>> Analysis for seed {seed} -> {seed_output_dir}")
            result = run_analysis(
                analysis_mode=analysis_mode,
                results_dir=results_dir,
                r0=r0,
                beta=beta,
                iss=iss,
                fixed_scenario=fixed_scenario,
                skip_bootstrap=skip_bootstrap,
                n_bootstrap=n_bootstrap,
                output_dir=seed_output_dir,
                supplementary=supplementary,
                seed_locations=[seed],
                per_seed=False,
            )
            per_seed_results[seed] = result

        print("=" * 60)
        print("Per-seed analysis complete!")
        print("=" * 60)
        print()
        return {"per_seed_results": per_seed_results}

    # --- Combined mode (default): all seeds in one analysis ---
    # --- Build dataset ---
    if analysis_mode == "parameter" and fixed_scenario:
        configs = get_param_combinations(
            results_dir, scenario=fixed_scenario, iss=iss,
            seed_locations=seed_locations,
        )
        if not configs:
            print(f"No simulations found for scenario {fixed_scenario}")
            return {}
    else:
        configs = None

    print("  Building survival dataset...")
    survival_df = build_survival_dataset(
        results_dir=results_dir,
        configs=configs,
        r0=r0,
        beta=beta,
        iss=iss,
        analysis_mode=analysis_mode,
        seed_locations=seed_locations,
    )
    print(f"  Dataset has {len(survival_df)} rows")

    if survival_df.empty:
        print("  ERROR: Empty dataset. Check simulation outputs.")
        return {}

    # --- Run primary analysis ---
    primary_results = run_primary_survival_analysis(
        survival_df=survival_df,
        analysis_mode=analysis_mode,
        r0=r0,
        beta=beta,
        iss=iss,
        fixed_scenario=fixed_scenario,
        output_dir=output_dir,
        seed_locations=seed_locations,
    )

    # --- Print supplementary-analysis status (always shown) ---
    _print_supplementary_status(supplementary)

    # --- Run requested supplementary analyses ---
    from .supplementary import (
        resolve_supplementary_requests,
        SUPPLEMENTARY_ANALYSES,
    )

    supplementary_results: Dict[str, Any] = {}

    if supplementary:
        supp_names = resolve_supplementary_requests(supplementary)
        supp_dir = _setup_supplementary_directory(output_dir)

        for name in supp_names:
            meta = SUPPLEMENTARY_ANALYSES[name]
            try:
                result = meta["run"](
                    survival_df=primary_results.get("survival_df", survival_df),
                    primary_model_results=primary_results.get("model_results", {}),
                    supplementary_dir=supp_dir,
                    analysis_mode=analysis_mode,
                    **({"n_bootstrap": n_bootstrap} if name == "bootstrap" else {}),
                )
                supplementary_results[name] = result
            except Exception as e:
                print(f"  {meta['name']} ... failed: {e}")
                import traceback
                traceback.print_exc()
                supplementary_results[name] = None
            print()

    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print()

    return {
        "survival_df": primary_results.get("survival_df", survival_df),
        "model_results": primary_results.get("model_results", {}),
        "comparison": primary_results.get("comparison", None),
        "naive_vs_clustered": primary_results.get("naive_vs_clustered", None),
        "frailty_results": supplementary_results.get("frailty", {}),
        "mediation_metrics": primary_results.get("mediation_metrics", None),
        "interpretation": primary_results.get("interpretation", ""),
        "supplementary_results": supplementary_results,
    }
