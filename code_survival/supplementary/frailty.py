"""
Supplementary analysis: Shared frailty (mixed-effects Cox) model.

This module wraps the frailty analysis implementation in ``frailty_analysis.py``
and integrates it into the supplementary-analysis framework.

The shared frailty model adds a Gaussian random intercept per simulation run
to account for unobserved between-run heterogeneity:

    h_ij(t) = h_0(t) * exp(X_ij . beta + u_j)
    u_j ~ Normal(0, sigma^2)

Outputs are saved to ``supplementary/frailty/`` with subdirectories:
    summaries/       – model-level summary CSV
    tables/          – coefficient, variance, and comparison CSVs
    figures/         – frailty-specific plots (future)
    interpretation/  – human-readable interpretation text
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..frailty_analysis import (
    run_shared_frailty_analysis,
    save_frailty_outputs,
    generate_frailty_interpretation,
    print_frailty_summary,
    FrailtyResult,
)

# Subdirectory names within supplementary/frailty/
SUMMARIES_DIR = "summaries"
TABLES_DIR = "tables"
FIGURES_DIR = "figures"
INTERPRETATION_DIR = "interpretation"


def run_supplementary_frailty(
    survival_df: pd.DataFrame,
    primary_model_results: Dict,
    supplementary_dir: Path,
    analysis_mode: str = "scenario",
    max_runs: int = 200,
    random_state: int = 42,
    debug: bool = False,
) -> Dict[str, FrailtyResult]:
    """
    Run the shared frailty supplementary analysis.

    This is a high-level entry point that:
    1. Fits the shared frailty (mixed-effects Cox) model via ``run_shared_frailty_analysis``.
    2. Saves all outputs to ``supplementary/frailty/`` with a structured subdirectory layout.
    3. Prints a concise summary to the console.

    Args:
        survival_df: Survival analysis DataFrame.
        primary_model_results: Primary (cluster-robust Cox) model results dict,
            used for side-by-side comparison in the frailty output.
        supplementary_dir: Path to the ``supplementary/`` directory (already created).
        analysis_mode: ``"scenario"`` (Mode A) or ``"parameter"`` (Mode B).
        max_runs: Maximum number of simulation-run clusters to use for frailty fitting.
        random_state: Random seed for reproducible subsampling.
        debug: If True, print intermediate debugging information.

    Returns:
        Dict mapping model names (e.g. ``M0_gathering``) to ``FrailtyResult``.
    """
    frailty_base = supplementary_dir / "frailty"
    frailty_base.mkdir(parents=True, exist_ok=True)

    # Create structured subdirectories
    for subdir in (SUMMARIES_DIR, TABLES_DIR, FIGURES_DIR, INTERPRETATION_DIR):
        (frailty_base / subdir).mkdir(parents=True, exist_ok=True)

    # --- Fit the shared frailty model ---
    frailty_results = run_shared_frailty_analysis(
        survival_df,
        cluster_col="simulation_run_id",
        max_runs=max_runs,
        random_state=random_state,
        debug=debug,
    )

    # --- Save outputs to structured subdirectories ---
    save_frailty_results(
        frailty_results,
        primary_model_results,
        str(frailty_base),
        analysis_mode,
    )

    # --- Print summary ---
    if frailty_results:
        for model_name, result in frailty_results.items():
            print_frailty_summary(result)
    else:
        print("  No frailty models could be fit (covariates may be constant in this mode).")

    return frailty_results


def save_frailty_results(
    frailty_results: Dict[str, FrailtyResult],
    primary_model_results: Dict,
    frailty_dir: str,
    analysis_mode: str = "scenario",
) -> Path:
    """
    Save frailty analysis outputs to structured subdirectories.

    Layout inside *frailty_dir*:

        summaries/frailty_model_summary.csv
        tables/frailty_coefficients.csv
        tables/frailty_variance.csv
        tables/frailty_model_comparison.csv
        figures/                           (reserved for future plots)
        interpretation/frailty_interpretation.txt

    Args:
        frailty_results: Dict from ``run_shared_frailty_analysis``.
        primary_model_results: Primary Cox model results for side-by-side comparison.
        frailty_dir: Path to the ``supplementary/frailty`` directory.
        analysis_mode: ``"scenario"`` or ``"parameter"``.

    Returns:
        Path to the frailty output directory.
    """
    # Delegate to the updated save_frailty_outputs in frailty_analysis.py,
    # which now creates the subdirectory structure.
    return save_frailty_outputs(
        frailty_results,
        primary_model_results,
        frailty_dir,
        analysis_mode,
    )


def generate_frailty_figures(
    frailty_results: Dict[str, FrailtyResult],
    frailty_dir: str,
    analysis_mode: str = "scenario",
) -> None:
    """
    Generate figures specific to the frailty analysis.

    Currently a placeholder for future frailty-specific plots (e.g., frailty
    distribution, frailty-effect forest plots). Reserved for future expansion.

    Args:
        frailty_results: Dict from ``run_shared_frailty_analysis``.
        frailty_dir: Path to the ``supplementary/frailty`` directory.
        analysis_mode: ``"scenario"`` or ``"parameter"``.
    """
    figures_dir = Path(frailty_dir) / FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    # Reserved for future frailty-specific figure generation


def interpret_frailty_models(
    frailty_results: Dict[str, FrailtyResult],
    primary_model_results: Dict,
    analysis_mode: str = "scenario",
) -> str:
    """
    Generate human-readable interpretation of the shared frailty analysis.

    Wraps ``generate_frailty_interpretation`` from ``frailty_analysis.py``.

    Args:
        frailty_results: Dict from ``run_shared_frailty_analysis``.
        primary_model_results: Primary Cox model results for comparison.
        analysis_mode: ``"scenario"`` or ``"parameter"``.

    Returns:
        Multi-line string with the interpretation.
    """
    return generate_frailty_interpretation(
        frailty_results,
        primary_model_results,
        analysis_mode,
    )
