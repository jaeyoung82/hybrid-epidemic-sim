"""
Supplementary analysis: Non-parametric bootstrap for hazard ratio uncertainty.

Re-samples districts within scenarios (with replacement) and refits Cox models
to obtain empirical standard errors and confidence intervals for hazard ratios.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..bootstrap import bootstrap_cox_models


def run_supplementary_bootstrap(
    survival_df: pd.DataFrame,
    primary_model_results: Dict,
    supplementary_dir: Path,
    analysis_mode: str = "scenario",
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Optional[pd.DataFrame]:
    """
    Run the bootstrap supplementary analysis.

    Args:
        survival_df: Survival analysis DataFrame.
        primary_model_results: Primary Cox model results dict.
        supplementary_dir: Path to the ``supplementary/`` directory.
        analysis_mode: ``"scenario"`` or ``"parameter"``.
        n_bootstrap: Number of bootstrap iterations.
        random_state: Random seed for reproducibility.

    Returns:
        Bootstrap results DataFrame, or None if insufficient data.
    """
    tables_dir = supplementary_dir / "bootstrap" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_results = bootstrap_cox_models(
        survival_df,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    bootstrap_results.to_csv(
        tables_dir / "bootstrap_HR_summary.csv", index=False
    )

    return bootstrap_results
