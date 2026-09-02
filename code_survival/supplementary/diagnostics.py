"""
Supplementary analysis: Model diagnostics.

Reserved for future model-diagnostics analyses (e.g., Schoenfeld residual
tests for proportional-hazards assumption, martingale residuals, scaled
residuals).  Currently a no-op placeholder so that ``"diagnostics"`` is a
valid ``--supplementary`` target.
"""

from pathlib import Path
from typing import Dict


def run_supplementary_diagnostics(
    survival_df,
    primary_model_results: Dict,
    supplementary_dir: Path,
    analysis_mode: str = "scenario",
    **kwargs,
) -> Dict:
    """
    Run model diagnostics as a supplementary analysis.

    Currently returns an empty dict; reserved for future implementation.

    Args:
        survival_df: Survival analysis DataFrame.
        primary_model_results: Primary Cox model results dict.
        supplementary_dir: Path to the ``supplementary/`` directory.
        analysis_mode: ``"scenario"`` or ``"parameter"``.

    Returns:
        Empty dict (placeholder).
    """
    diagnostics_dir = supplementary_dir / "diagnostics" / "tables"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Write a placeholder file so the directory is not empty
    placeholder = diagnostics_dir / "diagnostics_placeholder.txt"
    with open(placeholder, "w") as f:
        f.write("Diagnostics analysis is reserved for future implementation.\n")

    return {}
