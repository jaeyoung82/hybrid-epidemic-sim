"""
Supplementary analysis framework for survival analysis.

Each supplementary analysis is a self-contained module registered in
``SUPPLEMENTARY_ANALYSES``.  Analyses are executed only when explicitly
requested via the ``--supplementary`` CLI flag.

Available analyses:
    frailty     - Shared frailty (mixed-effects Cox) model
    bootstrap   - Non-parametric bootstrap for HR uncertainty
    diagnostics - Model diagnostics (reserved / future)
    all         - Run every registered supplementary analysis
"""

from typing import Callable, Dict

from .frailty import (
    run_supplementary_frailty,
    save_frailty_results,
    generate_frailty_figures,
    interpret_frailty_models,
)
from .bootstrap import run_supplementary_bootstrap
from .diagnostics import run_supplementary_diagnostics

#: Registry mapping supplementary-analysis names to their entry-point functions
#: and human-readable labels.
SUPPLEMENTARY_ANALYSES: Dict[str, Dict] = {
    "frailty": {
        "name": "Shared frailty model",
        "description": (
            "Shared frailty (mixed-effects Cox) model with Gaussian random "
            "intercepts for simulation runs"
        ),
        "run": run_supplementary_frailty,
    },
    "bootstrap": {
        "name": "Bootstrap analysis",
        "description": "Non-parametric bootstrap for hazard ratio uncertainty",
        "run": run_supplementary_bootstrap,
    },
    "diagnostics": {
        "name": "Diagnostics",
        "description": "Model diagnostics and residual analysis (future)",
        "run": run_supplementary_diagnostics,
    },
}

#: Convenience list of all supplementary analysis names (excluding ``"all"``).
SUPPORTED_SUPPLEMENTARY = list(SUPPLEMENTARY_ANALYSES.keys())


def resolve_supplementary_requests(
    requested: list,
) -> list:
    """
    Resolve a list of requested supplementary analyses.

    ``"all"`` expands to every registered analysis name.

    Args:
        requested: List of names (e.g. ``["frailty"]`` or ``["all"]``).

    Returns:
        Sorted list of concrete analysis names to run.
    """
    if not requested:
        return []

    resolved = []
    if "all" in requested:
        resolved = list(SUPPORTED_SUPPLEMENTARY)
    else:
        for name in requested:
            if name not in SUPPLEMENTARY_ANALYSES:
                raise ValueError(
                    f"Unknown supplementary analysis: '{name}'. "
                    f"Available: {', '.join(SUPPORTED_SUPPLEMENTARY)} or 'all'."
                )
            resolved.append(name)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for name in resolved:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


__all__ = [
    "SUPPLEMENTARY_ANALYSES",
    "SUPPORTED_SUPPLEMENTARY",
    "resolve_supplementary_requests",
    "run_supplementary_frailty",
    "save_frailty_results",
    "generate_frailty_figures",
    "interpret_frailty_models",
    "run_supplementary_bootstrap",
    "run_supplementary_diagnostics",
]
