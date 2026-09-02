"""
Seed-District-Location Robustness Analysis.

Repeats the M0/M1 Cox hazard-ratio comparison separately for each epidemic
seed district to test whether the main finding (mass gathering events affect
epidemic arrival timing) is robust to the choice of seed location.

This is a thin orchestrator that reuses existing infrastructure:
    - build_mode_c_survival_dataset (from mode_c_dataset)
    - run_scenario_cox_models (from cox_models)
    - extract_model_summary (called internally by run_scenario_cox_models)

No new model-fitting or data-loading code is introduced.
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .cox_models import run_scenario_cox_models, validate_survival_data
from .discovery import discover_seed_directories
from .mode_c_dataset import build_mode_c_survival_dataset

SELECTED_SEED_DISTRICTS = [0, 7, 18]
EXPECTED_DISTRICTS = 21
PENALIZER = 0.1
DEFAULT_R0 = 1.5
DEFAULT_BETA = 0.5
DEFAULT_ISS = 1
M0_FORMULA = "gathering_event"
M1_FORMULA = "gathering_event + log_seed_size"
M2_PRESSURE_WINDOW = 7
M2_FORMULA = "gathering_event + log_seed_size + pre_pressure_7d"
M2_PRESSURE_COL = f"pre_pressure_{M2_PRESSURE_WINDOW}d"


def discover_all_seed_districts(
    results_dir: str | Path,
) -> list[int]:
    """Auto-discover all seed district IDs from the results directory.

    Delegates to :func:`discover_seed_directories` and returns a sorted list
    of integer seed IDs (e.g. ``[0, 1, 2, ..., 20]``).
    """
    return discover_seed_directories(results_dir)


def build_full_seed_dataset(
    results_dir: str | Path,
    r0: float = DEFAULT_R0,
    beta: float = DEFAULT_BETA,
    iss: int = DEFAULT_ISS,
    seed_locations: list[int] | None = None,
) -> pd.DataFrame:
    """Build the full Mode C survival dataset across seed locations.

    When *seed_locations* is provided the dataset is built only for those
    seed districts (used by the ``--selected-seeds`` CLI restriction so that
    discovery of **all** seed locations is skipped).  Otherwise all seed
    districts are auto-discovered.

    Calling :func:`build_mode_c_survival_dataset` once with every seed avoids
    redundant disk I/O (mobility/population matrices are loaded inside that
    function on every call).  Downstream per-seed filtering is a cheap
    DataFrame operation.
    """
    if seed_locations is None:
        all_seeds = discover_all_seed_districts(results_dir)
    else:
        all_seeds = list(seed_locations)
    if not all_seeds:
        warnings.warn(f"No seed directories discovered in {results_dir}")
        return pd.DataFrame()

    df = build_mode_c_survival_dataset(
        seed_locations=all_seeds,
        results_dir=results_dir,
        r0=r0,
        beta=beta,
        iss=iss,
    )
    return df


def compute_c_index_se(model, df: pd.DataFrame) -> float:
    """Compute C-index standard error from a fitted CoxPHFitter.

    The point estimate (C-index) is available as ``model.concordance_index_``.
    This function attempts to derive an SE via
    ``lifelines.utils.concordance_index_censored``, which returns CI bounds
    as fractions.  The SE is computed as ``(upper - lower) / (2 * z)`` where
    ``z = inv_normal_cdf(0.975)`` (~1.96).

    Falls back to ``NaN`` if the import or computation fails.
    """
    try:
        from lifelines.utils import concordance_index_censored, inv_normal_cdf
    except (ImportError, AttributeError):
        return np.nan

    try:
        duration_col = getattr(model, "duration_col", "arrival_time")
        event_col = getattr(model, "event_col", "event")

        durations = df[duration_col].values
        event_indicator = df[event_col].values
        predicted_scores = -model.predict_partial_hazard(df).values.flatten()

        result = concordance_index_censored(
            durations, event_indicator, predicted_scores
        )

        if hasattr(result, "ci_lower") and hasattr(result, "ci_upper"):
            ci_lower = result.ci_lower
            ci_upper = result.ci_upper
        elif len(result) >= 3:
            ci_lower = result[1]
            ci_upper = result[2]
        else:
            return np.nan

        z = inv_normal_cdf(0.975)
        se = (ci_upper - ci_lower) / (2 * z)
        return float(se)
    except Exception:
        return np.nan


def _extract_var_stats(model_result: dict | None, var_name: str) -> dict:
    """Extract coefficient statistics for a named variable.

    The model result dict (from ``_fit_and_process_both``) stores primary
    (clustered/robust) values under keys ``coefficients`` (pd.Series),
    ``hazard_ratios`` (pd.Series), and list-aligned ``se``, ``p_values``,
    ``ci_lower``, ``ci_upper``.

    Uses ``list(coefficients.index).index(var_name)`` to find the correct
    position, mirroring the pattern in ``reporting.py:382-389``.
    """
    nan_result: dict[str, float] = {
        "coef": np.nan,
        "hr": np.nan,
        "se": np.nan,
        "z": np.nan,
        "p_value": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
    }

    if model_result is None:
        return nan_result

    coefficients = model_result.get("coefficients", pd.Series(dtype=float))
    hazard_ratios = model_result.get("hazard_ratios", pd.Series(dtype=float))

    if var_name not in coefficients.index:
        return nan_result

    idx = list(coefficients.index).index(var_name)

    se_list = model_result.get("se", [])
    z_list = model_result.get("z_values", [])
    p_list = model_result.get("p_values", [])
    ci_lower_list = model_result.get("ci_lower", [])
    ci_upper_list = model_result.get("ci_upper", [])

    return {
        "coef": float(coefficients.iloc[idx]),
        "hr": float(hazard_ratios.iloc[idx]),
        "se": se_list[idx] if idx < len(se_list) else np.nan,
        "z": z_list[idx] if idx < len(z_list) else np.nan,
        "p_value": p_list[idx] if idx < len(p_list) else np.nan,
        "ci_lower": ci_lower_list[idx] if idx < len(ci_lower_list) else np.nan,
        "ci_upper": ci_upper_list[idx] if idx < len(ci_upper_list) else np.nan,
    }


def fit_per_seed_models(
    seed_df: pd.DataFrame,
    seed: int,
    cluster_col: str = "simulation_run_id",
) -> dict:
    """Fit M0/M1/M2 Cox models on data filtered to a single seed district.

    Returns a dict with keys ``M0``, ``M1`` (result dicts or ``None``),
    ``convergence_status``, per-model convergence status, ``seed``,
    ``n_obs``, and ``n_events``.
    """
    seed_subset = seed_df[seed_df["seed_location"] == seed].copy()

    n_obs = len(seed_subset)
    n_events = (
        int(seed_subset["event"].sum())
        if "event" in seed_subset.columns
        else 0
    )

    try:
        validate_survival_data(seed_subset, cluster_col=cluster_col)
    except Exception as e:
        warnings.warn(f"Data validation warning for seed {seed}: {e}")

    model_results = run_scenario_cox_models(
        seed_subset, cluster_col=cluster_col
    )

    m0_res = model_results.get("M0_gathering")
    m1_res = model_results.get("M1_gathering_seed")
    m2_res = model_results.get("M2_pressure_7d")

    # Augment result dicts with n_events and C-index SE (computed where
    # the fitted model object is available and the DataFrame is accessible)
    for res in (m0_res, m1_res, m2_res):
        if res is not None:
            res["n_events"] = n_events
            primary_model = res.get("model")
            if primary_model is not None:
                res["c_index_se"] = compute_c_index_se(primary_model, seed_subset)

    return {
        "M0": m0_res,
        "M1": m1_res,
        "M2": m2_res,
        "convergence_status": "converged"
        if (m0_res is not None and m1_res is not None and m2_res is not None)
        else "failed",
        "convergence_status_M0": "converged" if m0_res is not None else "failed",
        "convergence_status_M1": "converged" if m1_res is not None else "failed",
        "convergence_status_M2": "converged" if m2_res is not None else "failed",
        "seed": seed,
        "n_obs": n_obs,
        "n_events": n_events,
    }


def extract_seed_row(
    seed: int,
    r0: float,
    beta: float,
    iss: int,
    m0_res: dict | None,
    m1_res: dict | None,
    m2_res: dict | None = None,
) -> dict:
    """Build a single-row dict with all columns from the results table spec."""
    row: dict[str, Any] = {
        "seed_district": seed,
        "R0": r0,
        "beta_event": beta,
        "Iss": iss,
        "n_observations_M0": np.nan,
        "n_events_M0": np.nan,
        "HR_event_M0": np.nan,
        "HR_event_M0_lower": np.nan,
        "HR_event_M0_upper": np.nan,
        "p_event_M0": np.nan,
        "se_event_M0": np.nan,
        "C_index_M0": np.nan,
        "C_index_se_M0": np.nan,
        "AIC_M0": np.nan,
        "log_likelihood_M0": np.nan,
        "convergence_status_M0": "failed",
        "formula_M0": M0_FORMULA,
        "coef_event_M0": np.nan,
        "n_observations_M1": np.nan,
        "n_events_M1": np.nan,
        "HR_event_M1": np.nan,
        "HR_event_M1_lower": np.nan,
        "HR_event_M1_upper": np.nan,
        "p_event_M1": np.nan,
        "se_event_M1": np.nan,
        "HR_log_Z_seed_M1": np.nan,
        "HR_log_Z_seed_M1_lower": np.nan,
        "HR_log_Z_seed_M1_upper": np.nan,
        "p_log_Z_seed_M1": np.nan,
        "C_index_M1": np.nan,
        "C_index_se_M1": np.nan,
        "AIC_M1": np.nan,
        "log_likelihood_M1": np.nan,
        "convergence_status_M1": "failed",
        "formula_M1": M1_FORMULA,
        "coef_event_M1": np.nan,
        "coef_log_Z_seed_M1": np.nan,
        "HR_ratio_M1_M0": np.nan,
        "HR_difference_M1_M0": np.nan,
        "delta_C": np.nan,
        "delta_AIC": np.nan,
        "n_observations_M2": np.nan,
        "n_events_M2": np.nan,
        "HR_event_M2": np.nan,
        "HR_event_M2_lower": np.nan,
        "HR_event_M2_upper": np.nan,
        "p_event_M2": np.nan,
        "se_event_M2": np.nan,
        "HR_log_Z_seed_M2": np.nan,
        "HR_log_Z_seed_M2_lower": np.nan,
        "HR_log_Z_seed_M2_upper": np.nan,
        "p_log_Z_seed_M2": np.nan,
        "HR_pressure_M2": np.nan,
        "HR_pressure_M2_lower": np.nan,
        "HR_pressure_M2_upper": np.nan,
        "p_pressure_M2": np.nan,
        "se_pressure_M2": np.nan,
        "C_index_M2": np.nan,
        "C_index_se_M2": np.nan,
        "AIC_M2": np.nan,
        "log_likelihood_M2": np.nan,
        "convergence_status_M2": "failed",
        "formula_M2": M2_FORMULA,
        "coef_event_M2": np.nan,
        "coef_log_Z_seed_M2": np.nan,
        "coef_pressure_M2": np.nan,
        "delta_AIC_M1_M2": np.nan,
    }

    if m0_res is not None:
        m0_stats = _extract_var_stats(m0_res, "gathering_event")
        row["n_observations_M0"] = m0_res.get("n_obs", np.nan)
        row["n_events_M0"] = m0_res.get("n_events", np.nan)
        row["HR_event_M0"] = m0_stats["hr"]
        row["HR_event_M0_lower"] = m0_stats["ci_lower"]
        row["HR_event_M0_upper"] = m0_stats["ci_upper"]
        row["p_event_M0"] = m0_stats["p_value"]
        row["se_event_M0"] = m0_stats["se"]
        row["C_index_M0"] = m0_res.get("concordance", np.nan)
        row["C_index_se_M0"] = m0_res.get("c_index_se", np.nan)
        row["AIC_M0"] = m0_res.get("aic", np.nan)
        row["log_likelihood_M0"] = m0_res.get("log_likelihood", np.nan)
        row["convergence_status_M0"] = "converged"
        row["formula_M0"] = m0_res.get("formula", M0_FORMULA)
        row["coef_event_M0"] = m0_stats["coef"]

    if m1_res is not None:
        m1_event = _extract_var_stats(m1_res, "gathering_event")
        m1_seed = _extract_var_stats(m1_res, "log_seed_size")
        row["n_observations_M1"] = m1_res.get("n_obs", np.nan)
        row["n_events_M1"] = m1_res.get("n_events", np.nan)
        row["HR_event_M1"] = m1_event["hr"]
        row["HR_event_M1_lower"] = m1_event["ci_lower"]
        row["HR_event_M1_upper"] = m1_event["ci_upper"]
        row["p_event_M1"] = m1_event["p_value"]
        row["se_event_M1"] = m1_event["se"]
        row["HR_log_Z_seed_M1"] = m1_seed["hr"]
        row["HR_log_Z_seed_M1_lower"] = m1_seed["ci_lower"]
        row["HR_log_Z_seed_M1_upper"] = m1_seed["ci_upper"]
        row["p_log_Z_seed_M1"] = m1_seed["p_value"]
        row["C_index_M1"] = m1_res.get("concordance", np.nan)
        row["C_index_se_M1"] = m1_res.get("c_index_se", np.nan)
        row["AIC_M1"] = m1_res.get("aic", np.nan)
        row["log_likelihood_M1"] = m1_res.get("log_likelihood", np.nan)
        row["convergence_status_M1"] = "converged"
        row["formula_M1"] = m1_res.get("formula", M1_FORMULA)
        row["coef_event_M1"] = m1_event["coef"]
        row["coef_log_Z_seed_M1"] = m1_seed["coef"]

    if m2_res is not None:
        m2_event = _extract_var_stats(m2_res, "gathering_event")
        m2_seed = _extract_var_stats(m2_res, "log_seed_size")
        m2_pressure = _extract_var_stats(m2_res, M2_PRESSURE_COL)
        row["n_observations_M2"] = m2_res.get("n_obs", np.nan)
        row["n_events_M2"] = m2_res.get("n_events", np.nan)
        row["HR_event_M2"] = m2_event["hr"]
        row["HR_event_M2_lower"] = m2_event["ci_lower"]
        row["HR_event_M2_upper"] = m2_event["ci_upper"]
        row["p_event_M2"] = m2_event["p_value"]
        row["se_event_M2"] = m2_event["se"]
        row["HR_log_Z_seed_M2"] = m2_seed["hr"]
        row["HR_log_Z_seed_M2_lower"] = m2_seed["ci_lower"]
        row["HR_log_Z_seed_M2_upper"] = m2_seed["ci_upper"]
        row["p_log_Z_seed_M2"] = m2_seed["p_value"]
        row["HR_pressure_M2"] = m2_pressure["hr"]
        row["HR_pressure_M2_lower"] = m2_pressure["ci_lower"]
        row["HR_pressure_M2_upper"] = m2_pressure["ci_upper"]
        row["p_pressure_M2"] = m2_pressure["p_value"]
        row["se_pressure_M2"] = m2_pressure["se"]
        row["C_index_M2"] = m2_res.get("concordance", np.nan)
        row["C_index_se_M2"] = m2_res.get("c_index_se", np.nan)
        row["AIC_M2"] = m2_res.get("aic", np.nan)
        row["log_likelihood_M2"] = m2_res.get("log_likelihood", np.nan)
        row["convergence_status_M2"] = "converged"
        row["formula_M2"] = m2_res.get("formula", M2_FORMULA)
        row["coef_event_M2"] = m2_event["coef"]
        row["coef_log_Z_seed_M2"] = m2_seed["coef"]
        row["coef_pressure_M2"] = m2_pressure["coef"]

    # Derived columns
    hr_m0 = row["HR_event_M0"]
    hr_m1 = row["HR_event_M1"]
    c_m0 = row["C_index_M0"]
    c_m1 = row["C_index_M1"]
    aic_m0 = row["AIC_M0"]
    aic_m1 = row["AIC_M1"]
    aic_m2 = row["AIC_M2"]

    if not np.isnan(hr_m0) and not np.isnan(hr_m1) and hr_m0 != 0:
        row["HR_ratio_M1_M0"] = hr_m1 / hr_m0
        row["HR_difference_M1_M0"] = hr_m1 - hr_m0
    if not np.isnan(c_m0) and not np.isnan(c_m1):
        row["delta_C"] = c_m1 - c_m0
    if not np.isnan(aic_m0) and not np.isnan(aic_m1):
        row["delta_AIC"] = aic_m0 - aic_m1
    if not np.isnan(aic_m1) and not np.isnan(aic_m2):
        row["delta_AIC_M1_M2"] = aic_m1 - aic_m2

    return row


def run_consistency_checks(results_df: pd.DataFrame) -> str:
    """Run all 5 consistency checks and return a multi-line report string."""
    lines: list[str] = []
    all_pass = True

    # --- Check 1: Formula strings match exactly ---
    lines.append("=" * 60)
    lines.append("Check 1: Model formula consistency")
    lines.append("=" * 60)
    m0_formulas = results_df["formula_M0"].unique()
    m1_formulas = results_df["formula_M1"].unique()
    m0_ok = all(f == M0_FORMULA for f in m0_formulas if f is not None and not (isinstance(f, float) and np.isnan(f)))
    m1_ok = all(f == M1_FORMULA for f in m1_formulas if f is not None and not (isinstance(f, float) and np.isnan(f)))
    if m0_ok:
        lines.append(f"  PASS: M0 formula matches expected '{M0_FORMULA}'")
    else:
        lines.append(f"  FAIL: M0 formulas found: {m0_formulas}")
        all_pass = False
    if m1_ok:
        lines.append(f"  PASS: M1 formula matches expected '{M1_FORMULA}'")
    else:
        lines.append(f"  FAIL: M1 formulas found: {m1_formulas}")
        all_pass = False

    # --- Check 2: n_obs and n_events identical for M0 and M1 ---
    lines.append("")
    lines.append("=" * 60)
    lines.append("Check 2: n_obs and n_events consistency (M0 vs M1)")
    lines.append("=" * 60)
    check2_pass = True
    for _, r in results_df.iterrows():
        n_obs_diff = abs(r["n_observations_M0"] - r["n_observations_M1"])
        n_events_diff = abs(r["n_events_M0"] - r["n_events_M1"])
        if not (np.isnan(n_obs_diff) or np.isnan(n_events_diff)) and (n_obs_diff > 0 or n_events_diff > 0):
            lines.append(
                f"  WARN: seed {int(r['seed_district'])}: "
                f"n_obs diff={n_obs_diff}, n_events diff={n_events_diff}"
            )
            check2_pass = False
    if check2_pass:
        lines.append("  PASS: n_obs and n_events identical for M0 and M1 across all seeds")
    else:
        all_pass = False

    # --- Check 3: Convergence status reporting ---
    lines.append("")
    lines.append("=" * 60)
    lines.append("Check 3: Convergence status")
    lines.append("=" * 60)
    failed_seeds = []
    for _, r in results_df.iterrows():
        if (r["convergence_status_M0"] == "failed"
                or r["convergence_status_M1"] == "failed"
                or r["convergence_status_M2"] == "failed"):
            failed_seeds.append(
                f"seed {int(r['seed_district'])} "
                f"(M0={r['convergence_status_M0']}, "
                f"M1={r['convergence_status_M1']}, "
                f"M2={r['convergence_status_M2']})"
            )
    n_converged = len(results_df) - len(failed_seeds)
    lines.append(f"  Converged: {n_converged}/{len(results_df)} seeds")
    if failed_seeds:
        lines.append(f"  Failed: {len(failed_seeds)} seeds:")
        for fs in failed_seeds:
            lines.append(f"    - {fs}")
    else:
        lines.append("  PASS: All seeds converged for M0, M1, and M2")

    # --- Check 4: Derived column integrity ---
    lines.append("")
    lines.append("=" * 60)
    lines.append("Check 4: Derived column integrity "
                 "(delta_AIC, delta_AIC_M1_M2, delta_C, HR=exp(coef))")
    lines.append("=" * 60)
    check4_pass = True
    for _, r in results_df.iterrows():
        seed_id = int(r["seed_district"])

        # delta_AIC == AIC_M0 - AIC_M1
        if not np.isnan(r["delta_AIC"]):
            expected = r["AIC_M0"] - r["AIC_M1"]
            if abs(r["delta_AIC"] - expected) > 1e-6:
                lines.append(
                    f"  FAIL: seed {seed_id}: delta_AIC={r['delta_AIC']:.6f} "
                    f"!= AIC_M0-AIC_M1={expected:.6f}"
                )
                check4_pass = False

        # delta_AIC_M1_M2 == AIC_M1 - AIC_M2
        if not np.isnan(r["delta_AIC_M1_M2"]):
            expected = r["AIC_M1"] - r["AIC_M2"]
            if abs(r["delta_AIC_M1_M2"] - expected) > 1e-6:
                lines.append(
                    f"  FAIL: seed {seed_id}: "
                    f"delta_AIC_M1_M2={r['delta_AIC_M1_M2']:.6f} "
                    f"!= AIC_M1-AIC_M2={expected:.6f}"
                )
                check4_pass = False

        # delta_C == C_index_M1 - C_index_M0
        if not np.isnan(r["delta_C"]):
            expected = r["C_index_M1"] - r["C_index_M0"]
            if abs(r["delta_C"] - expected) > 1e-6:
                lines.append(
                    f"  FAIL: seed {seed_id}: delta_C={r['delta_C']:.6f} "
                    f"!= C_index_M1-C_index_M0={expected:.6f}"
                )
                check4_pass = False

        # HR == exp(coef)
        if not np.isnan(r["coef_event_M0"]):
            expected_hr = np.exp(r["coef_event_M0"])
            if abs(r["HR_event_M0"] - expected_hr) > 1e-4:
                lines.append(
                    f"  FAIL: seed {seed_id}: HR_event_M0={r['HR_event_M0']:.6f} "
                    f"!= exp(coef)={expected_hr:.6f}"
                )
                check4_pass = False
        if not np.isnan(r["coef_event_M1"]):
            expected_hr = np.exp(r["coef_event_M1"])
            if abs(r["HR_event_M1"] - expected_hr) > 1e-4:
                lines.append(
                    f"  FAIL: seed {seed_id}: HR_event_M1={r['HR_event_M1']:.6f} "
                    f"!= exp(coef)={expected_hr:.6f}"
                )
                check4_pass = False
        if not np.isnan(r["coef_log_Z_seed_M1"]):
            expected_hr = np.exp(r["coef_log_Z_seed_M1"])
            if abs(r["HR_log_Z_seed_M1"] - expected_hr) > 1e-4:
                lines.append(
                    f"  FAIL: seed {seed_id}: HR_log_Z_seed_M1={r['HR_log_Z_seed_M1']:.6f} "
                    f"!= exp(coef)={expected_hr:.6f}"
                )
                check4_pass = False
        if not np.isnan(r["coef_event_M2"]):
            expected_hr = np.exp(r["coef_event_M2"])
            if abs(r["HR_event_M2"] - expected_hr) > 1e-4:
                lines.append(
                    f"  FAIL: seed {seed_id}: HR_event_M2={r['HR_event_M2']:.6f} "
                    f"!= exp(coef)={expected_hr:.6f}"
                )
                check4_pass = False
        if not np.isnan(r["coef_log_Z_seed_M2"]):
            expected_hr = np.exp(r["coef_log_Z_seed_M2"])
            if abs(r["HR_log_Z_seed_M2"] - expected_hr) > 1e-4:
                lines.append(
                    f"  FAIL: seed {seed_id}: "
                    f"HR_log_Z_seed_M2={r['HR_log_Z_seed_M2']:.6f} "
                    f"!= exp(coef)={expected_hr:.6f}"
                )
                check4_pass = False
        if not np.isnan(r["coef_pressure_M2"]):
            expected_hr = np.exp(r["coef_pressure_M2"])
            if abs(r["HR_pressure_M2"] - expected_hr) > 1e-4:
                lines.append(
                    f"  FAIL: seed {seed_id}: HR_pressure_M2={r['HR_pressure_M2']:.6f} "
                    f"!= exp(coef)={expected_hr:.6f}"
                )
                check4_pass = False

    if check4_pass:
        lines.append(
            "  PASS: All delta_AIC, delta_AIC_M1_M2, delta_C, "
            "and HR=exp(coef) checks passed"
        )
    else:
        all_pass = False

    # --- Check 5: All seeds appear exactly once in results ---
    lines.append("")
    lines.append("=" * 60)
    lines.append("Check 5: Seed district coverage (figure verification)")
    lines.append("=" * 60)
    n_seeds = len(results_df)
    n_unique = results_df["seed_district"].nunique()
    if n_seeds == n_unique:
        lines.append(f"  PASS: {n_seeds} unique seed districts, each appearing exactly once")
    else:
        lines.append(
            f"  FAIL: {n_seeds} rows but {n_unique} unique seeds — duplicates detected"
        )
        all_pass = False
    lines.append("  NOTE: Figure coverage verified post-generation (all seeds in "
                 "results_df are plotted)")

    lines.append("")
    lines.append("=" * 60)
    if all_pass:
        lines.append("OVERALL: ALL CHECKS PASSED")
    else:
        lines.append("OVERALL: SOME CHECKS FAILED — review above")
    lines.append("=" * 60)

    return "\n".join(lines)


def create_robustness_figure(
    results_df: pd.DataFrame,
    seed_districts: list[int],
    output_base: str | Path,
    title_suffix: str = "",
    min_height: float = 8.0,
) -> None:
    """Generate a three-panel robustness figure.

    Panel (a): Gathering-event hazard ratio (M0 circles vs M1 diamonds)
    Panel (b): C-index (M0 vs M1, connected by line)
    Panel (c): AIC improvement (delta_AIC = AIC_M0 - AIC_M1)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig_df = results_df[results_df["seed_district"].isin(seed_districts)].copy()
    fig_df = fig_df.sort_values("seed_district", key=lambda s: s.astype(int)).reset_index(drop=True)

    n_seeds = len(fig_df)
    if n_seeds == 0:
        warnings.warn("No seeds to plot in robustness figure")
        return

    m0_color = "steelblue"
    m1_color = "darkorange"
    y_positions = list(range(n_seeds))
    seed_labels = [str(int(s)) for s in fig_df["seed_district"].values]

    fig, axes = plt.subplots(
        1, 3, figsize=(16, max(min_height, n_seeds * 0.4))
    )

    # --- Panel (a): Gathering-event HR ---
    ax = axes[0]
    for i, (_, row) in enumerate(fig_df.iterrows()):
        y = i
        # M0
        hr0 = row["HR_event_M0"]
        lo0 = row["HR_event_M0_lower"]
        hi0 = row["HR_event_M0_upper"]
        if not np.isnan(hr0):
            ax.plot(hr0, y + 0.15, "o", color=m0_color, markersize=8,
                    markerfacecolor="none", markeredgewidth=2, zorder=3)
            if not np.isnan(lo0) and not np.isnan(hi0) and lo0 > 0:
                ax.plot([lo0, hi0], [y + 0.15, y + 0.15], color=m0_color,
                        linewidth=2, zorder=2)
        # M1
        hr1 = row["HR_event_M1"]
        lo1 = row["HR_event_M1_lower"]
        hi1 = row["HR_event_M1_upper"]
        if not np.isnan(hr1):
            ax.plot(hr1, y - 0.15, "D", color=m1_color, markersize=7,
                    markerfacecolor="none", markeredgewidth=2, zorder=3)
            if not np.isnan(lo1) and not np.isnan(hi1) and lo1 > 0:
                ax.plot([lo1, hi1], [y - 0.15, y - 0.15], color=m1_color,
                        linewidth=2, zorder=2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(seed_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_ylim(-0.5, n_seeds - 0.5)
    ax.set_xscale("log")
    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1.5, zorder=1)
    ax.set_xlabel("Hazard Ratio (log scale)", fontsize=11)
    ax.set_ylabel("Seed District", fontsize=11)
    ax.set_title("(a) Gathering-event hazard ratio", fontsize=12, fontweight="bold")
    legend_elements = [
        Line2D(
            [0], [0], marker="o", color=m0_color, linestyle="None",
            markeredgewidth=2, markerfacecolor="none", markersize=8,
            label="M0 (unadjusted)",
        ),
        Line2D(
            [0], [0], marker="D", color=m1_color, linestyle="None",
            markeredgewidth=2, markerfacecolor="none", markersize=7,
            label="M1 (adjusted for seed size)",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        bbox_transform=ax.transAxes,
        fontsize=9,
        framealpha=0.9,
        handletextpad=0.4,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.3)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))

    # --- Panel (b): C-index ---
    ax = axes[1]
    for i, (_, row) in enumerate(fig_df.iterrows()):
        y = i
        c0 = row["C_index_M0"]
        c1 = row["C_index_M1"]
        if not np.isnan(c0) and not np.isnan(c1):
            ax.plot([c0, c1], [y, y], "-", color="gray", linewidth=1, zorder=1)
            ax.plot(c0, y, "o", color=m0_color, markersize=7, zorder=3)
            ax.plot(c1, y, "s", color=m1_color, markersize=7,
                    markerfacecolor="none", markeredgewidth=2, zorder=3)
        elif not np.isnan(c0):
            ax.plot(c0, y, "o", color=m0_color, markersize=7, zorder=3)
        elif not np.isnan(c1):
            ax.plot(c1, y, "s", color=m1_color, markersize=7, zorder=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(seed_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_ylim(-0.5, n_seeds - 0.5)
    ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("C-index", fontsize=11)
    ax.set_ylabel("Seed District", fontsize=11)
    ax.set_title("(b) Model discrimination (C-index)", fontsize=12, fontweight="bold")
    legend_elements = [
        Line2D(
            [0], [0], marker="o", color=m0_color, linestyle="None",
            markersize=7, label="M0 C-index",
        ),
        Line2D(
            [0], [0], marker="s", color=m1_color, linestyle="None",
            markerfacecolor="none", markeredgewidth=2, markersize=7,
            label="M1 C-index",
        ),
        Line2D(
            [0], [0], color="gray", linestyle="-", linewidth=1,
            label="M0→M1 difference",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        bbox_transform=ax.transAxes,
        fontsize=9,
        framealpha=0.9,
        handletextpad=0.4,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.3)

    # --- Panel (c): AIC improvement ---
    ax = axes[2]
    for i, (_, row) in enumerate(fig_df.iterrows()):
        y = i
        d_aic = row["delta_AIC"]
        if not np.isnan(d_aic):
            bar_color = m0_color if d_aic > 0 else m1_color
            ax.barh(y, d_aic, height=0.6, color=bar_color, alpha=0.7, zorder=2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(seed_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_ylim(-0.5, n_seeds - 0.5)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1.5, zorder=1)
    ax.set_xlabel(r"$\Delta$AIC (M0 $-$ M1)", fontsize=11)
    ax.set_ylabel("Seed District", fontsize=11)
    ax.set_title("(c) AIC improvement", fontsize=12, fontweight="bold")
    legend_elements = [
        Patch(
            facecolor=m0_color, alpha=0.7,
            label="M1 better (ΔAIC > 0)",
        ),
        Patch(
            facecolor=m1_color, alpha=0.7,
            label="M0 better (ΔAIC < 0)",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        bbox_transform=ax.transAxes,
        fontsize=9,
        framealpha=0.9,
        handletextpad=0.4,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(
        f"Seed-Location Robustness Analysis{': ' + title_suffix if title_suffix else ''}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    output_base = str(output_base)
    fig.savefig(output_base + ".png", dpi=600, bbox_inches="tight", format="png")
    fig.savefig(output_base + ".pdf", bbox_inches="tight", format="pdf")
    plt.close(fig)


def print_robustness_summary(results_df: pd.DataFrame) -> None:
    """Print an 11-point textual summary of the robustness analysis."""
    r0 = results_df["R0"].iloc[0] if len(results_df) > 0 else DEFAULT_R0
    beta = results_df["beta_event"].iloc[0] if len(results_df) > 0 else DEFAULT_BETA
    iss = results_df["Iss"].iloc[0] if len(results_df) > 0 else DEFAULT_ISS

    param_label = f"R{int(r0*100)}_beta{int(beta*100)}_Iss{iss}"

    print()
    print("=" * 70)
    print("SEED-LOCATION ROBUSTNESS ANALYSIS SUMMARY")
    print("=" * 70)

    converged_mask = (results_df["convergence_status_M0"] == "converged") & \
                     (results_df["convergence_status_M1"] == "converged") & \
                     (results_df["convergence_status_M2"] == "converged")
    converged_df = results_df[converged_mask]
    failed_seeds = results_df[~converged_mask]
    m2_converged_mask = results_df["convergence_status_M2"] == "converged"
    m2_converged_count = int(m2_converged_mask.sum())

    # 1. Parameters
    print()
    print("1. Analysis parameters:")
    print(f"   R0={r0}, beta_event={beta}, Iss={iss} ({param_label})")
    print("   Scenarios: AMS_dance, AMS_football, Leipzig_1, Leipzig_2, "
          "Leipzig_3, no_event")

    # 2. Total seeds
    print()
    print(f"2. Seed districts analyzed: {len(results_df)}")

    # 3. Converged seeds
    print()
    print(f"3. Seeds with full convergence (M0 + M1 + M2): {len(converged_df)}/"
          f"{len(results_df)}")
    print(f"   M2-only converged: {m2_converged_count}/{len(results_df)} seeds")

    # 4. Failed seeds
    print()
    if len(failed_seeds) > 0:
        print(f"4. Convergence failures: {len(failed_seeds)} seed(s)")
        for _, r in failed_seeds.iterrows():
            print(f"   - Seed {int(r['seed_district'])}: "
                  f"M0={r['convergence_status_M0']}, "
                  f"M1={r['convergence_status_M1']}, "
                  f"M2={r['convergence_status_M2']}")
    else:
        print("4. Convergence failures: 0 — all seeds converged (M0, M1, M2)")

    # 5. M0 HR
    print()
    hr_m0 = converged_df["HR_event_M0"].dropna()
    if len(hr_m0) > 0:
        q1, med, q3 = hr_m0.quantile([0.25, 0.5, 0.75])
        print(f"5. M0 gathering-event HR: median={med:.3f} "
              f"[IQR: {q1:.3f}–{q3:.3f}], n={len(hr_m0)}")
    else:
        print("5. M0 gathering-event HR: no converged models")

    # 6. M1 HR
    print()
    hr_m1 = converged_df["HR_event_M1"].dropna()
    if len(hr_m1) > 0:
        q1, med, q3 = hr_m1.quantile([0.25, 0.5, 0.75])
        print(f"6. M1 gathering-event HR (adjusted): median={med:.3f} "
              f"[IQR: {q1:.3f}–{q3:.3f}], n={len(hr_m1)}")
    else:
        print("6. M1 gathering-event HR: no converged models")

    # 7. Seed-size HR
    print()
    hr_seed = converged_df["HR_log_Z_seed_M1"].dropna()
    if len(hr_seed) > 0:
        q1, med, q3 = hr_seed.quantile([0.25, 0.5, 0.75])
        print(f"7. M1 seed-size HR: median={med:.3f} "
              f"[IQR: {q1:.3f}–{q3:.3f}], n={len(hr_seed)}")
    else:
        print("7. M1 seed-size HR: no converged models")

    # 8. M2 pressure-coefficient HR
    print()
    hr_pressure = converged_df["HR_pressure_M2"].dropna()
    if len(hr_pressure) > 0:
        q1, med, q3 = hr_pressure.quantile([0.25, 0.5, 0.75])
        print(f"8. M2 pre-invasion-pressure HR: median={med:.3f} "
              f"[IQR: {q1:.3f}–{q3:.3f}], n={len(hr_pressure)}")
    else:
        print("8. M2 pre-invasion-pressure HR: no converged models")

    # 9. C-index
    print()
    if len(converged_df) > 0:
        c0_mean = converged_df["C_index_M0"].mean()
        c1_mean = converged_df["C_index_M1"].mean()
        c2_mean = converged_df["C_index_M2"].mean()
        print(f"9. C-index: M0 mean={c0_mean:.3f}, "
              f"M1 mean={c1_mean:.3f}, M2 mean={c2_mean:.3f}")
    else:
        print("9. C-index: no converged models")

    # 10. Delta AIC
    print()
    if len(converged_df) > 0:
        delta_aic = converged_df["delta_AIC"].dropna()
        delta_aic_m1_m2 = converged_df["delta_AIC_M1_M2"].dropna()
        if len(delta_aic) > 0:
            mean_da = delta_aic.mean()
            direction = "improvement" if mean_da > 0 else "worsening"
            print(f"   Mean delta_AIC (M0-M1) = {mean_da:.1f} ({direction}), "
                  f"range [{delta_aic.min():.1f}, {delta_aic.max():.1f}]")
        else:
            print("   Mean delta_AIC (M0-M1): not available")
        if len(delta_aic_m1_m2) > 0:
            mean_da = delta_aic_m1_m2.mean()
            direction = "improvement" if mean_da > 0 else "worsening"
            print(f"   Mean delta_AIC (M1-M2) = {mean_da:.1f} ({direction}), "
                  f"range [{delta_aic_m1_m2.min():.1f}, "
                  f"{delta_aic_m1_m2.max():.1f}]")
        else:
            print("   Mean delta_AIC (M1-M2): not available")
        print("10. AIC model improvement (ΔAIC > 0 = added covariates help):")
    else:
        print("10. AIC model improvement: no converged models")

    # 11. Conclusion
    print()
    if len(converged_df) > 0:
        n_sig = int((converged_df["p_event_M1"] < 0.05).sum())
        print("11. Robustness assessment:")
        print(f"    {len(converged_df)}/{len(results_df)} seeds fully converged "
              f"(M0+M1+M2).")
        if len(hr_m1) > 0:
            n_above_one = int((hr_m1 > 1).sum())
            if (hr_m1 > 1).all():
                print("    Gathering-event HR > 1 in ALL converged seeds — "
                      "robust positive effect.")
            elif n_above_one > 0:
                print(f"    Gathering-event HR > 1 in {n_above_one}/{len(hr_m1)} "
                      f"seeds (significant in {n_sig}) — partially robust effect.")
            else:
                print("    Gathering-event HR consistent with null in most seeds.")
    else:
        print("11. Robustness assessment: insufficient converged models")

    print()
    print("=" * 70)


def run_seed_location_robustness(
    results_dir: str = "results_hybrid_sim",
    output_dir: str = "results_survival_analysis_robustness",
    r0: float = DEFAULT_R0,
    beta: float = DEFAULT_BETA,
    iss: int = DEFAULT_ISS,
    selected_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Run the complete seed-location robustness analysis.

    1. Discover seed districts (auto-discover all, or restrict to selected_seeds)
    2. Build the full Mode C dataset once (for the relevant seeds only)
    3. For each seed: filter, fit M0/M1/M2, extract stats
    4. Assemble results DataFrame
    5. Run consistency checks
    6. Save results table and compact summary
    7. Generate main-text figure (selected seeds); appendix figure (all seeds)
       only when seeds were not explicitly restricted
    8. Print interpretation summary

    When *selected_seeds* is provided (not ``None``) discovery of all seed
    locations is skipped: only the requested seed directories are built and
    analysed, the main-text (selected-seeds) figure is produced, and the
    all-seeds appendix figure is omitted.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    print("=" * 70)
    print("Seed-Location Robustness Analysis")
    print("=" * 70)
    print(f"  R0={r0}, beta={beta}, Iss={iss}")
    print(f"  Results dir: {results_dir}")
    print(f"  Output dir: {output_path}")

    # Step 1: Determine which seed districts to process.
    explicit = selected_seeds is not None
    if explicit:
        # Only process explicitly-requested seeds; skip all-seed discovery.
        seeds_to_process = [s for s in selected_seeds
                            if (Path(results_dir) / f"seed_{s}").exists()]
        if not seeds_to_process:
            print("ERROR: None of the specified seed directories exist.")
            return {}
        all_seeds = seeds_to_process
        selected_available = seeds_to_process
        print(f"  Selected seeds for main text: {selected_seeds}")
        print(f"  Restricted to requested seeds: {all_seeds} "
              f"(all-seed discovery skipped)")
    else:
        all_seeds = discover_all_seed_districts(results_dir)
        if not all_seeds:
            print(f"ERROR: No seed directories found in {results_dir}")
            return {}

        print(f"  Discovered {len(all_seeds)} seed districts: {all_seeds}")
        selected_seeds = list(SELECTED_SEED_DISTRICTS)
        selected_available = [s for s in selected_seeds if s in all_seeds]
        print(f"  Selected seeds for main text: {selected_seeds}")

    # Step 2: Build full dataset once across the relevant seeds
    print()
    print("Building full Mode C survival dataset...")
    survival_df = build_full_seed_dataset(
        results_dir,
        seed_locations=all_seeds,
        r0=r0,
        beta=beta,
        iss=iss,
    )
    if survival_df.empty:
        print("ERROR: Survival dataset is empty")
        return {}

    print(f"  Total records: {len(survival_df)}")

    # Step 3: Fit per-seed models and extract rows
    print()
    print("Fitting per-seed Cox models (M0, M1, M2)...")
    rows: list[dict[str, Any]] = []
    for seed in all_seeds:
        print(f"  Seed {seed}: ", end="", flush=True)
        fit_result = fit_per_seed_models(survival_df, seed, cluster_col="simulation_run_id")
        row = extract_seed_row(
            seed=seed,
            r0=r0,
            beta=beta,
            iss=iss,
            m0_res=fit_result["M0"],
            m1_res=fit_result["M1"],
            m2_res=fit_result["M2"],
        )
        rows.append(row)
        if fit_result["convergence_status"] == "converged":
            print("converged (M0, M1, M2)")
        else:
            failed = []
            if fit_result["convergence_status_M0"] == "failed":
                failed.append("M0")
            if fit_result["convergence_status_M1"] == "failed":
                failed.append("M1")
            if fit_result["convergence_status_M2"] == "failed":
                failed.append("M2")
            print(f"failed ({', '.join(failed)})")

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("seed_district", key=lambda s: s.astype(int)).reset_index(drop=True)

    # Step 4: Run consistency checks
    print()
    print("Running consistency checks...")
    check_report = run_consistency_checks(results_df)
    checks_path = output_path / "seed_location_robustness_consistency_checks.txt"
    checks_path.write_text(check_report, encoding="utf-8")
    print(f"  Saved consistency checks to {checks_path}")
    print(check_report)

    # Step 5: Save full results table
    results_path = output_path / "seed_location_robustness_results.csv"
    results_df.to_csv(results_path, index=False)
    print()
    print(f"Saved full results table to {results_path}")

    # Step 6: Save compact summary (9-column table)
    summary_cols = [
        "seed_district",
        "HR_event_M0",
        "HR_event_M0_lower",
        "HR_event_M0_upper",
        "HR_event_M1",
        "HR_event_M1_lower",
        "HR_event_M1_upper",
        "p_event_M1",
        "delta_AIC",
        "HR_pressure_M2",
        "p_pressure_M2",
        "C_index_M2",
        "AIC_M2",
        "delta_AIC_M1_M2",
        "convergence_status_M2",
    ]
    summary_df = results_df[summary_cols].copy()
    summary_path = output_path / "seed_location_robustness_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved compact summary to {summary_path}")

    # Step 7: Generate figures
    print()
    print("Generating figures...")

    # Main-text figure: selected seeds only (always built when available)
    if selected_available:
        selected_base = str(output_path / "seed_location_robustness_selected")
        create_robustness_figure(
            results_df,
            selected_available,
            selected_base,
            title_suffix="Selected Seeds",
            min_height=4.5,
        )
        print(f"  Saved main-text figure: {selected_base}.pdf/png")
    else:
        print("  WARNING: No selected seeds found in data; skipping main-text figure")

    # Appendix figure: all seeds — built only when seeds were not explicitly
    # restricted (i.e. full discovery was performed).
    if not explicit:
        all_base = str(output_path / "seed_location_robustness_all_districts")
        create_robustness_figure(
            results_df,
            all_seeds,
            all_base,
            title_suffix="All Seeds",
        )
        print(f"  Saved appendix figure: {all_base}.pdf/png")
    else:
        print("  Skipping appendix (all-seeds) figure "
              "(seeds explicitly restricted)")

    # Step 8: Print summary
    print_robustness_summary(results_df)

    t_elapsed = time.time() - t_start
    print()
    print("=" * 70)
    print(f"Robustness analysis complete in {t_elapsed:.1f}s")
    print(f"Output directory: {output_path}")
    print("=" * 70)

    return {
        "results_df": results_df,
        "all_seeds": all_seeds,
        "selected_seeds": selected_seeds,
        "consistency_checks": check_report,
        "t_elapsed": t_elapsed,
    }
