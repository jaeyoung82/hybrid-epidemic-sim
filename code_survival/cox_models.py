"""
Cox proportional hazards regression models with cluster-robust standard errors.

Supports both scenario comparison (Mode A) and parameter sensitivity (Mode B) analyses.

Cluster-robust (Huber-White sandwich) standard errors are used because districts
within the same stochastic simulation run are NOT statistically independent.
They share the same epidemic realization, commuter mobility, and transmission
history. Cluster-robust sandwich standard errors account for this within-run
dependence while leaving coefficient estimates (hazard ratios) unchanged.

Both naive (model-based) and cluster-robust standard errors are computed for
every model so that a direct comparison can be produced.
"""

import copy
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import inv_normal_cdf
except ImportError:
    raise ImportError(
        "lifelines package required for Cox regression. "
        "Install with: pip install lifelines"
    )


# ============================================================================
# Data Validation
# ============================================================================

def validate_survival_data(df: pd.DataFrame,
                           cluster_col: str = 'simulation_run_id',
                           expected_districts: int = 21) -> bool:
    """
    Validate survival analysis data for clustered regression.

    Verifies:
      - ``arrival_time`` has no missing values
      - ``event`` indicator is binary (0/1)
      - every ``simulation_run_id`` contains exactly *expected_districts* districts
      - cluster IDs are unique (no duplicate district-run combinations)

    Raises informative ``warnings`` when inconsistencies are detected.

    Args:
        df: Survival analysis DataFrame.
        cluster_col: Column used for clustering (default ``'simulation_run_id'``).
        expected_districts: Expected number of districts per simulation run.

    Returns:
        ``True`` when all checks pass without warnings, ``False`` otherwise.
    """
    warnings_found = False

    # --- arrival_time must be present and non-null ---
    if 'arrival_time' not in df.columns:
        warnings.warn("arrival_time column not found in data.")
        warnings_found = True
    else:
        n_missing = int(df['arrival_time'].isna().sum())
        if n_missing > 0:
            warnings.warn(
                f"arrival_time has {n_missing} missing value(s) (expected 0). "
                f"Rows with missing arrival times may affect model fitting."
            )
            warnings_found = True

    # --- event indicator must be binary ---
    if 'event' not in df.columns:
        warnings.warn("event column not found in data.")
        warnings_found = True
    else:
        unique_events = set(df['event'].dropna().unique())
        if not unique_events.issubset({0, 1}):
            warnings.warn(
                f"Event indicator is not binary: found values {sorted(unique_events)}. "
                f"Expected only 0 and 1."
            )
            warnings_found = True

    # --- cluster column presence ---
    if cluster_col not in df.columns:
        warnings.warn(
            f"Cluster column '{cluster_col}' not found in data. "
            f"Cluster-robust standard errors cannot be computed."
        )
        return False

    # --- district_id for cluster-size checks ---
    district_col = 'district_id' if 'district_id' in df.columns else None
    if district_col is not None:
        cluster_sizes = df.groupby(cluster_col)[district_col].nunique()
        bad_clusters = cluster_sizes[cluster_sizes != expected_districts]
        if len(bad_clusters) > 0:
            sample_msg = bad_clusters.head(5).to_dict()
            mode_size = cluster_sizes.mode()
            mode_str = f"{mode_size.iloc[0]}" if len(mode_size) else "unknown"
            warnings.warn(
                f"{len(bad_clusters)} cluster(s) do not have exactly {expected_districts} "
                f"districts. Examples: {sample_msg}. Most clusters have {mode_str} districts."
            )
            warnings_found = True

    # --- duplicate (cluster, district) rows ---
    if district_col is not None:
        dup_mask = df.duplicated(subset=[cluster_col, district_col], keep=False)
        n_dup = int(dup_mask.sum())
        if n_dup > 0:
            warnings.warn(
                f"{n_dup} row(s) have duplicate (cluster, district_id) pairs. "
                f"Cluster IDs may not be unique across runs."
            )
            warnings_found = True

    n_obs = len(df)
    n_clusters = df[cluster_col].nunique()
    if district_col is not None:
        min_size = cluster_sizes.min()
        max_size = cluster_sizes.max()
        print(
            f"  Data validation: {n_obs} observations, {n_clusters} clusters, "
            f"clusters contain {min_size}-{max_size} districts each "
            f"(expected {expected_districts})."
        )
    else:
        print(
            f"  Data validation: {n_obs} observations, {n_clusters} clusters, "
            f"(district_id column not found for size check)."
        )

    return not warnings_found


# ============================================================================
# Model Fitting Helpers
# ============================================================================

def fit_naive_cox_model(df: pd.DataFrame,
                        formula: str,
                        duration_col: str = 'arrival_time',
                        event_col: str = 'event',
                        penalizer: float = 0.1) -> CoxPHFitter:
    """
    Fit a Cox proportional hazards model with standard model-based SEs.

    This version does **not** account for within-cluster dependence and is
    provided for comparison with :func:`fit_clustered_cox_model`.

    Args:
        df: DataFrame with survival data.
        formula: R-style formula string.
        duration_col: Duration column name.
        event_col: Event indicator column name.
        penalizer: L2 penalizer strength.

    Returns:
        Fitted ``CoxPHFitter`` (naive model-based SEs).
    """
    model = CoxPHFitter(penalizer=penalizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df, duration_col=duration_col, event_col=event_col, formula=formula)
    return model


def _reconstruct_normalized_X(
    model: CoxPHFitter,
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the normalized, sorted design matrix that lifelines uses internally.

    lifelines sorts the data by [duration, event] ascending, builds the design
    matrix from the formula, then normalizes each column by subtracting the
    mean and dividing by the std (stored on the model as ``_norm_mean`` and
    ``_norm_std``).

    Args:
        model: A fitted CoxPHFitter (naive, without clustering).
        df: The original (unsorted) DataFrame used for fitting.
        duration_col: Duration column name.
        event_col: Event column name.

    Returns:
        Tuple of (X_norm, T, E, W, original_index) where all arrays are in
        the same sorted order that lifelines uses internally.
    """
    sort_by = [duration_col, event_col] if event_col else [duration_col]
    df_sorted = df.sort_values(by=sort_by, kind='mergesort').copy()
    original_index = df_sorted.index.copy()

    X = model.regressors.transform_df(df_sorted)["beta_"].values.astype(float)
    norm_mean = model._norm_mean.values
    norm_std = model._norm_std.values
    X_norm = (X - norm_mean) / norm_std

    T = df_sorted[duration_col].values.astype(float)
    E = df_sorted[event_col].values.astype(float)
    if model.weights_col is not None and model.weights_col in df_sorted.columns:
        W = df_sorted[model.weights_col].values.astype(float)
    else:
        W = np.ones(len(df_sorted))

    return X_norm, T, E, W, original_index


def _compute_cluster_robust_variance(
    model: CoxPHFitter,
    X_norm: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    W: np.ndarray,
    clusters: np.ndarray,
) -> np.ndarray:
    """
    Compute the cluster-robust (Huber-White sandwich) variance matrix using
    a vectorized O(n) algorithm instead of lifelines' O(n²) loop.

    The sandwich formula is:

        V_robust = (sum_g U_g)(sum_g U_g)^T

    where U_g = sum_{i in cluster g} delta_beta_i, and

        delta_beta = score_residuals @ (variance_matrix * norm_std_col)

    The score residuals are computed via vectorized suffix-prefix sums,
    avoiding the per-observation Python loop in lifelines'
    ``_compute_score_within_strata``.

    Args:
        model: Fitted naive CoxPHFitter (coeffs + variance_matrix_ available).
        X_norm: Normalized design matrix, sorted by (T, E).
        T: Durations, sorted (same order as X_norm).
        E: Event indicators, sorted (0.0/1.0).
        W: Weights, sorted (same order as X_norm).
        clusters: Cluster IDs for each row (same order as X_norm).

    Returns:
        Robust variance matrix (d, d), same scale as ``model.variance_matrix_``.
    """
    d = X_norm.shape[1]

    beta = model.params_.values
    norm_std = model._norm_std.values
    beta_norm = beta * norm_std

    phi = np.exp(X_norm @ beta_norm)
    E_int = E.astype(int)

    # Suffix sums (risk set = {j >= i} in sorted order)
    S0 = np.cumsum((W * phi)[::-1])[::-1]
    S1 = np.cumsum((X_norm * (W * phi)[:, None])[::-1], axis=0)[::-1]

    # Guard against division by zero (empty risk set)
    S0_safe = np.where(S0 > 0, S0, 1.0)
    bar_x = S1 / S0_safe[:, None]
    bar_x[~np.isfinite(bar_x)] = 0.0

    # I_k = delta_k * w_k / S_0(t_k)
    I = E_int * W / S0_safe

    # Prefix sums (accumulate from k=0 to i)
    A = np.cumsum(I)
    B = np.cumsum(I[:, None] * bar_x, axis=0)

    # Vectorized score residuals
    score = (
        -phi[:, None] * X_norm * A[:, None]
        + phi[:, None] * B
        + E_int[:, None] * (X_norm - bar_x)
    )
    score_residuals = score * W[:, None]

    # Scale to delta_betas (same as lifelines' _compute_delta_beta)
    scaled_variance_matrix = model.variance_matrix_.values * np.tile(norm_std, (d, 1)).T
    delta_betas = score_residuals @ scaled_variance_matrix

    # Group by cluster and sum
    unique_clusters = np.unique(clusters)
    cluster_codes = pd.Categorical(clusters, categories=unique_clusters).codes
    G = len(unique_clusters)

    cluster_sums = np.zeros((G, d))
    np.add.at(cluster_sums, cluster_codes, delta_betas)

    # Sandwich estimator: sum_g U_g U_g^T
    sandwich = cluster_sums.T @ cluster_sums

    return sandwich


def _apply_cluster_robust_se(
    model: CoxPHFitter,
    df: pd.DataFrame,
    formula: str,
    duration_col: str,
    event_col: str,
    cluster_col: str,
) -> CoxPHFitter:
    """
    Take a fitted naive CoxPHFitter and replace its model-based standard
    errors with cluster-robust (Huber-White sandwich) standard errors.

    The model's coefficients, hazard ratios, log-likelihood, AIC, and
    concordance are unchanged — only the standard errors, z-values,
    p-values, and confidence intervals are updated.

    Args:
        model: A fitted *naive* CoxPHFitter.
        df: The DataFrame used for fitting (must contain *cluster_col*).
        formula: R-style formula string.
        duration_col: Duration column name.
        event_col: Event column name.
        cluster_col: Column identifying independent simulation runs.

    Returns:
        A **copy** of *model* with robust standard errors patched in.
    """
    robust_model = copy.deepcopy(model)
    robust_model.robust = True
    robust_model.cluster_col = cluster_col
    robust_model._model.robust = True
    robust_model._model.cluster_col = cluster_col
    robust_model._model._clusters = df[cluster_col].values

    X_norm, T, E, W, idx = _reconstruct_normalized_X(model, df, duration_col, event_col)
    clusters = df.loc[idx, cluster_col].values

    robust_var = _compute_cluster_robust_variance(model, X_norm, T, E, W, clusters)
    robust_se = np.sqrt(np.diag(robust_var))

    robust_model._model.standard_errors_ = pd.Series(
        robust_se, name="se", index=model.params_.index
    )

    alpha = model.alpha
    ci_pct = 100 * (1 - alpha)
    z_crit = inv_normal_cdf(1 - alpha / 2)

    hazards = model.params_.values
    ci_lower_name = "%g%% lower-bound" % ci_pct
    ci_upper_name = "%g%% upper-bound" % ci_pct
    robust_model._model.confidence_intervals_ = pd.DataFrame(
        np.c_[hazards - z_crit * robust_se, hazards + z_crit * robust_se],
        columns=[ci_lower_name, ci_upper_name],
        index=model.params_.index,
    )

    robust_model.variance_matrix_ = model.variance_matrix_

    return robust_model


def fit_clustered_cox_model(df: pd.DataFrame,
                            formula: str,
                            duration_col: str = 'arrival_time',
                            event_col: str = 'event',
                            cluster_col: str = 'simulation_run_id',
                            penalizer: float = 0.1,
                            prefit_model: Optional[CoxPHFitter] = None) -> CoxPHFitter:
    """
    Fit a Cox model with cluster-robust (Huber-White sandwich) SEs.

    Districts within the same stochastic simulation are correlated because they
    share the same epidemic realization, commuter mobility, and transmission
    history. Cluster-robust sandwich standard errors account for this within-run
    dependence while leaving coefficient estimates unchanged.

    This implementation fits a naive Cox model (without clustering) and then
    computes cluster-robust standard errors via a **vectorized** O(n) algorithm.
    This avoids the O(n²) score-residual loop in lifelines' built-in
    ``cluster_col`` / ``robust`` path, which is prohibitively slow for
    large datasets (e.g. 126k observations).

    Args:
        df: DataFrame with survival data (must contain *cluster_col*).
        formula: R-style formula string.
        duration_col: Duration column name.
        event_col: Event indicator column name.
        cluster_col: Column identifying independent simulation runs
            (default ``'simulation_run_id'``).
        penalizer: L2 penalizer strength.
        prefit_model: Optionally pass a pre-fitted naive ``CoxPHFitter``
            to avoid re-fitting. If ``None`` (default), a naive model is
            fit internally.

    Returns:
        Fitted ``CoxPHFitter`` whose ``summary`` reports cluster-robust SEs,
        z-values, p-values, and confidence intervals.
    """
    if prefit_model is not None:
        naive_model = prefit_model
    else:
        naive_model = CoxPHFitter(penalizer=penalizer)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            naive_model.fit(
                df,
                duration_col=duration_col,
                event_col=event_col,
                formula=formula,
            )

    clustered_model = _apply_cluster_robust_se(
        naive_model, df, formula, duration_col, event_col, cluster_col
    )
    return clustered_model


# ============================================================================
# Summary Extraction
# ============================================================================

def extract_model_summary(model: CoxPHFitter,
                          df: pd.DataFrame,
                          cluster_col: str = 'simulation_run_id') -> Dict[str, Any]:
    """
    Extract comprehensive summary statistics from a fitted ``CoxPHFitter``.

    When the model was fit with ``cluster_col`` and ``robust=True`` the returned
    standard errors, z-values, p-values, and confidence intervals are the
    cluster-robust (Huber-White sandwich) versions.  When fit without clustering
    they are the standard model-based quantities.

    Args:
        model: A fitted ``CoxPHFitter``.
        df: The DataFrame that was used for fitting (for *n_obs* and *n_clusters*).
        cluster_col: Column name for the clustering variable.

    Returns:
        Dictionary with keys: ``coefficients``, ``hazard_ratios``, ``se``,
        ``z_values``, ``p_values``, ``ci_lower``, ``ci_upper``, ``concordance``,
        ``aic``, ``log_likelihood``, ``n_obs``, ``n_clusters``.
    """
    summary = model.summary

    p_vals = list(summary['p'].values) if 'p' in summary.columns else []
    ses = list(summary['se(coef)'].values) if 'se(coef)' in summary.columns else []
    z_vals = list(summary['z'].values) if 'z' in summary.columns else []

    if hasattr(model, 'confidence_intervals_'):
        ci_lower = list(np.exp(model.confidence_intervals_.values[:, 0]))
        ci_upper = list(np.exp(model.confidence_intervals_.values[:, 1]))
    else:
        ci_lower = []
        ci_upper = []

    # Concordance requires at least one event
    event_df = df[df['event'] == 1] if 'event' in df.columns else pd.DataFrame()
    if len(event_df) > 0:
        try:
            ci = model.concordance_index_
        except Exception:
            ci = np.nan
    else:
        ci = np.nan
    if np.isnan(ci):
        ci = 0.0

    n_clusters = None
    if cluster_col in df.columns:
        n_clusters = int(df[cluster_col].nunique())

    return {
        'coefficients': model.params_.copy(),
        'hazard_ratios': np.exp(model.params_),
        'se': ses,
        'z_values': z_vals,
        'p_values': p_vals,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'concordance': ci,
        'aic': model.AIC_partial_ if hasattr(model, 'AIC_partial_') else np.nan,
        'log_likelihood': model.log_likelihood_ if hasattr(model, 'log_likelihood_') else np.nan,
        'n_obs': len(df),
        'n_clusters': n_clusters,
    }


def create_model_summary_df(model_result: Dict, version: str = 'clustered') -> pd.DataFrame:
    """
    Create a comprehensive one-row-per-variable summary DataFrame.

    Columns include coefficient, standard error, hazard ratio, 95% CI, z-statistic,
    p-value, plus model-level metrics (partial log-likelihood, partial AIC,
    concordance index, number of observations, number of clusters).

    Args:
        model_result: A single model's result dict from ``run_scenario_cox_models``
            or ``run_parameter_cox_models``.
        version: ``'clustered'`` (default) for robust SEs or ``'naive'`` for
            model-based SEs.

    Returns:
        DataFrame with one row per regression coefficient.
    """
    if version == 'clustered':
        summary_df = model_result.get('summary')
        se_list = model_result.get('se', [])
        z_list = model_result.get('z_values', [])
        p_list = model_result.get('p_values', [])
        ci_lo = model_result.get('ci_lower', [])
        ci_hi = model_result.get('ci_upper', [])
        se_type = ('Robust clustered (Huber-White sandwich, cluster=simulation_run_id)')
    else:
        summary_df = model_result.get('naive_summary')
        se_list = model_result.get('naive_se', [])
        z_list = model_result.get('naive_z_values', [])
        p_list = model_result.get('naive_p_values', [])
        ci_lo = model_result.get('naive_ci_lower', [])
        ci_hi = model_result.get('naive_ci_upper', [])
        se_type = 'Naive (model-based)'

    coefficients = model_result.get('coefficients', pd.Series(dtype=float))
    hazard_ratios = model_result.get('hazard_ratios', pd.Series(dtype=float))
    var_names = list(coefficients.index)

    rows = []
    for i, var in enumerate(var_names):
        rows.append({
            'Variable': var,
            'Coefficient': float(coefficients.iloc[i]) if i < len(coefficients) else np.nan,
            'Standard_Error': se_list[i] if i < len(se_list) else np.nan,
            'Hazard_Ratio': float(hazard_ratios.iloc[i]) if i < len(hazard_ratios) else np.nan,
            'CI_lower_95%': ci_lo[i] if i < len(ci_lo) else np.nan,
            'CI_upper_95%': ci_hi[i] if i < len(ci_hi) else np.nan,
            'z_statistic': z_list[i] if i < len(z_list) else np.nan,
            'p_value': p_list[i] if i < len(p_list) else np.nan,
        })

    result_df = pd.DataFrame(rows)

    # Model-level metrics (same for naive and clustered)
    result_df['Partial_Log_Likelihood'] = model_result.get('log_likelihood', np.nan)
    result_df['Partial_AIC'] = model_result.get('aic', np.nan)
    result_df['Concordance_Index'] = model_result.get('concordance', np.nan)
    result_df['N_Observations'] = model_result.get('n_obs', np.nan)
    result_df['N_Clusters'] = model_result.get('n_clusters', np.nan)
    result_df['SE_Type'] = se_type

    return result_df


# ============================================================================
# Naive vs. Clustered Comparison
# ============================================================================

def compare_naive_vs_clustered(model_results: Dict) -> pd.DataFrame:
    """
    Produce a side-by-side comparison of naive vs. cluster-robust results.

    Demonstrates that hazard ratios remain nearly unchanged while confidence
    intervals become wider with cluster-robust standard errors.

    Args:
        model_results: Dictionary of model results (from ``run_scenario_cox_models``
            or ``run_parameter_cox_models``).

    Returns:
        DataFrame with columns: ``Model``, ``Variable``, ``HR``, ``Naive_SE``,
        ``Robust_SE``, ``Naive_CI_lower``, ``Naive_CI_upper``, ``Robust_CI_lower``,
        ``Robust_CI_upper``.
    """
    rows = []

    for model_name, res in model_results.items():
        clustered_summary = res.get('summary')
        naive_summary = res.get('naive_summary')

        if clustered_summary is None or naive_summary is None:
            continue

        for var in clustered_summary.index:
            hr = clustered_summary.loc[var, 'exp(coef)']

            naive_se = (
                naive_summary.loc[var, 'se(coef)']
                if var in naive_summary.index and 'se(coef)' in naive_summary.columns
                else np.nan
            )
            robust_se = (
                clustered_summary.loc[var, 'se(coef)']
                if var in clustered_summary.index and 'se(coef)' in clustered_summary.columns
                else np.nan
            )

            naive_ci_lo = (
                naive_summary.loc[var, 'exp(coef) lower 95%']
                if var in naive_summary.index and 'exp(coef) lower 95%' in naive_summary.columns
                else np.nan
            )
            naive_ci_hi = (
                naive_summary.loc[var, 'exp(coef) upper 95%']
                if var in naive_summary.index and 'exp(coef) upper 95%' in naive_summary.columns
                else np.nan
            )
            robust_ci_lo = (
                clustered_summary.loc[var, 'exp(coef) lower 95%']
                if var in clustered_summary.index and 'exp(coef) lower 95%' in clustered_summary.columns
                else np.nan
            )
            robust_ci_hi = (
                clustered_summary.loc[var, 'exp(coef) upper 95%']
                if var in clustered_summary.index and 'exp(coef) upper 95%' in clustered_summary.columns
                else np.nan
            )

            rows.append({
                'Model': model_name,
                'Variable': var,
                'HR': hr,
                'Naive_SE': naive_se,
                'Robust_SE': robust_se,
                'Naive_CI_lower': naive_ci_lo,
                'Naive_CI_upper': naive_ci_hi,
                'Robust_CI_lower': robust_ci_lo,
                'Robust_CI_upper': robust_ci_hi,
            })

    return pd.DataFrame(rows)


# ============================================================================
# Internal: Dual-Fit Helper
# ============================================================================

def _fit_and_process_both(df: pd.DataFrame,
                          formula: str,
                          model_name: str,
                          cluster_col: str = 'simulation_run_id',
                          penalizer: float = 0.1,
                          duration_col: str = 'arrival_time',
                          event_col: str = 'event',
                          pressure_window: Optional[int] = None) -> Optional[Dict]:
    """
    Fit both naive and cluster-robust Cox models and return a combined result dict.

    Districts within the same stochastic simulation are correlated because they share
    the same epidemic realization, commuter mobility, and transmission history.
    Cluster-robust sandwich standard errors account for this within-run dependence
    while leaving coefficient estimates unchanged.

    The returned dict stores the **clustered (robust)** results under the primary
    keys (``coefficients``, ``se``, ``p_values``, ``ci_lower``, ``ci_upper`` etc.)
    and the **naive** results under ``naive_``-prefixed keys
    (``naive_se``, ``naive_p_values``, ``naive_ci_lower``, etc.).

    Args:
        df: Survival analysis DataFrame.
        formula: R-style formula string.
        model_name: Human-readable model name for warnings.
        cluster_col: Column identifying independent simulation runs.
        penalizer: L2 penalizer strength.
        duration_col: Duration column name.
        event_col: Event indicator column name.
        pressure_window: Optional pressure-window label to store.

    Returns:
        Combined result dict, or ``None`` if both fits fail.
    """
    naive_model = None
    clustered_model = None
    naive_stats = None
    clustered_stats = None

    # --- Fit naive model (model-based SEs) ---
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            naive_model = CoxPHFitter(penalizer=penalizer)
            naive_model.fit(
                df, duration_col=duration_col, event_col=event_col, formula=formula
            )
        naive_stats = extract_model_summary(naive_model, df, cluster_col=cluster_col)
    except Exception as e:
        warnings.warn(f"Failed to fit naive {model_name}: {e}")
        naive_model = None
        naive_stats = None

    # --- Compute cluster-robust SEs from the naive fit (vectorized, fast) ---
    if naive_model is not None and naive_stats is not None:
        try:
            clustered_model = fit_clustered_cox_model(
                df, formula, duration_col, event_col,
                cluster_col=cluster_col,
                penalizer=penalizer,
                prefit_model=naive_model,
            )
            clustered_stats = extract_model_summary(clustered_model, df, cluster_col=cluster_col)
        except Exception as e:
            warnings.warn(
                f"Failed to compute cluster-robust SEs for {model_name}: {e}. "
                f"Falling back to naive model-based standard errors."
            )
            clustered_model = None
            clustered_stats = None

    # --- Resolve which stats to use as primary ---
    if clustered_model is not None and clustered_stats is not None:
        # Clustered model succeeded — it is the primary
        primary_stats = clustered_stats
        primary_model = clustered_model
    elif naive_model is not None and naive_stats is not None:
        # Clustered failed — fall back to naive (robust_se_used = False)
        warnings.warn(
            f"Cluster-robust fit failed for {model_name}; "
            f"falling back to naive model-based standard errors."
        )
        primary_stats = naive_stats
        primary_model = naive_model
    else:
        warnings.warn(f"Both naive and cluster-robust fits failed for {model_name}.")
        return None

    n_vars = len(primary_stats['coefficients'])

    result = {
        # Primary (clustered/robust where available, otherwise naive):
        'model': primary_model,
        'summary': primary_model.summary,
        'coefficients': primary_stats['coefficients'],
        'hazard_ratios': primary_stats['hazard_ratios'],
        'se': primary_stats['se'],
        'z_values': primary_stats['z_values'],
        'p_values': primary_stats['p_values'],
        'ci_lower': primary_stats['ci_lower'],
        'ci_upper': primary_stats['ci_upper'],
        'concordance': primary_stats['concordance'],
        'aic': primary_stats['aic'],
        'log_likelihood': primary_stats['log_likelihood'],
        'n_obs': primary_stats['n_obs'],
        'n_clusters': primary_stats['n_clusters'],
        # Naive versions (may be None / NaN if naive fit failed):
        'naive_model': naive_model,
        'naive_summary': naive_model.summary if naive_model is not None else None,
        'naive_se': naive_stats['se'] if naive_stats else [np.nan] * n_vars,
        'naive_z_values': naive_stats['z_values'] if naive_stats else [np.nan] * n_vars,
        'naive_p_values': naive_stats['p_values'] if naive_stats else [np.nan] * n_vars,
        'naive_ci_lower': naive_stats['ci_lower'] if naive_stats else [np.nan] * n_vars,
        'naive_ci_upper': naive_stats['ci_upper'] if naive_stats else [np.nan] * n_vars,
        'naive_concordance': naive_stats['concordance'] if naive_stats else np.nan,
        'naive_aic': naive_stats['aic'] if naive_stats else np.nan,
        'naive_log_likelihood': naive_stats['log_likelihood'] if naive_stats else np.nan,
        # Metadata:
        'robust_se_used': clustered_model is not None,
        'cluster_col': cluster_col,
        'formula': formula,
    }

    if pressure_window is not None:
        result['pressure_window'] = pressure_window

    return result


# ============================================================================
# Scenario Comparison Models (Mode A)
# ============================================================================

def run_scenario_cox_models(survival_df: pd.DataFrame,
                            cluster_col: str = 'simulation_run_id') -> Dict:
    """
    Fit causal Cox models for scenario comparison (Mode A).

    Model hierarchy (both naive and cluster-robust SEs computed):

        M0: hazard ~ gathering_event
        M1: hazard ~ gathering_event + log_seed_size
        M2: hazard ~ gathering_event + log_seed_size + pre_invasion_pressure

    The cluster-robust (Huber-White sandwich) standard errors are the primary
    results. Naive model-based SEs are retained for comparison.

    Args:
        survival_df: DataFrame with arrival_time, event, covariates, and
            ``simulation_run_id``.
        cluster_col: Column identifying independent simulation runs.

    Returns:
        Dictionary keyed by model name. Each value is a result dict containing
        both naive and clustered summaries.
    """
    results = {}

    # --- Data validation ---
    # Districts within the same stochastic simulation are correlated because
    # they share the same epidemic realization, commuter mobility, and
    # transmission history. Cluster-robust sandwich SEs account for this.
    try:
        validate_survival_data(survival_df, cluster_col=cluster_col)
    except Exception as e:
        warnings.warn(f"Data validation issue: {e}")

    # M0: gathering_event only
    if 'gathering_event' in survival_df.columns:
        res = _fit_and_process_both(
            survival_df, 'gathering_event', 'M0_gathering',
            cluster_col=cluster_col
        )
        if res is not None:
            results['M0_gathering'] = res

    # M1: gathering_event + log_seed_size
    if 'gathering_event' in survival_df.columns and 'log_seed_size' in survival_df.columns:
        res = _fit_and_process_both(
            survival_df, 'gathering_event + log_seed_size', 'M1_gathering_seed',
            cluster_col=cluster_col
        )
        if res is not None:
            results['M1_gathering_seed'] = res

    # M2: gathering_event + log_seed_size + pre_pressure (multiple window sizes)
    pressure_windows = [3, 7, 14]
    for window in pressure_windows:
        col_name = f'pre_pressure_{window}d'
        if ('gathering_event' in survival_df.columns
                and 'log_seed_size' in survival_df.columns
                and col_name in survival_df.columns):
            res = _fit_and_process_both(
                survival_df,
                f'gathering_event + log_seed_size + {col_name}',
                f'M2_pressure_{window}d',
                cluster_col=cluster_col,
                pressure_window=window,
            )
            if res is not None:
                results[f'M2_pressure_{window}d'] = res

    return results


# ============================================================================
# Seed Location Models (Mode C)
# ============================================================================

def run_seed_cox_models(survival_df: pd.DataFrame,
                        cluster_col: str = 'simulation_run_id') -> Dict:
    """Fit causal Cox models for seed-location sensitivity analysis (Mode C).

    The Mode C analysis reuses the **same model hierarchy** as Mode A
    (scenario comparison).  The exposure variable is ``gathering_event``
    (MGE vs no_event), the key covariate is ``log_seed_size`` (initial
    epidemic seed size Z_seed), and ``pre_pressure_*`` captures the
    mediating effect of pre-invasion epidemic pressure.

    The ``seed_location`` variable is **not** a Cox covariate — it is
    captured in the ``simulation_run_id`` cluster ID (e.g.
    ``f"seed_{X}_{scenario}_{run_id}"``) so that cluster-robust standard
    errors correctly account for within-seed-run correlation.

    effective_distance is intentionally **excluded** from the Cox models
    per the paper scope; it remains available as an exploratory
    descriptive covariate only.

    Model hierarchy:

        M0: hazard ~ gathering_event
        M1: hazard ~ gathering_event + log_seed_size
        M2: hazard ~ gathering_event + log_seed_size + pre_invasion_pressure

    Args:
        survival_df: DataFrame with arrival_time, event, covariates, and
            ``simulation_run_id`` (prefixed with seed_location).
        cluster_col: Column identifying independent simulation runs.

    Returns:
        Dictionary keyed by model name.  Each value is a result dict
        containing both naive and clustered summaries.
    """
    return run_scenario_cox_models(survival_df, cluster_col=cluster_col)


# ============================================================================
# Parameter Sensitivity Models (Mode B)
# ============================================================================

def run_parameter_cox_models(survival_df: pd.DataFrame,
                             factorial: bool = False,
                             cluster_col: str = 'simulation_run_id') -> Dict:
    """
    Fit Cox models for parameter sensitivity analysis (Mode B).

    Models (both naive and cluster-robust SEs computed):

        P0: hazard ~ C(param_combo)
        P1: hazard ~ C(param_combo) + log_seed_size
        P2: hazard ~ C(param_combo) + log_seed_size + pre_pressure_7d

    If ``factorial=True`` continuous standardized versions are also fit:

        F1: hazard ~ z_R0 + z_beta
        F2: hazard ~ z_R0 + z_beta + z_R0:z_beta + log_seed_size

    Args:
        survival_df: DataFrame with survival data and ``simulation_run_id``.
        factorial: If True, also fit continuous factorial models.
        cluster_col: Column identifying independent simulation runs.

    Returns:
        Dictionary keyed by model name. Each value is a result dict containing
        both naive and clustered summaries.
    """
    results = {}

    # --- Data validation ---
    try:
        validate_survival_data(survival_df, cluster_col=cluster_col)
    except Exception as e:
        warnings.warn(f"Data validation issue: {e}")

    if 'param_combo' not in survival_df.columns:
        warnings.warn("param_combo column not found in DataFrame")
        return results

    if not factorial:
        # P0: param_combo only
        res = _fit_and_process_both(
            survival_df, 'C(param_combo)', 'P0_param_combo',
            cluster_col=cluster_col
        )
        if res is not None:
            results['P0_param_combo'] = res

        # P1: param_combo + log_seed_size
        if 'log_seed_size' in survival_df.columns:
            res = _fit_and_process_both(
                survival_df, 'C(param_combo) + log_seed_size', 'P1_param_seed',
                cluster_col=cluster_col
            )
            if res is not None:
                results['P1_param_seed'] = res

        # P2: param_combo + log_seed_size + pre_pressure_7d
        if 'log_seed_size' in survival_df.columns and 'pre_pressure_7d' in survival_df.columns:
            res = _fit_and_process_both(
                survival_df,
                'C(param_combo) + log_seed_size + pre_pressure_7d',
                'P2_param_seed_pressure',
                cluster_col=cluster_col,
                pressure_window=7,
            )
            if res is not None:
                results['P2_param_seed_pressure'] = res
    else:
        # Standardize R0 and beta for factorial models
        survival_df = survival_df.copy()
        if survival_df['R0'].std() > 0:
            survival_df['z_R0'] = (survival_df['R0'] - survival_df['R0'].mean()) / survival_df['R0'].std()
        if survival_df['beta'].std() > 0:
            survival_df['z_beta'] = (survival_df['beta'] - survival_df['beta'].mean()) / survival_df['beta'].std()

        # F1: R0 + beta
        res = _fit_and_process_both(
            survival_df, 'z_R0 + z_beta', 'F1_R0_beta',
            cluster_col=cluster_col
        )
        if res is not None:
            results['F1_R0_beta'] = res

        # F2: R0 + beta + R0*beta + log_seed_size
        if 'log_seed_size' in survival_df.columns:
            res = _fit_and_process_both(
                survival_df, 'z_R0 + z_beta + z_R0:z_beta + log_seed_size',
                'F2_R0_beta_interaction',
                cluster_col=cluster_col
            )
            if res is not None:
                results['F2_R0_beta_interaction'] = res

    return results


# ============================================================================
# Model Comparison Table
# ============================================================================

def create_causal_model_comparison_table(model_results: Dict) -> pd.DataFrame:
    """
    Create a comparison table of causal model performance.

    Columns: Model, Formula, Partial Log Likelihood, Partial AIC, Delta AIC,
    Concordance Index, Number of observations, Number of simulation clusters,
    plus per-variable hazard ratio columns and LRT column.
    """
    rows = []
    prev_ll = None

    aic_values = [res.get('aic', np.nan) for res in model_results.values() if not np.isnan(res.get('aic', np.nan))]
    min_aic = min(aic_values) if aic_values else np.nan
    conc_values = [res.get('concordance', np.nan) for res in model_results.values() if not np.isnan(res.get('concordance', np.nan))]
    max_concordance = max(conc_values) if conc_values else np.nan

    formula_map = {
        'M0_gathering': 'gathering_event',
        'M1_gathering_seed': 'gathering_event + log_seed_size',
        'M2_pressure_3d': 'gathering_event + log_seed_size + pre_pressure_3d',
        'M2_pressure_7d': 'gathering_event + log_seed_size + pre_pressure_7d',
        'M2_pressure_14d': 'gathering_event + log_seed_size + pre_pressure_14d',
        'P0_param_combo': 'C(param_combo)',
        'P1_param_seed': 'C(param_combo) + log_seed_size',
        'P2_param_seed_pressure': 'C(param_combo) + log_seed_size + pre_pressure_7d',
        'F1_R0_beta': 'z_R0 + z_beta',
        'F2_R0_beta_interaction': 'z_R0 + z_beta + z_R0:z_beta + log_seed_size',
    }

    for name, res in model_results.items():
        ll = res.get('log_likelihood', np.nan)
        lrt = 2 * (ll - prev_ll) if prev_ll is not None and not np.isnan(ll) and ll > prev_ll else np.nan
        aic = res.get('aic', np.nan)
        conc = res.get('concordance', np.nan)
        delta_aic = (aic - min_aic) if not np.isnan(aic) and not np.isnan(min_aic) else np.nan
        delta_conc = (max_concordance - conc) if not np.isnan(conc) and not np.isnan(max_concordance) else np.nan

        pressure_window = res.get('pressure_window', 7)
        pressure_col = f'pre_pressure_{pressure_window}d'

        row = {
            'Model': name,
            'Formula': formula_map.get(name, ''),
            'Partial_Log_Likelihood': ll,
            'Partial_AIC': aic,
            'Delta_AIC': delta_aic,
            'Concordance_Index': conc,
            'Delta_Concordance': delta_conc,
            'N_Observations': res.get('n_obs', np.nan),
            'N_Clusters': res.get('n_clusters', np.nan),
            'Log_likelihood': ll,
            'AIC': aic,
            'Concordance': conc,
            'Exposure_coef': res['coefficients'].get('gathering_event', res['coefficients'].get('param_combo', np.nan)),
            'Exposure_HR': res['hazard_ratios'].get('gathering_event', res['hazard_ratios'].get('param_combo', np.nan)),
            'Seed_coef': res['coefficients'].get('log_seed_size', np.nan),
            'Seed_HR': res['hazard_ratios'].get('log_seed_size', np.nan),
            'Pressure_coef': res['coefficients'].get(pressure_col, np.nan),
            'Pressure_HR': res['hazard_ratios'].get(pressure_col, np.nan),
            'LRT_vs_previous': lrt
        }
        rows.append(row)
        prev_ll = ll

    return pd.DataFrame(rows)
