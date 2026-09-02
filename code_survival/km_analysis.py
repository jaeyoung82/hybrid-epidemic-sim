"""
Kaplan-Meier analysis for survival data.

Supports both scenario comparison (Mode A) and parameter sensitivity (Mode B) analyses.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test, logrank_test
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


def compute_km_curves(survival_df: pd.DataFrame, 
                       grouping_col: str = 'scenario',
                       group_values: Optional[List] = None) -> Tuple[pd.DataFrame, pd.DataFrame, float, Dict, pd.DataFrame]:
    """
    Compute Kaplan-Meier curves for each group.
    
    Args:
        survival_df: DataFrame with arrival_time, event, and grouping column
        grouping_col: Column name to group by (scenario or param_combo)
        group_values: List of values to analyze. If None, uses all unique values in grouping_col.
    
    Returns:
        curve_data: DataFrame with survival probabilities
        km_data: DataFrame with KM statistics
        logrank_pvalue: float, log-rank test p-value
        global_test: dict with chi-square, df, p-value
        pairwise_results: DataFrame with pairwise log-rank test results
    """
    if not HAS_LIFELINES:
        raise ImportError("lifelines package required. Install with: pip install lifelines")
    
    if group_values is None:
        group_values = survival_df[grouping_col].unique().tolist()
    
    curve_records = []
    at_risk_records = []
    
    for group in group_values:
        group_df = survival_df[survival_df[grouping_col] == group]
        
        if group_df.empty:
            continue
        
        kmf = KaplanMeierFitter()
        kmf.fit(group_df['arrival_time'], group_df['event'])
        
        for t, s, lower, upper in zip(kmf.survival_function_.index,
                                     kmf.survival_function_.values.flatten(),
                                     kmf.confidence_interval_survival_function_.values[:, 0],
                                     kmf.confidence_interval_survival_function_.values[:, 1]):
            curve_records.append({
                'time': t,
                'group': group,
                'survival_probability': s,
                'lower_confidence_interval': lower,
                'upper_confidence_interval': upper
            })
        
        for t, n_at_risk in kmf.event_table['at_risk'].items():
            at_risk_records.append({'time': t, 'group': group, 'number_at_risk': n_at_risk})
    
    curve_data = pd.DataFrame(curve_records)
    km_data = pd.DataFrame(at_risk_records)
    
    # Global multivariate log-rank test
    global_test = {'chi_square': np.nan, 'df': np.nan, 'p_value': np.nan}
    df_event = survival_df[survival_df['event'] == 1].copy()
    if len(df_event) > 0:
        test_result = multivariate_logrank_test(df_event['arrival_time'], df_event[grouping_col], df_event['event'])
        logrank_pvalue = test_result.p_value
        global_test = {
            'chi_square': float(test_result.test_statistic),
            'df': int(test_result.degrees_of_freedom),
            'p_value': float(test_result.p_value)
        }
    else:
        logrank_pvalue = np.nan
    
    # Pairwise log-rank tests
    pairwise_results = run_pairwise_logrank_tests(survival_df, grouping_col, group_values)
    
    return curve_data, km_data, logrank_pvalue, global_test, pairwise_results


def run_pairwise_logrank_tests(survival_df: pd.DataFrame,
                                grouping_col: str = 'scenario',
                                group_values: Optional[List] = None) -> pd.DataFrame:
    """
    Perform pairwise log-rank tests comparing each group against baseline.
    
    Uses the first group as baseline for comparison.
    Applies Holm multiple-comparison correction to raw p-values.
    
    Args:
        survival_df: DataFrame with arrival_time, event, and grouping column
        grouping_col: Column name to group by
        group_values: List of group values. If None, uses all unique values.
    
    Returns:
        DataFrame with Comparison, n_districts, n_events, Test statistic, Raw p-value, 
        Holm-adjusted p-value, Significant
    """
    if not HAS_LIFELINES:
        raise ImportError("lifelines package required. Install with: pip install lifelines")
    
    if group_values is None:
        group_values = survival_df[grouping_col].unique().tolist()
    
    if len(group_values) < 2:
        return pd.DataFrame()
    
    baseline = group_values[0]
    comparisons = group_values[1:] if grouping_col == 'scenario' else group_values
    
    results = []
    raw_pvalues = []
    
    for group in comparisons:
        baseline_df = survival_df[survival_df[grouping_col] == baseline]
        group_df = survival_df[survival_df[grouping_col] == group]
        
        if baseline_df.empty or group_df.empty:
            continue
        
        baseline_events = baseline_df[baseline_df['event'] == 1].copy()
        group_events = group_df[group_df['event'] == 1].copy()
        
        if len(baseline_events) == 0 or len(group_events) == 0:
            continue
        
        test_result = logrank_test(
            baseline_events['arrival_time'], group_events['arrival_time'],
            baseline_events['event'], group_events['event']
        )
        
        comparison_label = f'{baseline} vs {group}'
        if grouping_col == 'param_combo':
            comparison_label = f'baseline vs {group}'
        
        results.append({
            'Comparison': comparison_label,
            'n_districts': int(len(baseline_df) + len(group_df)),
            'n_events': int(len(baseline_events) + len(group_events)),
            'Test statistic': float(test_result.test_statistic),
            'Raw p-value': float(test_result.p_value)
        })
        raw_pvalues.append(float(test_result.p_value))
    
    # Holm correction
    if len(results) > 0:
        n_tests = len(raw_pvalues)
        sorted_indices = np.argsort(raw_pvalues)
        holm_adjusted = [np.nan] * n_tests
        
        for rank, idx in enumerate(sorted_indices):
            holm_adjusted[idx] = min(raw_pvalues[idx] * (n_tests - rank), 1.0)
        
        for i, r in enumerate(results):
            r['Holm-adjusted p-value'] = holm_adjusted[i]
            r['Significant'] = holm_adjusted[i] < 0.05
        
        return pd.DataFrame(results)
    
    return pd.DataFrame(columns=['Comparison', 'n_districts', 'n_events', 'Test statistic', 'Raw p-value', 'Holm-adjusted p-value', 'Significant'])


def print_at_risk_table(km_data: pd.DataFrame, group_values: Optional[List] = None):
    """Print numbers at risk table to CLI."""
    if group_values is None:
        group_values = km_data['group'].unique().tolist()
    
    print("\nNumbers at Risk by Group (Initial -> Final):")
    print("=" * 50)
    
    for group in group_values:
        group_data = km_data[km_data['group'] == group]
        if len(group_data) > 0:
            max_at_risk = group_data['number_at_risk'].max()
            min_at_risk = group_data['number_at_risk'].min()
            events = int(max_at_risk - min_at_risk)
            print(f"{group}: {int(max_at_risk)} -> {int(min_at_risk)} (n={events} events)")


def compute_median_survival_times(curve_data: pd.DataFrame, group_values: Optional[List] = None) -> pd.DataFrame:
    """
    Compute median survival times for each group.
    
    Returns DataFrame with group, median_time, and n_events columns.
    """
    if group_values is None:
        group_values = curve_data['group'].unique().tolist()
    
    results = []
    for group in group_values:
        group_data = curve_data[curve_data['group'] == group]
        if group_data.empty:
            continue
        
        median_times = group_data[group_data['survival_probability'] <= 0.5]['time']
        median_time = median_times.min() if len(median_times) > 0 else np.nan
        
        n_events = group_data['number_at_risk'].iloc[-1] if len(group_data) > 0 else 0
        
        results.append({
            'group': group,
            'median_survival_time': median_time,
            'n_events': n_events
        })
    
    return pd.DataFrame(results)