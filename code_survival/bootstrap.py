"""
Bootstrap estimation for hazard ratios.

Provides non-parametric bootstrap for uncertainty quantification.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List

try:
    from lifelines import CoxPHFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


def bootstrap_cox_models(survival_df: pd.DataFrame, 
                         n_bootstrap: int = 1000, 
                         random_state: int = 42) -> pd.DataFrame:
    """
    Bootstrap hazard ratio estimation for Cox models.
    
    Resamples districts within scenarios to preserve invasion rate structure.
    For each bootstrap sample, refits models and extracts HRs.
    
    Args:
        survival_df: DataFrame with arrival_time, event, and relevant covariates
        n_bootstrap: Number of bootstrap iterations (default 1000)
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with mean, median, SE, 95% CI (percentile and bias-corrected) for each HR
    """
    if not HAS_LIFELINES:
        raise ImportError("lifelines package required. Install with: pip install lifelines")
    
    np.random.seed(random_state)
    
    models_to_fit = {}
    
    # Detect which models to fit based on available columns
    if 'gathering_event' in survival_df.columns:
        models_to_fit['M0_gathering'] = 'gathering_event'
        models_to_fit['M1_gathering_seed'] = 'gathering_event + log_seed_size'
    
    for window in [3, 7, 14]:
        col = f'pre_pressure_{window}d'
        if col in survival_df.columns:
            models_to_fit[f'M2_pressure_{window}d'] = f'gathering_event + log_seed_size + {col}'
    
    if 'param_combo' in survival_df.columns:
        models_to_fit['P0_param_combo'] = 'C(param_combo)'
    
    available_models = {name: formula for name, formula in models_to_fit.items() 
                        if all(v in survival_df.columns for v in formula.replace('C(param_combo)', 'param_combo').split())}
    
    hr_collections = {name: {'gathering_event': [], 'log_seed_size': [], 'pre_pressure': []} 
                        for name in available_models}
    
    grouping_col = 'scenario' if 'scenario' in survival_df.columns else 'param_combo'
    
    for b in range(n_bootstrap):
        bootstrap_dfs = []
        for group in survival_df[grouping_col].unique():
            group_df = survival_df[survival_df[grouping_col] == group]
            bootstrap_dfs.append(group_df.sample(n=len(group_df), replace=True, random_state=random_state + b))
        bootstrap_df = pd.concat(bootstrap_dfs, ignore_index=True)
        
        for model_name, formula in available_models.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = CoxPHFitter(penalizer=0.01)
                    model.fit(bootstrap_df, duration_col='arrival_time', event_col='event', formula=formula)
                
                for var in ['gathering_event', 'log_seed_size']:
                    if var in model.params_.index:
                        hr_collections[model_name][var].append(np.exp(model.params_.loc[var]))
                
                for window in [3, 7, 14]:
                    var = f'pre_pressure_{window}d'
                    if var in model.params_.index:
                        hr_collections[model_name]['pre_pressure'] = hr_collections[model_name].get('pre_pressure', [])
                        hr_collections[model_name]['pre_pressure'].append(np.exp(model.params_.loc[var]))
            except Exception:
                continue
    
    results = []
    for model_name in available_models:
        for var_key, var_label in [('gathering_event', 'gathering_event'), 
                                    ('log_seed_size', 'log_seed_size'),
                                    ('pre_pressure', 'pre_pressure_14d')]:
            values = hr_collections[model_name][var_key]
            if not values:
                continue
            values = np.array(values)
            
            mean_hr = np.mean(values)
            median_hr = np.median(values)
            se_hr = np.std(values, ddof=1)
            
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            
            sorted_vals = np.sort(values)
            n = len(values)
            z0 = 2 * 0.5 - np.mean(values < np.percentile(values, 50))
            z_alpha = 1.96
            ci_bc_lower = sorted_vals[max(0, int(np.exp(-z0 + 0.5) * n - 0.5 * z_alpha * np.sqrt(n)))]
            ci_bc_upper = sorted_vals[min(n - 1, int(np.exp(-z0 + 0.5) * n + 0.5 * z_alpha * np.sqrt(n)))]
            
            results.append({
                'model': model_name,
                'variable': var_label,
                'hr_mean': mean_hr,
                'hr_median': median_hr,
                'hr_se': se_hr,
                'ci_lower_percentile': ci_lower,
                'ci_upper_percentile': ci_upper,
                'ci_lower_bc': ci_bc_lower,
                'ci_upper_bc': ci_bc_upper,
                'n_bootstrap': len(values),
                'bias': mean_hr - values.mean() if var_label != 'pre_pressure_14d' else np.nan
            })
    
    return pd.DataFrame(results)