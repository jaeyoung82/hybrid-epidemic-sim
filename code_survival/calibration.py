"""
Model calibration assessment for survival analysis.

Computes calibration curves comparing predicted vs observed survival.
Calculates Integrated Brier Score and calibration slope/intercept.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


def assess_model_calibration(survival_df: pd.DataFrame, 
                            model_results: Dict,
                            time_points: Optional[np.ndarray] = None) -> Tuple[Dict, Dict]:
    """
    Assess model calibration by comparing predicted vs observed survival.
    
    For each model, computes predicted survival at time points and compares to
    observed Kaplan-Meier survival. Calculates calibration slope, intercept, and
    Integrated Brier Score.
    
    Args:
        survival_df: DataFrame with survival data
        model_results: Dictionary of fitted model results from run_causal_cox_models
        time_points: Array of time points for evaluation (default: 10-90th percentiles)
    
    Returns:
        calibration_data: Dict with survival predictions and observed for each model
        calibration_metrics: Dict with calibration slope, intercept, IBS for each model
    """
    if not HAS_LIFELINES:
        raise ImportError("lifelines package required. Install with: pip install lifelines")
    
    if time_points is None:
        time_points = np.percentile(survival_df[survival_df['event'] == 1]['arrival_time'], 
                                    np.arange(10, 91, 10))
    
    models_to_assess = ['M0_gathering', 'M1_gathering_seed', 'M2_pressure_14d']
    available_models = [m for m in models_to_assess if m in model_results]
    
    if not available_models:
        param_models = [m for m in model_results.keys() if m.startswith('P')]
        available_models = param_models[:3] if param_models else []
    
    calibration_data = {}
    calibration_metrics = {}
    
    overall_kmf = KaplanMeierFitter()
    overall_kmf.fit(survival_df['arrival_time'], survival_df['event'])
    
    overall_survival = {t: overall_kmf.survival_function_at_times(t).values[0] 
                        for t in time_points if t <= survival_df['arrival_time'].max()}
    
    for model_name in available_models:
        model = model_results[model_name].get('model')
        if model is None:
            continue
        
        try:
            surv_funcs = model.predict_survival_function(survival_df)
            mean_survival = surv_funcs.mean(axis=1).values if hasattr(surv_funcs.mean(axis=1), 'values') else surv_funcs.mean(axis=1).to_numpy()
            
            predicted_at_times = {}
            for t in time_points:
                idx = np.argmin(np.abs(surv_funcs.index - t))
                predicted_at_times[t] = mean_survival[idx]
            
            obs_times = overall_kmf.survival_function_.index.values
            obs_survival = overall_kmf.survival_function_.values.flatten()
            
            obs_interp = np.interp(time_points, obs_times, obs_survival, 
                                   left=1.0, right=0.0)
            
            obs_vals = np.array([overall_survival.get(t, 1.0) for t in time_points])
            pred_vals = np.array([predicted_at_times.get(t, 1.0) for t in time_points])
            
            valid_mask = ~(np.isnan(obs_vals) | np.isnan(pred_vals))
            if valid_mask.sum() > 2:
                X = np.vstack([np.ones(sum(valid_mask)), pred_vals[valid_mask]]).T
                y = obs_vals[valid_mask]
                try:
                    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                    slope = coeffs[1]
                    intercept = coeffs[0]
                except Exception:
                    slope = np.nan
                    intercept = np.nan
            else:
                slope = np.nan
                intercept = np.nan
            
            ibs_values = []
            for t_idx, t in enumerate(time_points[:-1]):
                if t_idx + 1 >= len(time_points):
                    break
                t_next = time_points[t_idx + 1]
                obs_surv_t = obs_interp[t_idx]
                pred_surv_t = np.mean([predicted_at_times.get(tt, 1.0) for tt in [t, t_next]])
                ibs_values.append((obs_surv_t - pred_surv_t) ** 2)
            ibs = np.mean(ibs_values) if ibs_values else np.nan
            
            calibration_data[model_name] = {
                'time_points': time_points,
                'predicted_survival': pred_vals,
                'observed_survival': obs_vals,
                'predicted_at_times': predicted_at_times,
                'overall_survival': overall_survival
            }
            calibration_metrics[model_name] = {
                'calibration_slope': slope,
                'calibration_intercept': intercept,
                'integrated_brier_score': ibs
            }
        except Exception as e:
            warnings.warn(f"Calibration assessment failed for {model_name}: {e}")
    
    return calibration_data, calibration_metrics


def create_model_validation_summary(model_results: Dict, 
                                   bootstrap_results: pd.DataFrame,
                                   calibration_metrics: Dict) -> pd.DataFrame:
    """
    Consolidate C-index, AIC, bootstrap HRs, and calibration metrics.
    
    Creates validation summary table for all models.
    """
    rows = []
    
    for model_name, res in model_results.items():
        row = {
            'Model': model_name,
            'AIC': res.get('aic', np.nan),
            'C_index': res.get('concordance', np.nan),
            'Log_likelihood': res.get('log_likelihood', np.nan)
        }
        
        exposure_col = None
        if 'gathering_event' in res.get('coefficients', {}):
            exposure_col = 'gathering_event'
        elif 'param_combo' in res.get('coefficients', {}):
            exposure_col = 'param_combo'
        
        for var in ['gathering_event', 'log_seed_size', 'pre_pressure_14d']:
            if var in res.get('coefficients', {}):
                bs_match = bootstrap_results[bootstrap_results['model'] == model_name] if bootstrap_results is not None else pd.DataFrame()
                bs_match = bs_match[bs_match['variable'].str.contains(var.split('_')[0])] if len(bs_match) > 0 else pd.DataFrame()
                if len(bs_match) > 0:
                    row[f'{var}_HR_mean'] = bs_match['hr_mean'].values[0]
                    row[f'{var}_HR_SE'] = bs_match['hr_se'].values[0]
                    row[f'{var}_CI_95_lower'] = bs_match['ci_lower_percentile'].values[0]
                    row[f'{var}_CI_95_upper'] = bs_match['ci_upper_percentile'].values[0]
        
        if model_name in calibration_metrics:
            cm = calibration_metrics[model_name]
            row['Calib_slope'] = cm.get('calibration_slope', np.nan)
            row['Calib_intercept'] = cm.get('calibration_intercept', np.nan)
            row['IBS'] = cm.get('integrated_brier_score', np.nan)
        
        rows.append(row)
    
    return pd.DataFrame(rows)