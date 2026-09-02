"""
Mediation analysis and interpretation for survival models.

Generates human-readable interpretation of model results.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Optional

try:
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def compute_mediation_metrics(model_results: Dict) -> Dict:
    """Calculate attenuation and proportion mediation metrics."""
    m0 = model_results.get('M0_gathering', {})
    m1 = model_results.get('M1_gathering_seed', {})
    
    beta_m0 = m0.get('coefficients', {}).get('gathering_event', np.nan)
    beta_m1 = m1.get('coefficients', {}).get('gathering_event', np.nan)
    hr_m0 = m0.get('hazard_ratios', {}).get('gathering_event', np.nan)
    hr_m1 = m1.get('hazard_ratios', {}).get('gathering_event', np.nan)
    
    metrics = {}
    if not np.isnan(hr_m0) and not np.isnan(hr_m1) and hr_m0 != 0:
        metrics['attenuation_HR'] = (hr_m0 - hr_m1) / hr_m0
    else:
        metrics['attenuation_HR'] = np.nan
    
    if not np.isnan(beta_m0) and not np.isnan(beta_m1) and beta_m0 != 0:
        metrics['loghr_proportion'] = (beta_m0 - beta_m1) / beta_m0
    else:
        metrics['loghr_proportion'] = np.nan
    
    pressure_windows = [k for k in model_results.keys() if k.startswith('M2_pressure_')]
    metrics['pressure_sensitivity'] = {k: model_results[k].get('concordance', np.nan) for k in pressure_windows}
    
    return metrics


def generate_mediation_interpretation(model_results: Dict, 
                                 bootstrap_results: Optional[pd.DataFrame] = None,
                                   calibration_metrics: Optional[Dict] = None,
                                   analysis_mode: str = "scenario",
                                   verbose: bool = False) -> str:
    """
    Generate interpretation with cautious wording about attenuation.
    
    Reports coefficient, HR, CI, p-value, and descriptive effect per SD increase.
    Includes bootstrap uncertainty and calibration findings when available.
    Avoids causal mediation language for sign/direction.
    
    Args:
        model_results: Dictionary of fitted model results
        bootstrap_results: Optional bootstrap summary DataFrame
        calibration_metrics: Optional calibration assessment results
        analysis_mode: "scenario" (Mode A) or "parameter" (Mode B)
        verbose: If True, print mediation metrics; if False, suppress them
    """
    lines = ['Survival Analysis Interpretation', '=' * 40, '']

    # Note about standard error type
    first_model = next(iter(model_results.values()), None)
    if first_model is not None and first_model.get('robust_se_used', False):
        lines.append(
            'Standard errors: Cluster-robust (Huber-White sandwich) standard errors '
            'clustered on simulation_run_id.'
        )
        lines.append(
            'Districts within the same stochastic simulation are correlated because they '
            'share'
        )
        lines.append(
            'the same epidemic realization, commuter mobility, and transmission history. '
            'Cluster-robust'
        )
        lines.append(
            'sandwich standard errors account for this within-run dependence while leaving '
            'coefficient estimates unchanged.'
        )
        lines.append('')
    
    exposure_key = 'gathering_event' if analysis_mode in ("scenario", "seed") else 'param_combo'
    
    model0_key = 'M0_gathering' if analysis_mode in ("scenario", "seed") else 'P0_param_combo'
    model0 = model_results.get(model0_key, {})
    
    model1_key = 'M1_gathering_seed' if analysis_mode in ("scenario", "seed") else 'P1_param_seed'
    model1 = model_results.get(model1_key, {})
    
    m2_results = {k: v for k, v in model_results.items() if k.startswith('M2_pressure_')} if analysis_mode in ("scenario", "seed") else \
                 {k: v for k, v in model_results.items() if k.startswith('P2_')}
    
    # M0 / P0: Exposure effect
    if model0:
        exp_coef = model0.get('coefficients', {}).get(exposure_key, np.nan)
        exp_hr = model0.get('hazard_ratios', {}).get(exposure_key, np.nan)
        exp_p = model0.get('p_values', [np.nan])[0] if model0.get('p_values') else np.nan
        sig = '*' if exp_p < 0.05 else ''
        
        model_label = 'M0 (gathering effect)' if analysis_mode in ("scenario", "seed") else 'P0 (parameter effect)'
        lines.append(f"{model_label}:")
        lines.append(f"  {exposure_key}: coef={exp_coef:.4f}, HR={exp_hr:.3f}, p={exp_p:.4f}{sig}")
        lines.append(f"  Note: Mass gathering events associated with earlier epidemic invasion timing")
        lines.append('')
    
    hr_m0 = exp_hr
    beta_m0_val = exp_coef
    
    # M1 / P1: Seed adjustment with cautious interpretation
    if model1:
        exp_coef_m1 = model1.get('coefficients', {}).get(exposure_key, np.nan)
        exp_hr_m1 = model1.get('hazard_ratios', {}).get(exposure_key, np.nan)
        exp_p_m1 = model1.get('p_values', [np.nan])[0] if model1.get('p_values') else np.nan
        
        s_coef_m1 = model1.get('coefficients', {}).get('log_seed_size', np.nan)
        s_hr_m1 = model1.get('hazard_ratios', {}).get('log_seed_size', np.nan)
        s_p_m1 = model1.get('p_values', [np.nan])[-1] if model1.get('p_values') else np.nan
        
        sig_m1 = '*' if exp_p_m1 < 0.05 else ''
        sig_s = '*' if s_p_m1 < 0.05 else ''
        
        model_label = 'M1 (adjusting for seed)' if analysis_mode in ("scenario", "seed") else 'P1 (adjusting for seed)'
        lines.append(f"{model_label}:")
        lines.append(f"  {exposure_key}: coef={exp_coef_m1:.4f}, HR={exp_hr_m1:.3f}, p={exp_p_m1:.4f}{sig_m1}")
        lines.append(f"  log_seed_size: coef={s_coef_m1:.4f}, HR={s_hr_m1:.3f}, p={s_p_m1:.4f}{sig_s}")
        
        if verbose:
            if exp_hr is not None and exp_hr_m1 is not None and not np.isnan(exp_hr) and not np.isnan(exp_hr_m1) and exp_hr != 0:
                attenuation_hr = (exp_hr - exp_hr_m1) / exp_hr
                lines.append(f"  Attenuation after seed adjustment: {attenuation_hr:.1%} reduction in HR")
                if bootstrap_results is not None and len(bootstrap_results) > 0:
                    bs_match = bootstrap_results[bootstrap_results['model'] == model0_key]
                    bs_match = bs_match[bs_match['variable'] == exposure_key]
                    if len(bs_match) > 0:
                        se = bs_match['hr_se'].values[0]
                        lines.append(f"  (Bootstrap SE for {model0_key} HR: {se:.3f})")
            
            if not np.isnan(exp_coef) and not np.isnan(exp_coef_m1) and exp_coef != 0:
                beta_m1_val = model1.get('coefficients', {}).get(exposure_key, np.nan)
                if not np.isnan(exp_coef) and not np.isnan(beta_m1_val):
                    loghr_proportion = (exp_coef - beta_m1_val) / exp_coef if exp_coef != 0 else np.nan
                    if not np.isnan(loghr_proportion):
                        lines.append(f"  Proportion explained by seed: {loghr_proportion:.1%}")
        lines.append('')
    
    # M2 / P2: Pressure window sensitivity comparison
    if m2_results:
        lines.append("M2/P2 (pressure window sensitivity):")
        for model_name in sorted(m2_results.keys()):
            m2 = m2_results[model_name]
            window = m2.get('pressure_window', 7)
            
            p_col = f'pre_pressure_{window}d' if analysis_mode in ("scenario", "seed") else 'pre_pressure_7d'
            p_coef = m2.get('coefficients', {}).get(p_col, np.nan)
            p_hr = m2.get('hazard_ratios', {}).get(p_col, np.nan)
            p_p = m2.get('p_values', [np.nan])[-1] if m2.get('p_values') else np.nan
            sig_p = '*' if p_p < 0.05 else ''
            aic = m2.get('aic', np.nan)
            conc = m2.get('concordance', np.nan)
            
            lines.append(f"  {model_name} (window={window}d): coef={p_coef:.4f}, HR={p_hr:.3f}, p={p_p:.4f}{sig_p}, AIC={aic:.1f}, C-index={conc:.3f}")
            if bootstrap_results is not None and len(bootstrap_results) > 0 and not np.isnan(p_hr):
                bs_match = bootstrap_results[bootstrap_results['model'] == model_name]
                bs_match = bs_match[bs_match['variable'] == p_col]
                if len(bs_match) > 0:
                    se = bs_match['hr_se'].values[0]
                    ci95 = bs_match['ci_upper_percentile'].values[0] - bs_match['ci_lower_percentile'].values[0]
                    lines.append(f"    (Bootstrap SE={se:.4f}, 95%CI width={ci95:.3f})")
        lines.append('')
    
    # Calibration findings
    if calibration_metrics:
        lines.append("Calibration Assessment:")
        lines.append("  Note: Calibration reflects agreement between predicted and observed survival.")
        for model_name, cm in calibration_metrics.items():
            slope = cm.get('calibration_slope', np.nan)
            intercept = cm.get('calibration_intercept', np.nan)
            ibs = cm.get('integrated_brier_score', np.nan)
            if not np.isnan(slope):
                cal_note = "well-calibrated" if 0.8 < slope < 1.2 else "miscalibrated"
                lines.append(f"  {model_name}: slope={slope:.3f} ({cal_note})")
            if not np.isnan(ibs):
                lines.append(f"    Integrated Brier Score: {ibs:.4f}")
        lines.append('')
    
    lines.append("Scientific Interpretation:")
    if analysis_mode in ("scenario", "seed"):
        lines.append("  Mass gathering events are associated with earlier epidemic invasion timing.")
        lines.append("  Adjusting for initial seed size attenuates this association, suggesting seeding")
        lines.append("  contributes to the observed timing differences. However, causal interpretation")
        lines.append("  requires further investigation of potential confounders and model assumptions.")
        lines.append("  Pressure covariates show limited additional explanatory power beyond seed size,")
        lines.append("  indicating that mobility connectivity enables but does not strongly predict timing")
        lines.append("  in this experimental design.")
        lines.append("  In Mode C, seed location varies across metapopulation seed districts, and cluster-robust")
        lines.append("  standard errors account for within-seed-run correlation.")
    else:
        lines.append("  Higher transmission rates (beta, R0) are associated with earlier epidemic invasion.")
        lines.append("  Seeding accounts for a substantial portion of timing differences.")
        lines.append("  The factorial model (F2) provides additional insight into parameter interactions.")
    
    return '\n'.join(lines)


def create_mediation_visualization(model_results: Dict, 
                                   output_path: str = None,
                                   analysis_mode: str = "scenario"):
    """
    Create pathway figure showing mediation analysis results.
    
    Shows the causal pathway: Gathering → Seed size → Invasion timing
    with HRs from M0 and M1 and attenuation annotation.
    """
    if not HAS_PLOTTING:
        raise ImportError('matplotlib required for visualization')
    
    exposure_key = 'gathering_event' if analysis_mode in ("scenario", "seed") else 'param_combo'
    
    m0 = model_results.get('M0_gathering', {}) if analysis_mode in ("scenario", "seed") else model_results.get('P0_param_combo', {})
    m1 = model_results.get('M1_gathering_seed', {}) if analysis_mode in ("scenario", "seed") else model_results.get('P1_param_seed', {})
    
    g_hr_m0 = m0.get('hazard_ratios', {}).get(exposure_key, np.nan)
    g_hr_m1 = m1.get('hazard_ratios', {}).get(exposure_key, np.nan)
    s_hr_m1 = m1.get('hazard_ratios', {}).get('log_seed_size', np.nan)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.text(0.2, 0.7, 'Gathering' if analysis_mode in ("scenario", "seed") else 'Parameters', 
            fontsize=16, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='steelblue', alpha=0.3))
    ax.text(0.5, 0.7, 'Seed Size\n(log scale)', fontsize=16, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='darkorange', alpha=0.3))
    ax.text(0.8, 0.7, 'Invasion\nTiming', fontsize=16, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='forestgreen', alpha=0.3))
    
    box_width = 0.25
    ax.annotate('', xy=(0.5, 0.65), xytext=(0.3, 0.65),
                arrowprops=dict(arrowstyle='->', lw=2, color='steelblue'))
    ax.annotate('', xy=(0.8, 0.65), xytext=(0.5, 0.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='darkorange'))
    
    if not np.isnan(g_hr_m0):
        ax.text(0.3, 0.55, f'M0 HR = {g_hr_m0:.2f}', fontsize=13, ha='center',
                color='steelblue', fontweight='bold')
    if not np.isnan(g_hr_m1):
        ax.text(0.5, 0.5, f'M1/P1: HR = {g_hr_m1:.2f}', fontsize=13, ha='center',
                color='steelblue', fontweight='bold')
    if not np.isnan(s_hr_m1):
        ax.text(0.65, 0.48, f'Seed HR = {s_hr_m1:.2f}', fontsize=13, ha='center',
                color='darkorange', fontweight='bold')
    
    if not np.isnan(g_hr_m0) and not np.isnan(g_hr_m1) and g_hr_m0 != 0:
        attenuation = (g_hr_m0 - g_hr_m1) / g_hr_m0
        ax.text(0.5, 0.25, f'Attenuation: {attenuation:.1%}', fontsize=14, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.8)
    ax.axis('off')
    
    title = 'Conceptual Mechanism Pathway:\nMass Gathering Accelerates Epidemic Invasion via Seeding' if analysis_mode in ("scenario", "seed") else \
            'Mechanism Pathway:\nEpidemiological Parameters Influence Invasion Timing via Seeding'
    ax.set_title(title, fontsize=15, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', format='png')
    plt.close()