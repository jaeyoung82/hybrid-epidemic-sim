"""
Forest plot and visualization functions for survival analysis.

Generates publication-quality hazard ratio plots with confidence intervals.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Mapping from model keys to human-readable labels for axis ticks
MODEL_LABELS = {
    'M0_gathering': 'M0',
    'M1_gathering_seed': 'M1',
    'M2_pressure_3d': 'M2 (3d)',
    'M2_pressure_7d': 'M2 (7d)',
    'M2_pressure_14d': 'M2 (14d)',
    'P0_param_combo': 'P0',
    'P1_param_seed': 'P1',
    'P2_param_seed_pressure': 'P2',
}


def create_forest_plot_hazard_ratios(model_results: Dict, 
                                     output_path: str = None,
                                     grouping_col: str = 'scenario',
                                     use_robust: bool = True):
    """Create publication-quality vertical forest plot of hazard ratios from Cox models.

    Districts within the same stochastic simulation are correlated because they share
    the same epidemic realization, commuter mobility, and transmission history.
    Cluster-robust sandwich standard errors account for this within-run dependence
    while leaving coefficient estimates unchanged.

    Args:
        model_results: Dictionary of fitted model results.
        output_path: Base path for output files (PNG and PDF saved).
        grouping_col: 'scenario' for Mode A or 'param_combo' for Mode B.
        use_robust: If True (default), use cluster-robust CIs. If False, use
            naive model-based CIs.
    """
    if not HAS_PLOTTING:
        raise ImportError("matplotlib required for plotting")
    
    if grouping_col == 'scenario' or grouping_col == 'seed_location':
        # Mode A and Mode C use the same model hierarchy (M0/M1/M2)
        model_hierarchy = [
            ('M0_gathering', 'M0: Gathering only', 'steelblue', ['gathering_event']),
            ('M1_gathering_seed', 'M1: Gathering + Seed', 'darkorange', ['gathering_event', 'log_seed_size']),
            ('M2_pressure_14d', 'M2: Gathering + Seed + Pressure', 'forestgreen', 
             ['gathering_event', 'log_seed_size', 'pre_pressure_14d'])
        ]
    else:
        model_hierarchy = [
            ('P0_param_combo', 'P0: Parameter Combo', 'steelblue', ['param_combo']),
            ('P1_param_seed', 'P1: Parameter + Seed', 'darkorange', ['param_combo', 'log_seed_size']),
            ('P2_param_seed_pressure', 'P2: Parameter + Seed + Pressure', 'forestgreen', 
             ['param_combo', 'log_seed_size', 'pre_pressure_7d'])
        ]
    
    entries = []
    y_pos = 0
    model_gap_ys = []  # Track gap positions between model groups
    
    for model_name, model_header, model_color, vars_list in model_hierarchy:
        model = model_results.get(model_name, {})
        if not model:
            continue
        
        coefficients = model.get('coefficients', {})
        hazard_ratios = model.get('hazard_ratios', {})
        if use_robust:
            ci_lower_list = model.get('ci_lower', [])
            ci_upper_list = model.get('ci_upper', [])
        else:
            ci_lower_list = model.get('naive_ci_lower', [])
            ci_upper_list = model.get('naive_ci_upper', [])
        
        entries.append({
            'y': y_pos,
            'is_header': True,
            'label': model_header,
            'color': model_color
        })
        y_pos += 1
        
        for var in vars_list:
            matching_vars = [v for v in coefficients.keys() if var in v or v == var]
            if not matching_vars:
                continue
            actual_var = matching_vars[0]
            
            hr = hazard_ratios.get(actual_var, np.nan)
            if np.isnan(hr):
                continue
            
            idx = list(coefficients.keys()).index(actual_var)
            lower = ci_lower_list[idx] if ci_lower_list and idx < len(ci_lower_list) else np.nan
            upper = ci_upper_list[idx] if ci_upper_list and idx < len(ci_upper_list) else np.nan
            
            var_label = 'Gathering event' if var == 'gathering_event' else \
                        ('log(seed size)' if var == 'log_seed_size' else 'Pre-invasion pressure (14 d)')
            
            entries.append({
                'y': y_pos,
                'is_header': False,
                'label': f"    {var_label}",
                'hr': hr,
                'ci_lower': lower,
                'ci_upper': upper,
                'color': model_color
            })
            y_pos += 1
        model_gap_ys.append(y_pos)
        y_pos += 0.3
    
    if not entries or all(e.get('is_header', False) for e in entries):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'No model results available', ha='center', va='center',
                fontsize=14, transform=ax.transAxes)
        ax.set_xlim(0.5, 2)
        ax.set_ylim(0, 1)
        ax.axis('off')
        base = output_path.rsplit('.', 1)[0] if output_path else 'survival_analysis_results/forest_plot_hazard_ratios'
        fig.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
        fig.savefig(f'{base}.pdf', bbox_inches='tight')
        plt.close()
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for entry in entries:
        if not entry.get('is_header', False) and not np.isnan(entry['hr']):
            ci_lower = entry['ci_lower'] if not np.isnan(entry['ci_lower']) else entry['hr']
            ci_upper = entry['ci_upper'] if not np.isnan(entry['ci_upper']) else entry['hr']
            
            ax.plot([ci_lower, ci_upper], [entry['y'], entry['y']],
                    color=entry['color'], linewidth=2)
            ax.plot(entry['hr'], entry['y'], 'o', color=entry['color'], markersize=12,
                    markerfacecolor='none', markeredgewidth=2)
            
            ci_str = f"({entry['ci_lower']:.2f}–{entry['ci_upper']:.2f})" if not np.isnan(entry['ci_lower']) else ""
            ax.text(entry['ci_upper'] * 1.2 if not np.isnan(entry['ci_upper']) else entry['hr'] * 1.3,
                    entry['y'], f"{entry['hr']:.2f} {ci_str}",
                    va='center', ha='left', fontsize=10)
    
    y_ticks = [e['y'] for e in entries]
    y_labels = [e['label'] for e in entries]
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=11)
    
    for i, label in enumerate(y_labels):
        if not label.startswith(' '):
            ax.get_yticklabels()[i].set_fontweight('bold')

    # Invert y-axis so M0/M1/M2 appear from top to bottom
    ax.invert_yaxis()

    # Expand y-limits with padding so all labels (especially the topmost M0) are fully visible
    y_min = min(e['y'] for e in entries)
    y_max = max(e['y'] for e in entries)
    ax.set_ylim(y_max + 0.5, y_min - 0.5)

    # Draw horizontal separator lines between model groups (behind plot elements)
    for gap_y in model_gap_ys:
        ax.axhline(y=gap_y + 0.15, color='gray', linestyle='-', linewidth=0.8, alpha=0.4, zorder=0)

    ax.set_xlabel('Hazard Ratio (log scale)', fontsize=12)
    
    if use_robust:
        ax.set_title('Hazard Ratios from Sequential Cox Models', fontsize=13, fontweight='bold', pad=15)
    else:
        ax.set_title('Hazard Ratios from Sequential Cox Models (Naive SEs)', fontsize=13, fontweight='bold', pad=15)
    
    ax.set_xscale('log')
    ax.set_xlim(0.4, 5)
    ax.set_xticks([0.5, 1, 2, 3, 4])
    
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    base = output_path.rsplit('.', 1)[0] if output_path else 'survival_analysis_results/forest_plot_hazard_ratios'
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight', format='png')
    plt.savefig(f'{base}.pdf', bbox_inches='tight', format='pdf')
    plt.close()


def create_discrimination_performance_plot(model_results: Dict, 
                                           output_path: str = None):
    if not HAS_PLOTTING:
        raise ImportError("matplotlib required for plotting")
    
    models_to_plot = ['M0_gathering', 'M1_gathering_seed', 'M2_pressure_3d', 'M2_pressure_7d', 'M2_pressure_14d']
    available_models = [m for m in models_to_plot if m in model_results]
    
    if len(available_models) == 0:
        param_models = [m for m in model_results.keys() if m.startswith('P')]
        available_models = param_models
    
    if len(available_models) == 0:
        return
    
    concordances = [model_results[m].get('concordance', np.nan) for m in available_models]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    model_colors = ['steelblue', 'darkorange', 'forestgreen', 'forestgreen', 'forestgreen']
    
    x_pos = np.arange(len(available_models))
    colors = [model_colors[min(i, len(model_colors)-1)] for i in range(len(available_models))]
    ax.bar(x_pos, concordances, color=colors, edgecolor='black', linewidth=1)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='No discrimination')
    
    ax.set_xticks(x_pos)
    display_labels = [MODEL_LABELS.get(m, m) for m in available_models]
    ax.set_xticklabels(display_labels, fontsize=11)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Concordance Index (C-index)', fontsize=12)
    ax.set_title('Discrimination Performance', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, c in zip(ax.patches, concordances):
        if not np.isnan(c):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{c:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        base = output_path.rsplit('.', 1)[0]
    else:
        base = 'survival_analysis_results/model_discrimination_performance'
    
    plt.savefig(f'{base}.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.close()


def create_model_parsimony_plot(model_results: Dict,
                                 output_path: str = None):
    """Create model parsimony plot showing AIC difference relative to best model."""
    if not HAS_PLOTTING:
        raise ImportError("matplotlib required for plotting")
    
    models_to_plot = ['M0_gathering', 'M1_gathering_seed', 'M2_pressure_3d', 'M2_pressure_7d', 'M2_pressure_14d']
    available_models = [m for m in models_to_plot if m in model_results]
    
    if len(available_models) == 0:
        param_models = [m for m in model_results.keys() if m.startswith('P')]
        available_models = param_models
    
    if len(available_models) == 0:
        return
    
    aics = [model_results[m].get('aic', np.nan) for m in available_models]
    
    valid_aics = [a for a in aics if not np.isnan(a)]
    if not valid_aics:
        return
    
    # Compute delta AIC (relative to best / lowest AIC)
    best_aic = min(valid_aics)
    delta_aics = [a - best_aic if not np.isnan(a) else np.nan for a in aics]
    max_delta = max(d for d in delta_aics if not np.isnan(d))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    model_colors = ['steelblue', 'darkorange', 'forestgreen', 'forestgreen', 'forestgreen']
    colors = [model_colors[min(i, len(model_colors)-1)] for i in range(len(available_models))]
    
    x_pos = np.arange(len(available_models))
    ax.bar(x_pos, delta_aics, color=colors, edgecolor='black', linewidth=1)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, label='Best model')
    
    ax.set_xticks(x_pos)
    display_labels = [MODEL_LABELS.get(m, m) for m in available_models]
    ax.set_xticklabels(display_labels, fontsize=11)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(r'$\Delta$AIC (relative to best)', fontsize=12)
    ax.set_title('Model Parsimony (AIC)', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, d in zip(ax.patches, delta_aics):
        if not np.isnan(d):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_delta * 0.02,
                    f'{d:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        base = output_path.rsplit('.', 1)[0]
    else:
        base = 'survival_analysis_results/model_parsimony'
    
    plt.savefig(f'{base}.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.close()


def create_parameter_sensitivity_heatmap(survival_df: pd.DataFrame, 
                                         output_path: str = None):
    if not HAS_PLOTTING:
        raise ImportError("matplotlib required for plotting")
    
    if 'R0' not in survival_df.columns or 'beta' not in survival_df.columns:
        warnings.warn("R0 and beta columns required for sensitivity heatmap")
        return
    
    summary = survival_df.groupby(['R0', 'beta']).agg({
        'arrival_time': 'median',
        'event': 'mean'
    }).reset_index()
    summary.columns = ['R0', 'beta', 'median_arrival_time', 'invasion_rate']
    
    r0_values = sorted(summary['R0'].unique())
    beta_values = sorted(summary['beta'].unique())
    
    median_matrix = np.zeros((len(r0_values), len(beta_values)))
    for _, row in summary.iterrows():
        i = r0_values.index(row['R0'])
        j = beta_values.index(row['beta'])
        median_matrix[i, j] = row['median_arrival_time']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(median_matrix, aspect='auto', cmap='RdYlGn_r')
    ax.invert_yaxis()  # Lower R0 at bottom, higher R0 at top
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Median Invasion Time (days)')
    
    ax.set_xticks(np.arange(len(beta_values)))
    ax.set_yticks(np.arange(len(r0_values)))
    ax.set_xticklabels([f'{b:.2f}' for b in beta_values], fontsize=10)
    ax.set_yticklabels([f'{r:.1f}' for r in r0_values], fontsize=10)
    
    ax.set_xlabel('Beta (Transmission Rate)', fontsize=12)
    ax.set_ylabel('R0 (Reproduction Number)', fontsize=12)
    ax.set_title('Parameter Sensitivity: Median Invasion Time', fontsize=14, fontweight='bold')
    
    for i in range(len(r0_values)):
        for j in range(len(beta_values)):
            if median_matrix[i, j] > 0:
                ax.text(j, i, f'{median_matrix[i, j]:.1f}', ha='center', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        base = output_path.rsplit('.', 1)[0]
    else:
        base = 'survival_analysis_results/parameter_sensitivity_heatmap'
    
    plt.savefig(f'{base}.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.close()


def extract_param_combo_hrs(model_results: Dict, model_name: str, 
                            survival_df: pd.DataFrame,
                            all_r0: List[float], all_beta: List[float],
                            scenario_prefix: str = 'AMS_dance',
                            use_robust: bool = True) -> pd.DataFrame:
    """Extract hazard ratios and CIs for parameter combinations from a model result.

    Args:
        use_robust: If True (default), extract cluster-robust CIs. If False,
            extract naive model-based CIs.
    """
    import re
    model = model_results.get(model_name, {})
    
    # If model failed to converge, still create records with reference values
    if not model:
        actual_combos = survival_df[['R0', 'beta']].drop_duplicates().sort_values(['R0', 'beta'])
        records = []
        for _, row in actual_combos.iterrows():
            records.append({
                'param_combo': f'{scenario_prefix}_R{row['R0']}_beta{row['beta']}',
                'R0': row['R0'],
                'beta': row['beta'],
                'hr': 1.0,
                'ci_lower': 1.0,
                'ci_upper': 1.0
            })
        return pd.DataFrame(records)
    
    coefficients = model.get('coefficients', {})
    hazard_ratios = model.get('hazard_ratios', {})
    if use_robust:
        ci_lower_list = model.get('ci_lower', [])
        ci_upper_list = model.get('ci_upper', [])
        summary_df = model.get('summary', pd.DataFrame())
    else:
        ci_lower_list = model.get('naive_ci_lower', [])
        ci_upper_list = model.get('naive_ci_upper', [])
        summary_df = model.get('naive_summary', pd.DataFrame())
    
    records = []
    
    hr_lookup = {}
    for key, hr in hazard_ratios.items():
        r_match = re.search(r'R(\d+\.?\d*)_beta(\d+\.?\d*)', key)
        if r_match:
            r0_val = float(r_match.group(1))
            beta_val = float(r_match.group(2))
            hr_lookup[(r0_val, beta_val)] = (hr, key)
    
    actual_combos = survival_df[['R0', 'beta']].drop_duplicates().sort_values(['R0', 'beta'])
    actual_combos = [(float(r), float(b)) for r, b in zip(actual_combos['R0'], actual_combos['beta'])]
    
    for r0_val, beta_val in actual_combos:
        if (r0_val, beta_val) in hr_lookup:
            hr, key = hr_lookup[(r0_val, beta_val)]
            if key in summary_df.index:
                lower = summary_df.loc[key, 'exp(coef) lower 95%'] if 'exp(coef) lower 95%' in summary_df.columns else np.nan
                upper = summary_df.loc[key, 'exp(coef) upper 95%'] if 'exp(coef) upper 95%' in summary_df.columns else np.nan
            else:
                idx = list(coefficients.keys()).index(key) if key in coefficients else -1
                lower = ci_lower_list[idx] if 0 <= idx < len(ci_lower_list) else np.nan
                upper = ci_upper_list[idx] if 0 <= idx < len(ci_upper_list) else np.nan
            records.append({
                'param_combo': f'{scenario_prefix}_R{r0_val}_beta{beta_val}',
                'R0': r0_val,
                'beta': beta_val,
                'hr': float(hr),
                'ci_lower': float(lower),
                'ci_upper': float(upper),
                'ci_margin': float((upper - lower) / 2) if not np.isnan(lower) and not np.isnan(upper) else np.nan
            })
        else:
            records.append({
                'param_combo': f'{scenario_prefix}_R{r0_val}_beta{beta_val}',
                'R0': r0_val,
                'beta': beta_val,
                'hr': 1.0,
                'ci_lower': 1.0,
                'ci_upper': 1.0,
                'ci_margin': 0.0
            })
    
    return pd.DataFrame(records)


def create_parameter_hazard_heatmap(model_results: Dict, 
                                     survival_df: pd.DataFrame,
                                     output_path: str = None):
    if not HAS_PLOTTING:
        raise ImportError("matplotlib required for plotting")
    
    r0_values = sorted(survival_df['R0'].dropna().unique().tolist())
    beta_values = sorted(survival_df['beta'].dropna().unique().tolist())
    
    hr_p0 = extract_param_combo_hrs(model_results, 'P0_param_combo', survival_df, r0_values, beta_values)
    hr_p1 = extract_param_combo_hrs(model_results, 'P1_param_seed', survival_df, r0_values, beta_values)
    
    if hr_p0.empty or hr_p1.empty:
        warnings.warn(f"Could not extract HR values. P0_empty={hr_p0.empty}, P1_empty={hr_p1.empty}, hr_p0_shape={hr_p0.shape}, hr_p1_shape={hr_p1.shape}")
        return
    
    merged = hr_p0.merge(hr_p1, on=['R0', 'beta'], suffixes=('_p0', '_p1'))
    
    merged['attenuation'] = merged['hr_p1'] / merged['hr_p0']
    
    p0_matrix = np.zeros((len(r0_values), len(beta_values)))
    p1_matrix = np.zeros((len(r0_values), len(beta_values)))
    atten_matrix = np.zeros((len(r0_values), len(beta_values)))
    
    for _, row in merged.iterrows():
        i = r0_values.index(row['R0'])
        j = beta_values.index(row['beta'])
        p0_matrix[i, j] = row['hr_p0']
        p1_matrix[i, j] = row['hr_p1']
        atten_matrix[i, j] = row['attenuation']
    
    all_hrs = np.concatenate([p0_matrix.flatten(), p1_matrix.flatten()])
    all_hrs = all_hrs[~np.isnan(all_hrs)]
    vmax = max(all_hrs.max(), 3.0)
    vmin = min(all_hrs.min(), 0.5)
    if vmin > 1.0:
        vmin = 0.5
    if vmax < 1.0:
        vmax = 2.0
    
    atten_vmax = atten_matrix.max()
    atten_vmin = atten_matrix.min()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    
    ax_a = axes[0]
    im_a = ax_a.imshow(p0_matrix, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax_a.invert_yaxis()  # Larger R0 at top, smaller R0 at bottom
    ax_a.set_xticks(np.arange(len(beta_values)))
    ax_a.set_yticks(np.arange(len(r0_values)))
    ax_a.set_xticklabels([f'{b:.1f}' for b in beta_values], fontsize=12)
    ax_a.set_yticklabels([f'{r:.1f}' for r in r0_values], fontsize=12)
    ax_a.set_xlabel('beta_event (Transmission probability)', fontsize=12)
    ax_a.set_ylabel('R0 (Reproduction number)', fontsize=12)
    ax_a.set_title('(a) P0: Parameter effect', fontsize=12, fontweight='bold')
    
    for i in range(len(r0_values)):
        for j in range(len(beta_values)):
            hr = p0_matrix[i, j]
            if not np.isnan(hr):
                norm_val = (hr - vmin) / (vmax - vmin)
                text_color = 'white' if norm_val < 0.3 or norm_val > 0.7 else 'black'
                ax_a.text(j, i, f'{hr:.2f}', ha='center', va='center', 
                         fontsize=13, fontweight='bold', color=text_color)
    
    ax_b = axes[1]
    im_b = ax_b.imshow(p1_matrix, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax_b.invert_yaxis()  # Larger R0 at top, smaller R0 at bottom
    ax_b.set_xticks(np.arange(len(beta_values)))
    ax_b.set_yticks(np.arange(len(r0_values)))
    ax_b.set_xticklabels([f'{b:.1f}' for b in beta_values], fontsize=12)
    ax_b.set_yticklabels([f'{r:.1f}' for r in r0_values], fontsize=12)
    ax_b.set_xlabel('beta_event (Transmission probability)', fontsize=12)
    ax_b.set_title('(b) P1: Parameter + seed effect', fontsize=12, fontweight='bold')
    
    for i in range(len(r0_values)):
        for j in range(len(beta_values)):
            hr = p1_matrix[i, j]
            if not np.isnan(hr):
                norm_val = (hr - vmin) / (vmax - vmin)
                text_color = 'white' if norm_val < 0.3 or norm_val > 0.7 else 'black'
                ax_b.text(j, i, f'{hr:.2f}', ha='center', va='center',
                         fontsize=13, fontweight='bold', color=text_color)
    
    ax_c = axes[2]
    im_c = ax_c.imshow(atten_matrix, aspect='auto', cmap='RdBu_r',
                       vmin=atten_vmin, vmax=atten_vmax)
    ax_c.invert_yaxis()  # Larger R0 at top, smaller R0 at bottom
    ax_c.set_xticks(np.arange(len(beta_values)))
    ax_c.set_yticks(np.arange(len(r0_values)))
    ax_c.set_xticklabels([f'{b:.1f}' for b in beta_values], fontsize=12)
    ax_c.set_yticklabels([f'{r:.1f}' for r in r0_values], fontsize=12)
    ax_c.set_xlabel('beta_event (Transmission probability)', fontsize=12)
    ax_c.set_title('(c) Relative change in hazard ratio', fontsize=12, fontweight='bold')
    
    for i in range(len(r0_values)):
        for j in range(len(beta_values)):
            att = atten_matrix[i, j]
            if not np.isnan(att):
                norm_val = (att - atten_vmin) / (atten_vmax - atten_vmin)
                text_color = 'white' if norm_val < 0.3 or norm_val > 0.7 else 'black'
                ax_c.text(j, i, f'{att:.2f}', ha='center', va='center',
                         fontsize=13, fontweight='bold', color=text_color)
    
    cbar_a = fig.colorbar(im_a, ax=ax_a, orientation='horizontal', shrink=0.8, aspect=40, pad=0.15)
    cbar_a.set_label('Hazard Ratio', fontsize=11)
    
    cbar_b = fig.colorbar(im_b, ax=ax_b, orientation='horizontal', shrink=0.8, aspect=40, pad=0.15)
    cbar_b.set_label('Hazard Ratio', fontsize=11)
    
    cbar_c = fig.colorbar(im_c, ax=ax_c, orientation='horizontal', shrink=0.8, aspect=40, pad=0.15)
    cbar_c.set_label('HR(P1)/HR(P0)', fontsize=11)
    
    fig.suptitle('Parameter sensitivity before and after adjustment for initial epidemic seed size',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if output_path:
        base = output_path.rsplit('.', 1)[0]
    else:
        base = 'survival_analysis_results/parameter_hazard_heatmap'
    
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight', format='png')
    plt.savefig(f'{base}.pdf', bbox_inches='tight', format='pdf')
    plt.close()


def _get_ci_bounds(row, ci_low_col, ci_up_col, ci_margin_col, hr):
    """Return (ci_lower, ci_upper) for a datapoint, preferring stored bounds.

    Falls back to a symmetric margin. On a log-scaled axis, drawing between the
    true bounds keeps intervals correctly shaped (avoids clamping at zero).
    Returns (None, None) when no CI is available.
    """
    ci_low = row.get(ci_low_col, np.nan)
    ci_up = row.get(ci_up_col, np.nan)
    if not np.isnan(ci_low) and not np.isnan(ci_up) and ci_low > 0:
        return float(ci_low), float(ci_up)
    ci_margin = row.get(ci_margin_col, np.nan)
    if not np.isnan(ci_margin) and not np.isnan(hr):
        lo = max(1e-3, hr - ci_margin)
        return float(lo), float(hr + ci_margin)
    return None, None


def create_parameter_forest_plot(model_results: Dict, 
                                     survival_df: pd.DataFrame,
                                     output_path: str = None,
                                     r0_values: List[float] = None,
                                     use_robust: bool = True):
    if not HAS_PLOTTING:
        raise ImportError("matplotlib required for plotting")
    
    if r0_values is None:
        r0_values = sorted(survival_df['R0'].dropna().unique().tolist())
    
    beta_values = sorted(survival_df['beta'].dropna().unique().tolist())
    
    hr_p0 = extract_param_combo_hrs(model_results, 'P0_param_combo', survival_df, r0_values, beta_values, use_robust=use_robust)
    hr_p1 = extract_param_combo_hrs(model_results, 'P1_param_seed', survival_df, r0_values, beta_values, use_robust=use_robust)
    
    if hr_p0.empty or hr_p1.empty:
        warnings.warn("Could not extract HR values from model results")
        return
    
    merged = hr_p0.merge(hr_p1, on=['R0', 'beta'], suffixes=('_p0', '_p1'))
    
    # Calculate symmetric CI endpoints for axis limits using ci_margin or fallback to bounds
    symmetric_mins = []
    symmetric_maxs = []
    for _, row in merged.iterrows():
        for hr_col, ci_margin_col, ci_low_col, ci_up_col in [
            ('hr_p0', 'ci_margin_p0', 'ci_lower_p0', 'ci_upper_p0'), 
            ('hr_p1', 'ci_margin_p1', 'ci_lower_p1', 'ci_upper_p1')
        ]:
            hr = row[hr_col]
            ci_margin = np.nan
            if ci_margin_col in merged.columns:
                ci_margin = row[ci_margin_col]
            elif ci_low_col in merged.columns and ci_up_col in merged.columns:
                ci_low = row[ci_low_col]
                ci_up = row[ci_up_col]
                if not np.isnan(ci_low) and not np.isnan(ci_up):
                    ci_margin = (ci_up - ci_low) / 2
            if not np.isnan(ci_margin) and not np.isnan(hr):
                symmetric_mins.append(max(0.01, hr - ci_margin))
                symmetric_maxs.append(hr + ci_margin)
     
    all_vals = list(merged['hr_p0'].values) + list(merged['hr_p1'].values)
    all_vals = [v for v in all_vals if not np.isnan(v)]
    all_vals.extend(symmetric_mins)
    all_vals.extend(symmetric_maxs)
    hr_min = max(0.1, min(all_vals) * 0.8)
    hr_max = min(10, max(all_vals) * 1.2)
    
    n_panels = len(r0_values)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 10), sharex=True)
    
    if n_panels == 1:
        axes = [axes]
    
    p0_color = '#1f77b4'
    p1_color = '#ff7f0e'
    
    for ax_idx, r0 in enumerate(r0_values):
        ax = axes[ax_idx]
        
        subset = merged[merged['R0'] == r0].sort_values('beta')
        
        y_positions = np.arange(len(beta_values))[::-1]
        
        for y_pos, (_, row) in enumerate(zip(y_positions, [subset[subset['beta'] == b] for b in beta_values])):
            if len(row) == 0:
                continue
            row = row.iloc[0]
            
            hr0 = row['hr_p0']
            ci_low0, ci_up0 = _get_ci_bounds(row, 'ci_lower_p0', 'ci_upper_p0', 'ci_margin_p0', hr0)
            if ci_low0 is not None and ci_up0 is not None and ci_up0 > ci_low0:
                ax.plot([ci_low0, ci_up0], [y_pos + 0.15, y_pos + 0.15], color=p0_color, linewidth=3, zorder=1)
                ax.plot([ci_low0, ci_low0], [y_pos + 0.10, y_pos + 0.20], color=p0_color, linewidth=3, zorder=1)
                ax.plot([ci_up0, ci_up0], [y_pos + 0.10, y_pos + 0.20], color=p0_color, linewidth=3, zorder=1)
            ax.plot(hr0, y_pos + 0.15, 'o', color=p0_color, markersize=10,
                    markerfacecolor='white', markeredgewidth=2, zorder=2)

            hr1 = row['hr_p1']
            ci_low1, ci_up1 = _get_ci_bounds(row, 'ci_lower_p1', 'ci_upper_p1', 'ci_margin_p1', hr1)
            if ci_low1 is not None and ci_up1 is not None and ci_up1 > ci_low1:
                ax.plot([ci_low1, ci_up1], [y_pos - 0.15, y_pos - 0.15], color=p1_color, linewidth=3, zorder=1)
                ax.plot([ci_low1, ci_low1], [y_pos - 0.20, y_pos - 0.10], color=p1_color, linewidth=3, zorder=1)
                ax.plot([ci_up1, ci_up1], [y_pos - 0.20, y_pos - 0.10], color=p1_color, linewidth=3, zorder=1)
            ax.plot(hr1, y_pos - 0.15, 's', color=p1_color, markersize=10,
                    markerfacecolor='white', markeredgewidth=2, zorder=2)
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f'{b:.2f}' for b in beta_values][::-1], fontsize=12)
        ax.set_ylim(-1.0, len(beta_values) - 0.0)
        ax.set_xlim(hr_min, hr_max)
        ax.set_xscale('log')
        ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.5)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if ax_idx == 0:
            ax.set_title('Forest Plot of Hazard Ratios by R0', fontsize=14, fontweight='bold', pad=15)
        
        ax.set_ylabel(f'R0 = {r0:.1f}', fontsize=12, fontweight='bold')
    
    axes[-1].set_xlabel('Hazard Ratio (log scale)', fontsize=12)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor=p0_color,
               markersize=10, markeredgewidth=2, label='P0 (parameter only)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='none', markeredgecolor=p1_color,
               markersize=10, markeredgewidth=2, label='P1 (parameter + seed)')
    ]
    axes[0].legend(handles=legend_elements, loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        base = output_path.rsplit('.', 1)[0]
    else:
        base = 'survival_analysis_results/parameter_forest_plot'
    
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight', format='png')
    plt.savefig(f'{base}.pdf', bbox_inches='tight', format='pdf')
    plt.close()


def create_parameter_forest_plot_by_beta(model_results: Dict, 
                                           survival_df: pd.DataFrame,
                                           output_path: str = None,
                                           beta_values: List[float] = None,
                                           use_robust: bool = True):
    if not HAS_PLOTTING:
        raise ImportError("matplotlib required for plotting")
    
    if beta_values is None:
        beta_values = sorted(survival_df['beta'].dropna().unique().tolist())
    
    r0_values = sorted(survival_df['R0'].dropna().unique().tolist())
    
    hr_p0 = extract_param_combo_hrs(model_results, 'P0_param_combo', survival_df, r0_values, beta_values, use_robust=use_robust)
    hr_p1 = extract_param_combo_hrs(model_results, 'P1_param_seed', survival_df, r0_values, beta_values, use_robust=use_robust)
    
    if hr_p0.empty or hr_p1.empty:
        warnings.warn("Could not extract HR values from model results")
        return
    
    merged = hr_p0.merge(hr_p1, on=['R0', 'beta'], suffixes=('_p0', '_p1'))
    
    # Calculate symmetric CI endpoints for axis limits using ci_margin or fallback to bounds
    symmetric_mins = []
    symmetric_maxs = []
    for _, row in merged.iterrows():
        for hr_col, ci_margin_col, ci_low_col, ci_up_col in [
            ('hr_p0', 'ci_margin_p0', 'ci_lower_p0', 'ci_upper_p0'), 
            ('hr_p1', 'ci_margin_p1', 'ci_lower_p1', 'ci_upper_p1')
        ]:
            hr = row[hr_col]
            ci_margin = np.nan
            if ci_margin_col in merged.columns:
                ci_margin = row[ci_margin_col]
            elif ci_low_col in merged.columns and ci_up_col in merged.columns:
                ci_low = row[ci_low_col]
                ci_up = row[ci_up_col]
                if not np.isnan(ci_low) and not np.isnan(ci_up):
                    ci_margin = (ci_up - ci_low) / 2
            if not np.isnan(ci_margin) and not np.isnan(hr):
                symmetric_mins.append(max(0.01, hr - ci_margin))
                symmetric_maxs.append(hr + ci_margin)
     
    all_vals = list(merged['hr_p0'].values) + list(merged['hr_p1'].values)
    all_vals = [v for v in all_vals if not np.isnan(v)]
    all_vals.extend(symmetric_mins)
    all_vals.extend(symmetric_maxs)
    hr_min = max(0.1, min(all_vals) * 0.8)
    hr_max = min(10, max(all_vals) * 1.2)
    
    n_panels = len(beta_values)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 10), sharex=True)
    
    if n_panels == 1:
        axes = [axes]
    
    p0_color = '#1f77b4'
    p1_color = '#ff7f0e'
    
    for ax_idx, beta in enumerate(beta_values):
        ax = axes[ax_idx]
        
        subset = merged[merged['beta'] == beta].sort_values('R0')
        
        y_positions = np.arange(len(r0_values))[::-1]
        
        for y_pos, (_, row) in enumerate(zip(y_positions, [subset[subset['R0'] == r] for r in r0_values])):
            if len(row) == 0:
                continue
            row = row.iloc[0]
            
            hr0 = row['hr_p0']
            ci_low0, ci_up0 = _get_ci_bounds(row, 'ci_lower_p0', 'ci_upper_p0', 'ci_margin_p0', hr0)
            if ci_low0 is not None and ci_up0 is not None and ci_up0 > ci_low0:
                ax.plot([ci_low0, ci_up0], [y_pos + 0.15, y_pos + 0.15], color=p0_color, linewidth=3, zorder=1)
                ax.plot([ci_low0, ci_low0], [y_pos + 0.10, y_pos + 0.20], color=p0_color, linewidth=3, zorder=1)
                ax.plot([ci_up0, ci_up0], [y_pos + 0.10, y_pos + 0.20], color=p0_color, linewidth=3, zorder=1)
            ax.plot(hr0, y_pos + 0.15, 'o', color=p0_color, markersize=10,
                    markerfacecolor='white', markeredgewidth=2, zorder=2)

            hr1 = row['hr_p1']
            ci_low1, ci_up1 = _get_ci_bounds(row, 'ci_lower_p1', 'ci_upper_p1', 'ci_margin_p1', hr1)
            if ci_low1 is not None and ci_up1 is not None and ci_up1 > ci_low1:
                ax.plot([ci_low1, ci_up1], [y_pos - 0.15, y_pos - 0.15], color=p1_color, linewidth=3, zorder=1)
                ax.plot([ci_low1, ci_low1], [y_pos - 0.20, y_pos - 0.10], color=p1_color, linewidth=3, zorder=1)
                ax.plot([ci_up1, ci_up1], [y_pos - 0.20, y_pos - 0.10], color=p1_color, linewidth=3, zorder=1)
            ax.plot(hr1, y_pos - 0.15, 's', color=p1_color, markersize=10,
                    markerfacecolor='white', markeredgewidth=2, zorder=2)
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f'{r:.1f}' for r in r0_values][::-1], fontsize=12)
        ax.set_ylim(-1.0, len(r0_values) - 0.0)
        ax.set_xlim(hr_min, hr_max)
        ax.set_xscale('log')
        ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.5)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if ax_idx == 0:
            ax.set_title('Forest Plot of Hazard Ratios by beta_event', fontsize=14, fontweight='bold', pad=15)
        
        ax.set_ylabel(f'beta = {beta:.1f}', fontsize=12, fontweight='bold')
    
    axes[-1].set_xlabel('Hazard Ratio (log scale)', fontsize=12)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor=p0_color,
               markersize=10, markeredgewidth=2, label='P0 (parameter only)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='none',         markeredgecolor=p1_color,
               markersize=10, markeredgewidth=2, label='P1 (parameter + seed)')
    ]
    axes[0].legend(handles=legend_elements, loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        base = output_path.rsplit('.', 1)[0]
    else:
        base = 'survival_analysis_results/parameter_forest_plot_by_beta'
    
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight', format='png')
    plt.savefig(f'{base}.pdf', bbox_inches='tight', format='pdf')
    plt.close()


def generate_clustered_forest_plot(model_results: Dict,
                                   output_path: str = None,
                                   grouping_col: str = 'scenario'):
    """
    Generate a forest plot using cluster-robust (Huber-White sandwich) confidence intervals.

    Districts within the same stochastic simulation are correlated because they share
    the same epidemic realization, commuter mobility, and transmission history.
    Cluster-robust sandwich standard errors account for this within-run dependence
    while leaving coefficient estimates (hazard ratios) unchanged.

    The hazard ratios remain identical to the naive model, but the confidence
    intervals are correctly widened to account for within-simulation correlation.

    Args:
        model_results: Dictionary of fitted model results (with both naive and
            clustered summaries).
        output_path: Base path for output files (PNG and PDF saved).
        grouping_col: 'scenario' for Mode A or 'param_combo' for Mode B.
    """
    if not HAS_PLOTTING:
        raise ImportError("matplotlib required for plotting")

    create_forest_plot_hazard_ratios(
        model_results,
        output_path=output_path,
        grouping_col=grouping_col,
        use_robust=True,
    )
