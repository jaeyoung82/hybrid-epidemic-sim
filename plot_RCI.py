import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Try to import config to get results dir, otherwise default
try:
    import config as cfg
    import data_loader
    RESULTS_DIR = getattr(cfg, 'results_dir', Path("results_hybrid_sim"))
except ImportError:
    RESULTS_DIR = Path("results_hybrid_sim")
    cfg = None

SUMMARY_DIR = Path(RESULTS_DIR/"figures_relative_ratio")

# Global for population
TOTAL_POPULATION = 1.0

def load_population():
    global TOTAL_POPULATION
    if cfg is None:
        print("Config not available, skipping population load.")
        return

    try:
        print("Loading population data for normalization...")
        sim_data = data_loader.load_and_prepare_data(cfg)
        TOTAL_POPULATION = sim_data['population_df']['population'].sum()
        print(f"Total Population: {TOTAL_POPULATION}")
    except Exception as e:
        print(f"Warning: Could not load population data: {e}")
        TOTAL_POPULATION = 1.0

def calculate_global_max_amplification(df, metric_col, iss_col_candidates=['initially_infected', 'I_ss']):
    """
    Calculates the maximum amplification factor across all scenarios in the dataframe.
    """
    if df.empty: return None
    
    # Find the I_ss column
    iss_col = None
    for col in iss_col_candidates:
        if col in df.columns:
            iss_col = col
            break
    if not iss_col: return None

    # Ensure numeric
    df = df.copy()
    df['R0'] = pd.to_numeric(df['R0'], errors='coerce')
    df[iss_col] = pd.to_numeric(df[iss_col], errors='coerce')
    df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')

    baseline = df[df['scenario_name'] == 'no_event'].groupby(['R0', iss_col])[metric_col].mean().reset_index()
    events = df[df['scenario_name'] != 'no_event']
    
    if baseline.empty or events.empty: return None
    
    merged = pd.merge(events, baseline, on=['R0', iss_col], suffixes=('', '_base'))
    if merged.empty: return None
    
    # Calculate ratio
    merged['ratio'] = merged[metric_col] / merged[f'{metric_col}_base'].replace(0, np.nan)
    
    valid_ratios = merged['ratio'][np.isfinite(merged['ratio'])]
    if valid_ratios.empty: return None
    
    return valid_ratios.max()

def plot_metric_vs_beta(df, x_col, y_col, hue_col, facet_col, ylabel, title_suffix, filename_prefix):
    """
    Generates line plots of Metric vs Beta_Event, grouped by R0 (hue).
    """
    if df.empty:
        print(f"Dataframe for {filename_prefix} is empty.")
        return

    # Ensure numeric types
    df[x_col] = pd.to_numeric(df[x_col])
    df[y_col] = pd.to_numeric(df[y_col])
    df[hue_col] = pd.to_numeric(df[hue_col])

    # Get unique values for faceting (e.g. I_ss)
    if facet_col in df.columns:
        facet_values = df[facet_col].unique()
    else:
        facet_values = [None]

    scenarios = df['scenario_name'].unique()

    for scenario in scenarios:
        for facet_val in facet_values:
            # Filter data
            mask = (df['scenario_name'] == scenario)
            if facet_val is not None:
                mask &= (df[facet_col] == facet_val)
            
            subset = df[mask].copy()
            if subset.empty: continue

            # Sort for proper line plotting
            subset = subset.sort_values(by=[hue_col, x_col])

            plt.figure(figsize=(8, 6))
            
            # Plot
            sns.lineplot(
                data=subset, 
                x=x_col, 
                y=y_col, 
                hue=hue_col, 
                palette='viridis', 
                marker='o',
                legend='full'
            )
            
            facet_str = f", $I_{{ss}}$={facet_val}" if facet_val is not None else ""
            plt.title(f'{scenario}: {title_suffix}{facet_str}')
            plt.xlabel(r'Event Transmission Rate ($\beta_{event}$)')
            plt.ylabel(ylabel)
            plt.legend(title='$R_0$')
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Save
            facet_file_str = f"_Iss{facet_val}" if facet_val is not None else ""
            out_file = SUMMARY_DIR / f"{filename_prefix}_{scenario}{facet_file_str}.png"
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {out_file}")

def plot_scenario_heatmaps(df_peak, df_final, scenario, iss_val):
    """
    Generates a combined figure with two heatmaps:
    1. Peak Active Cases vs (R0, Beta_Event)
    2. Final Size vs (R0, Beta_Event)
    """
    # Skip baseline scenario as it does not vary with beta_event
    if scenario == 'no_event':
        return

    # Filter Peak Data
    peak_subset = df_peak[
        (df_peak['scenario_name'] == scenario) & 
        (df_peak['initially_infected'] == iss_val)
    ].copy()

    # Filter Final Size Data
    final_subset = df_final[
        (df_final['scenario_name'] == scenario) & 
        (df_final['I_ss'] == iss_val)
    ].copy()

    if peak_subset.empty or final_subset.empty:
        return

    # Prepare Pivot Tables
    # Rounding to avoid float mismatch issues
    peak_subset['R0_r'] = peak_subset['R0'].round(4)
    peak_subset['beta_r'] = peak_subset['beta_event'].round(4)
    
    final_subset['R0_r'] = final_subset['R0'].round(4)
    final_subset['beta_r'] = final_subset['beta_event'].round(4)

    try:
        peak_pivot = peak_subset.pivot_table(index='R0_r', columns='beta_r', values='peak_value_avg')
        final_pivot = final_subset.pivot_table(index='R0_r', columns='beta_r', values='final_size_mean')
    except Exception as e:
        print(f"Error creating pivot tables for {scenario}, I_ss={iss_val}: {e}")
        return

    # Sort indices (High R0 at top for heatmap display)
    peak_pivot = peak_pivot.sort_index(ascending=False)
    final_pivot = final_pivot.sort_index(ascending=False)

    # Setup Figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot 1: Peak Active Cases
    sns.heatmap(peak_pivot, ax=axes[0], cmap="Reds", annot=True, fmt=".1f", cbar_kws={'label': 'Peak Active Cases (per 1000)'})
    axes[0].set_title(f"Peak Active Cases\nScenario: {scenario}, $I_{{ss}}$={iss_val}")
    axes[0].set_xlabel(r"Event Transmission Rate ($\beta_{event}$)")
    axes[0].set_ylabel(r"Basic Reproduction Number ($R_0$)")

    # Plot 2: Final Size
    sns.heatmap(final_pivot, ax=axes[1], cmap="Reds", annot=True, fmt=".1f", cbar_kws={'label': 'Cumulative Incidence (per 1000)'})
    axes[1].set_title(f"Cumulative Incidence\nScenario: {scenario}, $I_{{ss}}$={iss_val}")
    axes[1].set_xlabel(r"Event Transmission Rate ($\beta_{event}$)")
    axes[1].set_ylabel(r"Basic Reproduction Number ($R_0$)")

    plt.tight_layout()
    
    out_file = SUMMARY_DIR / f"heatmap_combined_{scenario}_Iss{iss_val}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved combined heatmap: {out_file}")

def plot_iss_r0_heatmaps(df, metric_col, metric_label, filename_prefix):
    """
    Generates heatmaps of Metric vs (I_ss, R0).
    One heatmap per Scenario (and per Beta_Event for event scenarios).
    Also generates Difference heatmaps (Scenario - Baseline).
    """
    if df.empty:
        return

    # Ensure numeric types
    df['R0'] = pd.to_numeric(df['R0'], errors='coerce')
    df['beta_event'] = pd.to_numeric(df['beta_event'], errors='coerce')
    
    # Normalize I_ss column
    if 'initially_infected' in df.columns:
        df['I_ss'] = pd.to_numeric(df['initially_infected'], errors='coerce')
    elif 'I_ss' in df.columns:
        df['I_ss'] = pd.to_numeric(df['I_ss'], errors='coerce')
    
    # Identify Baseline
    baseline_df = df[df['scenario_name'] == 'no_event'].copy()
    
    baseline_pivot = None
    if not baseline_df.empty:
        try:
            # Baseline depends only on R0 and I_ss. Average if multiple entries exist.
            baseline_pivot = baseline_df.pivot_table(index='R0', columns='I_ss', values=metric_col, aggfunc='mean')
            baseline_pivot = baseline_pivot.sort_index(ascending=False) # High R0 at top
            
            # Plot Baseline
            plt.figure(figsize=(8, 6))
            sns.heatmap(baseline_pivot, annot=True, fmt=".1f", cmap="viridis", cbar_kws={'label': metric_label})
            plt.title(f"Baseline (No Event): {metric_label}")
            plt.xlabel(r"Initial Seed Size ($I_{ss}$)")
            plt.ylabel(r"Basic Reproduction Number ($R_0$)")
            plt.tight_layout()
            out_file = SUMMARY_DIR / f"{filename_prefix}_heatmap_baseline.png"
            plt.savefig(out_file, dpi=300)
            plt.close()
            print(f"Saved baseline heatmap: {out_file}")
        except Exception as e:
            print(f"Error plotting baseline heatmap: {e}")

    # Process Event Scenarios
    event_scenarios = df[df['scenario_name'] != 'no_event']['scenario_name'].unique()
    
    for sc in sorted(event_scenarios):
        sc_df = df[df['scenario_name'] == sc]
        unique_betas = sorted(sc_df['beta_event'].unique())
        
        for beta in unique_betas:
            subset = sc_df[np.isclose(sc_df['beta_event'], beta)]
            if subset.empty: continue
            
            try:
                pivot = subset.pivot_table(index='R0', columns='I_ss', values=metric_col, aggfunc='mean')
                pivot = pivot.sort_index(ascending=False)
                
                # Plot Absolute
                plt.figure(figsize=(8, 6))
                sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis", cbar_kws={'label': metric_label})
                plt.title(f"{sc} ($\\beta_{{event}}$={beta}): {metric_label}")
                plt.xlabel(r"Initial Seed Size ($I_{ss}$)")
                plt.ylabel(r"Basic Reproduction Number ($R_0$)")
                plt.tight_layout()
                out_file = SUMMARY_DIR / f"{filename_prefix}_heatmap_{sc}_beta{int(beta*100)}.png"
                plt.savefig(out_file, dpi=300)
                plt.close()
                print(f"Saved heatmap: {out_file}")
                
                # Plot Difference (Event - Baseline)
                if baseline_pivot is not None:
                    common_index = pivot.index.intersection(baseline_pivot.index)
                    common_cols = pivot.columns.intersection(baseline_pivot.columns)
                    
                    if not common_index.empty and not common_cols.empty:
                        p_aligned = pivot.loc[common_index, common_cols]
                        b_aligned = baseline_pivot.loc[common_index, common_cols]
                        diff = p_aligned - b_aligned
                        
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(diff, annot=True, fmt=".1f", cmap="coolwarm", center=0, cbar_kws={'label': f"Difference in {metric_label}"})
                        plt.title(f"Difference: {sc} ($\\beta_{{event}}$={beta}) - Baseline")
                        plt.xlabel(r"Initial Seed Size ($I_{ss}$)")
                        plt.ylabel(r"Basic Reproduction Number ($R_0$)")
                        plt.tight_layout()
                        out_file_diff = SUMMARY_DIR / f"{filename_prefix}_diff_heatmap_{sc}_beta{int(beta*100)}.png"
                        plt.savefig(out_file_diff, dpi=300)
                        plt.close()
                        print(f"Saved difference heatmap: {out_file_diff}")
            except Exception as e:
                print(f"Error plotting heatmap for {sc}, beta={beta}: {e}")

def plot_amplification_heatmaps(df, metric_col, metric_label, filename_prefix, vmax=None):
    """
    Generates heatmaps of the Amplification Factor (Event / Baseline) vs (I_ss, R0).
    Amplification = Metric_Event / Metric_Baseline.
    """
    if df.empty:
        return

    # Ensure numeric types
    df['R0'] = pd.to_numeric(df['R0'], errors='coerce')
    df['beta_event'] = pd.to_numeric(df['beta_event'], errors='coerce')
    if 'initially_infected' in df.columns:
        df['I_ss'] = pd.to_numeric(df['initially_infected'], errors='coerce')
    elif 'I_ss' in df.columns:
        df['I_ss'] = pd.to_numeric(df['I_ss'], errors='coerce')

    # Identify Baseline
    baseline_df = df[df['scenario_name'] == 'no_event'].copy()
    if baseline_df.empty:
        print(f"No baseline data found for {filename_prefix}, cannot compute amplification.")
        return

    try:
        baseline_pivot = baseline_df.pivot_table(index='R0', columns='I_ss', values=metric_col, aggfunc='mean')
    except Exception as e:
        print(f"Error creating baseline pivot for amplification: {e}")
        return

    # Process Event Scenarios
    event_scenarios = df[df['scenario_name'] != 'no_event']['scenario_name'].unique()
    
    for sc in sorted(event_scenarios):
        sc_df = df[df['scenario_name'] == sc]
        unique_betas = sorted(sc_df['beta_event'].unique())
        
        for beta in unique_betas:
            subset = sc_df[np.isclose(sc_df['beta_event'], beta)]
            if subset.empty: continue
            
            try:
                pivot = subset.pivot_table(index='R0', columns='I_ss', values=metric_col, aggfunc='mean')
                
                # Align and Compute Ratio
                common_index = pivot.index.intersection(baseline_pivot.index)
                common_cols = pivot.columns.intersection(baseline_pivot.columns)
                
                if not common_index.empty and not common_cols.empty:
                    p_aligned = pivot.loc[common_index, common_cols]
                    b_aligned = baseline_pivot.loc[common_index, common_cols]
                    
                    # --- Robust Ratio Calculation ---
                    p_vals = p_aligned.values
                    b_vals = b_aligned.values
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        r_vals = p_vals / b_vals
                    
                    # Case 1: 0 / 0 -> 1.0 (No change)
                    mask_0_0 = (p_vals < 1e-9) & (b_vals < 1e-9)
                    r_vals[mask_0_0] = 1.0
                    
                    # Case 2: X / 0 -> Infinite Amplification. Cap at a high value for visualization.
                    mask_inf = np.isinf(r_vals)
                    if np.any(mask_inf):
                        r_vals[mask_inf] = np.nanmax(r_vals[~mask_inf]) * 1.5 if np.any(~mask_inf) else 10.0

                    ratio = pd.DataFrame(r_vals, index=p_aligned.index, columns=p_aligned.columns)
                    ratio = ratio.sort_index(ascending=False)
                    
                    # Prepare annotation labels including baseline values for context
                    # Ensure baseline is aligned and sorted same as ratio
                    b_aligned_sorted = b_aligned.loc[ratio.index, ratio.columns]
                    
                    r_vals = ratio.values
                    b_vals = b_aligned_sorted.values
                    annotations = np.empty_like(r_vals, dtype=object)
                    
                    for r in range(r_vals.shape[0]):
                        for c in range(r_vals.shape[1]):
                            annotations[r, c] = f"{r_vals[r, c]:.1f}\n(B:{b_vals[r, c]:.1f})"

                    plt.figure(figsize=(8, 6))
                    # vmin=1.0 ensures 1.0 (no change) is the lightest color
                    sns.heatmap(ratio, annot=annotations, fmt="", cmap="OrRd", vmin=1.0, vmax=vmax, cbar_kws={'label': f"Relative {metric_label}"})
                    
                    plt.title(f"Relative {metric_label}: {sc} ($\\beta_{{event}}$={beta})\n(Cell: Ratio, B: Baseline Value)")
                    plt.xlabel(r"Initial Seed Size ($I_{ss}$)")
                    plt.ylabel(r"Basic Reproduction Number ($R_0$)")
                    plt.tight_layout()
                    
                    out_file = SUMMARY_DIR / f"{filename_prefix}_{sc}_beta{int(beta*100)}.png"
                    plt.savefig(out_file, dpi=300)
                    plt.close()
                    print(f"Saved amplification heatmap: {out_file}")
            except Exception as e:
                print(f"Error plotting amplification for {sc}, beta={beta}: {e}")

def plot_amplification_scenario_heatmap(df, metric_col, metric_label, filename_prefix, vmax=None):
    """
    Generates heatmaps of Amplification Factor vs (Scenario, R0).
    Fixed I_ss and Beta_Event per plot.
    X-axis: Scenario
    Y-axis: R0
    """
    if df.empty:
        return

    # Ensure numeric types
    df['R0'] = pd.to_numeric(df['R0'], errors='coerce')
    df['beta_event'] = pd.to_numeric(df['beta_event'], errors='coerce')
    if 'initially_infected' in df.columns:
        df['I_ss'] = pd.to_numeric(df['initially_infected'], errors='coerce')
    elif 'I_ss' in df.columns:
        df['I_ss'] = pd.to_numeric(df['I_ss'], errors='coerce')

    # Identify Baseline
    baseline_df = df[df['scenario_name'] == 'no_event'].copy()
    if baseline_df.empty:
        print(f"No baseline data found for {filename_prefix}, cannot compute amplification.")
        return

    # Get unique parameters
    unique_iss = sorted(df['I_ss'].dropna().unique())
    # Get betas from event scenarios only
    event_df = df[df['scenario_name'] != 'no_event']
    if event_df.empty:
        return
    unique_betas = sorted(event_df['beta_event'].dropna().unique())

    for iss in unique_iss:
        # Baseline for this I_ss
        try:
            base_subset = baseline_df[baseline_df['I_ss'] == iss]
            if base_subset.empty: continue
            # Average over duplicates if any (e.g. multiple runs)
            base_pivot = base_subset.groupby('R0')[metric_col].mean()
        except Exception as e:
            print(f"Error processing baseline for I_ss={iss}: {e}")
            continue

        for beta in unique_betas:
            # Build a matrix: Rows=R0, Cols=Scenario
            ratio_matrix = pd.DataFrame()
            
            # Get all scenarios present for this beta/iss
            scenarios_for_beta = event_df[
                (event_df['I_ss'] == iss) & 
                (np.isclose(event_df['beta_event'], beta))
            ]['scenario_name'].unique()
            
            if len(scenarios_for_beta) == 0:
                continue

            for sc in sorted(scenarios_for_beta):
                sc_subset = event_df[
                    (event_df['scenario_name'] == sc) & 
                    (event_df['I_ss'] == iss) & 
                    (np.isclose(event_df['beta_event'], beta))
                ]
                if sc_subset.empty: continue
                
                sc_pivot = sc_subset.groupby('R0')[metric_col].mean()
                
                # Align with baseline
                common_r0 = sc_pivot.index.intersection(base_pivot.index)
                if common_r0.empty: continue
                
                sc_vals = sc_pivot.loc[common_r0]
                base_vals = base_pivot.loc[common_r0]
                
                # --- Robust Ratio Calculation ---
                sc_v = sc_vals.values
                base_v = base_vals.values
                with np.errstate(divide='ignore', invalid='ignore'):
                    r_v = sc_v / base_v
                
                # 0/0 -> 1.0
                mask_0_0 = (sc_v < 1e-9) & (base_v < 1e-9)
                r_v[mask_0_0] = 1.0
                
                # X/0 -> Inf
                mask_inf = np.isinf(r_v)
                if np.any(mask_inf):
                    r_v[mask_inf] = np.nanmax(r_v[~mask_inf]) * 1.5 if np.any(~mask_inf) else 10.0
                
                ratio = pd.Series(r_v, index=sc_vals.index)
                ratio_matrix[sc] = ratio

            if ratio_matrix.empty:
                continue

            # Sort index (R0) descending
            ratio_matrix = ratio_matrix.sort_index(ascending=False)

            # Prepare annotations (Ratio + Baseline)
            annotations = np.empty(ratio_matrix.shape, dtype=object)
            for r_idx, r0 in enumerate(ratio_matrix.index):
                b_val = base_pivot.get(r0, np.nan)
                for c_idx in range(ratio_matrix.shape[1]):
                    val = ratio_matrix.iloc[r_idx, c_idx]
                    if pd.isna(val):
                        annotations[r_idx, c_idx] = ""
                    else:
                        annotations[r_idx, c_idx] = f"{val:.1f}\n(B:{b_val:.1f})"

            # Plot
            plt.figure(figsize=(max(8, len(ratio_matrix.columns) * 1.5), 6))
            sns.heatmap(ratio_matrix, annot=annotations, fmt="", cmap="OrRd", vmin=1.0, vmax=vmax, cbar_kws={'label': f"Relative {metric_label}"})
            
            # plt.title(f"Relative {metric_label} Comparison: Scenarios vs $R_0$\n($I_{{ss}}$={iss}, $\\beta_{{event}}$={beta})\n(Cell: Ratio, B: Baseline Value)")
            plt.title(f"Relative {metric_label} Comparison ($I_{{ss}}$={iss}, $\\beta_{{event}}$={beta})\n(Cell: Ratio, B: Baseline Value)")
            plt.xlabel("Scenario")
            plt.ylabel(r"Basic Reproduction Number ($R_0$)")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            out_file = SUMMARY_DIR / f"{filename_prefix}_compare_Iss{iss}_beta{int(beta*100)}.png"
            plt.savefig(out_file, dpi=300)
            plt.close()
            print(f"Saved scenario comparison heatmap: {out_file}")

def plot_amplification_beta_r0_heatmap(df, metric_col, metric_label, filename_prefix, vmax=None):
    """
    Generates heatmaps of Amplification Factor vs (Beta_Event, R0).
    Fixed I_ss and Scenario per plot.
    X-axis: Beta_Event
    Y-axis: R0
    """
    if df.empty:
        return

    # Ensure numeric types
    df['R0'] = pd.to_numeric(df['R0'], errors='coerce')
    df['beta_event'] = pd.to_numeric(df['beta_event'], errors='coerce')
    if 'initially_infected' in df.columns:
        df['I_ss'] = pd.to_numeric(df['initially_infected'], errors='coerce')
    elif 'I_ss' in df.columns:
        df['I_ss'] = pd.to_numeric(df['I_ss'], errors='coerce')

    # Identify Baseline
    baseline_df = df[df['scenario_name'] == 'no_event'].copy()
    if baseline_df.empty:
        print(f"No baseline data found for {filename_prefix}, cannot compute amplification.")
        return

    # Get unique parameters for iteration
    unique_iss = sorted(df['I_ss'].dropna().unique())
    event_scenarios = sorted([s for s in df['scenario_name'].unique() if s != 'no_event'])

    for iss in unique_iss:
        # Baseline for this I_ss, indexed by R0
        base_subset = baseline_df[baseline_df['I_ss'] == iss]
        if base_subset.empty: continue
        base_pivot = base_subset.groupby('R0')[metric_col].mean()

        for scenario in event_scenarios:
            # Event data for this I_ss and scenario
            event_subset = df[(df['I_ss'] == iss) & (df['scenario_name'] == scenario)]
            if event_subset.empty: continue

            try:
                # Pivot to get Beta_Event vs R0
                event_pivot = event_subset.pivot_table(index='R0', columns='beta_event', values=metric_col, aggfunc='mean')
            except Exception as e:
                print(f"Error creating pivot for {scenario}, I_ss={iss}: {e}")
                continue

            # --- Robust Ratio Calculation ---
            # Align indices
            common_idx = event_pivot.index.intersection(base_pivot.index)
            e_aligned = event_pivot.loc[common_idx]
            b_aligned = base_pivot.loc[common_idx]
            
            # Broadcast baseline across columns
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_matrix = e_aligned.div(b_aligned, axis=0)
            
            # Fix Infs and NaNs
            ratio_matrix = ratio_matrix.fillna(1.0) # NaNs usually mean 0/0 here if aligned correctly
            # Note: div(0) gives inf. We should cap infs.
            ratio_matrix = ratio_matrix.replace([np.inf, -np.inf], 10.0) # Cap high
            ratio_matrix = ratio_matrix.sort_index(ascending=False)

            # Prepare annotations
            annotations = np.empty(ratio_matrix.shape, dtype=object)
            for r_idx, r0 in enumerate(ratio_matrix.index):
                b_val = base_pivot.get(r0, np.nan)
                for c_idx, beta in enumerate(ratio_matrix.columns):
                    val = ratio_matrix.iloc[r_idx, c_idx]
                    annotations[r_idx, c_idx] = f"{val:.2f}\n({b_val:.1f})" if not pd.isna(val) else ""
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(ratio_matrix, annot=annotations, fmt="", cmap="Reds", vmin=1.0, vmax=vmax, cbar_kws={'label': f"Relative {metric_label}"})
            # plt.title(f"Relative {metric_label} vs Event Risk: {scenario}\n($I_{{ss}}$={iss})\n(Cell: Ratio, Parentheses: Baseline per 1000)")
            plt.title(f"Relative {metric_label}: {scenario}\n($I_{{ss}}$={iss})\n(Cell: Ratio, Parentheses: Baseline per 1000)")
            plt.xlabel(r"Event Transmission Rate ($\beta_{event}$)")
            plt.ylabel(r"Basic Reproduction Number ($R_0$)")
            plt.tight_layout()
            
            out_file = SUMMARY_DIR / f"{filename_prefix}_beta_r0_{scenario}_Iss{iss}.png"
            plt.savefig(out_file, dpi=300)
            plt.close()
            print(f"Saved beta vs R0 amplification heatmap: {out_file}")

def plot_amplification_vs_difference_log_by_beta(df, metric_col, metric_label, filename_prefix):
    """
    Generates a log-log scatter plot of Difference vs Amplification Factor.
    X-axis: Amplification Factor (Event / Baseline) (Log Scale)
    Y-axis: Difference (Event - Baseline) (Log Scale)
    Color (Hue): Beta_Event
    """
    if df.empty: return

    # Ensure numeric types
    df = df.copy()
    df['R0'] = pd.to_numeric(df['R0'], errors='coerce')
    df['beta_event'] = pd.to_numeric(df['beta_event'], errors='coerce')
    if 'initially_infected' in df.columns:
        df['I_ss'] = pd.to_numeric(df['initially_infected'], errors='coerce')
    elif 'I_ss' in df.columns:
        df['I_ss'] = pd.to_numeric(df['I_ss'], errors='coerce')
    df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')

    # Identify Baseline
    baseline_df = df[df['scenario_name'] == 'no_event'].copy()
    if baseline_df.empty:
        print(f"No baseline data found for {filename_prefix}, cannot compute metrics.")
        return

    # Aggregate baseline if multiple runs exist
    try:
        baseline_agg = baseline_df.groupby(['R0', 'I_ss'])[metric_col].mean().reset_index()
    except Exception as e:
        print(f"Error aggregating baseline: {e}")
        return

    # Process Event Scenarios
    event_df = df[df['scenario_name'] != 'no_event'].copy()
    if event_df.empty: return

    # Merge Event and Baseline
    merged = pd.merge(event_df, baseline_agg, on=['R0', 'I_ss'], suffixes=('', '_base'))
    
    # Calculate Metrics
    # Avoid division by zero for Amplification
    merged[f'{metric_col}_base_safe'] = merged[f'{metric_col}_base'].replace(0, np.nan)
    merged['Relative'] = merged[metric_col] / merged[f'{metric_col}_base_safe']
    merged['Difference'] = merged[metric_col] - merged[f'{metric_col}_base']
    
    # Filter for positive values for log scale
    merged = merged[(merged['Relative'] > 0) & (merged['Difference'] > 0)]
    
    if merged.empty:
        return

    # Plotting
    plt.figure(figsize=(8, 6))
    
    sns.scatterplot(data=merged, x='Relative', y='Difference', hue='beta_event', style='scenario_name', palette='viridis', s=100, alpha=0.7)
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.title(f"Relative {metric_label} vs Absolute Increase (Log-Log) by beta")
    plt.xlabel(f"Relative {metric_label} (Ratio)")
    plt.ylabel(f"Difference per 1000 persons")
    plt.grid(True, linestyle='--', alpha=0.6, which='both')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title=r'$\beta_{event}$')
    
    plt.tight_layout()
    out_file = SUMMARY_DIR / f"{filename_prefix}_scatter_vs_diff_log_beta.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved log-log scatter plot (by beta): {out_file}")

def plot_amplification_vs_difference_log_per_beta(df, metric_col, metric_label, filename_prefix):
    """
    Generates a log-log scatter plot of Difference vs Amplification Factor for each beta_event value.
    X-axis: Amplification Factor (Event / Baseline) (Log Scale)
    Y-axis: Difference (Event - Baseline) (Log Scale)
    Color (Hue): R0 (since beta is fixed per plot)
    Global axis limits are used for comparison.
    """
    if df.empty: return

    # Ensure numeric types
    df = df.copy()
    df['R0'] = pd.to_numeric(df['R0'], errors='coerce')
    df['beta_event'] = pd.to_numeric(df['beta_event'], errors='coerce')
    if 'initially_infected' in df.columns:
        df['I_ss'] = pd.to_numeric(df['initially_infected'], errors='coerce')
    elif 'I_ss' in df.columns:
        df['I_ss'] = pd.to_numeric(df['I_ss'], errors='coerce')
    df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')

    # Identify Baseline
    baseline_df = df[df['scenario_name'] == 'no_event'].copy()
    if baseline_df.empty:
        print(f"No baseline data found for {filename_prefix}, cannot compute metrics.")
        return

    # Aggregate baseline if multiple runs exist
    try:
        baseline_agg = baseline_df.groupby(['R0', 'I_ss'])[metric_col].mean().reset_index()
    except Exception as e:
        print(f"Error aggregating baseline: {e}")
        return

    # Process Event Scenarios
    event_df_all = df[df['scenario_name'] != 'no_event'].copy()
    if event_df_all.empty: return

    # --- Pre-calculate metrics for ALL data to determine global axis limits ---
    merged_all = pd.merge(event_df_all, baseline_agg, on=['R0', 'I_ss'], suffixes=('', '_base'))
    
    merged_all[f'{metric_col}_base_safe'] = merged_all[f'{metric_col}_base'].replace(0, np.nan)
    merged_all['Relative'] = merged_all[metric_col] / merged_all[f'{metric_col}_base_safe']
    merged_all['Difference'] = merged_all[metric_col] - merged_all[f'{metric_col}_base']
    
    # Filter for positive values for log scale
    merged_all = merged_all[(merged_all['Relative'] > 0) & (merged_all['Difference'] > 0)]
    
    if merged_all.empty:
        return

    # Determine Global Limits with some padding
    x_min, x_max = merged_all['Relative'].min(), merged_all['Relative'].max()
    y_min, y_max = merged_all['Difference'].min(), merged_all['Difference'].max()
    
    # Add padding (multiplicative for log scale)
    x_min = x_min / 1.1 if x_min > 0 else x_min
    x_max = x_max * 1.1
    y_min = y_min / 1.1 if y_min > 0 else y_min
    y_max = y_max * 1.1

    unique_betas = sorted(event_df_all['beta_event'].dropna().unique())

    for beta in unique_betas:
        # Filter the pre-calculated dataframe
        subset = merged_all[np.isclose(merged_all['beta_event'], beta)]
        
        if subset.empty: continue

        # Plotting
        plt.figure(figsize=(8, 6))
        
        sns.scatterplot(data=subset, x='Relative', y='Difference', hue='R0', style='scenario_name', palette='viridis', s=100, alpha=0.7)
        
        plt.xscale('log')
        plt.yscale('log')
        
        # Apply Global Limits
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        plt.title(f"Relative {metric_label} vs Absolute Increase (Log-Log)\n$\\beta_{{event}}$={beta}")
        plt.xlabel(f"Relative {metric_label} (Ratio)")
        plt.ylabel(f"Difference per 1000 persons")
        plt.grid(True, linestyle='--', alpha=0.6, which='both')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        plt.tight_layout()
        out_file = SUMMARY_DIR / f"{filename_prefix}_scatter_vs_diff_log_beta{int(beta*100)}.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved log-log scatter plot for beta={beta}: {out_file}")

def plot_amplification_vs_difference_per_beta(df, metric_col, metric_label, filename_prefix):
    """
    Generates a scatter plot of Difference vs Amplification Factor for each beta_event value.
    X-axis: Amplification Factor (Event / Baseline)
    Y-axis: Difference (Event - Baseline)
    Color (Hue): R0
    Shape (Style): Scenario
    Global axis limits are used for comparison.
    """
    if df.empty: return

    # Ensure numeric types
    df = df.copy()
    df['R0'] = pd.to_numeric(df['R0'], errors='coerce')
    df['beta_event'] = pd.to_numeric(df['beta_event'], errors='coerce')
    if 'initially_infected' in df.columns:
        df['I_ss'] = pd.to_numeric(df['initially_infected'], errors='coerce')
    elif 'I_ss' in df.columns:
        df['I_ss'] = pd.to_numeric(df['I_ss'], errors='coerce')
    df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')

    # Identify Baseline
    baseline_df = df[df['scenario_name'] == 'no_event'].copy()
    if baseline_df.empty:
        print(f"No baseline data found for {filename_prefix}, cannot compute metrics.")
        return

    # Aggregate baseline if multiple runs exist
    try:
        baseline_agg = baseline_df.groupby(['R0', 'I_ss'])[metric_col].mean().reset_index()
    except Exception as e:
        print(f"Error aggregating baseline: {e}")
        return

    # Process Event Scenarios
    event_df_all = df[df['scenario_name'] != 'no_event'].copy()
    if event_df_all.empty: return

    # --- Pre-calculate metrics for ALL data to determine global axis limits ---
    merged_all = pd.merge(event_df_all, baseline_agg, on=['R0', 'I_ss'], suffixes=('', '_base'))
    
    merged_all[f'{metric_col}_base_safe'] = merged_all[f'{metric_col}_base'].replace(0, np.nan)
    merged_all['Relative'] = merged_all[metric_col] / merged_all[f'{metric_col}_base_safe']
    merged_all['Difference'] = merged_all[metric_col] - merged_all[f'{metric_col}_base']
    
    # Drop NaNs
    merged_all = merged_all.dropna(subset=['Relative', 'Difference'])
    
    if merged_all.empty:
        return

    # Determine Global Limits with some padding
    x_min, x_max = merged_all['Relative'].min(), merged_all['Relative'].max()
    y_min, y_max = merged_all['Difference'].min(), merged_all['Difference'].max()
    
    # Add padding
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0
    x_min -= x_range * 0.05
    x_max += x_range * 0.05
    y_min -= y_range * 0.05
    y_max += y_range * 0.05

    unique_betas = sorted(event_df_all['beta_event'].dropna().unique())

    for beta in unique_betas:
        subset = merged_all[np.isclose(merged_all['beta_event'], beta)]
        if subset.empty: continue

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=subset, x='Relative', y='Difference', hue='R0', style='scenario_name', palette='viridis', s=100, alpha=0.7)
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(f"Relative {metric_label} vs Absolute Increase\n$\\beta_{{event}}$={beta}")
        plt.xlabel(f"Relative {metric_label} (Ratio)")
        plt.ylabel(f"Difference per 1000 persons")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        out_file = SUMMARY_DIR / f"{filename_prefix}_scatter_vs_diff_beta{int(beta*100)}.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved scatter plot for beta={beta}: {out_file}")

def plot_relative_peak_vs_cumulative_scatter(df_peak, df_final, filename_prefix):
    """
    Generates scatter plots of Relative Peak Prevalence vs Relative Cumulative Incidence.
    One plot per I_ss value.
    """
    if df_peak.empty or df_final.empty:
        return

    print("Generating Relative Peak vs Relative Cumulative scatter plots...")
    
    # --- Prepare Peak Data ---
    df_p = df_peak.copy()
    # Standardize I_ss column name
    if 'initially_infected' in df_p.columns:
        if 'I_ss' in df_p.columns:
            df_p = df_p.drop(columns=['initially_infected'])
        else:
            df_p = df_p.rename(columns={'initially_infected': 'I_ss'})
    
    # Ensure numeric types for key columns
    for col in ['R0', 'beta_event', 'I_ss', 'peak_value_avg']:
        if col in df_p.columns:
            df_p[col] = pd.to_numeric(df_p[col], errors='coerce')

    # Calculate Baseline Peak (Scenario = 'no_event')
    # Group by R0 and I_ss to handle multiple runs or beta entries for baseline
    base_p = df_p[df_p['scenario_name'] == 'no_event'].groupby(['R0', 'I_ss'])['peak_value_avg'].mean().reset_index()
    
    # Merge Baseline back to calculate relative peak
    df_p = pd.merge(df_p, base_p, on=['R0', 'I_ss'], suffixes=('', '_base'))
    df_p['Relative_Peak'] = df_p['peak_value_avg'] / df_p['peak_value_avg_base'].replace(0, np.nan)

    # --- Prepare Final Size Data ---
    df_f = df_final.copy()
    # Standardize I_ss column name (df_final usually has 'I_ss')
    if 'initially_infected' in df_f.columns:
        if 'I_ss' in df_f.columns:
            df_f = df_f.drop(columns=['initially_infected'])
        else:
            df_f = df_f.rename(columns={'initially_infected': 'I_ss'})

    # Ensure numeric types
    for col in ['R0', 'beta_event', 'I_ss', 'final_size_mean']:
        if col in df_f.columns:
            df_f[col] = pd.to_numeric(df_f[col], errors='coerce')

    # Calculate Baseline Final Size
    base_f = df_f[df_f['scenario_name'] == 'no_event'].groupby(['R0', 'I_ss'])['final_size_mean'].mean().reset_index()

    # Merge Baseline back
    df_f = pd.merge(df_f, base_f, on=['R0', 'I_ss'], suffixes=('', '_base'))
    df_f['Relative_Cumulative'] = df_f['final_size_mean'] / df_f['final_size_mean_base'].replace(0, np.nan)

    # --- Merge Peak and Final Data ---
    # Select relevant columns to avoid collisions
    cols_p = ['scenario_name', 'R0', 'beta_event', 'I_ss', 'Relative_Peak']
    cols_f = ['scenario_name', 'R0', 'beta_event', 'I_ss', 'Relative_Cumulative']
    
    merged = pd.merge(
        df_p[cols_p],
        df_f[cols_f],
        on=['scenario_name', 'R0', 'beta_event', 'I_ss']
    )
    
    # Filter out baseline rows
    merged = merged[merged['scenario_name'] != 'no_event']

    if merged.empty:
        print("No data available for Relative Peak vs Cumulative scatter plot.")
        return

    # --- Plotting ---
    unique_iss = sorted(merged['I_ss'].unique())
    
    for iss in unique_iss:
        subset = merged[merged['I_ss'] == iss]
        if subset.empty: continue
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=subset, x='Relative_Peak', y='Relative_Cumulative', hue='scenario_name', style='scenario_name', s=120, alpha=0.8, palette='viridis')
        plt.axhline(1.0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        plt.axvline(1.0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        plt.title(f"Relative Peak Prevalence vs Relative Cumulative Incidence\n($I_{{ss}}$={iss})", fontsize=16)
        plt.xlabel("Relative Peak Prevalence (Ratio)", fontsize=14)
        plt.ylabel("Relative Cumulative Incidence (Ratio)", fontsize=14)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="Scenario")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        out_file = SUMMARY_DIR / f"{filename_prefix}_scatter_peak_vs_cumulative_Iss{iss}.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved scatter plot: {out_file}")

def main():
    if not RESULTS_DIR.exists():
        print(f"Results directory '{RESULTS_DIR}' does not exist.")
        return

    if not SUMMARY_DIR.exists():
        SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created summary directory: {SUMMARY_DIR}")

    # Load population for normalization
    load_population()

    # User Configuration: List of I_ss values to visualize (e.g., [1, 10]). 
    # Set to None or empty list [] to visualize all available values.
    target_iss = []

    df_peak = pd.DataFrame()
    df_final = pd.DataFrame()

    print(f"Looking for result files in: {RESULTS_DIR}")
    if target_iss:
        print(f"Filtering for I_ss values: {target_iss}")

    # --- 1. Peak Active Cases (from peak-time-distribution.txt) ---
    peak_file = RESULTS_DIR / "peak-time-distribution_all.csv" # <-- modify the filename if needed!
    if peak_file.exists():
        print(f"Processing {peak_file}...")
        try:
            df_peak = pd.read_csv(peak_file)
            
            # Normalize scenario names to match final_size_summary (baseline -> no_event)
            if 'scenario_name' in df_peak.columns:
                df_peak['scenario_name'] = df_peak['scenario_name'].replace({'baseline': 'no_event'})
            
            # Robustness: Round float columns to avoid merge/pivot mismatches
            if 'R0' in df_peak.columns: df_peak['R0'] = df_peak['R0'].round(6)
            if 'beta_event' in df_peak.columns: df_peak['beta_event'] = df_peak['beta_event'].round(6)

            # Normalize to per 1000
            if TOTAL_POPULATION > 1:
                df_peak['peak_value_avg'] = (df_peak['peak_value_avg'] / TOTAL_POPULATION) * 1000
                print("Normalized Peak Active Cases to per 1000 people.")

            # Filter by target I_ss if specified
            if target_iss:
                df_peak['initially_infected'] = pd.to_numeric(df_peak['initially_infected'], errors='coerce')
                df_peak = df_peak[df_peak['initially_infected'].isin(target_iss)]

            # Deduplicate keeping last run in case of multiple runs appended
            df_peak = df_peak.drop_duplicates(
                subset=['scenario_name', 'R0', 'beta_event', 'initially_infected'], 
                keep='last'
            )
            
            # plot_metric_vs_beta(
            #     df_peak, 
            #     x_col='beta_event', 
            #     y_col='peak_value_avg', 
            #     hue_col='R0', 
            #     facet_col='initially_infected',
            #     ylabel='Peak Active Cases (per 1000)', 
            #     title_suffix='Peak Active Cases vs Event Risk', 
            #     filename_prefix='summary_peak_cases'
            # )

            # --- NEW: Plot Heatmaps of Peak Cases vs (Iss, R0) ---
            plot_iss_r0_heatmaps(df_peak, 'peak_value_avg', 'Peak Active Cases (per 1000)', 'summary_peak_cases_iss_r0')

            # Calculate global max amplification for consistent colorbar
            peak_max_amp = calculate_global_max_amplification(df_peak, 'peak_value_avg', ['initially_infected'])
            print(f"Global Max Amplification (Peak): {peak_max_amp}")

            # --- NEW: Plot Amplification Heatmaps ---
            plot_amplification_heatmaps(df_peak, 'peak_value_avg', 'Peak Prevalence', 'summary_peak_cases', vmax=peak_max_amp)

            # --- NEW: Plot Amplification Comparison Heatmaps (X=Scenario, Y=R0) ---
            plot_amplification_scenario_heatmap(df_peak, 'peak_value_avg', 'Peak Prevalence', 'summary_peak_cases', vmax=peak_max_amp)

            # --- NEW: Plot Amplification vs Beta/R0 Heatmaps ---
            plot_amplification_beta_r0_heatmap(df_peak, 'peak_value_avg', 'Peak Prevalence', 'summary_peak_cases', vmax=peak_max_amp)

            # --- NEW: Plot Amplification vs Difference Scatter (Log Scale) by Beta ---
            plot_amplification_vs_difference_log_by_beta(df_peak, 'peak_value_avg', 'Peak Prevalence', 'summary_peak_cases')

            # --- NEW: Plot Amplification vs Difference Scatter (Log Scale) PER Beta ---
            plot_amplification_vs_difference_log_per_beta(df_peak, 'peak_value_avg', 'Peak Prevalence', 'summary_peak_cases')

            # --- NEW: Plot Amplification vs Difference Scatter (Linear Scale) PER Beta ---
            plot_amplification_vs_difference_per_beta(df_peak, 'peak_value_avg', 'Peak Prevalence', 'summary_peak_cases')

        except Exception as e:
            print(f"Error processing peak file: {e}")
    else:
        print(f"Skipping Peak Cases: {peak_file} not found.")

    # --- 2. Cumulative Recovered (from final_size_summary.csv) ---
    final_size_file = RESULTS_DIR / "final_size_summary_all.csv" # <-- modify the filename if needed!
    if final_size_file.exists():
        print(f"Processing {final_size_file}...")
        try:
            df_final = pd.read_csv(final_size_file)
            
            # Robustness: Round float columns to avoid merge/pivot mismatches
            if 'R0' in df_final.columns: df_final['R0'] = df_final['R0'].round(6)
            if 'beta_event' in df_final.columns: df_final['beta_event'] = df_final['beta_event'].round(6)

            # Normalize to per 1000 (It is fraction 0-1, so just * 1000)
            df_final['final_size_mean'] = df_final['final_size_mean'] * 1000
            print("Normalized Final Size to per 1000 people.")
            
            # Filter by target I_ss if specified
            if target_iss:
                df_final['I_ss'] = pd.to_numeric(df_final['I_ss'], errors='coerce')
                df_final = df_final[df_final['I_ss'].isin(target_iss)]

            # Deduplicate
            df_final = df_final.drop_duplicates(
                subset=['scenario_name', 'R0', 'beta_event', 'I_ss'], 
                keep='last'
            )

            plot_metric_vs_beta(
                df_final, 
                x_col='beta_event', 
                y_col='final_size_mean', 
                hue_col='R0', 
                facet_col='I_ss',
                ylabel='Cumulative Incidence (per 1000)', 
                title_suffix='Cumulative Incidence', 
                filename_prefix='summary_final_size'
            )

            # --- NEW: Plot Heatmaps of Final Size vs (Iss, R0) ---
            plot_iss_r0_heatmaps(df_final, 'final_size_mean', 'Cumulative Incidence (per 1000)', 'summary_final_size_iss_r0')

            # Calculate global max amplification for consistent colorbar
            final_max_amp = calculate_global_max_amplification(df_final, 'final_size_mean', ['I_ss'])
            print(f"Global Max Amplification (Final Size): {final_max_amp}")

            # --- Plot Cumulative Incidence Heatmaps ---
            plot_amplification_heatmaps(df_final, 'final_size_mean', 'Cumulative Incidence', 'RCI', vmax=final_max_amp)

            # --- Plot Cumulative Incidence Comparison Heatmaps (X=Scenario, Y=R0) ---
            plot_amplification_scenario_heatmap(df_final, 'final_size_mean', 'Cumulative Incidence', 'RCI', vmax=final_max_amp)

            # --- Plot Cumulative Incidence vs Beta/R0 Heatmaps ---
            plot_amplification_beta_r0_heatmap(df_final, 'final_size_mean', 'Cumulative Incidence', 'RCI', vmax=final_max_amp)

            # --- Plot Cumulative Incidence vs Difference Scatter (Log Scale) by beta ---
            plot_amplification_vs_difference_log_by_beta(df_final, 'final_size_mean', 'Cumulative Incidence', 'RCI')

            # --- Plot Cumulative Incidence vs Difference Scatter (Log Scale) PER beta ---
            plot_amplification_vs_difference_log_per_beta(df_final, 'final_size_mean', 'Cumulative Incidence', 'RCI')

            # --- Cumulative Incidence vs Difference Scatter (Linear Scale) PER beta ---
            plot_amplification_vs_difference_per_beta(df_final, 'final_size_mean', 'Cumulative Incidence', 'RCI')

        except Exception as e:
            print(f"Error processing final size file: {e}")
    else:
        print(f"Skipping Final Size: {final_size_file} not found.")

    # --- 3. Combined Heatmaps ---
    if not df_peak.empty and not df_final.empty:
        print("\nGenerating combined heatmaps...")
        # Get intersection of scenarios
        scenarios = set(df_peak['scenario_name'].unique()) & set(df_final['scenario_name'].unique())
        
        if not scenarios:
            print("Warning: No common scenarios found between peak data and final size data.")
            print(f"Peak scenarios: {df_peak['scenario_name'].unique()}")
            print(f"Final size scenarios: {df_final['scenario_name'].unique()}")

        # Get I_ss values present in both
        iss_peak = set(df_peak['initially_infected'].unique())
        iss_final = set(df_final['I_ss'].unique())
        iss_values = iss_peak & iss_final
        
        for sc in sorted(list(scenarios)):
            for iss in sorted(list(iss_values)):
                plot_scenario_heatmaps(df_peak, df_final, sc, iss)
        
        # --- NEW: Plot Relative Peak vs Relative Cumulative Scatter ---
        plot_relative_peak_vs_cumulative_scatter(df_peak, df_final, 'summary_combined')
    else:
        print("\nSkipping combined heatmaps because one of the dataframes is empty.")
        if df_peak.empty: print("  -> Peak data is empty or filtered out.")
        if df_final.empty: print("  -> Final size data is empty or filtered out.")

if __name__ == "__main__":
    main()