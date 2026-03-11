import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
from pathlib import Path
import glob
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

# Try to import config to get results directory, otherwise default
try:
    import config as cfg
    RESULTS_DIR = getattr(cfg, 'results_dir', Path("results_hybrid_sim"))
except ImportError:
    RESULTS_DIR = Path("results_hybrid_sim")

OUTPUT_DIR = RESULTS_DIR / "final-size-dist"
OUTPUT_FILE = "final_size_summary_all.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_merge_summaries(results_dir=RESULTS_DIR, output_file=OUTPUT_FILE):
    """
    Loads all final_size_summary*.csv files, merges, cleans, sorts, and saves.
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory '{results_dir}' does not exist.")
        return None

    # Pattern to match summary files
    files = list(results_path.glob("final_size_summary*.csv"))
    
    # Exclude the output file itself if it exists to avoid recursive duplication
    files = [f for f in files if f.name != output_file]
    
    if not files:
        print(f"No 'final_size_summary*.csv' files found in {results_path}.")
        return None

    print(f"Found {len(files)} files to merge: {[f.name for f in files]}")
    
    dfs = []
    for f in files:
        try:
            # Use python engine for robustness against bad lines
            df = pd.read_csv(f, engine='python', on_bad_lines='skip')
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    if not dfs:
        return None

    combined_df = pd.concat(dfs, ignore_index=True)

    # Standardize columns
    required_cols = ['scenario_name', 'R0', 'beta_event', 'I_ss', 'values']
    missing = [c for c in required_cols if c not in combined_df.columns]
    if missing:
        print(f"Error: Combined data missing columns: {missing}")
        return None

    # Ensure numeric types for sorting keys
    combined_df['R0'] = pd.to_numeric(combined_df['R0'], errors='coerce').round(6)
    combined_df['beta_event'] = pd.to_numeric(combined_df['beta_event'], errors='coerce').round(6)
    combined_df['I_ss'] = pd.to_numeric(combined_df['I_ss'], errors='coerce')

    # Handle no_event beta_event = 0.0 explicitly
    combined_df.loc[combined_df['scenario_name'] == 'no_event', 'beta_event'] = 0.0

    # Drop duplicates (keep last occurrence)
    combined_df.drop_duplicates(subset=['scenario_name', 'R0', 'beta_event', 'I_ss'], keep='last', inplace=True)

    # Sort: beta_event (asc), then scenario_name (asc)
    combined_df.sort_values(by=['beta_event', 'scenario_name'], inplace=True)

    # Save
    out_path = OUTPUT_DIR / output_file
    combined_df.to_csv(out_path, index=False)
    print(f"Successfully created merged file: {out_path}")
    
    return combined_df

def visualize_distributions(df, results_dir=RESULTS_DIR):
    """
    Generates line plots of average epidemic outbreak size against R0 for given beta_event values.
    """
    print("Generating average size vs R0 curves...")

    # Ensure final_size_mean exists
    if 'final_size_mean' not in df.columns:
        print("Calculating final_size_mean from values...")
        def get_mean(x):
            try:
                if isinstance(x, str):
                    val_list = ast.literal_eval(x)
                else:
                    val_list = x
                return np.mean(val_list)
            except:
                return np.nan
        df['final_size_mean'] = df['values'].apply(get_mean)

    # Get unique parameter sets to generate separate plots
    iss_values = sorted(df['I_ss'].unique())
    
    # Get event betas (exclude 0.0 if it is only for no_event)
    # We assume no_event has beta_event = 0.0
    event_betas = sorted([b for b in df['beta_event'].unique() if b > 0])
    if not event_betas:
        # If only baseline exists or everything is 0, just plot for 0
        event_betas = [0.0]

    # Define markers for different scenarios
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    sns.set_theme(style="whitegrid")

    for iss in iss_values:
        # Get baseline data for this I_ss (beta_event=0)
        baseline_df = df[(df['scenario_name'] == 'no_event') & (df['I_ss'] == iss)].sort_values('R0')
        
        for beta in event_betas:
            # Get event data for this I_ss and beta
            # We include all scenarios that match this beta (excluding no_event which is handled separately)
            event_df = df[
                (df['scenario_name'] != 'no_event') & 
                (df['I_ss'] == iss) & 
                (np.isclose(df['beta_event'], beta))
            ]
            
            if event_df.empty and baseline_df.empty:
                continue

            plt.figure(figsize=(10, 6))
            
            # Plot Baseline
            if not baseline_df.empty:
                plt.plot(
                    baseline_df['R0'], 
                    baseline_df['final_size_mean'], 
                    label='Baseline (No Event)', 
                    color='black', 
                    linestyle='--', 
                    marker='x',
                    linewidth=2,
                    markersize=8
                )

            # Plot Event Scenarios
            scenarios = sorted(event_df['scenario_name'].unique())
            for idx, scenario in enumerate(scenarios):
                subset = event_df[event_df['scenario_name'] == scenario].sort_values('R0')
                marker = markers[idx % len(markers)]
                
                plt.plot(
                    subset['R0'], 
                    subset['final_size_mean'], 
                    label=scenario, 
                    marker=marker,
                    linewidth=1.5,
                    alpha=0.8,
                    markersize=6
                )
            
            plt.title(f'Average epidemic size vs $R_0$\n($I_{{ss}}$={iss}, $\\beta_{{event}}$={beta})', fontsize=16)
            plt.xlabel('$R_0$', fontsize=14)
            plt.ylabel('Average final size (fraction recovered)', fontsize=14)
            plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
            plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            plt.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            out_file = OUTPUT_DIR / f"avg_size_vs_R0_beta{int(beta*100)}_Iss{iss}.png"
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {out_file}")

def visualize_density_heatmaps(df, results_dir=RESULTS_DIR):
    """
    Generates heatmaps of the normalized outbreak size distributions against R0.
    X-axis: R0, Y-axis: Normalized Outbreak Size, Color: Frequency/Density.
    """
    print("Generating outbreak size density heatmaps...")

    # Ensure values are parsed
    if 'values_list' not in df.columns:
        def safe_eval(x):
            if isinstance(x, list): return x
            try:
                return ast.literal_eval(x)
            except:
                return []
        df['values_list'] = df['values'].apply(safe_eval)

    # Explode list to rows for plotting
    cols = ['scenario_name', 'R0', 'beta_event', 'I_ss', 'values_list']
    # Filter columns to avoid issues if other columns are present
    plot_df = df[[c for c in cols if c in df.columns]].copy()
    plot_df = plot_df.explode('values_list')
    plot_df['final_size'] = pd.to_numeric(plot_df['values_list'], errors='coerce')
    plot_df = plot_df.dropna(subset=['final_size'])

    # Group by (Scenario, Beta, Iss)
    groups = plot_df.groupby(['scenario_name', 'beta_event', 'I_ss'])

    for (scenario, beta, iss), group in groups:
        unique_r0 = sorted(group['R0'].unique())
        if len(unique_r0) < 2:
            continue

        # Define Y-axis bins (0 to 1)
        y_bins = np.linspace(0, 1, 51) # 50 bins
        
        # Matrix: Rows=Size, Cols=R0
        heatmap_matrix = np.zeros((len(y_bins)-1, len(unique_r0)))
        
        for i, r0 in enumerate(unique_r0):
            vals = group[group['R0'] == r0]['final_size']
            if len(vals) == 0: continue
            
            hist, _ = np.histogram(vals, bins=y_bins)
            if hist.sum() > 0:
                heatmap_matrix[:, i] = hist / hist.sum()
        
        # Create DataFrame for plotting
        y_labels = np.round(y_bins[:-1], 2)
        heatmap_df = pd.DataFrame(heatmap_matrix, index=y_labels, columns=unique_r0)
        heatmap_df = heatmap_df.sort_index(ascending=False)
        
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(heatmap_df, cmap="Reds", cbar_kws={'label': 'Frequency'}, yticklabels=5, vmin=0)
        
        plt.title(f'Outbreak Size Distribution vs $R_0$\nScenario: {scenario}, $\\beta_{{event}}$={beta}, $I_{{ss}}$={iss}', fontsize=16)
        plt.xlabel('$R_0$', fontsize=14)
        plt.ylabel('Final Size (Fraction Recovered)', fontsize=14)
        
        plt.tight_layout()
        
        out_file = OUTPUT_DIR / f"heatmap_size_dist_{scenario}_beta{int(beta*100)}_Iss{iss}.png"
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"Saved density heatmap: {out_file}")

def visualize_cv(df, results_dir=RESULTS_DIR):
    """
    Generates line plots of the Coefficient of Variation (CV) of outbreak size against R0.
    CV is computed using the first and second moments: CV = sqrt(E[X^2] - E[X]^2) / E[X].
    """
    print("Generating CV vs R0 curves...")

    # Ensure values are parsed
    if 'values_list' not in df.columns:
        def safe_eval(x):
            if isinstance(x, list): return x
            try:
                return ast.literal_eval(x)
            except:
                return []
        df['values_list'] = df['values'].apply(safe_eval)

    # Calculate CV using moments
    def calculate_cv_from_moments(values):
        if not values:
            return np.nan
        vals = np.array(values)
        if len(vals) == 0:
            return np.nan
        
        # First moment (Mean)
        m1 = np.mean(vals)
        
        # Second moment
        m2 = np.mean(vals**2)
        
        # Variance = E[X^2] - (E[X])^2
        variance = m2 - m1**2
        
        # Handle floating point precision issues where variance might be slightly negative
        if variance < 0: 
            variance = 0.0
        
        std_dev = np.sqrt(variance)
        
        if m1 > 1e-9: # Avoid division by zero
            return std_dev / m1
        else:
            return 0.0 

    df['cv'] = df['values_list'].apply(calculate_cv_from_moments)

    # Get unique parameter sets
    iss_values = sorted(df['I_ss'].unique())
    
    # Get event betas (exclude 0.0 if it is only for no_event)
    event_betas = sorted([b for b in df['beta_event'].unique() if b > 0])
    if not event_betas:
        event_betas = [0.0]

    # Define markers for different scenarios
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    sns.set_theme(style="whitegrid")

    for iss in iss_values:
        # Get baseline data for this I_ss (beta_event=0)
        baseline_df = df[(df['scenario_name'] == 'no_event') & (df['I_ss'] == iss)].sort_values('R0')
        
        for beta in event_betas:
            # Get event data for this I_ss and beta
            event_df = df[
                (df['scenario_name'] != 'no_event') & 
                (df['I_ss'] == iss) & 
                (np.isclose(df['beta_event'], beta))
            ]
            
            if event_df.empty and baseline_df.empty:
                continue

            plt.figure(figsize=(10, 6))
            
            # Plot Baseline
            if not baseline_df.empty:
                plt.plot(
                    baseline_df['R0'], 
                    baseline_df['cv'], 
                    label='Baseline (No Event)', 
                    color='black', 
                    linestyle='--', 
                    marker='x',
                    linewidth=2,
                    markersize=8
                )

            # Plot Event Scenarios
            scenarios = sorted(event_df['scenario_name'].unique())
            for idx, scenario in enumerate(scenarios):
                subset = event_df[event_df['scenario_name'] == scenario].sort_values('R0')
                marker = markers[idx % len(markers)]
                
                plt.plot(
                    subset['R0'], 
                    subset['cv'], 
                    label=scenario, 
                    marker=marker,
                    linewidth=1.5,
                    alpha=0.8,
                    markersize=6
                )
            
            plt.title(f'Coefficient of Variation (CV) vs $R_0$\n($I_{{ss}}$={iss}, $\\beta_{{event}}$={beta})', fontsize=16)
            plt.xlabel('$R_0$', fontsize=14)
            plt.ylabel('Coefficient of Variation (CV)', fontsize=14)
            plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
            plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            plt.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            out_file = OUTPUT_DIR / f"cv_vs_R0_beta{int(beta*100)}_Iss{iss}.png"
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {out_file}")

def visualize_cv_vs_beta(df, results_dir=RESULTS_DIR):
    """
    Generates line plots of the Coefficient of Variation (CV) of outbreak size against beta_event.
    """
    print("Generating CV vs beta_event curves...")

    # Ensure cv is calculated
    if 'cv' not in df.columns:
        if 'values_list' not in df.columns:
            def safe_eval(x):
                if isinstance(x, list): return x
                try: return ast.literal_eval(x)
                except: return []
            df['values_list'] = df['values'].apply(safe_eval)

        def calculate_cv_from_moments(values):
            if not values: return np.nan
            vals = np.array(values)
            if len(vals) == 0: return np.nan
            m1 = np.mean(vals)
            m2 = np.mean(vals**2)
            variance = m2 - m1**2
            if variance < 0: variance = 0.0
            std_dev = np.sqrt(variance)
            return std_dev / m1 if m1 > 1e-9 else 0.0
        df['cv'] = df['values_list'].apply(calculate_cv_from_moments)

    # Get unique parameter sets
    iss_values = sorted(df['I_ss'].unique())
    r0_values = sorted(df['R0'].unique())

    # Define markers for different scenarios
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    sns.set_theme(style="whitegrid")

    for iss in iss_values:
        for r0 in r0_values:
            # Get baseline CV for this R0/Iss
            baseline_row = df[(df['scenario_name'] == 'no_event') & (df['I_ss'] == iss) & (np.isclose(df['R0'], r0))]
            baseline_cv = baseline_row['cv'].iloc[0] if not baseline_row.empty else None

            # Get event data for this R0/Iss
            event_df = df[
                (df['scenario_name'] != 'no_event') &
                (df['I_ss'] == iss) &
                (np.isclose(df['R0'], r0))
            ]

            if event_df.empty:
                continue

            plt.figure(figsize=(10, 6))
            ax = plt.gca()

            # Plot Baseline as a horizontal line
            if baseline_cv is not None:
                ax.axhline(baseline_cv, color='black', linestyle='--', label='Baseline (No Event)', linewidth=2, zorder=1)

            # Plot Event Scenarios
            scenarios = sorted(event_df['scenario_name'].unique())
            for idx, scenario in enumerate(scenarios):
                subset = event_df[event_df['scenario_name'] == scenario].sort_values('beta_event')
                marker = markers[idx % len(markers)]
                
                plt.plot(subset['beta_event'], subset['cv'], label=scenario, marker=marker, linewidth=1.5, alpha=0.8, markersize=6)
            
            plt.title(f'Coefficient of Variation (CV) vs $\\beta_{{event}}$\n($R_0$={r0}, $I_{{ss}}$={iss})', fontsize=16)
            plt.xlabel('$\\beta_{event}$', fontsize=14)
            plt.ylabel('Coefficient of Variation (CV)', fontsize=14)
            plt.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            out_file = OUTPUT_DIR / f"cv_vs_beta_R{int(r0*100)}_Iss{iss}.png"
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {out_file}")

if __name__ == "__main__":
    df = load_and_merge_summaries()
    if df is not None:
        visualize_distributions(df)
        visualize_density_heatmaps(df)
        visualize_cv(df)
        # visualize_cv_vs_beta(df)