import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import data_loader
import simulation

# --- Configuration ---
# Try to import the main config to get the correct number of iterations and results directory.
try:
    import config as cfg
    N_ITERATIONS = cfg.n_iterations
    RESULTS_DIR = getattr(cfg, 'results_dir', Path("results_hybrid_sim"))
except ImportError:
    print("Warning: 'config.py' not found. Using default settings (N=100).")
    N_ITERATIONS = 100
    RESULTS_DIR = Path("results_hybrid_sim")

def load_die_out_data(filepath):
    """
    Robustly loads the die-out records.
    """
    if not filepath.exists():
        print(f"Error: Log file not found at {filepath}")
        return None
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
        
    return df

def generate_extinction_diagrams():
    """
    Generates 2D heatmaps (phase diagrams) of epidemic extinction frequency based on R0 and beta_event.
    """
    die_out_file = RESULTS_DIR / "disease_die_out_records.txt"

    # --- Calculate Theoretical Baseline Invasion Threshold ---
    sim_data = data_loader.load_and_prepare_data(cfg)
    S_mat = simulation.compute_structural_ngm(sim_data)
    eigenvalues = np.linalg.eigvals(S_mat)
    spectral_radius = np.max(np.abs(eigenvalues))
    baseline_invasion_threshold_R0 = 1 / spectral_radius if spectral_radius > 0 else np.inf
    print(f"\nTheoretical Baseline Invasion Threshold (R* > 1) at R0 = {baseline_invasion_threshold_R0:.3f}")

    print(f"Reading data from: {die_out_file}")
    
    df = load_die_out_data(die_out_file)
    if df is None or df.empty:
        print("No die-out data available to plot.")
        return

    # Clean data for consistent grouping
    df['R0'] = df['R0'].round(4)
    df['beta_event'] = df['beta_event'].round(4)
    
    # Get unique parameter values present in the logs
    unique_iss = sorted(df['I_ss'].unique())
    unique_scenarios = sorted(df['scenario_name'].unique())
    # Get all beta values used across all event scenarios to build the Y-axis
    all_betas = sorted(df[~df['scenario_name'].isin(['no_event', 'baseline'])]['beta_event'].unique())
    if not all_betas: # Fallback if only baseline runs exist
        all_betas = sorted(df['beta_event'].unique())

    # --- Generate one heatmap per (I_ss, Scenario) combination ---
    for iss in unique_iss:
        # --- Special Handling for Baseline Scenario ---
        baseline_df = df[(df['I_ss'] == iss) & (df['scenario_name'].isin(['no_event', 'baseline']))].copy()
        if not baseline_df.empty:
            # For baseline, frequency only depends on R0
            baseline_counts = baseline_df.groupby('R0').size()
            baseline_freq = baseline_counts / N_ITERATIONS
            
            # Create a full 2D matrix for the phase diagram
            # Columns are R0 values from the baseline results
            r0_cols = sorted(baseline_freq.index.unique())
            
            # Create an empty df with the correct index and columns
            baseline_heatmap_data = pd.DataFrame(index=all_betas, columns=r0_cols, dtype=float)
            
            # Fill the dataframe: each column (R0) gets its frequency value repeated for all betas
            for r0_val, freq in baseline_freq.items():
                if r0_val in baseline_heatmap_data.columns:
                    baseline_heatmap_data[r0_val] = freq
            
            baseline_heatmap_data.fillna(0, inplace=True)
            
            # Sort for a clean plot
            baseline_heatmap_data = baseline_heatmap_data.sort_index(ascending=False) # High beta at top

            # --- Plotting for Baseline ---
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                baseline_heatmap_data, 
                annot=True, fmt=".2f", cmap="Reds", linewidths=.5, ax=ax,
                vmin=0, vmax=1, cbar_kws={'label': 'Extinction Frequency (Probability of Die-Out)'}
            )
            ax.set_title(f"Extinction Phase Diagram: Baseline (No Event)\n($I_{{ss}}$={int(iss)})", fontsize=16)
            ax.set_xlabel("$R_0$ (Community Transmission)", fontsize=14)
            ax.set_ylabel("$\\beta_{event}$ (Event Transmission Rate)", fontsize=14)
            plt.tight_layout()
            
            out_filename = RESULTS_DIR / f"phase_diagram_extinction_freq_baseline_Iss{int(iss)}.png"
            plt.savefig(out_filename, dpi=300)
            plt.close(fig)
            print(f"Saved extinction phase diagram: {out_filename}")

        # --- Handling for Event Scenarios ---
        for scenario in unique_scenarios:
            # Skip baseline as it's handled above
            if scenario in ['no_event', 'baseline']:
                continue

            # Filter data for the current combination
            df_subset = df[(df['I_ss'] == iss) & (df['scenario_name'] == scenario)].copy()
            
            if df_subset.empty:
                continue

            # --- Calculate Frequency and Create Pivot Table ---
            die_out_counts = df_subset.groupby(['R0', 'beta_event']).size()
            die_out_freq = die_out_counts / N_ITERATIONS
            
            # Pivot to create the heatmap structure
            heatmap_data = die_out_freq.unstack(level='R0')

            # Reindex to include all betas, filling missing with 0. This ensures all heatmaps have the same Y-axis.
            heatmap_data = heatmap_data.reindex(all_betas, fill_value=0)
            
            # Fill NaNs with 0, assuming combinations not in the log had 0 die-outs.
            heatmap_data.fillna(0, inplace=True)

            # Sort for a clean plot
            heatmap_data = heatmap_data.sort_index(ascending=False) # High beta at top
            heatmap_data = heatmap_data.sort_index(axis=1, ascending=True) # Low R0 on left

            # --- Plotting ---
            fig, ax = plt.subplots(figsize=(8, 6))
            
            sns.heatmap(
                heatmap_data, 
                annot=True, 
                fmt=".2f",
                cmap="Reds", # High frequency = dark red, consistent with plot_die-out_heatmap.py
                linewidths=.5, 
                ax=ax,
                vmin=0, # Fix color scale from 0%
                vmax=1, # to 100%
                cbar_kws={'label': 'Extinction Frequency (Probability of Die-Out)'}
            )
            
            # --- Add Contour Line for the 50% Invasion Threshold ---
            # The contour function needs X, Y, Z data.
            # Z is our heatmap data. X and Y are the R0 and beta values.
            if heatmap_data.shape[0] > 1 and heatmap_data.shape[1] > 1:
                X, Y = np.meshgrid(heatmap_data.columns, heatmap_data.index)
                # The contour is drawn where the data value equals the level.
                CS = ax.contour(
                    X, Y, heatmap_data.values,
                    levels=[0.5], # The 50% threshold
                    colors='white',
                    linestyles='--',
                    linewidths=2.0
                )
                ax.clabel(CS, inline=True, fontsize=12, fmt='Invasion Threshold')

            # --- Add Vertical Line for the Baseline Theoretical Threshold ---
            ax.axvline(
                baseline_invasion_threshold_R0,
                color='cyan', linestyle=':', lw=3,
                label=f'Baseline Threshold ($R_0 \\approx {baseline_invasion_threshold_R0:.2f}$)'
            )

            ax.set_title(f"Extinction Phase Diagram: {scenario}\n($I_{{ss}}$={int(iss)})", fontsize=16)
            ax.set_xlabel("$R_0$ (Community Transmission)", fontsize=14)
            ax.set_ylabel("$\\beta_{event}$ (Event Transmission Rate)", fontsize=14)
            ax.legend() # Add legend to include the new line
            plt.tight_layout()
            
            out_filename = RESULTS_DIR / f"phase_diagram_extinction_freq_{scenario}_Iss{int(iss)}.png"
            plt.savefig(out_filename, dpi=300)
            plt.close(fig)
            print(f"Saved extinction phase diagram: {out_filename}")

if __name__ == "__main__":
    generate_extinction_diagrams()