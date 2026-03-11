import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

try:
    import config as cfg
    RESULTS_DIR = getattr(cfg, 'results_dir', Path("results_hybrid_sim"))
except ImportError:
    print("Warning: 'config.py' not found. Using default results directory.")
    RESULTS_DIR = Path("results_hybrid_sim")

def load_peak_data(filepath):
    """
    Robustly loads the peak time and value records.
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

def generate_phase_diagrams():
    """
    Generates 2D heatmaps (phase diagrams) of epidemic outcomes based on R0 and beta_event.
    """
    peak_file = RESULTS_DIR / "peak-time-distribution.txt"
    print(f"Reading data from: {peak_file}")
    
    df = load_peak_data(peak_file)
    if df is None or df.empty:
        print("No peak time data available to plot.")
        return

    # Clean data for consistent grouping
    df['R0'] = df['R0'].round(4)
    df['beta_event'] = df['beta_event'].round(4)
    
    # Get unique parameter values present in the logs
    unique_iss = sorted(df['initially_infected'].unique())
    unique_scenarios = sorted(df['scenario_name'].unique())

    # --- Generate one heatmap per (I_ss, Scenario) combination ---
    for iss in unique_iss:
        for scenario in unique_scenarios:
            # Skip baseline for this plot as it's independent of beta_event
            if scenario == 'baseline':
                continue

            # Filter data for the current combination
            df_subset = df[(df['initially_infected'] == iss) & (df['scenario_name'] == scenario)].copy()
            
            if df_subset.empty:
                continue

            # --- Create Pivot Table for the Heatmap ---
            # Index (Y-axis) = beta_event
            # Columns (X-axis) = R0
            # Values (Color) = peak_value_avg
            try:
                heatmap_data = df_subset.pivot_table(
                    index='beta_event', 
                    columns='R0', 
                    values='peak_value_avg'
                )
            except Exception as e:
                print(f"Could not create pivot table for I_ss={iss}, Scenario={scenario}: {e}")
                continue

            # Sort for a clean plot
            heatmap_data = heatmap_data.sort_index(ascending=False) # High beta at top
            heatmap_data = heatmap_data.sort_index(axis=1, ascending=True) # Low R0 on left

            # --- Plotting ---
            fig, ax = plt.subplots(figsize=(8, 6))
            
            sns.heatmap(
                heatmap_data, 
                annot=True, 
                fmt=".0f",  # Format annotations as integers
                cmap="Reds", # Use a red color scale from light to dark
                linewidths=.5, 
                ax=ax,
                cbar_kws={'label': 'Mean Peak Active Cases (E+I)'}
            )
            
            ax.set_title(f"Epidemic Phase Diagram: {scenario}\n($I_{{ss}}$={int(iss)})", fontsize=16)
            ax.set_xlabel("$R_0$ (Community Transmission)", fontsize=14)
            ax.set_ylabel("$\\beta_{event}$ (Event Transmission Rate)", fontsize=14)
            plt.tight_layout()
            
            out_filename = RESULTS_DIR / f"heatmap_peak_cases_{scenario}_Iss{int(iss)}.png"
            plt.savefig(out_filename, dpi=300)
            plt.close(fig)
            print(f"Saved phase diagram: {out_filename}")

if __name__ == "__main__":
    generate_phase_diagrams()