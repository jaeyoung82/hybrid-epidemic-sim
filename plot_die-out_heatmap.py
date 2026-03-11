import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# --- Configuration ---
# Try to import the main config to get the correct number of iterations and results directory.
# If not found, default to standard values.
try:
    import config as cfg
    N_ITERATIONS = cfg.n_iterations
    RESULTS_DIR = getattr(cfg, 'results_dir', Path("results_hybrid_sim"))
except ImportError:
    print("Warning: 'config.py' not found. Using default settings (N=100).")
    N_ITERATIONS = 100
    RESULTS_DIR = Path("results")

def load_die_out_data(filepath):
    """
    Robustly loads the die-out records, handling potential schema changes.
    """
    if not os.path.exists(filepath):
        print(f"Error: Log file not found at {filepath}")
        return None
    
    # Define the expected columns (matching the latest simulation.py)
    cols = [
        "scenario_name", "I_ss", "run_id", "event_model_type", "recruitment_model",
        "initially_infected", "event_patch_id", "beta_event", "R0",
        "initially_infected_seed", "initially_infected_patch", "die_out_day", "recovered_at_die_out"
    ]
    
    try:
        # Use python engine and skip bad lines to handle mixed-format files
        df = pd.read_csv(filepath, names=cols, header=0, on_bad_lines='skip', engine='python')
    except TypeError:
        # Fallback for older pandas versions
        df = pd.read_csv(filepath, names=cols, header=0, error_bad_lines=False, engine='python')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
        
    return df

def generate_heatmaps():
    filename = RESULTS_DIR / "disease_die_out_records.txt"
    print(f"Reading data from: {filename}")
    
    df = load_die_out_data(filename)
    if df is None or df.empty:
        print("No die-out data available to plot.")
        return

    # Clean R0 to ensure consistent grouping (avoid float precision issues)
    df['R0'] = df['R0'].round(4)
    
    # Get unique parameter values present in the logs
    unique_iss = sorted(df['I_ss'].unique())
    
    # Identify unique beta_event values used in EVENT scenarios
    # (Baseline 'no_event' rows might have arbitrary beta values, so we ignore them for this list)
    event_rows = df[df['scenario_name'] != 'no_event']
    if not event_rows.empty:
        unique_betas = sorted(event_rows['beta_event'].unique())
    else:
        unique_betas = sorted(df['beta_event'].unique())
        
    if not unique_betas:
        unique_betas = [0.0] # Fallback if only baseline exists

    print(f"Found data for I_ss: {unique_iss}")
    print(f"Found data for Beta_Event: {unique_betas}")

    # --- Generate one heatmap per (I_ss, Beta_Event) combination ---
    for iss in unique_iss:
        for beta in unique_betas:
            
            # Filter Data:
            # 1. Match the current I_ss
            # 2. Include Event scenarios ONLY if they match the current beta_event
            # 3. Always include Baseline ('no_event') regardless of its logged beta (since baseline is beta-independent)
            mask_iss = (df['I_ss'] == iss)
            mask_event = (df['scenario_name'] != 'no_event') & (np.isclose(df['beta_event'], beta))
            mask_baseline = (df['scenario_name'] == 'no_event')
            
            df_subset = df[mask_iss & (mask_event | mask_baseline)].copy()
            
            # Filter for Early Extinction (Minor Outbreaks)
            # We exclude runs where the recovered count is high (indicating a major outbreak finished).
            # Threshold: 2000 (approx 15-20% of a typical 12k population, safe upper bound for early extinction)
            if 'recovered_at_die_out' in df_subset.columns and not df_subset['recovered_at_die_out'].isnull().all():
                df_subset = df_subset[df_subset['recovered_at_die_out'] < 2000]
            
            if df_subset.empty:
                continue

            # Calculate Frequency
            # Count unique run_ids for each (R0, Scenario) pair.
            # This represents the number of times the disease died out.
            die_out_counts = df_subset.groupby(['R0', 'scenario_name'])['run_id'].nunique()
            
            # Convert to frequency (0.0 to 1.0)
            die_out_freq = die_out_counts / N_ITERATIONS
            
            # Pivot table for Heatmap: Rows=R0, Cols=Scenario
            heatmap_data = die_out_freq.reset_index().pivot(index='R0', columns='scenario_name', values='run_id')
            
            # Fill NaNs with 0 (implies 0 die-outs recorded for that combination)
            heatmap_data = heatmap_data.fillna(0)
            
            # Sort R0 descending so high R0 is at the top (or bottom, depending on preference).
            # Standard matrix convention: Index 0 at top. 
            # Let's sort Descending so higher R0 (more infectious) is at the top? 
            # Actually, usually low R0 (high extinction) is at the top in a matrix if index is sorted ascending.
            # Let's sort index ascending (low R0 at top) to match standard heatmap behavior, 
            # or descending to put high R0 at top. Let's do Descending R0 at top.
            heatmap_data = heatmap_data.sort_index(ascending=False)

            # Ensure 'no_event' is the first column if it exists
            cols = heatmap_data.columns.tolist()
            if 'no_event' in cols:
                cols.remove('no_event')
                cols.sort()
                cols.insert(0, 'no_event')
                heatmap_data = heatmap_data[cols]

            # --- Plotting ---
            plt.figure(figsize=(8, 6))
            
            # Create Heatmap
            # cmap="YlGnBu" -> Yellow (Low) to Blue (High)
            # vmin=0, vmax=1 ensures the color scale is fixed from 0% to 100%
            ax = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="Reds_r", 
                             vmin=0, vmax=1, linewidths=.5, cbar_kws={'label': 'Early Extinction Probability'})
            
            ax.set_title(f"Early Extinction Probability (Minor Outbreaks)\n($I_{{ss}}$={iss}, $\\beta_{{event}}$={beta})", fontsize=16)
            ax.set_xlabel("Scenario", fontsize=14)
            ax.set_ylabel("$R_0$", fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save
            out_filename = RESULTS_DIR / f"heatmap_die_out_Iss{iss}_beta{int(beta*100)}.png"
            plt.savefig(out_filename, dpi=300)
            plt.close()
            print(f"Saved heatmap: {out_filename}")

if __name__ == "__main__":
    generate_heatmaps()
