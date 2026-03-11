import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import numpy as np

# --- Configuration ---
RESULTS_DIR = Path("results_hybrid_sim")
OUTPUT_DIR = Path("results_hybrid_sim")

def parse_filename_params(filename):
    """
    Extracts beta, R0, and Iss from the filename.
    Expected format: n_new_infections_betaEvent{int}_R{int}_Iss{int}.txt
    """
    # Regex to capture integers after specific keywords
    match = re.search(r"betaEvent(\d+)_R(\d+)_Iss(\d+)", filename.name)
    if match:
        beta = int(match.group(1)) / 100.0
        r0 = int(match.group(2)) / 100.0
        iss = int(match.group(3))
        return beta, r0, iss
    return None, None, None

def main():
    if not RESULTS_DIR.exists():
        print(f"Error: Results directory '{RESULTS_DIR}' not found.")
        return

    # Find all infection log files
    files = list(RESULTS_DIR.glob("n_new_infections_*.txt"))
    if not files:
        print("No 'n_new_infections_*.txt' files found.")
        return

    print(f"Found {len(files)} infection log files. Processing...")

    results = []

    for f in files:
        beta, r0, iss = parse_filename_params(f)
        if beta is None:
            continue

        try:
            # Read CSV. Handle potential mixed types or errors robustly.
            # Header expected: scenario_name,run_id,event_day,n_all,n_0,n_1...
            df = pd.read_csv(f)
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
            continue

        if df.empty:
            continue

        # Filter for the main event day (Day 0)
        # If you have multiple event days, you might want to group by event_day too,
        # but usually we care about the initial spark at day 0.
        df_day0 = df[df['event_day'] == 0]

        if df_day0.empty:
            continue

        # Group by scenario (e.g., AMS_dance, Leipzig_1) contained in this file
        for scenario, group in df_day0.groupby('scenario_name'):
            total_runs = group['run_id'].nunique()
            
            # Sum n_all per run (in case of duplicate rows, though unlikely for day 0)
            infections_per_run = group.groupby('run_id')['n_all'].sum()
            
            # Count runs where ZERO infections occurred
            zero_infection_runs = (infections_per_run == 0).sum()
            
            failure_prob = zero_infection_runs / total_runs if total_runs > 0 else 0.0

            results.append({
                'scenario_name': scenario,
                'beta_event': beta,
                'R0': r0,
                'I_ss': iss,
                'failure_prob': failure_prob,
                'total_runs': total_runs
            })

    if not results:
        print("No valid data extracted.")
        return

    df_res = pd.DataFrame(results)
    
    # --- Generate Heatmaps ---
    unique_iss = sorted(df_res['I_ss'].unique())
    unique_scenarios = sorted(df_res['scenario_name'].unique())

    print("\nGenerating Event Failure Probability Heatmaps (Zero Primary Infections)...")

    for scenario in unique_scenarios:
        for iss in unique_iss:
            subset = df_res[(df_res['scenario_name'] == scenario) & (df_res['I_ss'] == iss)]
            
            if subset.empty:
                continue

            # Pivot: Rows=R0, Cols=Beta_Event
            try:
                pivot = subset.pivot_table(
                    index='R0', 
                    columns='beta_event', 
                    values='failure_prob'
                )
            except Exception as e:
                print(f"Error pivoting data for {scenario}, Iss={iss}: {e}")
                continue
            
            # Sort R0 descending (High R0 at top)
            pivot = pivot.sort_index(ascending=False)

            plt.figure(figsize=(6, 4))
            
            # cmap="Greens": Dark Green = High Failure Prob (Good), Light/White = Low Failure Prob (Bad)
            # Or "Reds_r": Dark Red = Low Failure (Bad), White = High Failure (Good)
            # Let's use "Blues" where Dark Blue = High Probability of Zero Infections (Event Failed)
            sns.heatmap(
                pivot, 
                annot=True, 
                fmt=".2f", 
                cmap="Blues", 
                vmin=0, 
                vmax=1,
                cbar_kws={'label': 'Probability of Zero Event Infections'}
            )
            
            plt.title(f"Event Failure Probability (Zero Primary Infections)\nScenario: {scenario}, $I_{{ss}}$={iss}")
            plt.ylabel("Basic Reproduction Number ($R_0$)")
            plt.xlabel("Event Transmission Rate ($\\beta_{event}$)")
            plt.tight_layout()

            out_filename = OUTPUT_DIR / f"event_failure_heatmap_{scenario}_Iss{iss}.png"
            plt.savefig(out_filename, dpi=300)
            plt.close()
            print(f"Saved: {out_filename}")

if __name__ == "__main__":
    main()
