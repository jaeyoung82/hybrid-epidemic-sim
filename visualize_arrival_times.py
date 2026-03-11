import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path

# Configuration
RESULTS_DIR = Path("results_hybrid_sim")
OUTPUT_DIR = RESULTS_DIR / "arrival_times_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

def parse_filename_info(filename):
    """Extracts scenario parameters from the standard filename format."""
    name = filename.stem
    parts = name.split('_')
    # Expected format: run_daily_active_cases_{scenario}_R{r0}_beta{beta}_Iss{iss}
    try:
        iss_part = parts[-1]   # e.g., Iss10
        beta_part = parts[-2]  # e.g., beta20
        r0_part = parts[-3]    # e.g., R150
        
        iss = int(iss_part.replace('Iss', ''))
        beta = int(beta_part.replace('beta', '')) / 100.0
        r0 = int(r0_part.replace('R', '')) / 100.0
        
        # Scenario name is everything between 'cases' and 'R...'
        start_idx = 4
        end_idx = -3
        scenario = "_".join(parts[start_idx:end_idx])
        
        return scenario, r0, beta, iss
    except Exception:
        return "Unknown", 0, 0, 0

def analyze_file(filepath):
    print(f"Processing {filepath.name}...")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  Error reading file: {e}")
        return

    scenario, r0, beta, iss = parse_filename_info(filepath)
    
    # Identify day columns (e.g., day_0.0, day_1.0)
    day_cols = [c for c in df.columns if c.startswith('day_')]
    # Sort columns numerically by day
    day_cols.sort(key=lambda x: float(x.split('_')[1]))
    days = np.array([float(c.split('_')[1]) for c in day_cols])
    
    # --- 1. Calculate Arrival Times ---
    # We want to find the first day where cases > 0 for each (run, district)
    arrival_times = []
    
    # Iterate over runs
    for run_id in df['run_id'].unique():
        # Subset for this run: rows=districts, cols=days
        run_df = df[df['run_id'] == run_id].set_index('district_id')[day_cols]
        
        # Create boolean mask of infections
        infected_mask = run_df.values > 0
        
        # Find index of first True along the time axis
        first_indices = np.argmax(infected_mask, axis=1)
        
        # Check if district was ever infected (argmax returns 0 if all False)
        any_infected = np.any(infected_mask, axis=1)
        
        for i, district_id in enumerate(run_df.index):
            if any_infected[i]:
                arrival_time = days[first_indices[i]]
                arrival_times.append({
                    'run_id': run_id,
                    'district_id': district_id,
                    'arrival_time': arrival_time
                })
            else:
                # Never infected
                arrival_times.append({
                    'run_id': run_id,
                    'district_id': district_id,
                    'arrival_time': np.nan
                })
                
    res_df = pd.DataFrame(arrival_times)
    
    # --- 2. Quantify Affected Patches ---
    # Count non-NaN arrival times per run
    affected_counts = res_df.groupby('run_id')['arrival_time'].count()
    print(f"  -> Mean patches affected: {affected_counts.mean():.2f} +/- {affected_counts.std():.2f}")

    # --- 3. Visualization: Spread Curve ---
    # Plot number of affected patches over time
    spread_curves = []
    
    for run_id in res_df['run_id'].unique():
        run_arrivals = res_df[res_df['run_id'] == run_id]['arrival_time'].dropna().values
        run_arrivals.sort()
        # Count how many arrivals occurred by each time step t
        counts = np.searchsorted(run_arrivals, days, side='right')
        spread_curves.append(counts)
        
    spread_curves = np.array(spread_curves)
    mean_spread = np.mean(spread_curves, axis=0)
    std_spread = np.std(spread_curves, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(days, mean_spread, label='Mean Affected Patches', color='blue', lw=2)
    plt.fill_between(days, mean_spread - std_spread, mean_spread + std_spread, color='blue', alpha=0.2)
    plt.xlabel('Time (Days)')
    plt.ylabel('Number of Affected Districts')
    plt.title(f'Spread Curve: {scenario}\n($R_0$={r0}, $\\beta_{{event}}$={beta}, $I_{{ss}}$={iss})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_name = OUTPUT_DIR / f"spread_curve_{scenario}_R{int(r0*100)}_beta{int(beta*100)}_Iss{iss}.png"
    plt.savefig(out_name, dpi=150)
    plt.close()
    print(f"  -> Saved spread curve to {out_name.name}")
    
    # --- 4. Visualization: Arrival Time Boxplot ---
    plt.figure(figsize=(12, 6))
    
    # Pivot: Index=run_id, Columns=district_id, Values=arrival_time
    pivot_arrival = res_df.pivot(index='run_id', columns='district_id', values='arrival_time')
    
    # Sort districts by median arrival time for cleaner plot
    medians = pivot_arrival.median().sort_values()
    pivot_arrival = pivot_arrival[medians.index]
    
    # Filter to show only districts that were infected in at least one run
    pivot_arrival = pivot_arrival.dropna(axis=1, how='all')
    
    if not pivot_arrival.empty:
        pivot_arrival.boxplot(rot=90, fontsize=8)
        plt.xlabel('District ID (Sorted by Median Arrival)')
        plt.ylabel('Arrival Time (Days)')
        plt.title(f'Arrival Time Distribution per District: {scenario}\n($R_0$={r0}, $\\beta_{{event}}$={beta}, $I_{{ss}}$={iss})')
        plt.tight_layout()
        
        out_name_box = OUTPUT_DIR / f"arrival_boxplot_{scenario}_R{int(r0*100)}_beta{int(beta*100)}_Iss{iss}.png"
        plt.savefig(out_name_box, dpi=150)
        plt.close()
        print(f"  -> Saved boxplot to {out_name_box.name}")
    else:
        print("  -> No infections spread to other districts. Skipping boxplot.")

def main():
    files = list(RESULTS_DIR.glob("run_daily_active_cases_*.csv"))
    if not files:
        print("No 'run_daily_active_cases_*.csv' files found in results directory.")
        print("Please run the simulation again with the updated code to generate these files.")
        return
        
    print(f"Found {len(files)} daily case files. Generating visualizations...")
    for f in files:
        analyze_file(f)

if __name__ == "__main__":
    main()
