import numpy as np
import os
import pandas as pd
from pathlib import Path
import time
import json
import re
import warnings
import ast
from typing import List, Dict, Any, Optional, Tuple, Union

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.patches import Polygon, Patch, Rectangle
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    import seaborn as sns
    HAS_PLOTTING = True
except Exception:
    plt = None
    ticker = None
    Polygon = None
    Patch = None
    Rectangle = None
    Line2D = None
    LinearSegmentedColormap = None
    TwoSlopeNorm = None
    sns = None
    HAS_PLOTTING = False

try:
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform
    HAS_CLUSTERING = True
except ImportError:
    HAS_CLUSTERING = False

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

try:
    from scipy import stats
except ImportError:
    stats = None

def update_summary_csv(filename: Path, data_list: List[Dict[str, Any]]):
    """
    Updates a summary CSV file with new data, handling duplicates.
    This function reads the existing file, appends new data, removes duplicates,
    and overwrites the file, making it safe for continuous updates.
    """
    if not data_list:
        return

    new_df = pd.DataFrame(data_list)
    
    # Ensure results directory exists
    filename.parent.mkdir(parents=True, exist_ok=True)

    if filename.exists() and os.path.getsize(filename) > 0:
        try:
            existing_df = pd.read_csv(filename)
            # Convert any column that looks like a list-string back to a list
            for col in existing_df.columns:
                if col in ['values', 'extinction_times', 'final_sizes']:
                     existing_df[col] = existing_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
            
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            combined_df = new_df
    else:
        combined_df = new_df

    # Define key columns for identifying unique runs
    key_cols = ['scenario_name', 'R0', 'beta_event', 'I_ss']
    present_key_cols = [col for col in key_cols if col in combined_df.columns]

    # Round float columns to avoid precision issues during deduplication
    for col in ['R0', 'beta_event']:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].round(6)

    if present_key_cols:
        combined_df.drop_duplicates(subset=present_key_cols, keep='last', inplace=True)
        combined_df.sort_values(by=present_key_cols, inplace=True)

    combined_df.to_csv(filename, index=False)

def save_realizations(realizations: Dict[str, np.ndarray], cfg: Any):
    """Saves aggregated SEIR realizations to CSV files."""
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    scenario = cfg.event_model_type
    if not scenario.startswith('no_event'):
        scenario = getattr(cfg, 'current_event_scenario_name', scenario)
    
    base_name = f"realizations_{scenario}_R{int(cfg.R_0*100)}_beta{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}" if not scenario.startswith('no_event') else f"realizations_{scenario}_R{int(cfg.R_0*100)}_Iss{cfg.I_ss}"
    
    for comp, data in realizations.items():
        df = pd.DataFrame(data)
        # Column 0 is time, others are run_ids
        cols = ['time'] + [f'run_{i}' for i in range(data.shape[1]-1)]
        df.columns = cols
        df.to_csv(results_dir / f"{base_name}_{comp}.csv", index=False)

def save_event_recruitment_records(recruitment_data: np.ndarray, infected_attendee_counts: np.ndarray, cfg: Any):
    """Saves the spatial distribution of recruited attendees and infected seeds."""
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    if cfg.event_model_type.startswith('no_event'):
        scenario = cfg.event_model_type
    else:
        scenario = getattr(cfg, 'current_event_scenario_name', cfg.event_model_type)
    
    base_name = f"spatial_records_{scenario}_beta{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}"
    
    df = pd.DataFrame({
        'district_id': range(len(recruitment_data)),
        'total_attendees': recruitment_data,
        'infected_attendees': infected_attendee_counts
    })
    df.to_csv(results_dir / f"{base_name}.csv", index=False)

def get_arrival_times_data(all_runs_results, n_patches, n_iterations):
    """Public wrapper for _get_arrival_times to allow post-hoc data access."""
    return _get_arrival_times(all_runs_results, n_patches, n_iterations)

def create_summary_plot(S_realizations, E_realizations, I_realizations, R_realizations, data, cfg, event_total_exposures=None):
    """
    Creates a single figure summarizing simulation results for any scenario type.
    The layout changes depending on whether it's an event scenario or not.
    """
    if not getattr(cfg, 'plot_on_the_fly', True):
        return

    scenario_type = cfg.event_model_type
    is_event_scenario = scenario_type in ['large_venue', 'multi_venue']

    time = S_realizations[:, 0]
    total_population = data['population_df']['population'].sum()
    n_iterations = cfg.n_iterations

    # --- Figure and Subplot Layout ---
    if is_event_scenario:
        fig = plt.figure(figsize=(22, 12))
        gs = fig.add_gridspec(2, 2)
        ax_seir = fig.add_subplot(gs[0, 0])
        ax_zoom = fig.add_subplot(gs[0, 1])
        ax_event = fig.add_subplot(gs[1, 0])
        ax_peak = fig.add_subplot(gs[1, 1])
    else: # no_event
        fig = plt.figure(figsize=(30, 7))
        gs = fig.add_gridspec(1, 3)
        ax_seir = fig.add_subplot(gs[0, 0])
        ax_zoom = fig.add_subplot(gs[0, 1])
        ax_peak = fig.add_subplot(gs[0, 2])

    # --- Plotting Data ---
    plot_seir_evolution(ax_seir, time, total_population, n_iterations, S_realizations, E_realizations, I_realizations, R_realizations)
    plot_exposed_infected_zoom(ax_zoom, time, total_population, n_iterations, E_realizations, I_realizations)
    plot_peak_time_distribution(ax_peak, time, n_iterations, E_realizations, I_realizations)

    if is_event_scenario and event_total_exposures is not None:
        plot_event_infection_distribution(ax_event, event_total_exposures)

    # --- Finalize and Save ---
    beta_event = cfg.event_base_transmission_rate
    R0 = cfg.R_0

    # --- Titles ---
    title_map = {
        'large_venue': f'Large Venue Simulation Summary (Scenario: {getattr(cfg, "current_event_scenario_name", "N/A")}, Model: {cfg.attendee_recruitment_model})',
        'multi_venue': f'Multi-Venue Simulation Summary (Scenario: {getattr(cfg, "current_event_scenario_name", "N/A")}, All Groups, Model: {cfg.attendee_recruitment_model})',
        'no_event': 'Baseline Simulation Summary (No Events, Single-Patch Seeding)',
        'no_event_distributed': 'Baseline Simulation Summary (No Events, Distributed Seeding)'
    }
    title = title_map.get(scenario_type, "Simulation Summary")
    fig.suptitle(f'{title}\n($\\beta_0$ = {beta_event:.2f}, $R_0$ = {R0:.2f}, $I_{{ss}}$ = {cfg.I_ss})', fontsize=18)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # type: ignore
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'summary_{scenario_type}_R{int(R0*100)}_betaEvent{int(beta_event*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Summary plot for '{scenario_type}' saved to '{filename}'")
    plt.close(fig)


def create_combined_summary_plot(all_scenario_results, data, cfg, group_name=None):
    """
    Creates a single figure summarizing a group of scenarios.
    A group typically consists of a baseline ('no_event') and one or more event scenarios.
    """
    if not getattr(cfg, 'plot_on_the_fly', True):
        return

    plot_title_name = group_name if group_name else getattr(cfg, "current_event_scenario_name", "All")
    print(f"\n--- Generating combined summary plot for group: {plot_title_name}. ---")

    # Dynamically determine the order of scenarios to plot, ensuring 'no_event' is first.
    active_scenarios = list(all_scenario_results.keys())
    if 'no_event' in active_scenarios:
        active_scenarios.remove('no_event')
        # Sort event scenarios alphabetically for consistent plotting order
        active_scenarios.sort()
        active_scenarios.insert(0, 'no_event')

    nrows = len(active_scenarios)

    if nrows == 0:
        print("No scenarios found to plot.")
        return

    # Adjust figure height dynamically (e.g. 3 inches per row)
    fig_height = 3 * nrows
    fig, axes = plt.subplots(nrows, 2, figsize=(14, fig_height), constrained_layout=True, squeeze=False)

    total_population = data['population_df']['population'].sum()
    n_iterations = cfg.n_iterations

    for row_idx, scenario_name in enumerate(active_scenarios):
        results = all_scenario_results[scenario_name]
        title_text = "Baseline" if scenario_name == 'no_event' else scenario_name
        
        realizations = results['realizations']

        # Extract data for plotting
        E_realizations = realizations['E']
        I_realizations = realizations['I']
        time = E_realizations[:, 0]

        # --- Column 1: Exposed & Infected Zoom ---
        ax_zoom = axes[row_idx, 0]
        plot_exposed_infected_zoom(ax_zoom, time, total_population, n_iterations, E_realizations, I_realizations)
        ax_zoom.set_title(f"Exposed & Infected ({title_text})", fontsize=14)

        # --- Column 2: Peak Time Distribution ---
        ax_peak = axes[row_idx, 1]
        plot_peak_time_distribution(ax_peak, time, n_iterations, E_realizations, I_realizations)
        ax_peak.set_title(f"Peak Time Distribution ({title_text})", fontsize=14)

    # --- Finalize and Save ---
    R0 = cfg.R_0
    beta_event = cfg.event_base_transmission_rate
    fig.suptitle(f'Combined Simulation Summary (Group: {plot_title_name}, Recruitment: {cfg.attendee_recruitment_model}, $\\beta_0$ = {beta_event:.2f}, $R_0$ = {R0:.2f}, $I_{{ss}}$ = {cfg.I_ss})', fontsize=22)

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    if group_name:
        filename_group_part = group_name
    else: # Fallback for old behavior or single-scenario runs
        filename_group_part = getattr(cfg, "current_event_scenario_name", "combined")
    filename = results_dir / f'summary_combined_{filename_group_part}_R{int(R0*100)}_betaEvent{int(beta_event*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Combined summary plot saved to '{filename}'")
    plt.close(fig)

def plot_seir_evolution(ax, time, total_population, n_iterations, S_realizations, E_realizations, I_realizations, R_realizations):
    """Helper to plot full SEIR evolution on a given axis."""
    colors = {
        'S': {'mean': 'blue', 'runs': 'lightblue'}, 'E': {'mean': 'orange', 'runs': 'moccasin'},
        'I': {'mean': 'red', 'runs': 'lightcoral'}, 'R': {'mean': 'green', 'runs': 'lightgreen'}
    }
    def plot_curves(data, color_map, label):
        for i in range(1, n_iterations + 1):
            ax.plot(time, data[:, i] / total_population, color=color_map['runs'], alpha=0.1)
        mean_prop = np.mean(data[:, 1:], axis=1) / total_population
        ax.plot(time, mean_prop, color=color_map['mean'], lw=2.5, label=f'Mean {label}')

    plot_curves(S_realizations, colors['S'], 'Susceptible')
    plot_curves(E_realizations, colors['E'], 'Exposed')
    plot_curves(I_realizations, colors['I'], 'Infected')
    plot_curves(R_realizations, colors['R'], 'Recovered')
    ax.set_title('Full SEIR Evolution', fontsize=14)
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Proportion of Population', fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)


def plot_exposed_infected_zoom(ax, time, total_population, n_iterations, E_realizations, I_realizations):
    """Helper to plot zoomed-in E, I, and E+I curves."""
    colors = {
        'E': {'mean': 'orange', 'runs': 'moccasin'}, 'I': {'mean': 'red', 'runs': 'lightcoral'},
        'E+I': {'mean': 'purple', 'runs': 'thistle'}
    }
    def plot_curves(data, color_map, label):
        for i in range(1, n_iterations + 1):
            ax.plot(time, (data[:, i] / total_population) * 1000, color=color_map['runs'], alpha=0.1)
        mean_prop = (np.mean(data[:, 1:], axis=1) / total_population) * 1000
        ax.plot(time, mean_prop, color=color_map['mean'], lw=2.5, label=f'Mean {label}')

    plot_curves(E_realizations, colors['E'], 'Exposed')
    plot_curves(I_realizations, colors['I'], 'Infected')
    EI_realizations = E_realizations[:, 1:] + I_realizations[:, 1:]
    EI_with_time = np.hstack((time[:, np.newaxis], EI_realizations))
    plot_curves(EI_with_time, colors['E+I'], 'Exposed + Infected')
    ax.set_title('Exposed & Infected Dynamics', fontsize=14)
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Incidence per 1000', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)


def plot_event_infection_distribution(ax, event_results):
    """Helper to plot the distribution of new exposures from the event."""
    if not event_results or max(event_results) == min(event_results):
        bins = 20
    else:
        bins = range(min(event_results), max(event_results) + 2)

    ax.hist(event_results, bins=bins, align='left', rwidth=0.8, color='skyblue', edgecolor='black')
    mean_infections = np.mean(event_results)
    ax.axvline(mean_infections, color='red', linestyle='--', lw=2, label=f'Mean: {mean_infections:.2f}')
    ax.set_title('Distribution of New Exposures from Event', fontsize=14)
    ax.set_xlabel('Number of Newly Exposed Individuals')
    ax.set_ylabel('Frequency (Number of Runs)')
    ax.legend()
    ax.grid(axis='y', alpha=0.75)


def plot_peak_time_distribution(ax, time, n_iterations, E_realizations, I_realizations):
    """Helper to plot the distribution of epidemic peak times."""
    peak_times = [time[np.argmax(E_realizations[:, i] + I_realizations[:, i])] for i in range(1, n_iterations + 1)]
    ax.hist(peak_times, bins=25, color='mediumpurple', edgecolor='black', rwidth=0.85)
    mean_peak_time = np.mean(peak_times)
    ax.axvline(mean_peak_time, color='red', linestyle='--', lw=2, label=f'Mean: {mean_peak_time:.1f} days')
    ax.set_title('Distribution of Epidemic Peak Time (E+I)', fontsize=14)
    ax.set_xlabel('Time to Peak (days)')
    ax.set_ylabel('Frequency (Number of Runs)')
    ax.legend()
    ax.grid(axis='y', alpha=0.75)


def plot_attendee_origins_on_map(recruitment_counts, data, cfg):
    """
    Plots the number of recruited attendees from each district on a map.
    """
    # print("Generating attendee origin map...")
    population_df = data['population_df']
    plot_df = population_df.copy()
    plot_df['recruited_attendees'] = recruitment_counts

    fig, ax = plt.subplots(figsize=(12, 12))

    scatter = ax.scatter(
        plot_df['lon'],
        plot_df['lat'],
        s=plot_df['recruited_attendees'] / 5,
        alpha=0.6,
        cmap='viridis',
        c=plot_df['recruited_attendees']
    )

    for i, txt in enumerate(plot_df['id']):
        ax.annotate(txt, (plot_df['lon'][i], plot_df['lat'][i]), ha='center', va='center', fontsize=10, color='black')

    cbar = plt.colorbar(scatter, shrink=0.7)
    cbar.set_label('Number of Attendees')

    scenario_label = getattr(cfg, "current_event_scenario_name", "N/A")
    ax.set_title(f'Geographic Origin of Event Attendees\n(Scenario: {scenario_label}, Venue: {cfg.event_model_type}, Recruitment: {cfg.attendee_recruitment_model})', fontsize=16)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)

    if cfg.event_model_type == 'large_venue':
        venue_rows = population_df.loc[population_df['id'] == cfg.large_venue_patch_id]
        
        # Fallback: if ID lookup fails, try index lookup
        if venue_rows.empty:
            try:
                idx = int(cfg.large_venue_patch_id)
                if 0 <= idx < len(population_df):
                    venue_rows = population_df.iloc[[idx]]
            except (ValueError, TypeError):
                pass

        if not venue_rows.empty:
            venue_lon = venue_rows['lon'].iloc[0]
            venue_lat = venue_rows['lat'].iloc[0]
            for _, row in plot_df.iterrows():
                if row['recruited_attendees'] > 0:
                    line_width = row['recruited_attendees'] / 500
                    ax.plot([row['lon'], venue_lon], [row['lat'], venue_lat], color='red', alpha=0.5, linewidth=line_width, zorder=0)
        else:
            print(f"Warning: Large venue patch ID {cfg.large_venue_patch_id} not found in population data (by ID or index). Skipping origin lines.")
    elif cfg.event_model_type == 'multi_venue':
        group_to_venue_coords = {
            group: (population_df.loc[population_df['id'] == venue_id, 'lon'].iloc[0], population_df.loc[population_df['id'] == venue_id, 'lat'].iloc[0])
            for group, venue_id in cfg.multi_venue_group_venues.items()
        }
        for _, row in plot_df.iterrows():
            if row['recruited_attendees'] > 0 and row['group'] in group_to_venue_coords:
                venue_lon, venue_lat = group_to_venue_coords[row['group']]
                line_width = row['recruited_attendees'] / 100
                ax.plot([row['lon'], venue_lon], [row['lat'], venue_lat], color='red', alpha=0.5, linewidth=line_width, zorder=0)

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    beta_event = cfg.event_base_transmission_rate
    filename = results_dir / f'attendee-origins_{scenario_label}_{cfg.event_model_type}_{cfg.attendee_recruitment_model}_R{int(cfg.R_0*100)}_betaEvent{int(beta_event*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Attendee origin map saved to '{filename}'")
    plt.close(fig)

def _get_peak_times_df(all_runs_results, n_patches, n_iterations):
    """Helper to extract peak times for each district across runs."""
    district_peak_times = {i: [] for i in range(n_patches)}

    for run_id in range(n_iterations):
        results = all_runs_results[run_id]
        time_points = np.array(results['t'])
        
        # Check for aggregated data first (E_j, I_j), fall back to full matrix (E_ij, I_ij)
        if 'E_j' in results and 'I_j' in results:
            E_in_j = np.array(results['E_j'])
            I_in_j = np.array(results['I_j'])
        elif 'E_ij' in results and 'I_ij' in results:
            E_ij_hist = np.array(results['E_ij'])
            I_ij_hist = np.array(results['I_ij'])
            # Reshape to (n_steps, n_patches, n_patches) and sum over axis 1 (home patch)
            E_in_j = E_ij_hist.reshape(-1, n_patches, n_patches).sum(axis=1)
            I_in_j = I_ij_hist.reshape(-1, n_patches, n_patches).sum(axis=1)
        else:
            continue

        # Calculate total infected (E+I) in each district 'j' at each time step
        # Sum over axis 1 (the 'i' or home patch axis)
        EI_in_j = E_in_j + I_in_j 

        for district_id in range(n_patches):
            district_curve = EI_in_j[:, district_id]
            if np.sum(district_curve) > 0:
                peak_index = np.argmax(district_curve)
                if peak_index < len(time_points):
                    peak_time = time_points[peak_index]
                    district_peak_times[district_id].append(peak_time)
    
    return pd.DataFrame.from_dict(district_peak_times, orient='index').transpose()

def plot_peak_time_distribution_by_district(all_runs_results, data, cfg):
    """
    Calculates and plots the distribution of epidemic peak times (E+I) for each district
    as a boxplot across all simulation runs.
    """
    if not getattr(cfg, 'plot_on_the_fly', True):
        return

    print(f"Generating peak time distribution by district for '{cfg.event_model_type}'...")
    n_patches = data['n_patches']
    n_iterations = cfg.n_iterations
    
    peak_times_df = _get_peak_times_df(all_runs_results, n_patches, n_iterations)

    if peak_times_df.empty:
        print("No peak time data to plot for districts.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    peak_times_df.boxplot(ax=ax, grid=True, vert=True, patch_artist=False)
    
    R0 = cfg.R_0
    beta_event = cfg.event_base_transmission_rate
    ax.set_title(f'Distribution of Epidemic Peak Time (E+I) per District\n(Scenario: {cfg.event_model_type}, Recruitment: {cfg.attendee_recruitment_model}, $\\beta_0$ = {beta_event:.2f},  $R_0$ = {R0:.2f})', fontsize=16)
    fig.suptitle('') # Remove default title
    ax.set_xlabel('District ID', fontsize=12)
    ax.set_ylabel('Time to Peak (days)', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'peak_time_by_district_boxplot_{cfg.event_model_type}_R{int(R0*100)}_betaEvent{int(beta_event*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"District peak time boxplot saved to '{filename}'")

def _get_daily_active_cases(all_runs_results, n_patches, n_iterations, n_days):
    """Helper to extract daily active cases for each district across runs."""
    daily_times = np.arange(n_days + 1)
    # Shape: (n_iterations, n_days + 1, n_patches)
    all_runs_daily_cases = np.zeros((n_iterations, n_days + 1, n_patches))

    for run_id in range(n_iterations):
        results = all_runs_results[run_id]
        time_points = np.array(results['t'])
        
        if 'E_j' in results and 'I_j' in results:
            E_in_j = np.array(results['E_j'])
            I_in_j = np.array(results['I_j'])
        elif 'E_ij' in results and 'I_ij' in results:
            E_ij_hist = np.array(results['E_ij'])
            I_ij_hist = np.array(results['I_ij'])
            E_in_j = E_ij_hist.reshape(-1, n_patches, n_patches).sum(axis=1)
            I_in_j = I_ij_hist.reshape(-1, n_patches, n_patches).sum(axis=1)
        else:
            continue

        Active_j = E_in_j + I_in_j # Shape: (time_steps, n_patches)

        for p in range(n_patches):
            # Interpolate to get daily values
            interpolated_cases = np.interp(daily_times, time_points, Active_j[:, p])
            all_runs_daily_cases[run_id, :, p] = interpolated_cases
            
    # Calculate mean across runs
    return daily_times, all_runs_daily_cases

def plot_spacetime_heatmap(all_runs_results, data, cfg):
    """
    Plots a heatmap of active cases (E+I) for all districts over time.
    Districts are sorted by the time of their peak infection to visualize the wave propagation.
    """
    # Explicitly use event_model_type for baseline to avoid using a lingering scenario name
    if cfg.event_model_type.startswith('no_event'):
        scenario_name = cfg.event_model_type
    else:
        scenario_name = getattr(cfg, "current_event_scenario_name", cfg.event_model_type)
    print(f"Generating space-time heatmap for '{scenario_name}'...")

    n_patches = data['n_patches']
    n_iterations = cfg.n_iterations
    n_days = cfg.n_days

    # Get daily active cases (mean over runs) -> Shape: (n_days+1, n_patches)
    daily_times, all_runs_daily_cases = _get_daily_active_cases(all_runs_results, n_patches, n_iterations, n_days)
    mean_daily_cases = np.mean(all_runs_daily_cases, axis=0)
    
    # Transpose for heatmap: Y=Districts, X=Time
    heatmap_data = mean_daily_cases.T  # Shape: (n_patches, n_days+1)

    # Sort districts by peak time to visualize the "wave"
    peak_times = np.argmax(heatmap_data, axis=1)
    sorted_indices = np.argsort(peak_times)
    sorted_data = heatmap_data[sorted_indices, :]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use a log scale for color if the range is large, otherwise linear
    if np.max(sorted_data) > 100:
        norm = plt.matplotlib.colors.LogNorm(vmin=max(1, np.min(sorted_data[sorted_data>0])), vmax=np.max(sorted_data))
    else:
        norm = plt.Normalize(vmin=0, vmax=np.max(sorted_data))

    im = ax.imshow(sorted_data, aspect='auto', cmap='Reds', interpolation='nearest', 
                   origin='lower', extent=[0, n_days, 0, n_patches], norm=norm)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Active Cases (E+I)')

    ax.set_title(f'Space-Time Heatmap of Epidemic Spread\n(Sorted by Peak Time) - {scenario_name}', fontsize=16)
    ax.set_xlabel('Time (Days)', fontsize=12)
    ax.set_ylabel('District ID (Sorted)', fontsize=12)

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    # filename = results_dir / f'spacetime_heatmap_{scenario_name}_R{int(cfg.R_0*100)}_betaEvent{int(cfg.event_base_transmission_rate*100)}.png'
    # plt.savefig(filename, dpi=300)
    plt.close(fig)
    # print(f"Space-time heatmap saved to '{filename}'")

def plot_spatial_spread_snapshots(all_runs_results, data, cfg, vmax=None, scenario_name_override=None, filename_suffix="", mean_daily_cases_override=None):
    """
    Plots choropleth map snapshots of the epidemic at different time points for the mean of runs.
    """
    if not HAS_GEOPANDAS:
        print("Warning: 'geopandas' not found. Skipping choropleth spatial snapshots.")
        return

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    
    if scenario_name_override:
        scenario_name = scenario_name_override
    else:
        # Explicitly use event_model_type for baseline to avoid using a lingering scenario name
        if cfg.event_model_type.startswith('no_event'):
            scenario_name = cfg.event_model_type
        else:
            scenario_name = getattr(cfg, "current_event_scenario_name", cfg.event_model_type)
    print(f"Generating spatial spread choropleth snapshots for '{scenario_name}'...")

    # --- Load GeoJSON ---
    mobility_dir = Path(f"data_mobility_{cfg.dataset_name}")
    geojson_filename = f"{cfg.dataset_name.lower()}-districts.geojson"
    geojson_path = mobility_dir / geojson_filename

    if not geojson_path.exists():
        # Fallback: Check for any geojson in the directory
        if mobility_dir.exists():
            geojsons = list(mobility_dir.glob("*.geojson"))
            if geojsons:
                geojson_path = geojsons[0]
                print(f"Note: Default GeoJSON '{geojson_filename}' not found. Using '{geojson_path.name}' instead.")

    if not geojson_path.exists():
        print(f"Warning: GeoJSON file not found at '{geojson_path}'. Cannot generate choropleth maps.")
        return

    try:
        geodf = gpd.read_file(geojson_path)
    except Exception as e:
        print(f"Warning: Could not read GeoJSON file '{geojson_path}'. Cannot generate choropleth maps. Error: {e}")
        return

    if mean_daily_cases_override is not None:
        mean_daily_cases = mean_daily_cases_override
        n_days = mean_daily_cases.shape[0] - 1
    else:
        n_patches = data['n_patches']
        n_days = cfg.n_days
        n_iterations = cfg.n_iterations
        # Get daily active cases
        _, all_runs_daily_cases = _get_daily_active_cases(all_runs_results, n_patches, n_iterations, n_days)
        mean_daily_cases = np.mean(all_runs_daily_cases, axis=0)
    
    # Select 6 time points spread across the simulation
    time_points = getattr(cfg, 'custom_spatial_comparison_days', [0, 50, 100, 150, 200, 250])
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    # Determine global max for color scaling
    if vmax is None:
        vmax = np.max(mean_daily_cases)
        if vmax == 0: vmax = 1
    
    # Prepare for colorbar
    norm = plt.matplotlib.colors.Normalize(vmin=0, vmax=vmax)
    mappable = plt.cm.ScalarMappable(cmap='Reds', norm=norm)

    n_patches = mean_daily_cases.shape[1]
    # --- Find and prepare the ID column for merging ---
    id_col_found = None
    # Check for common ID column names in the GeoDataFrame
    id_candidates = ['id', 'patch_id', 'ID', 'cartodb_id']
    for col_name in id_candidates:
        if col_name in geodf.columns:
            id_col_found = col_name
            break
    
    if not id_col_found:
        print(f"Error: Could not find a suitable ID column in '{geojson_path}'.")
        print(f"       Checked for: {id_candidates}. Available columns: {geodf.columns.tolist()}")
        return

    # If the found column is not 'id', rename it for consistent merging
    if id_col_found != 'id':
        geodf = geodf.rename(columns={id_col_found: 'id'})

    # Ensure 'id' column is numeric for merging
    if not pd.api.types.is_numeric_dtype(geodf['id']):
        try:
            geodf['id'] = pd.to_numeric(geodf['id'], errors='coerce')
            geodf.dropna(subset=['id'], inplace=True)
            geodf['id'] = geodf['id'].astype(int)
        except Exception as e:
            print(f"Warning: Could not convert 'id' column in GeoJSON to numeric. Cannot create choropleth. Error: {e}")
            return

    # Adjust 1-based IDs (common in CartoDB/GeoJSON) to 0-based if they match the patch count
    if geodf['id'].min() == 1 and geodf['id'].max() == n_patches:
        print("Note: Detected 1-based IDs in GeoJSON. Adjusting to 0-based to match simulation data.")
        geodf['id'] = geodf['id'] - 1

    # --- Get Seed Patch Coordinates ---
    seed_patch_id = getattr(cfg, 'initial_infection_patch_id', 0)
    seed_coords = None
    try:
        seed_patch_row = geodf[geodf['id'] == seed_patch_id]
        if not seed_patch_row.empty:
            # Re-project to a projected CRS (3857) to calculate centroid accurately and avoid UserWarning
            centroid_geom = seed_patch_row.to_crs(epsg=3857).centroid.to_crs(geodf.crs).iloc[0]
            seed_coords = (centroid_geom.x, centroid_geom.y)
        else:
            print(f"Warning: Seed patch ID {seed_patch_id} not found in GeoDataFrame for spatial spread plot.")
    except Exception as e:
        print(f"Warning: Could not get seed patch coordinates for spatial spread plot: {e}")

    for idx, t in enumerate(time_points):
        ax = axes[idx]
        if t >= len(mean_daily_cases): t = len(mean_daily_cases) - 1
        
        cases_at_t = mean_daily_cases[t, :]
        cases_df = pd.DataFrame({'id': range(n_patches), 'cases': cases_at_t})
        
        # Merge geodata with case data
        merged_gdf = geodf.merge(cases_df, on='id', how='left').fillna(0)
        
        # Plot choropleth map
        merged_gdf.plot(column='cases', cmap='Reds', linewidth=0.5, ax=ax, edgecolor='0.3', norm=norm)

        # --- Add Seed Marker ---
        if seed_coords:
            seed_case_value = cases_at_t[seed_patch_id]
            
            # Determine marker color based on the background.
            # If the case value is in the upper half of the color range, the background is dark.
            marker_color = 'black' # Default for light background
            if not np.isnan(seed_case_value):
                # Use a threshold of 50% of the color range.
                color_threshold = 0 + (vmax - 0) * 0.5
                if seed_case_value >= color_threshold:
                    marker_color = 'white' # For dark background
            ax.scatter(seed_coords[0], seed_coords[1], marker='x', color=marker_color, s=50, lw=2.5, zorder=10)

        ax.set_title(f'Day {t}', fontsize=14)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_aspect('equal', adjustable='box')

    # Add a global colorbar
    fig.tight_layout(rect=[0, 0, 1, 0.9]) # Adjust rect to make space at the top for the colorbar
    cbar_ax = fig.add_axes([0.1, 0.92, 0.8, 0.02]) # Position: [left, bottom, width, height]
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label('Mean Active Cases (E+I)', fontsize=12)
    
    fig.suptitle(f'Spatial Spread of Epidemic Over Time - {scenario_name}', fontsize=20, y=0.98) # Adjust suptitle y-position
    filename = results_dir / f'spatial_spread_{scenario_name}{filename_suffix}_R{int(cfg.R_0*100)}_betaEvent{int(cfg.event_base_transmission_rate*100)}_Iss{int(cfg.I_ss)}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Spatial choropleth snapshots saved to '{filename}'")

def plot_spatial_spread_comparison(group_results: Dict[str, Any], data: Dict[str, Any], cfg: Any, group_name: str):
    """
    Generates spatial choropleth maps for multiple scenarios with a unified color scale.
    """
    if not HAS_GEOPANDAS:
        return

    print(f"\nGenerating comparable spatial snapshots for group: {group_name}...")

    n_patches = data['n_patches']
    n_days = cfg.n_days
    n_iterations = cfg.n_iterations
    
    # 1. Calculate and store the mean daily cases for each scenario
    scenario_mean_cases = {}
    for sc_name, res in group_results.items():
        _, all_runs = _get_daily_active_cases(res['all_runs_results'], n_patches, n_iterations, n_days)
        mean_cases = np.mean(all_runs, axis=0)
        scenario_mean_cases[sc_name] = mean_cases

    # 2. Calculate global max across all stored mean cases
    global_vmax = 0
    if scenario_mean_cases:
        # Use a generator expression to find the max of all maxes
        max_of_all = max(np.max(cases) for cases in scenario_mean_cases.values())
        global_vmax = max(global_vmax, max_of_all)
        
    if global_vmax == 0: 
        global_vmax = 1
    
    # 3. Generate plots for each scenario with the fixed scale
    for sc_name, mean_cases in scenario_mean_cases.items():
        # Pass the pre-calculated mean cases to avoid recalculation
        plot_spatial_spread_snapshots(
            None, # all_runs_results is not needed as we provide mean_daily_cases_override
            data, 
            cfg, 
            vmax=global_vmax, 
            filename_suffix="", # Keep this empty as per previous request
            mean_daily_cases_override=mean_cases
        )

def plot_custom_spatial_comparison(group_results, data, cfg, metric='mean', scenarios=None, titles=None, filename_suffix="", vmax_override=None):
    """
    Generates a custom spatial choropleth map grid comparing specific scenarios and days.
    Rows: configurable via scenarios or cfg.custom_spatial_comparison_scenarios
    Cols: configurable via cfg.custom_spatial_comparison_days
    Metric: 'mean' or 'std'
    """
    if not HAS_GEOPANDAS:
        print("Warning: 'geopandas' not found. Skipping custom spatial comparison plot.")
        return

    scenarios_to_plot = scenarios if scenarios is not None else getattr(cfg, 'custom_spatial_comparison_scenarios', ["no_event", "AMS_dance", "AMS_football", "Leipzig_1", "Leipzig_2", "Leipzig_3"])
    time_points = getattr(cfg, 'custom_spatial_comparison_days', [100, 150, 200])
    scenario_titles = titles if titles is not None else getattr(cfg, 'custom_spatial_comparison_titles', {
        "no_event": "Baseline",
        "AMS_dance": "AMS Dance",
        "AMS_football": "AMS Football",
        "Leipzig_1": "Leipzig 1",
        "Leipzig_2": "Leipzig 2",
        "Leipzig_3": "Leipzig 3"
    })

    # Filter to only include scenarios present in group_results to ensure the plot is generated
    original_count = len(scenarios_to_plot)
    scenarios_to_plot = [s for s in scenarios_to_plot if s in group_results]
    
    if not scenarios_to_plot:
        print(f"Warning: Skipping custom spatial comparison plot. None of the requested scenarios found in group results.")
        return
    
    if len(scenarios_to_plot) < original_count:
        print(f"Note: Custom spatial comparison plot will only include {len(scenarios_to_plot)}/{original_count} scenarios found in group results.")

    metric_tag = "Mean" if metric == 'mean' else "Std Dev"
    cmap_name = "Reds" if metric == 'mean' else "Blues"
    print(f"\nGenerating custom spatial comparison plot ({metric_tag}) for {len(scenarios_to_plot)} scenarios...")

    # --- Load GeoJSON ---
    mobility_dir = Path(f"mobility_{getattr(cfg, 'dataset_name', 'unknown')}")
    geojson_filename = f"{cfg.dataset_name.lower()}-districts.geojson"
    geojson_path = mobility_dir / geojson_filename
    if not geojson_path.exists():
        if mobility_dir.exists():
            geojsons = list(mobility_dir.glob("*.geojson"))
            if geojsons: geojson_path = geojsons[0]
    if not geojson_path.exists():
        print(f"Warning: GeoJSON file not found at '{geojson_path}'. Cannot generate choropleth maps.")
        return
    try:
        geodf = gpd.read_file(geojson_path)
    except Exception as e:
        print(f"Warning: Could not read GeoJSON file '{geojson_path}'. Error: {e}")
        return

    # --- Calculate metric (mean or std) daily cases for all required scenarios ---
    scenario_data = {}
    n_patches = data['n_patches']
    n_days = cfg.n_days
    n_iterations = cfg.n_iterations
    for sc_name in scenarios_to_plot:
        res = group_results[sc_name]
        _, all_runs = _get_daily_active_cases(res['all_runs_results'], n_patches, n_iterations, n_days)
        if metric == 'std':
            scenario_data[sc_name] = np.std(all_runs, axis=0)
        else:
            scenario_data[sc_name] = np.mean(all_runs, axis=0)

    # --- Calculate global vmax ---
    if vmax_override is not None:
        global_vmax = vmax_override
    else:
        global_vmax = 0
        for sc_name in scenarios_to_plot:
            metric_cases = scenario_data[sc_name]
            for t in time_points:
                if t < metric_cases.shape[0]:
                    vmax_t = np.max(metric_cases[t, :])
                    if vmax_t > global_vmax:
                        global_vmax = vmax_t
        if global_vmax == 0: global_vmax = 1

    # --- Prepare GeoDataFrame and Seed Coordinates ---
    id_col_found = None
    id_candidates = ['id', 'patch_id', 'ID', 'cartodb_id']
    for col_name in id_candidates:
        if col_name in geodf.columns:
            id_col_found = col_name
            break
    if not id_col_found: return
    if id_col_found != 'id':
        geodf = geodf.rename(columns={id_col_found: 'id'})
    if not pd.api.types.is_numeric_dtype(geodf['id']):
        try:
            geodf['id'] = pd.to_numeric(geodf['id'], errors='coerce')
            geodf.dropna(subset=['id'], inplace=True)
            geodf['id'] = geodf['id'].astype(int)
        except Exception: return
    if geodf['id'].min() == 1 and geodf['id'].max() == n_patches:
        geodf['id'] = geodf['id'] - 1

    seed_patch_id = getattr(cfg, 'initial_infection_patch_id', 0)
    seed_coords = None
    try:
        seed_patch_row = geodf[geodf['id'] == seed_patch_id]
        if not seed_patch_row.empty:
            # Re-project to a projected CRS (3857) to calculate centroid accurately and avoid UserWarning
            centroid_geom = seed_patch_row.to_crs(epsg=3857).centroid.to_crs(geodf.crs).iloc[0]
            seed_coords = (centroid_geom.x, centroid_geom.y)
    except Exception: pass

    # --- Create Subplots ---
    nrows = len(scenarios_to_plot)
    ncols = len(time_points)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows), squeeze=False)
    norm = plt.matplotlib.colors.Normalize(vmin=0, vmax=global_vmax)
    mappable = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)

    # --- Plotting Loop ---
    for row_idx, sc_name in enumerate(scenarios_to_plot):
        metric_daily_cases = scenario_data[sc_name]
        for col_idx, t in enumerate(time_points):
            ax = axes[row_idx, col_idx]
            if t >= len(metric_daily_cases): t = len(metric_daily_cases) - 1
            
            cases_at_t = metric_daily_cases[t, :]
            cases_df = pd.DataFrame({'id': range(n_patches), 'cases': cases_at_t})
            merged_gdf = geodf.merge(cases_df, on='id', how='left').fillna(0)
            merged_gdf.plot(column='cases', cmap=cmap_name, linewidth=0.5, ax=ax, edgecolor='0.3', norm=norm)

            # Add Seed Marker
            if seed_coords:
                seed_case_value = cases_at_t[seed_patch_id]
                marker_color = 'black'
                if not np.isnan(seed_case_value):
                    color_threshold = 0 + (global_vmax - 0) * 0.5
                    if seed_case_value >= color_threshold:
                        marker_color = 'white'
                ax.scatter(seed_coords[0], seed_coords[1], marker='x', color=marker_color, s=50, lw=2.5, zorder=10)

            # Set titles and labels
            if row_idx == 0:
                ax.set_title(f'Day {t}', fontsize=16)
            if col_idx == 0:
                ax.set_ylabel(f"{scenario_titles.get(sc_name, sc_name)}", fontsize=16)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_aspect('equal', adjustable='box')

    # --- Finalize and Save ---
    fig.suptitle(f'Spatial Spread Comparison ({metric_tag})\n($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=20, y=0.98) # Adjust suptitle y-position
    
    fig.tight_layout(rect=[0, 0, 1, 0.82]) # Lower subplots further to create more room
    cbar_ax = fig.add_axes([0.1, 0.83, 0.8, 0.015]) # Lower the colorbar to avoid suptitle overlap
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
    cbar_ax.set_title(f'{metric_tag} Active Cases (E+I)', fontsize=15, pad=10)

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    metric_file_tag = "std" if metric == 'std' else "mean"
    filename = results_dir / f'spatial_spread_custom_{metric_file_tag}_comparison{filename_suffix}_R{int(cfg.R_0*100)}_betaEvent{int(cfg.event_base_transmission_rate*100)}_Iss{int(cfg.I_ss)}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Custom spatial comparison plot saved to '{filename}'")

def _get_arrival_times(all_runs_results, n_patches, n_iterations):
    """Helper to extract arrival times (first time I>0) for all runs.
    Uses I_j which contains active cases (E+I) as a proxy for infectious individuals.
    Epidemic arrival is defined as the first time infectious-capable individuals appear in a district."""
    arrival_times = np.full((n_iterations, n_patches), np.nan)

    for run_id, res in enumerate(all_runs_results):
        if run_id >= n_iterations: break

        if 'precomputed_arrival_times' in res:
            run_arrivals = res['precomputed_arrival_times']
            for p in range(n_patches):
                if p < len(run_arrivals) and not np.isnan(run_arrivals[p]):
                    arrival_times[run_id, p] = run_arrivals[p]
            continue

        t = np.array(res['t'])

        if 'I_j' in res:
            I_j = np.array(res['I_j'])
        elif 'I_ij' in res:
             I_ij = np.array(res['I_ij'])
             I_j = I_ij.reshape(-1, n_patches, n_patches).sum(axis=1)
        else:
            continue

        # Find first time I > 0 for each patch
        is_infected = (I_j > 0)

        if not np.any(is_infected):
             continue

        first_indices = np.argmax(is_infected, axis=0)

        for p in range(n_patches):
            idx = first_indices[p]
            if is_infected[idx, p]:
                arrival_times[run_id, p] = t[idx]

    return arrival_times



def plot_arrival_time_comparison(group_results, data, cfg, group_name, metric='mean'):
    """
    Generates a map comparison of mean or std arrival time for multiple scenarios.
    Uses I-based arrival times (first time infectious individuals > 0).
    
    Args:
        metric: 'mean' for mean arrival time, 'std' for standard deviation of arrival time.
    """
    if not HAS_GEOPANDAS:
        return

    print(f"Generating arrival time comparison map ({metric}) for group: {group_name}...")
    
    # Load scenarios from config if available, otherwise use all results in group
    scenarios = getattr(cfg, 'arrival_time_comparison_scenarios', list(group_results.keys()))
    scenarios = [s for s in scenarios if s in group_results]
    
    scenario_titles = getattr(cfg, 'arrival_time_comparison_titles', {})

    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
    
    n_scenarios = len(scenarios)
    if n_scenarios == 0: return

    # --- Load GeoJSON (Reusing logic for consistency) ---
    mobility_dir = Path(f"data_mobility_{cfg.dataset_name}")
    geojson_filename = f"{cfg.dataset_name.lower()}-districts.geojson"
    geojson_path = mobility_dir / geojson_filename

    if not geojson_path.exists():
        if mobility_dir.exists():
             geojsons = list(mobility_dir.glob("*.geojson"))
             if geojsons: geojson_path = geojsons[0]
    
    if not geojson_path.exists():
        print("GeoJSON not found, skipping map comparison.")
        return

    try:
        geodf = gpd.read_file(geojson_path)
    except Exception as e:
        print(f"Error reading GeoJSON: {e}")
        return

    # Align IDs
    id_col_found = None
    id_candidates = ['id', 'patch_id', 'ID', 'cartodb_id']
    for col_name in id_candidates:
        if col_name in geodf.columns:
            id_col_found = col_name
            break
    
    if id_col_found:
        if id_col_found != 'id':
            geodf = geodf.rename(columns={id_col_found: 'id'})
        
        if not pd.api.types.is_numeric_dtype(geodf['id']):
                geodf['id'] = pd.to_numeric(geodf['id'], errors='coerce')
                geodf.dropna(subset=['id'], inplace=True)
                geodf['id'] = geodf['id'].astype(int)

        n_patches = data['n_patches']
        if geodf['id'].min() == 1 and geodf['id'].max() == n_patches:
                geodf['id'] = geodf['id'] - 1
    else:
        print("No suitable ID column in GeoJSON.")
        return

    # --- Get Seed Patch Coordinates ---
    seed_patch_id = getattr(cfg, 'initial_infection_patch_id', 0) # Use getattr for safety
    seed_coords = None
    try:
        seed_patch_row = geodf[geodf['id'] == seed_patch_id]
        if not seed_patch_row.empty:
            centroid_geom = seed_patch_row.to_crs(epsg=3857).centroid.to_crs(geodf.crs).iloc[0]
            seed_coords = (centroid_geom.x, centroid_geom.y)
        else:
            print(f"Warning: Seed patch ID {seed_patch_id} not found in GeoDataFrame.")
    except Exception as e:
        print(f"Warning: Could not get seed patch coordinates: {e}")

    n_iterations = cfg.n_iterations

    # Prepare data and find global min/max for color scale
    scenario_data = {}
    global_min, global_max = 0.0, float('-inf')
    global_std_max = float('-inf')

    for sc in scenarios:
        res = group_results[sc]
        arrival_times = _get_arrival_times(res['all_runs_results'], n_patches, n_iterations)

        with np.errstate(invalid='ignore', divide='ignore'):
            if arrival_times.size > 0 and not np.all(np.isnan(arrival_times)):
                mean_times = np.nanmean(arrival_times, axis=0)
                std_times = np.nanstd(arrival_times, axis=0)
            else:
                mean_times = np.full(n_patches, np.nan)
                std_times = np.full(n_patches, np.nan)
        
        scenario_data[sc] = {'mean': mean_times, 'std': std_times}
        
        valid_times = mean_times[~np.isnan(mean_times)]
        if len(valid_times) > 0:
            global_max = max(global_max, np.max(valid_times))
            
        valid_std = std_times[~np.isnan(std_times)]
        if len(valid_std) > 0:
            global_std_max = max(global_std_max, np.max(valid_std))

    if global_max == float('-inf'): global_max = 1
    if global_std_max == float('-inf'): global_std_max = 1

    # Create subplots (only for the requested metric)
    cols = min(n_scenarios, 3)
    rows = int(np.ceil(n_scenarios / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows), squeeze=False)

    for i, sc in enumerate(scenarios):
        row_in_grid = i // cols
        col_in_grid = i % cols
        ax = axes[row_in_grid, col_in_grid]
        
        if metric == 'mean':
            times = scenario_data[sc]['mean']
            col_name = 'arrival_time'
            cmap = 'Reds_r'
            vmin = global_min
            vmax = global_max
            
            arrival_df = pd.DataFrame({'id': range(n_patches), col_name: times})
            merged_gdf = geodf.merge(arrival_df, on='id', how='left')
            
            merged_gdf.plot(column=col_name, cmap=cmap, linewidth=0.5, ax=ax, edgecolor='0.3',
                            vmin=vmin, vmax=vmax,
                            missing_kwds={'color': 'lightgrey', 'label': 'Never Infected'})
            
            if seed_coords:
                seed_time = times[seed_patch_id]
                marker_color = 'black'
                if not np.isnan(seed_time):
                    color_threshold = vmin + (vmax - vmin) * 0.5
                    if seed_time <= color_threshold:
                        marker_color = 'white'
                ax.scatter(seed_coords[0], seed_coords[1], marker='x', color=marker_color, s=50, lw=2.5, zorder=10)
        else:  # metric == 'std'
            times = scenario_data[sc]['std']
            col_name = 'std_dev'
            cmap = 'Blues'
            vmin = 0
            vmax = global_std_max
            
            std_df = pd.DataFrame({'id': range(n_patches), col_name: times})
            merged_gdf = geodf.merge(std_df, on='id', how='left')
            
            merged_gdf.plot(column=col_name, cmap=cmap, linewidth=0.5, ax=ax, edgecolor='0.3',
                            vmin=vmin, vmax=vmax,
                            missing_kwds={'color': 'lightgrey'})
            
            if seed_coords:
                ax.scatter(seed_coords[0], seed_coords[1], marker='x', color='black', s=50, lw=2.5, zorder=10)

        title = scenario_titles.get(sc, "Baseline (No Event)" if sc == 'no_event' else sc)
        ax.set_title(f"{title}", fontsize=14)
        ax.set_axis_off()

    # Hide unused subplots
    for i in range(n_scenarios, rows * cols):
        row_in_grid = i // cols
        col_in_grid = i % cols
        axes[row_in_grid, col_in_grid].set_visible(False)

    # Shared Colorbar
    cmap_used = 'Reds_r' if metric == 'mean' else 'Blues'
    vmin_used = global_min if metric == 'mean' else 0
    vmax_used = global_max if metric == 'mean' else global_std_max
    sm = plt.cm.ScalarMappable(cmap=cmap_used, norm=plt.Normalize(vmin=vmin_used, vmax=vmax_used))
    sm._A = []
    
    beta_event = cfg.event_base_transmission_rate
    metric_label = "Mean" if metric == 'mean' else "Std Dev"
    fig.suptitle(f'Epidemic Arrival Time {metric_label} - Comparison ($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={beta_event})', fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 0.88, 0.94])
    
    cbar_ax = fig.add_axes([0.91, 0.10, 0.03, 0.80])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f'Arrival Time {metric_label} (Days)', fontsize=12)

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'arrival_time_comparison_{metric}_{group_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved arrival time comparison map ({metric}) to '{filename}'")

def plot_grouped_peak_time_distributions(group_results, data, cfg, group_name):
    """
    Plots peak time distributions for a group of scenarios in subplots.
    """
    print(f"Generating grouped peak time distribution boxplots for {group_name}...")
    
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
    
    n_scenarios = len(scenarios)
    if n_scenarios == 0:
        return

    n_patches = data['n_patches']
    n_iterations = cfg.n_iterations
    
    fig, axes = plt.subplots(n_scenarios, 1, figsize=(15, 5 * n_scenarios), sharex=True)
    if n_scenarios == 1:
        axes = [axes]
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        results = group_results[scenario]
        all_runs = results['all_runs_results']
        
        df = _get_peak_times_df(all_runs, n_patches, n_iterations)
        
        if not df.empty:
            df.boxplot(ax=ax, grid=True, vert=True, patch_artist=False)
        
        title = "Baseline" if scenario == 'no_event' else scenario
        ax.set_title(f"{title}", fontsize=14)
        ax.set_ylabel('Time to Peak (days)', fontsize=12)
        
        if idx == n_scenarios - 1:
            ax.set_xlabel('District ID', fontsize=12)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    beta_event = cfg.event_base_transmission_rate
    fig.suptitle(f'Peak Time Distribution by District - Group: {group_name}\n($R_0$={cfg.R_0}, $\\beta_{{event}}$={beta_event})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'peak_time_by_district_boxplot_{group_name}_R{int(cfg.R_0*100)}_betaEvent{int(beta_event*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Grouped peak time boxplot saved to '{filename}'")

def plot_event_exposure_distribution_boxplot(event_exposures_df, cfg):
    """
    Creates a boxplot of new exposures per event day across all simulation runs.
    """
    if event_exposures_df.empty:
        print("No event exposure data to plot for boxplot.")
        return

    # print("Generating event exposure distribution boxplot...")
    fig, ax = plt.subplots(figsize=(8, 6))

    event_exposures_df.boxplot(ax=ax, grid=False)

    ax.set_title(
        f'Distribution of New Exposures per Event Day\n'
        f'(Model: {cfg.event_model_type}, Recruitment: {cfg.attendee_recruitment_model})',
        fontsize=16
    )
    ax.set_xlabel('Event Day', fontsize=12)
    ax.set_ylabel('Number of Newly Exposed Individuals', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    beta_event = cfg.event_base_transmission_rate
    filename = results_dir / f'event-exposure-boxplot_{cfg.event_model_type}_{cfg.attendee_recruitment_model}_R{int(cfg.R_0*100)}_betaEvent{int(beta_event*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Event exposure boxplot saved to '{filename}'")
    plt.close(fig)

def plot_scenario_exposure_comparison_boxplot(all_exposures_data, cfg):
    """
    Creates a boxplot comparing new exposures on the first event day across different scenarios.
    
    Args:
        all_exposures_data (dict): A dictionary where keys are scenario names and values are lists
                                   of new exposure counts for each run.
        cfg: The configuration module.
    """
    if not getattr(cfg, 'plot_on_the_fly', True):
        return

    if not all_exposures_data:
        print("No scenario exposure data provided for comparison boxplot.")
        return

    print("Generating scenario exposure comparison boxplot...")
    # Pad with NaN if lists are uneven (e.g., due to die-out)
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in all_exposures_data.items() ]))

    fig, ax = plt.subplots(figsize=(8, 6))
    
    df.boxplot(ax=ax, grid=False)

    ax.set_title(
        f'Comparison of New Exposures on First Event Day\n'
        f'($I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})',
        fontsize=16
    )
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Number of Newly Exposed Individuals', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    beta_event = cfg.event_base_transmission_rate
    filename = results_dir / f'scenario_exposure_comparison_boxplot_Iss{cfg.I_ss}_betaEvent{int(beta_event*100)}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Scenario exposure comparison boxplot saved to '{filename}'")
    plt.close(fig)

def plot_infected_attendees_recruited_boxplot(infected_attendees_df, cfg):
    """
    Creates a boxplot of infected attendees recruited per event day across all simulation runs.
    """
    if infected_attendees_df.empty:
        print("No infected attendee data to plot for boxplot.")
        return

    if 0 in infected_attendees_df.columns:
        day0_data = infected_attendees_df[0]
        print(f"Infected attendees recruited on Day 0 (from boxplot data): Mean = {day0_data.mean():.2f} +/- {day0_data.std():.2f}")

    # print("Generating infected attendees recruited distribution boxplot...")
    fig, ax = plt.subplots(figsize=(8, 6))

    infected_attendees_df.boxplot(ax=ax, grid=False)

    ax.set_title(
        f'Distribution of Infected Attendees Recruited per Event Day\n'
        f'(Model: {cfg.event_model_type}, Recruitment: {cfg.attendee_recruitment_model})',
        fontsize=16
    )
    ax.set_xlabel('Event Day', fontsize=12)
    ax.set_ylabel('Number of Infected Attendees Recruited', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    beta_event = cfg.event_base_transmission_rate
    filename = results_dir / f'infected-attendees-recruited-boxplot_{cfg.attendee_recruitment_model}_R{int(cfg.R_0*100)}_betaEvent{int(beta_event*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Infected attendees recruited boxplot saved to '{filename}'")
    plt.close(fig)

def plot_infected_attendees_from_specific_districts(infected_attendee_counts, population_df, cfg):
    """
    Creates a bar chart showing the number of infected attendees from specific districts in subplots.
    """
    if not cfg.districts_to_visualize_attendees:
        print("No specific districts defined for infected attendee visualization.")
        return

    # print("Generating infected attendee visualization for specific districts...")

    selected_districts_data = []
    for district_id in cfg.districts_to_visualize_attendees:
        if district_id < len(infected_attendee_counts):
            district_name = population_df.loc[population_df['id'] == district_id, 'name'].iloc[0]
            attendees = infected_attendee_counts[district_id]
            selected_districts_data.append({'District': district_name, 'Infected Attendees': attendees})

    if not selected_districts_data:
        print("No infected attendee data available for the specified districts.")
        return

    df_plot = pd.DataFrame(selected_districts_data)

    if df_plot.empty or df_plot['Infected Attendees'].sum() == 0:
        print("No infected attendees to plot for the specified districts.")
        return

    num_plots = len(df_plot)
    ncols = 3
    nrows = int(np.ceil(num_plots / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, row in df_plot.iterrows():
        ax = axes[i]
        ax.bar(row['District'], row['Infected Attendees'], color='crimson', edgecolor='black')
        ax.set_title(f"{row['District']}", fontsize=12)
        ax.set_ylabel('Infected Attendees')
        ax.tick_params(axis='x', rotation=15)
        # Set y-axis to start from 0 and add a small margin
        if row['Infected Attendees'] > 0:
            ax.set_ylim(0, row['Infected Attendees'] * 1.1)
        else:
            ax.set_ylim(0, 1)

    # Hide unused subplots
    for i in range(num_plots, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'Infected Attendees from Selected Districts (First Event Day)\n({cfg.attendee_recruitment_model} Model)', fontsize=18, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 1]) # type: ignore

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    beta_event = cfg.event_base_transmission_rate
    filename = results_dir / f'infected_attendees_selected_districts_{cfg.attendee_recruitment_model}_R{int(cfg.R_0*100)}_betaEvent{int(beta_event*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Infected attendee visualization saved to '{filename}'")

def analyze_and_save_peak_time(E_realizations, I_realizations, cfg):
    """
    Calculates peak time statistics and appends them to a results file.
    """
    peak_times = []
    peak_values = []
    time_points = E_realizations[:, 0]

    for i in range(1, cfg.n_iterations + 1):
        total_infected_curve = E_realizations[:, i] + I_realizations[:, i]
        peak_index = np.argmax(total_infected_curve)
        peak_time = time_points[peak_index]
        peak_value = total_infected_curve[peak_index]
        peak_times.append(peak_time)
        peak_values.append(peak_value)

    mean_peak_time = np.mean(peak_times)
    std_peak_time = np.std(peak_times)
    mean_peak_value = np.mean(peak_values)

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / "peak-time-distribution.txt"
    file_exists = os.path.isfile(filename)

    if cfg.event_model_type.startswith('no_event'):
        scenario_name = "baseline"
    else:
        scenario_name = getattr(cfg, 'current_event_scenario_name', 'N/A')

    with open(filename, 'a') as f:
        if not file_exists:
            header = "scenario_name,event_model,recruitment_model,initially_infected,event_patch_id,beta_event,R0,initially_infected_seed,initially_infected_patch,n_iterations,peak_time_avg,peak_time_std,peak_value_avg\n"
            f.write(header)

        # Determine the number of initially infected/exposed based on the scenario
        if cfg.event_model_type.startswith('no_event') and cfg.no_event_seeding_mode == 'single_patch':
            init_inf = cfg.I_ss
            patch_id = cfg.initial_infection_patch_id
        elif cfg.event_model_type in ['large_venue', 'multi_venue', 'no_event_distributed']:
            init_inf = cfg.large_venue_initially_infected
            patch_id = cfg.large_venue_patch_id if cfg.event_model_type == 'large_venue' else "N/A"
        else:
            init_inf = -1 # Should not happen with current setup
            patch_id = -1

        data_row = (
            f"{scenario_name},{cfg.event_model_type},{cfg.attendee_recruitment_model},{init_inf},{patch_id},"
            f"{cfg.event_base_transmission_rate},{cfg.R_0:.2f},"
            f"{cfg.large_venue_ini_infected_attendee_seed},{cfg.large_venue_ini_infected_attendee_patch},"
            f"{cfg.n_iterations},{mean_peak_time:.4f},{std_peak_time:.2f},{mean_peak_value:.2f}\n"
        )
        f.write(data_row)

    print(f"\nPeak time distribution summary saved to '{filename}'.")


def plot_initial_infection_distribution_by_district(infection_log_file: str, population_df: pd.DataFrame, cfg: any):
    """
    Loads the n_ini_infections file and creates boxplots showing the distribution
    of initially infected attendees recruited from each district across all runs.
    """
    if not os.path.exists(infection_log_file):
        print(f"Warning: Infection log file not found, cannot generate district infection boxplot: {infection_log_file}")
        return

    df = pd.read_csv(infection_log_file)
    if df.empty:
        print(f"Warning: Infection log file is empty: {infection_log_file}")
        return

    # Identify district columns (e.g., 'n_0', 'n_1', ...)
    district_cols = [col for col in df.columns if col.startswith('n_') and col != 'n_all']
    if not district_cols:
        print("Warning: No district-specific infection columns (n_X) found in the log file.")
        return

    # Calculate and print total initial infections statistics for Day 0
    df_day0 = df[df['event_day'] == 0]
    if not df_day0.empty:
        if 'n_all' in df_day0.columns:
            totals = df_day0['n_all']
        else:
            totals = df_day0[district_cols].sum(axis=1)
        print(f"Initial infections at event (Day 0) across {len(totals)} runs: Mean = {totals.mean():.2f} +/- {totals.std():.2f}")

    # Melt the dataframe from a wide to a long format for easier plotting
    df_long = df.melt(id_vars=['run_id', 'event_day'],
                      value_vars=district_cols,
                      var_name='district_col',
                      value_name='n_infected')

    # Extract the integer district ID from the column name (e.g., 'n_1' -> 1)
    df_long['district_id'] = df_long['district_col'].str.replace('n_', '').astype(int)

    # Filter for only the first day of the event (day 0)
    df_long = df_long[df_long['event_day'] == 0]

    # We only need to plot districts that contributed at least one infected attendee
    df_plot = df_long[df_long['n_infected'] > 0]

    if df_plot.empty:
        print(f"No initially infected attendees were recorded in {infection_log_file}. Skipping boxplot.")
        return

    fig, ax = plt.subplots(figsize=(20, 10))
    # Set patch_artist=True to allow for custom styling of the boxes
    bp = df_plot.boxplot(column='n_infected', by='district_id', ax=ax, grid=True, vert=True, patch_artist=True)

    # Make the boxes have no fill color so the lines are clear
    for box in bp.findobj(plt.matplotlib.patches.PathPatch):
        box.set_facecolor('none')
    fig.suptitle('') # Remove the default suptitle from pandas boxplot
    ax.set_title(f'Distribution of Initially Infected Attendees per District on Day 0\n(Scenario: {cfg.event_model_type}, Recruitment: {cfg.attendee_recruitment_model})', fontsize=14)
    ax.set_xlabel('District ID', fontsize=10)
    ax.set_ylabel('Number of Initially Infected Attendees Recruited', fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    beta_event = cfg.event_base_transmission_rate
    filename = results_dir / f'ini_infections_by_district_boxplot_{cfg.event_model_type}_R{int(cfg.R_0*100)}_betaEvent{int(beta_event*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Initial infection distribution boxplot saved to '{filename}'")





def plot_event_induced_infections_map(all_runs_results, data, cfg):
    """
    Plots a choropleth map showing the mean number of event-induced exposures per district.
    """
    if not HAS_GEOPANDAS or not getattr(cfg, 'plot_on_the_fly', True):
        return

    n_patches = data['n_patches']
    exposure_data = np.stack([run.get('event_exposures_by_patch', np.zeros(n_patches)) for run in all_runs_results])
    mean_exposures = np.mean(exposure_data, axis=0)
    
    if np.sum(mean_exposures) == 0:
        return

    # Load GeoJSON
    mobility_dir = Path(f"data_mobility_{cfg.dataset_name}")
    geojson_path = mobility_dir / f"{cfg.dataset_name.lower()}-districts.geojson"
    
    # Fallback discovery logic
    if not geojson_path.exists():
        if mobility_dir.exists():
            geojsons = list(mobility_dir.glob("*.geojson"))
            if geojsons:
                geojson_path = geojsons[0]

    if not geojson_path.exists():
        return

    try:
        geodf = gpd.read_file(geojson_path)
        id_col = next((c for c in ['id', 'patch_id', 'ID'] if c in geodf.columns), None)
        if id_col:
            if id_col != 'id': geodf = geodf.rename(columns={id_col: 'id'})
            geodf['id'] = pd.to_numeric(geodf['id'], errors='coerce')
            if geodf['id'].min() == 1 and geodf['id'].max() == n_patches:
                geodf['id'] = geodf['id'] - 1
        else:
            # Fallback: create id column from index
            geodf['id'] = range(len(geodf))

        exposure_df = pd.DataFrame({'id': range(n_patches), 'mean_exposures': mean_exposures})
        merged = geodf.merge(exposure_df, on='id', how='left')

        fig, ax = plt.subplots(figsize=(8, 6))
        merged.plot(column='mean_exposures', cmap='Oranges', legend=True,
                    legend_kwds={'label': "Mean New Exposures per Run"},
                    edgecolor='0.3', linewidth=0.5, ax=ax)
        
        """
        # Mark the venue
        venue_patch_id = getattr(cfg, 'large_venue_patch_id', 0)
        venue_row = geodf[geodf['id'] == venue_patch_id]
        if not venue_row.empty:
            centroid = venue_row.to_crs(epsg=3857).centroid.to_crs(geodf.crs).iloc[0]
            ax.scatter(centroid.x, centroid.y, marker='*', color='red', s=150, label='Event Venue', edgecolors='black')
        """

        # Mark the seed
        seed_patch_id = getattr(cfg, 'large_venue_ini_infected_attendee_patch', 0)
        seed_row = geodf[geodf['id'] == seed_patch_id]
        if not seed_row.empty:
            centroid = seed_row.to_crs(epsg=3857).centroid.to_crs(geodf.crs).iloc[0]
            ax.scatter(centroid.x, centroid.y, marker='x', color='black', s=80, lw=2, label='Initial Seed Source')

        scenario_label = getattr(cfg, "current_event_scenario_name", cfg.event_model_type)
        ax.set_title(f'Geographic Distribution of Event-Induced Infections\n'
                     f'({scenario_label}: Mean exposures, $\\beta_{{event}}$ = {cfg.event_base_transmission_rate:.2f})', fontsize=14)
        ax.set_axis_off()
        # ax.legend(loc='lower right')

        results_dir = getattr(cfg, 'results_dir', Path("results"))
        filename = results_dir / f'event_induced_infections_map_{scenario_label}_betaEvent{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Event-induced infection map saved to '{filename.name}'")
        
    except Exception as e:
        print(f"  -> Failed to generate event-induced infection map: {e}")


def plot_infected_attendee_distribution_by_district_all_days(infection_log_file: str, population_df: pd.DataFrame, cfg: any):
    """
    Loads the n_ini_infections file and creates boxplots showing the distribution
    of infected attendees recruited from each district across all runs and all event days.
    """
    if not os.path.exists(infection_log_file):
        print(f"Warning: Infection log file not found, cannot generate district infection boxplot: {infection_log_file}")
        return

    df = pd.read_csv(infection_log_file)
    if df.empty:
        print(f"Warning: Infection log file is empty: {infection_log_file}")
        return

    # Identify district columns (e.g., 'n_0', 'n_1', ...)
    district_cols = [col for col in df.columns if col.startswith('n_') and col != 'n_all']
    if not district_cols:
        print("Warning: No district-specific infection columns (n_X) found in the log file.")
        return

    # Melt the dataframe from a wide to a long format for easier plotting
    df_long = df.melt(id_vars=['run_id', 'event_day'],
                      value_vars=district_cols,
                      var_name='district_col',
                      value_name='n_infected')

    # Extract the integer district ID from the column name (e.g., 'n_1' -> 1)
    df_long['district_id'] = df_long['district_col'].str.replace('n_', '').astype(int)

    # We only need to plot districts that contributed at least one infected attendee
    df_plot = df_long[df_long['n_infected'] > 0]

    if df_plot.empty:
        print(f"No infected attendees were recorded in {infection_log_file}. Skipping boxplot for all days.")
        return

    fig, ax = plt.subplots(figsize=(20, 10))
    df_plot.boxplot(column='n_infected', by='district_id', ax=ax, grid=True, vert=True, patch_artist=False)
    fig.suptitle('') # Remove the default suptitle from pandas boxplot
    ax.set_title(f'Distribution of Infected Attendees Recruited per District (All Event Days)\n(Scenario: {cfg.event_model_type}, Recruitment: {cfg.attendee_recruitment_model})', fontsize=14)
    ax.set_xlabel('District ID', fontsize=10)
    ax.set_ylabel('Number of Infected Attendees Recruited', fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    beta_event = cfg.event_base_transmission_rate
    filename = results_dir / f'infected_attendees_by_district_all_days_boxplot_{cfg.event_model_type}_R{int(cfg.R_0*100)}_betaEvent{int(beta_event*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Infected attendee distribution boxplot for all days saved to '{filename}'")


def save_full_results(run_id, results, cfg):
    """Saves the detailed subpopulation results for a single run."""
    if not cfg.write_full_results:
        return

    t = results['t']
    n_subpops = results['S_ij'][0].shape[0]

    # Create a time column that repeats for each subpopulation
    time_col = np.repeat(t, n_subpops)

    # Stack all data columns horizontally
    output_data = np.stack([
        time_col,
        np.vstack(results['S_ij']).ravel(), np.vstack(results['E_ij']).ravel(),
        np.vstack(results['I_ij']).ravel(), np.vstack(results['R_ij']).ravel(),
        np.vstack(results['N_ij']).ravel()
    ], axis=1)

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    beta_event = cfg.event_base_transmission_rate
    filename = results_dir / f"results_full_{cfg.dataset_name}_{cfg.event_model_type}_R{int(cfg.R_0*100)}_betaEvent{int(beta_event*100)}_Iss{cfg.I_ss}_{run_id}.npy"
    np.save(filename, output_data.astype(int))
    print(f"Full results for run {run_id} saved to '{filename}'.")

def plot_peak_scatter_comparison(all_scenario_results, cfg, group_name=None):
    """
    Generates a scatter plot comparing Peak Time vs Peak Value (E+I) for different scenarios.
    """
    print(f"Generating peak time vs peak value scatter plot (Group: {group_name})...")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define baseline style
    baseline_color = 'blue'
    baseline_marker = 'x'

    # Get list of scenarios, ensuring no_event is first if present
    scenarios = list(all_scenario_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort() # Sort event scenarios alphabetically
        scenarios.insert(0, 'no_event')

    if not scenarios:
        print("No scenarios available for scatter plot.")
        plt.close(fig)
        return

    # Use a colormap for event scenarios to ensure distinct colors
    cmap = plt.get_cmap('tab10')

    for idx, scenario in enumerate(scenarios):
        results = all_scenario_results[scenario]
        realizations = results['realizations']
        E_realizations = realizations['E']
        I_realizations = realizations['I']
        time = E_realizations[:, 0]
        n_iterations = cfg.n_iterations
        
        peak_times = []
        peak_values = []

        for i in range(1, n_iterations + 1):
            # Calculate E+I curve (Active cases)
            total_active = E_realizations[:, i] + I_realizations[:, i]
            peak_idx = np.argmax(total_active)
            peak_times.append(time[peak_idx])
            peak_values.append(total_active[peak_idx])

        if scenario == 'no_event':
            c = baseline_color
            m = baseline_marker
            label = 'Baseline'
            alpha = 0.6
            edgecolors = None
        else:
            # Determine color and label
            if scenario == 'large_venue':
                c = 'red'
                label = getattr(cfg, "current_event_scenario_name", "Large Gathering")
            elif scenario == 'multi_venue':
                c = 'green'
                label = 'Multi-Venue'
            else:
                # Dynamic color for grouped scenarios
                # Skip index 0 (no_event) for color mapping if present
                color_index = (idx - 1) if 'no_event' in scenarios else idx
                c = cmap(color_index % 10)
                label = scenario
            
            m = 'o'
            alpha = 0.6
            edgecolors = 'none'

        ax.scatter(peak_times, peak_values, color=c, marker=m, label=label, alpha=alpha, edgecolors=edgecolors)

    if group_name:
        title_part = f"Group: {group_name}"
        filename_prefix = f"peak_scatter_{group_name}"
    else:
        scenario_name = getattr(cfg, "current_event_scenario_name", "combined")
        title_part = f"Scenario: {scenario_name}"
        filename_prefix = f"peak_scatter_comparison_{scenario_name}"

    ax.set_title(f'Epidemic Peak Characteristics (E+I)\n({title_part}, $R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=16)
    ax.set_xlabel('Peak Time (Days)', fontsize=14)
    ax.set_ylabel('Peak Value (Total E+I)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'{filename_prefix}_R{int(cfg.R_0*100)}_betaEvent{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Peak scatter plot saved to '{filename}'")
    plt.close(fig)

def plot_patch_peak_comparison(all_scenario_results, data, cfg):
    """
    Generates two figures comparing peak time and peak value distributions per patch
    between Baseline and the active Event scenario using boxplots.
    """
    print("Generating patch-level peak comparison plots...")
    
    baseline_name = 'no_event'
    event_name = None
    if 'large_venue' in all_scenario_results:
        event_name = 'large_venue'
    elif 'multi_venue' in all_scenario_results:
        event_name = 'multi_venue'
    
    if baseline_name not in all_scenario_results or event_name is None:
        print("Comparison requires 'no_event' and an event scenario (large_venue or multi_venue). Skipping patch comparison.")
        return

    n_patches = data['n_patches']
    
    def extract_peaks(results_list):
        p_times = {i: [] for i in range(n_patches)}
        p_values = {i: [] for i in range(n_patches)}
        
        for res in results_list:
            t = np.array(res['t'])
            
            if 'E_j' in res and 'I_j' in res:
                Total_j = np.array(res['E_j']) + np.array(res['I_j'])
            elif 'E_ij' in res and 'I_ij' in res:
                # Reshape to (time, home_patch, current_patch)
                E_reshaped = np.array(res['E_ij']).reshape(-1, n_patches, n_patches)
                I_reshaped = np.array(res['I_ij']).reshape(-1, n_patches, n_patches)
                # Sum over home patches (axis 1) to get total people present in patch j
                Total_j = E_reshaped.sum(axis=1) + I_reshaped.sum(axis=1)
            else:
                continue
            
            for j in range(n_patches):
                curve = Total_j[:, j]
                if np.max(curve) > 0:
                    idx = np.argmax(curve)
                    p_times[j].append(t[idx])
                    p_values[j].append(curve[idx])
                else:
                    p_times[j].append(np.nan)
                    p_values[j].append(0)
        return pd.DataFrame(p_times), pd.DataFrame(p_values)

    base_res = all_scenario_results[baseline_name]['all_runs_results']
    event_res = all_scenario_results[event_name]['all_runs_results']
    
    df_base_time, df_base_val = extract_peaks(base_res)
    df_event_time, df_event_val = extract_peaks(event_res)

    # Calculate differences (Event - Baseline) per run
    df_diff_time = df_event_time - df_base_time
    df_diff_val = df_event_val - df_base_val

    beta_event = cfg.event_base_transmission_rate
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    scenario_label = getattr(cfg, "current_event_scenario_name", event_name)

    # --- Figure 1: Peak Time ---
    fig1, axes1 = plt.subplots(1, 3, figsize=(36, 8))
    df_base_time.boxplot(ax=axes1[0], grid=True, showfliers=False)
    axes1[0].set_title(f"Baseline ({baseline_name})", fontsize=16)
    axes1[0].set_ylabel("Peak Time (Days)", fontsize=14)
    axes1[0].set_xlabel("Patch ID", fontsize=14)
    
    df_event_time.boxplot(ax=axes1[1], grid=True, showfliers=False)
    axes1[1].set_title(f"Event ({scenario_label})", fontsize=16)
    axes1[1].set_xlabel("Patch ID", fontsize=14)
    axes1[1].sharey(axes1[0]) # Share Y-axis with Baseline for direct comparison

    df_diff_time.boxplot(ax=axes1[2], grid=True, showfliers=False)
    axes1[2].set_title(f"Difference (Event - Baseline)", fontsize=16)
    axes1[2].set_xlabel("Patch ID", fontsize=14)
    axes1[2].set_ylabel("Difference (Days)", fontsize=14)
    axes1[2].axhline(0, color='red', linestyle='--', alpha=0.5)
    
    fig1.suptitle(f"Distribution of Epidemic Peak Time (E+I) per Patch\n($R_0$={cfg.R_0}, $\\beta_{{event}}$={beta_event})", fontsize=16)
    plt.savefig(results_dir / f'patch_peak_time_comparison_{scenario_label}_R{int(cfg.R_0*100)}_betaEvent{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png', bbox_inches='tight', dpi=300)
    plt.close(fig1)

    # --- Figure 2: Peak Value ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(36, 8))
    df_base_val.boxplot(ax=axes2[0], grid=True, showfliers=False)
    axes2[0].set_title(f"Baseline ({baseline_name})", fontsize=16)
    axes2[0].set_ylabel("Peak Value (Active Cases)", fontsize=14)
    axes2[0].set_xlabel("Patch ID", fontsize=14)
    
    df_event_val.boxplot(ax=axes2[1], grid=True, showfliers=False)
    axes2[1].set_title(f"Event ({scenario_label})", fontsize=16)
    axes2[1].set_xlabel("Patch ID", fontsize=14)
    axes2[1].sharey(axes2[0]) # Share Y-axis with Baseline for direct comparison

    df_diff_val.boxplot(ax=axes2[2], grid=True, showfliers=False)
    axes2[2].set_title(f"Difference (Event - Baseline)", fontsize=16)
    axes2[2].set_xlabel("Patch ID", fontsize=14)
    axes2[2].set_ylabel("Difference (Active Cases)", fontsize=14)
    axes2[2].axhline(0, color='red', linestyle='--', alpha=0.5)
    
    fig2.suptitle(f"Distribution of Epidemic Peak Value (E+I) per Patch\n($R_0$={cfg.R_0}, $\\beta_{{event}}$={beta_event})", fontsize=16)
    plt.savefig(results_dir / f'patch_peak_value_comparison_{scenario_label}_R{int(cfg.R_0*100)}_betaEvent{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png', bbox_inches='tight', dpi=300)
    plt.close(fig2)
    print("Saved patch-level peak comparison figures.")

def plot_early_cumulative_cases(all_scenario_results, data, cfg, days=60):
    """
    Plots active cases E for the first 'days' days for AMS, Leipzig, and Baseline scenarios.
    """
    print(f"Generating early epidemic active case plot (first {days} days)...")
    
    # Filter scenarios to include only Baseline, AMS*, and Leipzig*
    scenarios_to_plot = []
    for name in all_scenario_results.keys():
        if name == 'no_event' or 'AMS' in name or 'Leipzig' in name:
            scenarios_to_plot.append(name)
    
    if not scenarios_to_plot:
        return

    # Sort: Baseline first, then alphabetical
    scenarios_to_plot.sort(key=lambda x: (0 if x == 'no_event' else 1, x))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for scenario in scenarios_to_plot:
        res = all_scenario_results[scenario]
        E = res['realizations']['E']
        # I = res['realizations']['I']
        time = E[:, 0]
        
        # Limit to 'days'
        mask = time <= days
        time_segment = time[mask]
        E_segment = E[mask, 1:] # Exclude time column (index 0)
        # I_segment = I[mask, 1:] # Exclude time column (index 0)
        
        # Calculate mean active cases: E + I
        mean_E = np.mean(E_segment, axis=1)
        # mean_I = np.mean(I_segment, axis=1)
        active_cases = mean_E # + mean_I
        
        label = "Baseline" if scenario == 'no_event' else scenario
        linestyle = '--' if scenario == 'no_event' else '-'
        linewidth = 3 if scenario == 'no_event' else 2
        color = 'black' if scenario == 'no_event' else None
        
        if color:
            ax.plot(time_segment, active_cases, label=label, linestyle=linestyle, linewidth=linewidth, color=color)
        else:
            ax.plot(time_segment, active_cases, label=label, linestyle=linestyle, linewidth=linewidth)
        
    ax.set_title(f'Cumulative active cases: First {days} Days\n($R_0$={cfg.R_0}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=16)
    ax.set_xlabel('Time (Days)', fontsize=14)
    ax.set_ylabel('Active Cases', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'active_cases_early_{days}d_AMS_Leipzig_R{int(cfg.R_0*100)}_betaEvent{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Early active case plot saved to '{filename}'")

def plot_die_out_frequency(cfg, scenarios_to_plot):
    """
    Visualizes the frequency of disease die-out for a set of scenarios
    based on the current configuration parameters.
    """
    print("Generating die-out frequency plot...")
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / "disease_die_out_records.txt"

    if not os.path.exists(filename):
        print("  -> Die-out log file not found. Skipping plot.")
        return

    try:
        df = pd.read_csv(filename)
    except Exception as e:
        # Handle mixed-format files (e.g., old 10-col vs new 12-col) by forcing the new schema
        # print(f"  -> Standard read failed ({e}). Attempting robust read...")
        cols = ["scenario_name","I_ss","run_id","event_model_type","recruitment_model","initially_infected","event_patch_id","beta_event","R0","initially_infected_seed","initially_infected_patch","die_out_day"]
        try:
            # Try reading with python engine and skipping bad lines
            df = pd.read_csv(filename, names=cols, header=0, on_bad_lines='skip', engine='python')
        except TypeError:
            # Fallback for older pandas versions
            try:
                df = pd.read_csv(filename, names=cols, header=0, error_bad_lines=False, engine='python')
            except Exception as e2:
                print(f"  -> Error reading die-out log file: {e2}. Skipping plot.")
                return
        except Exception as e3:
            print(f"  -> Error reading die-out log file: {e3}. Skipping plot.")
            return

    if df.empty:
        # If the file is empty, it means no die-outs have occurred yet.
        df = pd.DataFrame(columns=['scenario_name', 'R0', 'I_ss', 'beta_event'])

    # Filter for the current parameter set using floating point-safe comparison
    # FIX: Baseline ('no_event') is independent of beta_event, but logged with a specific one.
    # We must include it if R0 and Iss match, regardless of the current beta_event filter.
    mask_common = (np.isclose(df['R0'], cfg.R_0)) & (df['I_ss'] == cfg.I_ss)
    mask_event = (df['scenario_name'] != 'no_event') & (np.isclose(df['beta_event'], cfg.event_base_transmission_rate))
    mask_baseline = (df['scenario_name'] == 'no_event')
    df_filtered = df[mask_common & (mask_event | mask_baseline)]

    if df_filtered.empty:
        # No die-out events recorded for the current parameter set.
        die_out_counts = pd.Series(0, index=scenarios_to_plot)
    else:
        # Count die-outs for each scenario
        die_out_counts = df_filtered.groupby('scenario_name').size()

    # Ensure all scenarios from the current run are represented, even if they had 0 die-outs
    all_counts = pd.Series(0, index=scenarios_to_plot, dtype=float)
    all_counts.update(die_out_counts)
    
    # Calculate frequency
    frequencies = all_counts / cfg.n_iterations

    # Sort scenarios: baseline first, then alphabetically
    frequencies = frequencies.reindex(sorted(frequencies.index, key=lambda x: (0 if x == 'no_event' else 1, x)))

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    frequencies.plot(kind='bar', ax=ax, color='teal', edgecolor='black', alpha=0.8, zorder=3)

    ax.set_title(f'Disease Die-Out Frequency\n($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=16)
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Die-Out Frequency (Fraction of Runs)', fontsize=12)
    ax.set_ylim(0, 1.05)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=10)

    fig.tight_layout()
    
    plot_filename = results_dir / f'die_out_frequency_R{int(cfg.R_0*100)}_betaEvent{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)
    print(f"Die-out frequency plot saved to '{plot_filename}'")

def plot_event_risk_metrics(risk_data, cfg):
    """
    Plots a comparison of static event risk metrics (R_event, GCC) across scenarios.
    """
    if not getattr(cfg, 'plot_on_the_fly', True):
        return

    if not risk_data:
        return
    
    print("Generating event risk metrics comparison plot...")
    df = pd.DataFrame(risk_data)
    # df columns: scenario, gcc, nodes, R_avg, R_max
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: R_event (Avg and Max)
    x = np.arange(len(df))
    width = 0.35
    
    axes[0].bar(x - width/2, df['R_avg'], width, label='Avg $R_{event}$', color='skyblue', edgecolor='black')
    axes[0].bar(x + width/2, df['R_max'], width, label='Max $R_{event}$', color='salmon', edgecolor='black')
    
    axes[0].set_ylabel('Expected Secondary Infections ($R_{event}$)')
    axes[0].set_title(f'First-Generation Transmission Potential\n($\\beta_{{event}}$={cfg.event_base_transmission_rate})')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df['scenario'], rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 2: GCC
    axes[1].bar(x, df['gcc'], width*1.5, color='mediumpurple', edgecolor='black')
    axes[1].set_ylabel('Giant Connected Component Fraction')
    axes[1].set_title('Theoretical Network Connectivity (GCC)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df['scenario'], rotation=45, ha='right')
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate GCC node counts
    for i, row in df.iterrows():
        axes[1].text(i, row['gcc'] + 0.02, f"{int(row['gcc']*row['nodes'])}/{row['nodes']}", ha='center', fontsize=9)

    plt.tight_layout()
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'event_risk_metrics_comparison_betaEvent{int(cfg.event_base_transmission_rate*100)}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Event risk metrics plot saved to '{filename}'")

def plot_phase_plane_comparison(all_scenario_results, data, cfg):
    """
    Generates a phase plane plot (S vs E+I) comparing trajectories of different scenarios.
    X-axis: Proportion of Susceptible (S)
    Y-axis: Proportion of Active Cases (E+I)
    Generates linear scale version.
    """
    if not getattr(cfg, 'plot_on_the_fly', True):
        return

    print("Generating S vs (E+I) phase plane comparison plots...")
    
    scenarios = list(all_scenario_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
    
    if not scenarios:
        return

    total_population = data['population_df']['population'].sum()
    cmap = plt.get_cmap('tab10')
    results_dir = getattr(cfg, 'results_dir', Path("results"))

    # Define markers for end points to distinguish trajectories
    # markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for scale in ['linear']:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for idx, scenario in enumerate(scenarios):
            results = all_scenario_results[scenario]
            realizations = results['realizations']
            
            # We want the mean trajectory across runs
            # S_realizations shape: (time_steps, n_iterations + 1) where col 0 is time
            # We average across columns 1 to end (the runs)
            S_mean = np.mean(realizations['S'][:, 1:], axis=1)
            E_mean = np.mean(realizations['E'][:, 1:], axis=1)
            I_mean = np.mean(realizations['I'][:, 1:], axis=1)
            
            S_prop = S_mean / total_population
            EI_prop = (E_mean + I_mean) / total_population
            
            label = "Baseline" if scenario == 'no_event' else scenario
            
            if scenario == 'no_event':
                color = 'black'
                linestyle = '--'
                linewidth = 2.5
                zorder = 10
                marker = 'X'
            else:
                color = cmap(idx % 10)
                linestyle = '-'
                linewidth = 2
                zorder = 5
                marker = markers[(idx - 1) % len(markers)]
                
            ax.plot(S_prop, EI_prop, label=label, color=color, linestyle=linestyle, linewidth=linewidth, zorder=zorder)
            
            # Mark the start point (Time 0)
            ax.scatter(S_prop[0], EI_prop[0], color=color, s=50, marker='.', zorder=zorder)
            
            # Mark the end point with a unique symbol
            ax.scatter(S_prop[-1], EI_prop[-1], color=color, s=80, marker=marker, zorder=zorder+1, edgecolors='white')

        ax.set_xlabel('Proportion Susceptible (S)', fontsize=14)
        ax.set_ylabel('Proportion Active Cases (E+I)', fontsize=14)
        
        if scale == 'log':
            ax.set_xscale('log')
            ax.set_yscale('log')
            title_suffix = " (Log Scale)"
            file_suffix = "_log"
        else:
            title_suffix = "" # " (Linear Scale)"
            file_suffix = "_linear"

        # ax.set_title(f'Phase Plane Trajectory: S vs (E+I){title_suffix}\n($R_0$={cfg.R_0}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=16)
        ax.set_title(f'Phase Plane Trajectory\n($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=16)
        ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.6, which='both' if scale == 'log' else 'major')
        
        ax.set_xlim(0, 1.2)
        # ax.set_ylim(0, 1)
        
        # Invert x-axis so time flows left-to-right (High S -> Low S)
        # ax.invert_xaxis()
        
        filename = results_dir / f'phase_plane_S_vs_EI{file_suffix}_R{int(cfg.R_0*100)}_betaEvent{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Phase plane plot ({scale}) saved to '{filename}'")

def plot_active_cases_comparison(group_results, data, cfg, group_name, show_variation=True):
    """
    Plots the time evolution of active cases ((E+I)/N * 1000) for a group of scenarios.
    Includes individual run curves (light) and average trend (solid with markers).
    """
    print(f"Generating active cases comparison plot for group: {group_name} (Variation: {show_variation})...")
    
    # Sort scenarios: Baseline first, then alphabetical
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
    
    total_population = data['population_df']['population'].sum()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Define styles
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p'] #, '*', 'h']
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    
    for idx, scenario in enumerate(scenarios):
        results = group_results[scenario]
        realizations = results['realizations']
        E = realizations['E']
        I = realizations['I']
        time = E[:, 0] # Time vector (interpolated frames)
        
        # Filter for successful runs: cumulative infections at the end > I_ss
        # This ensures the statistics reflect the dynamics of outbreaks that successfully emerge.
        if 'R' in realizations:
            # cumulative infections = Recovered at end
            successful_runs_mask = realizations['R'][-1, 1:] > cfg.I_ss
        else:
            # Fallback: check for any active cases during the run
            successful_runs_mask = np.max(E[:, 1:] + I[:, 1:], axis=0) > 0

        if not np.any(successful_runs_mask):
            print(f"  -> Scenario '{scenario}': No successful runs found (all died out). Skipping in plot.")
            continue

        n_excluded = len(successful_runs_mask) - np.sum(successful_runs_mask)
        if n_excluded > 0:
            print(f"  -> Scenario '{scenario}': Excluding {n_excluded} runs with zero secondary infections from comparison.")

        # Calculate (E+I)/N * 1000 for all runs (Columns 1 to end are the runs)
        EI_runs = ((E[:, 1:][:, successful_runs_mask] + I[:, 1:][:, successful_runs_mask]) / total_population) * 1000
        EI_median = np.median(EI_runs, axis=1)
        EI_low = np.percentile(EI_runs, 25, axis=1)
        EI_high = np.percentile(EI_runs, 75, axis=1)
        n_runs = EI_runs.shape[1]
        
        label = "Baseline" if scenario == 'no_event' else scenario
        
        # Assign color and marker
        if scenario == 'no_event':
            color = 'black'
            # marker = 'X'
            marker = 'x'
            zorder_mean = 10
        else:
            color = colors[idx % len(colors)]
            if color == 'black': color = colors[(idx+1) % len(colors)] # Avoid black for non-baseline
            marker = markers[(idx-1) % len(markers)] if scenario != 'no_event' else 'X'
            zorder_mean = 5
        
        # Plot individual runs
        if show_variation:
            if "All_Scenarios" in group_name:
                ax.fill_between(time, EI_low, EI_high, 
                               color=color, alpha=0.15, zorder=zorder_mean - 1)
            else:
                for i in range(n_runs):
                    ax.plot(time, EI_runs[:, i], color=color, alpha=0.1, linewidth=1, zorder=1)
        
        # Plot mean trend without markers
        ax.plot(time, EI_median, color=color, label=label, linewidth=2.5, zorder=zorder_mean)

    ax.set_xlabel('Time (Days)', fontsize=14)
    ax.set_ylabel('Active cases per 1000 persons', fontsize=14)
    #ax.set_title(f'Active cases: {group_name}\n($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=16)
    ax.set_title(f'Active case curves\n($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=16)
    ax.legend(fontsize=10, loc='best', framealpha=0.7)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'active_cases_comparison_{group_name}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Active cases comparison plot saved to '{filename}'")

def plot_cumulative_incidence_comparison(group_results, data, cfg, group_name, show_variation=True):
    """
    Plots the time evolution of cumulative incidence (total infected) for a group of scenarios.
    Includes individual run curves (light) and average trend (solid with markers).
    """
    print(f"Generating cumulative incidence comparison plot for group: {group_name} (Variation: {show_variation})...")
    
    # Sort scenarios: Baseline first, then alphabetical
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
    
    total_population = data['population_df']['population'].sum()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define styles
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p'] #, '*', 'h']
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    marker_interval = 10
    
    for idx, scenario in enumerate(scenarios):
        results = group_results[scenario]
        realizations = results['realizations']
        E = realizations['E']
        I = realizations['I']
        R = realizations['R']
        time = E[:, 0]
        
        # Filter for successful runs: cumulative infections at the end > I_ss
        # This ensures the statistics reflect the dynamics of outbreaks that successfully emerge.
        if 'R' in realizations:
            # cumulative infections = Recovered at end
            successful_runs_mask = realizations['R'][-1, 1:] > cfg.I_ss
        else:
            # Fallback: check for any active cases during the run
            successful_runs_mask = np.max(E[:, 1:] + I[:, 1:], axis=0) > 0

        if not np.any(successful_runs_mask):
            print(f"  -> Scenario '{scenario}': No successful runs found. Skipping.")
            continue

        n_excluded = len(successful_runs_mask) - np.sum(successful_runs_mask)
        if n_excluded > 0:
            print(f"  -> Scenario '{scenario}': Excluding {n_excluded} runs with zero secondary infections from comparison.")

        # Calculate R/N * 1000 for all runs (Columns 1 to end are the runs)
        # Using Cumulative Recovered as the metric for cumulative infections as requested
        Cum_runs = (R[:, 1:][:, successful_runs_mask] / total_population) * 1000
        Cum_median = np.median(Cum_runs, axis=1)
        Cum_low = np.percentile(Cum_runs, 25, axis=1)
        Cum_high = np.percentile(Cum_runs, 75, axis=1)
        n_runs = Cum_runs.shape[1]
        
        label = "Baseline" if scenario == 'no_event' else scenario
        
        # Assign color and marker
        if scenario == 'no_event':
            color = 'black'
            marker = 'x'
            zorder_mean = 10
        else:
            color = colors[idx % len(colors)]
            if color == 'black': color = colors[(idx+1) % len(colors)] # Avoid black for non-baseline
            marker = markers[(idx-1) % len(markers)] if scenario != 'no_event' else 'X'
            zorder_mean = 5
        
        # Plot individual runs
        if show_variation:
            if "All_Scenarios" in group_name:
                ax.fill_between(time, Cum_low, Cum_high, 
                               color=color, alpha=0.15, zorder=zorder_mean - 1)
            else:
                for i in range(n_runs):
                    ax.plot(time, Cum_runs[:, i], color=color, alpha=0.1, linewidth=1, zorder=1)
        
        # Plot mean (solid curve with markers at ticks)
        max_time = time[-1]
        target_times = np.arange(0, max_time + 1, marker_interval)
        mark_indices = [np.abs(time - t_val).argmin() for t_val in target_times]
        
        ax.plot(time, Cum_median, color=color, label=label, linewidth=2.5, 
                marker=marker, markevery=mark_indices, markersize=8, zorder=zorder_mean)
        # Plot mean trend
        if show_variation:
            # Plot mean with markers at intervals
            max_time = time[-1]
            target_times = np.arange(0, max_time + 1, marker_interval)
            mark_indices = [np.abs(time - t_val).argmin() for t_val in target_times]
            ax.plot(time, Cum_median, color=color, label=label, linewidth=2.5, 
                    marker=marker, markevery=mark_indices, markersize=8, zorder=zorder_mean)
        else:
            # Cleaner version for 'mean only' plots: just the line without markers
            ax.plot(time, Cum_median, color=color, label=label, linewidth=2.5, zorder=zorder_mean)

    ax.set_xlabel('Time (Days)', fontsize=14)
    ax.set_ylabel('Cumulative cases per 1000 persons', fontsize=14)
    ax.set_title(f'Cumulative Incidence: {group_name}\n($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'cumulative_incidence_comparison_{group_name}_R{int(cfg.R_0*100)}_betaEvent{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Cumulative incidence comparison plot saved to '{filename}'")

def plot_rt_comparison(group_results, data, cfg, group_name):
    """
    Plots the time evolution of the effective reproduction number (Rt) for a group of scenarios.
    """
    print(f"Generating Rt comparison plot for group: {group_name}...")
    
    # Sort scenarios: Baseline first, then alphabetical
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define styles
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    for idx, scenario in enumerate(scenarios):
        results = group_results[scenario]
        realizations = results['realizations']
        
        if 'rt' not in realizations:
            print(f"Warning: Rt data not found for scenario '{scenario}'. Skipping.")
            continue
            
        rt_runs = realizations['rt']
        time = rt_runs[:, 0]
        
        # Filter for successful runs: cumulative infections at the end > I_ss
        # This is critical for Rt because failed runs stay at R0, biasing the average.
        if 'R' in realizations:
            successful_runs_mask = realizations['R'][-1, 1:] > cfg.I_ss
        else:
            successful_runs_mask = np.max(realizations['E'][:, 1:] + realizations['I'][:, 1:], axis=0) > 0

        if not np.any(successful_runs_mask):
            print(f"  -> Scenario '{scenario}': No successful runs found. Skipping.")
            continue

        rt_data = rt_runs[:, 1:][:, successful_runs_mask]
        n_runs = rt_data.shape[1]

        rt_mean = np.mean(rt_data, axis=1)
        rt_std = np.std(rt_data, axis=1)
        ci = 1.96 * rt_std / np.sqrt(n_runs) if n_runs > 0 else 0
        
        label = "Baseline" if scenario == 'no_event' else scenario
        
        if scenario == 'no_event':
            color = 'black'
            zorder_mean = 10
        else:
            color = colors[idx % len(colors)]
            if color == 'black': color = colors[(idx+1) % len(colors)]
            zorder_mean = 5
        
        ax.fill_between(time, rt_mean - ci, rt_mean + ci, color=color, alpha=0.2, zorder=1)
        ax.plot(time, rt_mean, color=color, label=label, linewidth=2.5, zorder=zorder_mean)

    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='$R_t=1$')
    ax.set_xlabel('Time (Days)', fontsize=14)
    ax.set_ylabel('Effective Reproduction Number ($R_t$)', fontsize=14)
    ax.set_title(f'Effective Reproduction Number ($R_t$) Evolution: {group_name}\n($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0)
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'rt_comparison_{group_name}_R{int(cfg.R_0*100)}_betaEvent{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Rt comparison plot saved to '{filename}'")

def plot_peak_incidence_comparison_boxplot(group_results, data, cfg, group_name, use_log_scale=False):
    """
    Generates a boxplot comparison of peak incidence ((E+I)/N * 1000)
    across different scenarios, excluding runs with zero secondary infections.
    """
    print(f"Generating peak incidence boxplot for group: {group_name} (Log: {use_log_scale})...")
    
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
    
    total_population = data['population_df']['population'].sum()
    
    plot_data = []
    labels = []
    
    # Define styles
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    for idx, scenario in enumerate(scenarios):
        realizations = group_results[scenario]['realizations']
        E = realizations['E']
        I = realizations['I']
        
        # Filter for successful runs: cumulative infections at end > I_ss
        if 'R' in realizations:
            successful_runs_mask = realizations['R'][-1, 1:] > cfg.I_ss
        else:
            successful_runs_mask = np.max(E[:, 1:] + I[:, 1:], axis=0) > 0

        if not np.any(successful_runs_mask):
            continue

        # Calculate peak (E+I)/N * 1000 for each successful run
        EI_runs = ((E[:, 1:][:, successful_runs_mask] + I[:, 1:][:, successful_runs_mask]) / total_population) * 1000
        peaks = np.max(EI_runs, axis=0)
        
        plot_data.append(peaks)
        labels.append("Baseline" if scenario == 'no_event' else scenario)

    if not plot_data:
        print(f"  -> No successful runs found in any scenario for peak incidence boxplot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    if use_log_scale:
        ax.set_yscale('log')

    # Hide the default median line so we can draw a single marker instead
    bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, 
                    medianprops={'visible': False})
    
    for i, line in enumerate(bp['medians']):
        median_x = np.mean(line.get_xdata())
        median_y = np.mean(line.get_ydata())
        ax.plot(median_x, median_y, 'rx', markersize=10, markeredgewidth=2.5, zorder=5)

    for idx, box in enumerate(bp['boxes']):
        color = 'black' if labels[idx] == "Baseline" else colors[idx % len(colors)]
        box.set_facecolor(color)
        box.set_alpha(0.6)

    ax.set_ylabel('Peak Incidence per 1000 persons', fontsize=12)
    ax.set_title(f'Distribution of Peak Incidence: {group_name}\n($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6, axis='y', which='both' if use_log_scale else 'major')
    plt.xticks(rotation=45, ha='right')
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    log_suffix = "_log" if use_log_scale else ""
    filename = results_dir / f'peak_incidence_boxplot_{group_name}{log_suffix}_R{int(cfg.R_0*100)}_beta{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  -> Peak incidence boxplot saved to '{filename}'")

def plot_peak_incidence_comparison_violin(group_results, data, cfg, group_name, use_log_scale=False):
    """
    Generates a violin plot comparison of peak incidence ((E+I)/N * 1000)
    across different scenarios, excluding runs with zero secondary infections.
    """
    print(f"Generating peak incidence violin plot for group: {group_name} (Log: {use_log_scale})...")
    
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
    
    total_population = data['population_df']['population'].sum()
    
    plot_data = []
    labels = []
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    for idx, scenario in enumerate(scenarios):
        realizations = group_results[scenario]['realizations']
        E = realizations['E']
        I = realizations['I']
        
        if 'R' in realizations:
            successful_runs_mask = realizations['R'][-1, 1:] > cfg.I_ss
        else:
            successful_runs_mask = np.max(E[:, 1:] + I[:, 1:], axis=0) > 0

        if not np.any(successful_runs_mask):
            continue

        EI_runs = ((E[:, 1:][:, successful_runs_mask] + I[:, 1:][:, successful_runs_mask]) / total_population) * 1000
        peaks = np.max(EI_runs, axis=0)
        
        plot_data.append(peaks)
        labels.append("Baseline" if scenario == 'no_event' else scenario)

    if not plot_data:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    if use_log_scale:
        ax.set_yscale('log')

    vps = ax.violinplot(plot_data, showmedians=False, showextrema=True)
    
    for idx, body in enumerate(vps['bodies']):
        color = 'black' if labels[idx] == "Baseline" else colors[idx % len(colors)]
        body.set_facecolor(color)
        body.set_alpha(0.6)
        body.set_edgecolor('black')

    # Plot thick red x marks for medians
    for idx, data_subset in enumerate(plot_data):
        median_val = np.median(data_subset)
        ax.plot(idx + 1, median_val, 'rx', markersize=10, markeredgewidth=2.5, zorder=5)

    ax.set_ylabel('Peak Incidence per 1000 persons', fontsize=12)
    ax.set_title(f'Distribution of Peak Incidence (Violin): {group_name}\n($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6, axis='y', which='both' if use_log_scale else 'major')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    log_suffix = "_log" if use_log_scale else ""
    filename = results_dir / f'peak_incidence_violin_{group_name}{log_suffix}_R{int(cfg.R_0*100)}_beta{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  -> Peak incidence violin plot saved to '{filename}'")

def plot_peak_time_comparison_boxplot(group_results, data, cfg, group_name):
    """
    Generates a boxplot comparison of peak timing (Day of maximum E+I)
    across different scenarios, excluding runs with zero secondary infections.
    """
    print(f"Generating peak time boxplot for group: {group_name}...")
    
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
    scenarios.reverse()
    
    plot_data = []
    labels = []
    
    # Define styles
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    for idx, scenario in enumerate(scenarios):
        realizations = group_results[scenario]['realizations']
        E = realizations['E']
        I = realizations['I']
        time = E[:, 0]
        
        # Filter for successful runs: cumulative infections at end > I_ss
        if 'R' in realizations:
            successful_runs_mask = realizations['R'][-1, 1:] > cfg.I_ss
        else:
            successful_runs_mask = np.max(E[:, 1:] + I[:, 1:], axis=0) > 0

        if not np.any(successful_runs_mask):
            continue

        # Calculate peak time for each successful run
        EI_runs = E[:, 1:][:, successful_runs_mask] + I[:, 1:][:, successful_runs_mask]
        peak_indices = np.argmax(EI_runs, axis=0)
        peak_times = time[peak_indices]
        
        plot_data.append(peak_times)
        labels.append("Baseline" if scenario == 'no_event' else scenario)

    if not plot_data:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    # Hide the default median line so we can draw a single marker instead
    bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, vert=False,
                    medianprops={'visible': False})
    
    for i, line in enumerate(bp['medians']):
        median_x = np.mean(line.get_xdata())
        median_y = np.mean(line.get_ydata())
        ax.plot(median_x, median_y, 'rx', markersize=10, markeredgewidth=2.5, zorder=5)

    for idx, box in enumerate(bp['boxes']):
        color = 'black' if labels[idx] == "Baseline" else colors[idx % len(colors)]
        box.set_facecolor(color)
        box.set_alpha(0.6)

    ax.set_xlabel('Day of Peak Incidence (E+I)', fontsize=12)
    ax.set_ylabel('Scenario', fontsize=12)
    ax.set_title(f'Distribution of Peak Timing: {group_name}\n($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6, axis='x')
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'peak_timing_boxplot_{group_name}_R{int(cfg.R_0*100)}_beta{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  -> Peak timing boxplot saved to '{filename}'")

def plot_peak_summary_rectangles(group_results: Dict[str, Any], data: Dict[str, Any], cfg: Any, group_name: str):
    """
    Visualizes the mean and standard deviation of peak characteristics for each scenario.
    X-axis: Peak Time (Days)
    Y-axis: Peak Incidence (per 1000)
    Rectangle dimensions: standard deviations.
    """
    print(f"Generating peak summary rectangles plot for group: {group_name}...")
    
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
    
    total_population = data['population_df']['population'].sum()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'] # Define a list of markers
    
    for idx, scenario in enumerate(scenarios):
        realizations = group_results[scenario]['realizations']
        E = realizations['E']
        I = realizations['I']
        time = E[:, 0]
        
        # Filter for successful runs: cumulative infections at end > I_ss
        if 'R' in realizations:
            successful_runs_mask = realizations['R'][-1, 1:] > cfg.I_ss
        else:
            successful_runs_mask = np.max(E[:, 1:] + I[:, 1:], axis=0) > 0

        if not np.any(successful_runs_mask):
            continue

        # Extract data for successful runs
        EI_runs = (E[:, 1:][:, successful_runs_mask] + I[:, 1:][:, successful_runs_mask])
        peaks_runs = np.max(EI_runs, axis=0) 
        peak_inc_runs = (peaks_runs / total_population) * 1000 # Normalize to per 1000
        
        peak_indices = np.argmax(EI_runs, axis=0)
        peak_time_runs = time[peak_indices]
        
        # Calculate statistics
        mu_x, sigma_x = np.mean(peak_time_runs), np.std(peak_time_runs)
        mu_y, sigma_y = np.mean(peak_inc_runs), np.std(peak_inc_runs)
        
        label = "Baseline" if scenario == 'no_event' else scenario
        color = 'black' if scenario == 'no_event' else colors[idx % len(colors)]
        
        # Assign a specific marker for baseline, and different markers for other scenarios
        if scenario == 'no_event':
            marker = 'x' # Distinct marker for baseline
        else:
            # Use a different marker for each MGE scenario
            marker = markers[(idx - 1) % len(markers)] # -1 to account for baseline being idx 0
        
        # Rectangle dimensions correspond to standard deviation, centered at the mean
        light_color = plt.matplotlib.colors.to_rgba(color, alpha=0.3)
        rect = Rectangle((mu_x - 0.5 * sigma_x, mu_y - 0.5 * sigma_y), 
                         sigma_x, sigma_y, 
                         facecolor=light_color, edgecolor='none', linewidth=0)
        ax.add_patch(rect)
        
        # Mark the mean center point with hollow marker
        ax.plot(mu_x, mu_y, marker=marker, color=color, markersize=10, markeredgewidth=1.5, 
                markerfacecolor='none',
                zorder=5, label=label, linestyle='None')

    ax.set_xlabel('Peak Time (Days)', fontsize=12)
    ax.set_ylabel('Peak Incidence (per 1000 persons)', fontsize=12)
    # ax.set_title(f'Summary of active case curves: {group_name}\n($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=14)
    ax.set_title(f'Summary of active case curves\n($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title="Scenario", frameon=True)
    
    ax.set_xlim(left=0, right=250)
    ax.set_ylim(bottom=0)
    ax.autoscale_view()
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'peak_summary_rectangles_{group_name}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  -> Peak summary rectangles plot saved to '{filename}'")

def plot_mean_arrival_time_by_district_and_scenario(group_results: Dict[str, Any], data: Dict[str, Any], cfg: Any, group_name: str):
    """
    Generates a scatter plot visualizing the mean epidemic arrival time for each district
    across MGE scenarios (including baseline).
    Uses I-based arrival times (first time infectious individuals > 0).
    X-axis indicates time in days and Y-axis indicates MGE scenarios, with baseline on top,
    followed by AMS scenarios, and then Leipzig scenarios.
    Each scenario has a horizontal line, and different symbols are used for different scenarios,
    with each symbol indicating the mean epidemic arrival time for a district.
    """
    if not getattr(cfg, 'plot_on_the_fly', True):
        return

    print(f"  -> Generating mean epidemic arrival time plot for group: {group_name}")

    # Order scenarios: baseline (no_event) first, then AMS, then Leipzig
    baseline_scen = ['no_event'] if 'no_event' in group_results else []
    ams_scen = sorted([s for s in group_results.keys() if s.startswith('AMS_')])
    leipzig_scen = sorted([s for s in group_results.keys() if s.startswith('Leipzig_')])
    other_scen = sorted([s for s in group_results.keys() if s not in baseline_scen + ams_scen + leipzig_scen])
    ordered_scenarios = baseline_scen + ams_scen + leipzig_scen + other_scen

    if not ordered_scenarios:
        print("     [Warning] No scenarios found for arrival time plot.")
        return

    n_patches = data['n_patches']
    n_iterations = cfg.n_iterations

    # Compute mean arrival time per district for each scenario
    scenario_mean_arrival = {}  # scenario -> array of mean arrival times per district
    max_mean_arrival = 0
    for sc in ordered_scenarios:
        res = group_results[sc]
        arrival_times = _get_arrival_times(res['all_runs_results'], n_patches, n_iterations)
        with np.errstate(invalid='ignore', divide='ignore'):
            if arrival_times.size > 0 and not np.all(np.isnan(arrival_times)):
                mean_times = np.nanmean(arrival_times, axis=0)  # average over runs for each district
            else:
                mean_times = np.full(n_patches, np.nan)
        scenario_mean_arrival[sc] = mean_times
        # Update global max (ignore NaN)
        valid_times = mean_times[~np.isnan(mean_times)]
        if len(valid_times) > 0:
            max_mean_arrival = max(max_mean_arrival, np.nanmax(valid_times))

    # If all are NaN, set a default max
    if np.isnan(max_mean_arrival) or max_mean_arrival == 0:
        max_mean_arrival = cfg.n_days if hasattr(cfg, 'n_days') else 100

    # Add some padding to the x-axis
    xmax = max_mean_arrival + 2
    if xmax < 5:
        xmax = 5

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))

    # Define markers for scenarios (cycle through if needed)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X']

    # Plot each scenario
    for idx, sc in enumerate(ordered_scenarios):
        y_pos = idx  # baseline at y=0, then AMS, then Leipzig (will invert later)
        mean_times = scenario_mean_arrival[sc]

        # Draw horizontal line for this scenario
        ax.axhline(y=y_pos, xmin=0, xmax=1, color='lightgray', linewidth=0.8, linestyle='--')

        # Plot each district's mean arrival time as a point
        for district_id in range(n_patches):
            t_mean = mean_times[district_id]
            if not np.isnan(t_mean):
                ax.scatter(t_mean, y_pos, marker=markers[idx % len(markers)],
                            s=40, alpha=0.8, edgecolors='black', linewidth=0.5)

    # Set y-axis to scenario names and invert
    ax.set_yticks(range(len(ordered_scenarios)))
    ax.set_yticklabels(ordered_scenarios)
    ax.invert_yaxis() # Invert y-axis so baseline is at the top

    # Set labels, title, and limits
    ax.set_xlabel('Mean epidemic arrival time (days)')
    # ax.set_ylabel('Scenarios')
    ax.set_title(f'Mean epidemic arrival time by MGE scenario')
    ax.set_xlim(0, xmax)
    ax.grid(True, axis='x', linestyle=':', alpha=0.6)

    # Save figure
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'mean_arrival_time_by_district_and_scenario_{group_name}_R{int(cfg.R_0*100)}_betaEvent{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"     Saved mean arrival time plot to '{filename}'")

def plot_mean_arrival_time_by_district_and_scenario_clustered(group_results: Dict[str, Any], data: Dict[str, Any], cfg: Any, group_name: str, cluster_threshold: Optional[float] = None):
    """
    Another version of the mean arrival time plot where districts within each scenario
    are clustered based on their arrival time using a natural "gap" detection method.
    Uses I-based arrival times (first time infectious individuals > 0).
    """
    if not HAS_CLUSTERING:
        print(f"  -> Skipping clustered arrival time plot for {group_name}: 'scipy' is required for clustering.")
        return

    if not getattr(cfg, 'plot_on_the_fly', True):
        return

    clustering_type_str = f"fixed threshold (dist <= {cluster_threshold}d)" if cluster_threshold is not None else "natural"
    print(f"  -> Generating {clustering_type_str} clustered mean epidemic arrival time plot for group: {group_name}")

    # Order scenarios: baseline (no_event) first, then AMS, then Leipzig
    baseline_scen = ['no_event'] if 'no_event' in group_results else []
    ams_scen = sorted([s for s in group_results.keys() if s.startswith('AMS_')])
    leipzig_scen = sorted([s for s in group_results.keys() if s.startswith('Leipzig_')])
    other_scen = sorted([s for s in group_results.keys() if s not in baseline_scen + ams_scen + leipzig_scen])
    ordered_scenarios = baseline_scen + ams_scen + leipzig_scen + other_scen

    if not ordered_scenarios:
        return

    n_patches = data['n_patches']
    n_iterations = cfg.n_iterations
    seed_patch_id = getattr(cfg, 'initial_infection_patch_id', 0)

    # Pre-calculate data and global max
    scenario_data = {}
    max_mean_arrival = 0
    for sc in ordered_scenarios:
        res = group_results[sc]
        arrival_times = _get_arrival_times(res['all_runs_results'], n_patches, n_iterations)
        with np.errstate(invalid='ignore', divide='ignore'):
            if arrival_times.size > 0 and not np.all(np.isnan(arrival_times)):
                mean_times = np.nanmean(arrival_times, axis=0)
            else:
                mean_times = np.full(n_patches, np.nan)
        scenario_data[sc] = mean_times
        valid_times = mean_times[~np.isnan(mean_times)]
        
        # --- New: Print arrival time range for each scenario ---
        if len(valid_times) > 0:
            min_arr = np.nanmin(valid_times)
            max_arr = np.nanmax(valid_times)
            print(f"    Scenario '{sc}': Arrival time range = [{min_arr:.2f}, {max_arr:.2f}] days (Total spread: {max_arr - min_arr:.2f} days)")
        else:
            print(f"    Scenario '{sc}': No valid arrival times found.")

        if len(valid_times) > 0:
            max_mean_arrival = max(max_mean_arrival, np.nanmax(valid_times))

    if np.isnan(max_mean_arrival) or max_mean_arrival == 0:
        max_mean_arrival = cfg.n_days if hasattr(cfg, 'n_days') else 100

    fig, ax = plt.subplots(figsize=(5, 3.5))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X']
    cmap = plt.get_cmap('tab20')

    for idx, sc in enumerate(ordered_scenarios):
        y_pos = idx
        mean_times = scenario_data[sc]
        ax.axhline(y=y_pos, color='lightgray', linewidth=0.8, linestyle='--', zorder=1)

        valid_mask = ~np.isnan(mean_times)
        cluster_mask = valid_mask.copy()

        # Exclude the seed district from clustering and visualization
        if 0 <= seed_patch_id < n_patches:
            cluster_mask[seed_patch_id] = False

        if np.any(cluster_mask):
            cluster_patch_ids = np.where(cluster_mask)[0]
            cluster_vals = mean_times[cluster_mask].reshape(-1, 1)
            
            if len(cluster_vals) > 1:
                # Perform 1D hierarchical clustering
                Z = hierarchy.linkage(cluster_vals, method='single')
                
                # If no threshold is provided, determine a 'natural' one dynamically for this scenario.
                # We analyze the merge heights in the dendrogram to find the largest 'jump'.
                # This corresponds to the most distinct separation between groups of districts.
                if cluster_threshold is None:
                    merge_heights = Z[:, 2]
                    if len(merge_heights) > 1:
                        # Calculate the increase in distance at each merge step
                        jumps = np.diff(merge_heights)
                        # Find the merge level just before the most significant jump
                        # This preserves the groups that are 'nearby' relative to the rest of the data.
                        max_jump_idx = np.argmax(jumps)
                        t_cutoff = merge_heights[max_jump_idx] + 1e-5
                    else:
                        t_cutoff = merge_heights[0] + 1e-5
                else:
                    t_cutoff = cluster_threshold
                
                clusters = hierarchy.fcluster(Z, t=t_cutoff, criterion='distance')
            else:
                clusters = np.ones(len(cluster_vals), dtype=int)
            
            # Add text indicating the number of clusters for this scenario
            n_clusters = len(np.unique(clusters))
            ax.text(2, y_pos + 0.3, f"{n_clusters} clusters", fontsize=10, color='black', ha='left', va='center')

            for i, p_id in enumerate(cluster_patch_ids):
                t_arrival = cluster_vals[i][0]
                color = cmap(idx % 20)
                ax.scatter(t_arrival, y_pos, marker=markers[clusters[i] % len(markers)],
                            c=[color], s=40, alpha=0.9, edgecolors='black', linewidth=0.5, zorder=3)

    ax.set_yticks(range(len(ordered_scenarios)))
    ax.set_yticklabels(ordered_scenarios)
    ax.invert_yaxis()
    ax.set_xlabel('Mean epidemic arrival time (days)')
    # title_suffix = f" (dist <= {cluster_threshold}d)" if cluster_threshold is not None else " (Natural Clustering)"
    # title_suffix = f" (dist <= {cluster_threshold}d)" if cluster_threshold is not None else ""
    # ax.set_title(f'Mean Arrival Times {title_suffix}')
    ax.set_title(f'Mean epidemic arrival time by MGE scenarios')
    ax.set_xlim(0, max_mean_arrival + 2)
    ax.set_ylim(len(ordered_scenarios) +0.2, -0.5) 	# Add extra space below the lowest scenario for cluster labels
    ax.grid(True, axis='x', linestyle=':', alpha=0.6)
    
    # Add legend to explain color/marker scheme
    legend_elements = []
    for idx, sc in enumerate(ordered_scenarios):
        color = cmap(idx % 20)
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                      markersize=8, label=sc, markeredgecolor='black', markeredgewidth=0.5))
    # ax.legend(handles=legend_elements, loc='upper right', fontsize=8, title='Scenarios (colors)')

    results_dir = getattr(cfg, 'results_dir', Path("results"))

    filename_parts = [f'mean_arrival_time_clustered_{group_name}']
    if cluster_threshold is not None:
        filename_parts.append(f'_{int(cluster_threshold)}d')
    filename_parts.append('.png')
    filename = results_dir / "".join(filename_parts)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"     Saved clustered mean arrival time plot to '{filename}'")

def plot_mean_arrival_interval_distribution(group_results: Dict[str, Any], data: Dict[str, Any], cfg: Any, group_name: str, show_outliers: bool = True):
    """
    Visualizes the distribution of gaps between successive mean epidemic arrival times
    for each scenario using horizontal boxplots.
    Uses I-based arrival times (first time infectious individuals > 0).
    """
    if not getattr(cfg, 'plot_on_the_fly', True):
        return

    outlier_str = "" if show_outliers else " (no outliers)"
    print(f"  -> Generating mean arrival interval distribution plot{outlier_str} for group: {group_name}")

    # Order scenarios: baseline first, then AMS, then Leipzig, then others
    baseline_scen = ['no_event'] if 'no_event' in group_results else []
    ams_scen = sorted([s for s in group_results.keys() if s.startswith('AMS_')])
    leipzig_scen = sorted([s for s in group_results.keys() if s.startswith('Leipzig_')])
    other_scen = sorted([s for s in group_results.keys() if s not in baseline_scen + ams_scen + leipzig_scen])
    ordered_scenarios = baseline_scen + ams_scen + leipzig_scen + other_scen

    if not ordered_scenarios:
        return

    n_patches = data['n_patches']
    n_iterations = cfg.n_iterations

    plot_data = []
    labels = []

    for sc in ordered_scenarios:
        res = group_results[sc]
        arr_times = _get_arrival_times(res['all_runs_results'], n_patches, n_iterations)

        # Calculate mean arrival time for each district, averaged over runs
        with np.errstate(invalid='ignore', divide='ignore'):
            if arr_times.size > 0 and not np.all(np.isnan(arr_times)):
                mean_times = np.nanmean(arr_times, axis=0)
            else:
                mean_times = np.full(n_patches, np.nan)

        # Filter out NaN values (districts never infected)
        valid_mean_times = mean_times[~np.isnan(mean_times)]
        
        if len(valid_mean_times) > 1: # Need at least two valid arrival times to calculate a gap
            # Sort the mean arrival times and calculate the differences (gaps)
            sorted_mean_times = np.sort(valid_mean_times)
            gaps = np.diff(sorted_mean_times)
            
            plot_data.append(gaps.tolist())
            labels.append("Baseline" if sc == 'no_event' else sc)

    if not plot_data:
        print(f"  -> No sufficient data to plot mean arrival time gaps for group: {group_name}")
        return

    # Reverse for consistent top-down ordering in horizontal plots (Baseline at top)
    plot_data.reverse()
    labels.reverse()

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(plot_data, labels=labels, vert=False, patch_artist=True, 
                    medianprops={'visible': False}, showfliers=show_outliers)

    for i, line in enumerate(bp['medians']):
        median_x = np.mean(line.get_xdata())
        median_y = np.mean(line.get_ydata())
        ax.plot(median_x, median_y, 'rx', markersize=9, markeredgewidth=2, zorder=5)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for idx, box in enumerate(bp['boxes']):
        color = 'black' if labels[idx] == "Baseline" else colors[idx % len(colors)]
        box.set_facecolor(color)
        box.set_alpha(0.6)

    ax.set_xlabel('Inter-arrival time gap (days)')
    ax.set_title(f'Infection inter-arrival time intervals (Mean Arrival Times)')
    ax.grid(True, axis='x', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    outlier_suffix = "" if show_outliers else "_no_outliers"
    filename = results_dir / f'arrival_interval_mean_boxplot_{group_name}{outlier_suffix}_R{int(cfg.R_0*100)}_beta{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"     Saved mean arrival interval distribution boxplot to '{filename}'")

def plot_infection_arrival_interval_distribution(group_results: Dict[str, Any], data: Dict[str, Any], cfg: Any, group_name: str, show_outliers: bool = False):
    """
    Visualizes the distribution of infection inter-arrival times (time gaps between
    successive district infections) for each scenario using horizontal boxplots.
    Uses I-based arrival times (first time infectious individuals > 0).
    """
    if not getattr(cfg, 'plot_on_the_fly', True):
        return

    outlier_str = "" if show_outliers else " (no outliers)"
    print(f"  -> Generating infection arrival interval distribution plot{outlier_str} for group: {group_name}")

    # Order scenarios: baseline first, then AMS, then Leipzig, then others
    baseline_scen = ['no_event'] if 'no_event' in group_results else []
    ams_scen = sorted([s for s in group_results.keys() if s.startswith('AMS_')])
    leipzig_scen = sorted([s for s in group_results.keys() if s.startswith('Leipzig_')])
    other_scen = sorted([s for s in group_results.keys() if s not in baseline_scen + ams_scen + leipzig_scen])
    ordered_scenarios = baseline_scen + ams_scen + leipzig_scen + other_scen

    if not ordered_scenarios:
        return

    n_patches = data['n_patches']
    n_iterations = cfg.n_iterations
    seed_patch_id = getattr(cfg, 'initial_infection_patch_id', 0)

    plot_data = []
    labels = []

    for sc in ordered_scenarios:
        res = group_results[sc]
        arr_times = _get_arrival_times(res['all_runs_results'], n_patches, n_iterations)

        intervals_all_runs = []
        for r in range(arr_times.shape[0]):
            run_arr = arr_times[r, :].copy()
            # Exclude the seed district from interval calculations
            if 0 <= seed_patch_id < n_patches:
                run_arr[seed_patch_id] = np.nan
            valid_arr = run_arr[~np.isnan(run_arr)]
            if len(valid_arr) > 1:
                # Calculate time gaps between successive arrivals in this realization
                intervals = np.diff(np.sort(valid_arr))
                intervals_all_runs.extend(intervals.tolist())
        
        if intervals_all_runs:
            plot_data.append(intervals_all_runs)
            labels.append("Baseline" if sc == 'no_event' else sc)

    if not plot_data:
        return

    # Reverse for consistent top-down ordering in horizontal plots
    plot_data.reverse()
    labels.reverse()

    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(plot_data, labels=labels, vert=False, patch_artist=True, 
                    medianprops={'visible': False}, showfliers=show_outliers)

    for i, line in enumerate(bp['medians']):
        median_x = np.mean(line.get_xdata())
        median_y = np.mean(line.get_ydata())
        ax.plot(median_x, median_y, 'rx', markersize=9, markeredgewidth=2, zorder=5)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for idx, box in enumerate(bp['boxes']):
        color = 'black' if labels[idx] == "Baseline" else colors[idx % len(colors)]
        box.set_facecolor(color)
        box.set_alpha(0.6)

    ax.set_xlabel('Inter-arrival time gap (days)')
    ax.set_title(f'Infection inter-arrival time intervals')
    ax.grid(True, axis='x', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    outlier_suffix = "" if show_outliers else "_no_outliers"
    filename = results_dir / f'infection_arrival_interval_boxplot_{group_name}.png'
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"     Saved arrival interval distribution boxplot to '{filename}'")

def plot_exposed_infected_distribution_boxplot(group_results: Dict[str, Any], data: Dict[str, Any], cfg: Any, group_name: str):
    """
    Generates two box plots side-by-side: one for the distribution of peak 'E' (Exposed)
    and one for the distribution of peak 'I' (Infected) individuals, normalized per 1000 persons,
    across different scenarios.
    """
    if not getattr(cfg, 'plot_on_the_fly', True):
        return

    print(f"Generating peak Exposed and Infected distribution boxplots for group: {group_name}...")
    
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
    
    total_population = data['population_df']['population'].sum()
    
    plot_data_E = []
    plot_data_I = []
    labels = []
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    for idx, scenario in enumerate(scenarios):
        realizations = group_results[scenario]['realizations']
        E = realizations['E']
        I = realizations['I']
        
        # Filter for successful runs: cumulative infections at the end > I_ss
        if 'R' in realizations:
            successful_runs_mask = realizations['R'][-1, 1:] > cfg.I_ss
        else:
            # Fallback: check for any active cases during the run
            successful_runs_mask = np.max(E[:, 1:] + I[:, 1:], axis=0) > 0

        if not np.any(successful_runs_mask):
            continue

        # Extract E and I for successful runs
        E_runs = E[:, 1:][:, successful_runs_mask]
        I_runs = I[:, 1:][:, successful_runs_mask]

        # Calculate peak E and peak I for each successful run
        peak_E_values = np.max(E_runs, axis=0)
        peak_I_values = np.max(I_runs, axis=0)

        # Normalize to per 1000 persons
        peak_E_per_1000 = (peak_E_values / total_population) * 1000
        peak_I_per_1000 = (peak_I_values / total_population) * 1000

        plot_data_E.append(peak_E_per_1000)
        plot_data_I.append(peak_I_per_1000)
        labels.append("Baseline" if scenario == 'no_event' else scenario)

    if not plot_data_E or not plot_data_I:
        print(f"  -> No successful runs found in any scenario for E/I peak distribution boxplot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True) # Share Y-axis for easier comparison

    # Plot for Peak Exposed
    bp_E = axes[0].boxplot(plot_data_E, labels=labels, patch_artist=True, 
                           medianprops={'color': 'black', 'linewidth': 2})
    for idx, box in enumerate(bp_E['boxes']):
        color = 'black' if labels[idx] == "Baseline" else colors[idx % len(colors)]
        box.set_facecolor(color)
        box.set_alpha(0.6)
    axes[0].set_ylabel('Peak Exposed Individuals per 1000 persons', fontsize=12)
    axes[0].set_title('Distribution of Peak Exposed (E)', fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')

    # Plot for Peak Infected
    bp_I = axes[1].boxplot(plot_data_I, labels=labels, patch_artist=True, 
                           medianprops={'color': 'black', 'linewidth': 2})
    for idx, box in enumerate(bp_I['boxes']):
        color = 'black' if labels[idx] == "Baseline" else colors[idx % len(colors)]
        box.set_facecolor(color)
        box.set_alpha(0.6)
    axes[1].set_title('Distribution of Peak Infected (I)', fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')

    fig.suptitle(f'Peak Exposed and Infected Individuals: {group_name}\n($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'peak_E_I_boxplot_{group_name}_R{int(cfg.R_0*100)}_beta{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  -> Peak E and I boxplot saved to '{filename}'")

def plot_initial_exposed_infected_distribution_boxplot(group_results: Dict[str, Any], data: Dict[str, Any], cfg: Any, group_name: str, target_t: float = 1.0, show_outliers: bool = False):
    """
    Generates a box plot for the sum of Exposed (E) and Infected (I) counts at a specific early time point
    (e.g., beginning of Day 1) to visualize the initial seeding impact of different scenarios.
    """
    if not getattr(cfg, 'plot_on_the_fly', True):
        return

    outlier_str = "" if show_outliers else " (no outliers)"
    print(f"Generating initial seeding distribution boxplot (sum E+I, t={target_t}) for group: {group_name}{outlier_str}...")
    
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
    
    plot_data = []
    labels = []
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    actual_t = target_t
    for idx, scenario in enumerate(scenarios):
        realizations = group_results[scenario]['realizations']
        E = realizations['E']
        I = realizations['I']
        time = E[:, 0]
        
        t_idx = np.abs(time - target_t).argmin()
        actual_t = time[t_idx]
        
        # Sum of E and I for all realizations at this time point
        sum_EI = E[t_idx, 1:] + I[t_idx, 1:]
        
        plot_data.append(sum_EI)
        labels.append("Baseline" if scenario == 'no_event' else scenario)

    if not plot_data:
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, 
                    medianprops={'visible': False},
                    showfliers=show_outliers)
    
    for i, line in enumerate(bp['medians']):
        median_x = np.mean(line.get_xdata())
        median_y = np.mean(line.get_ydata())
        ax.plot(median_x, median_y, 'rx', markersize=10, markeredgewidth=2.5, zorder=5)
    
    for idx, box in enumerate(bp['boxes']):
        color = 'black' if labels[idx] == "Baseline" else colors[idx % len(colors)]
        box.set_facecolor(color)
        box.set_alpha(0.6)
        
    ax.set_ylabel('Epidemic seed size', fontsize=12)
    # ax.set_title('MGE impact on epidemic seed', fontsize=14)
    ax.set_title(f'MGE impact on epidemic seeding\n($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # fig.suptitle(f'Initial Epidemic Seeding Impact\n($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    outlier_suffix = "" if show_outliers else "_no_outliers"
    filename = results_dir / f'MGE_seeding_boxplot_{group_name}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  -> Initial seeding sum E+I distribution boxplot saved to '{filename}'")

def plot_max_infectious_ratio_vs_beta(data_list, cfg):
    """
    Plots Max Infectious Ratio vs Beta_Event for each scenario, with curves for different R0.
    Also saves the data to a CSV file.
    """
    if not data_list:
        print("No data for max infectious ratio plot.")
        return

    print("Generating max infectious ratio vs beta plots...")
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    df = pd.DataFrame(data_list)
    
    # Plotting
    scenarios = df['scenario_name'].unique()
    
    for scenario in scenarios:
        scenario_df = df[df['scenario_name'] == scenario]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Group by R0
        r0_values = sorted(scenario_df['R0'].unique())
        
        for r0 in r0_values:
            subset = scenario_df[scenario_df['R0'] == r0].sort_values('beta_event')
            ax.plot(subset['beta_event'], subset['max_infectious_ratio_mean'], 
                    marker='o', label=f'$R_0$={r0}')
            
        ax.set_xlabel(r'Event Transmission Rate ($\beta_{event}$)', fontsize=12)
        ax.set_ylabel('Max Infectious Ratio ((E+I)/N)', fontsize=12)
        ax.set_title(f'Peak Epidemic Size vs Event Risk\nScenario: {scenario}', fontsize=14)
        ax.legend(title='$R_0$')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        filename = results_dir / f'max_infectious_ratio_vs_beta_{scenario}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved max infectious ratio plot for {scenario} to '{filename}'")

def plot_infectious_ratio_difference_vs_beta(data_list, cfg):
    """
    Plots the difference in Max Infectious Ratio (Event - Baseline) vs Beta_Event using boxplots.
    Generates one plot per (R0, I_ss) combination.
    """
    return # Disable plotting as requested
    if not data_list:
        return

    print("Generating infectious ratio difference plots...")
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    df = pd.DataFrame(data_list)
    
    # Unique combinations of R0 and I_ss
    params = df[['R0', 'I_ss']].drop_duplicates()
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for _, row in params.iterrows():
        r0 = row['R0']
        iss = row['I_ss']
        
        # Get baseline value for this R0/Iss
        baseline_row = df[(df['scenario_name'] == 'no_event') & 
                          (df['R0'] == r0) & 
                          (df['I_ss'] == iss)]
        
        if baseline_row.empty:
            continue
            
        baseline_vals = np.array(baseline_row.iloc[0]['values'])
        
        # Get event scenarios
        subset = df[(df['scenario_name'] != 'no_event') & 
                    (df['R0'] == r0) & 
                    (df['I_ss'] == iss)]
        
        if subset.empty:
            continue
            
        scenarios = subset['scenario_name'].unique()
        scenarios.sort()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Determine boxplot layout
        betas = sorted(subset['beta_event'].unique())
        if len(betas) > 1:
            min_gap = min(np.diff(betas))
        else:
            min_gap = 0.1
        
        n_scenarios = len(scenarios)
        total_width = min_gap * 0.8
        box_width = total_width / n_scenarios

        for idx, scenario in enumerate(scenarios):
            scen_data = subset[subset['scenario_name'] == scenario].sort_values('beta_event')
            
            # Calculate difference for each beta
            plot_data = []
            positions = []
            
            for _, row_data in scen_data.iterrows():
                event_vals = np.array(row_data['values'])
                # Ensure lengths match (should be n_iterations)
                if len(event_vals) == len(baseline_vals):
                    diffs = event_vals - baseline_vals
                    plot_data.append(diffs)
                    positions.append(row_data['beta_event'] + (idx - n_scenarios/2 + 0.5) * box_width)
            
            if plot_data:
                color = colors[idx % len(colors)]
                bp = ax.boxplot(plot_data, positions=positions, widths=box_width*0.9, 
                                patch_artist=True, manage_ticks=False,
                                boxprops=dict(facecolor=color, alpha=0.6),
                                medianprops=dict(color='black'),
                                showfliers=False)
                
                # Add dummy line for legend
                ax.plot([], [], color=color, label=scenario, linewidth=4, alpha=0.6)
            
        ax.set_xlabel(r'Event Transmission Rate ($\beta_{event}$)', fontsize=12)
        ax.set_ylabel('Difference in Max Infectious Ratio\n(Event - Baseline)', fontsize=12)
        ax.set_title(f'Impact of Mass Gathering on Peak Epidemic Size\n($R_0$={r0}, $I_{{ss}}$={iss})', fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        
        filename = results_dir / f'infectious_ratio_difference_vs_beta_R{int(r0*100)}_Iss{iss}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved difference plot to '{filename}'")

def plot_final_size_vs_beta(data_list, cfg):
    """
    Plots Final Fraction of Recovered Individuals vs Beta_Event using boxplots.
    Generates one plot per (R0, I_ss) combination.
    Different scenarios are depicted by different colors.
    """
    if not data_list:
        print("No data for final size plot.")
        return

    print("Generating final size vs beta plots...")
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    df = pd.DataFrame(data_list)
    
    return # Disable plotting as requested
    
    # Unique combinations of R0 and I_ss
    params = df[['R0', 'I_ss']].drop_duplicates()
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    for _, row in params.iterrows():
        r0 = row['R0']
        iss = row['I_ss']
        
        # Get baseline value for this R0/Iss
        baseline_row = df[(df['scenario_name'] == 'no_event') & (df['R0'] == r0) & (df['I_ss'] == iss)]
        baseline_val = baseline_row.iloc[0]['final_size_mean'] if not baseline_row.empty else None
        
        # Get event scenarios
        subset = df[(df['scenario_name'] != 'no_event') & (df['R0'] == r0) & (df['I_ss'] == iss)]
        if subset.empty: continue
            
        scenarios = sorted(subset['scenario_name'].unique())
        
        fig, ax = plt.subplots(figsize=(8, 6))
        if baseline_val is not None:
            ax.axhline(baseline_val, color='black', linestyle='--', label='Baseline (No Event)')

        # Determine boxplot layout
        betas = sorted(subset['beta_event'].unique())
        if len(betas) > 1:
            min_gap = min(np.diff(betas))
        else:
            min_gap = 0.1
            
        n_scenarios = len(scenarios)
        total_width = min_gap * 0.8
        box_width = total_width / n_scenarios

        for idx, scenario in enumerate(scenarios):
            scen_data = subset[subset['scenario_name'] == scenario].sort_values('beta_event')
            
            plot_data = []
            positions = []
            
            for _, row_data in scen_data.iterrows():
                plot_data.append(row_data['values'])
                positions.append(row_data['beta_event'] + (idx - n_scenarios/2 + 0.5) * box_width)
            
            if plot_data:
                color = colors[idx % len(colors)]
                bp = ax.boxplot(plot_data, positions=positions, widths=box_width*0.9, 
                                patch_artist=True, manage_ticks=False,
                                boxprops=dict(facecolor=color, alpha=0.6),
                                medianprops=dict(color='black'),
                                showfliers=False)
                
                ax.plot([], [], color=color, label=scenario, linewidth=4, alpha=0.6)
            
        ax.set_xlabel(r'Event Transmission Rate ($\beta_{event}$)', fontsize=12)
        ax.set_ylabel('Final Fraction of Recovered Individuals (R/N)', fontsize=12)
        ax.set_title(f'Final Epidemic Size vs Event Risk\n($R_0$={r0}, $I_{{ss}}$={iss})', fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(bottom=0)
        
        filename = results_dir / f'final_size_vs_beta_R{int(r0*100)}_Iss{iss}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved final size plot to '{filename}'")

def plot_final_size_heatmap(data_list, cfg):
    """
    Generates a heatmap of the final epidemic size (fraction recovered).
    X-axis: beta_event
    Y-axis: R0
    One heatmap per scenario.
    """
    if not data_list:
        return

    print("Generating final size heatmaps...")
    try:
        import seaborn as sns
    except ImportError:
        print("Seaborn not found, skipping heatmaps.")
        return

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    df = pd.DataFrame(data_list)
    
    scenarios = df['scenario_name'].unique()
    iss_values = df['I_ss'].unique()

    for scenario in scenarios:
        for iss in iss_values:
            subset = df[(df['scenario_name'] == scenario) & (df['I_ss'] == iss)].copy()
            if subset.empty: continue
            
            subset['R0_rounded'] = subset['R0'].round(4)
            subset['beta_rounded'] = subset['beta_event'].round(4)
            
            pivot_table = subset.pivot_table(index='R0_rounded', columns='beta_rounded', values='final_size_mean')
            pivot_table = pivot_table.sort_index(ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="Blues", 
                        cbar_kws={'label': 'Final Size (Fraction Recovered)'}, ax=ax, vmin=0, vmax=1)
            ax.set_title(f'Final Epidemic Size Heatmap\nScenario: {scenario}, $I_{{ss}}$={iss}')
            ax.set_xlabel(r'Event Transmission Rate ($\beta_{event}$)')
            ax.set_ylabel(r'Basic Reproduction Number ($R_0$)')
            
            filename = results_dir / f'final_size_heatmap_{scenario}_Iss{iss}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved final size heatmap to '{filename}'")

def plot_peak_ratio_vs_beta(data_list, cfg):
    """
    Plots the ratio of Peak Infectious Ratio (Event / Baseline) vs Beta_Event.
    Generates one plot per (R0, I_ss) combination.
    """
    if not data_list:
        return

    print("Generating peak infectious ratio comparison curves (Event / Baseline)...")
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    df = pd.DataFrame(data_list)
    
    # Unique combinations of R0 and I_ss
    params = df[['R0', 'I_ss']].drop_duplicates()
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    for _, row in params.iterrows():
        r0 = row['R0']
        iss = row['I_ss']
        
        # Get baseline value for this R0/Iss
        baseline_row = df[(df['scenario_name'] == 'no_event') & 
                          (df['R0'] == r0) & 
                          (df['I_ss'] == iss)]
        
        if baseline_row.empty:
            continue
            
        baseline_vals = np.array(baseline_row.iloc[0]['values'])
        
        # Get event scenarios
        subset = df[(df['scenario_name'] != 'no_event') & 
                    (df['R0'] == r0) & 
                    (df['I_ss'] == iss)]
        
        if subset.empty:
            continue
            
        scenarios = sorted(subset['scenario_name'].unique())
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for idx, scenario in enumerate(scenarios):
            scen_data = subset[subset['scenario_name'] == scenario].sort_values('beta_event')
            
            betas = []
            mean_ratios = []
            
            for _, row_data in scen_data.iterrows():
                event_vals = np.array(row_data['values'])
                if len(event_vals) == len(baseline_vals):
                    # Calculate ratio per run
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratios = event_vals / baseline_vals
                    
                    # Filter out infs/nans if any (e.g. if baseline was 0)
                    valid_ratios = ratios[np.isfinite(ratios)]
                    
                    if len(valid_ratios) > 0:
                        betas.append(row_data['beta_event'])
                        mean_ratios.append(np.mean(valid_ratios))
            
            if betas:
                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)]
                ax.plot(betas, mean_ratios, marker=marker, color=color, label=scenario)
            
        ax.set_xlabel(r'Event Transmission Rate ($\beta_{event}$)', fontsize=12)
        ax.set_ylabel('Peak Ratio (Event / Baseline)', fontsize=12)
        ax.set_title(f'Relative Increase in Peak Epidemic Size\n($R_0$={r0}, $I_{{ss}}$={iss})', fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        filename = results_dir / f'peak_ratio_vs_beta_R{int(r0*100)}_Iss{iss}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved peak ratio plot to '{filename}'")

def plot_early_extinction_heatmap(data_list, cfg):
    """
    Generates a heatmap of Early Extinction Probability based on final epidemic size.
    Early extinction is defined as the final epidemic size being less than a threshold (e.g. 5% of population).
    """
    if not data_list:
        return

    print("Generating early extinction probability heatmaps...")
    try:
        import seaborn as sns
    except ImportError:
        print("Seaborn not found, skipping heatmaps.")
        return

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    df = pd.DataFrame(data_list)
    
    # Define threshold for "Early Extinction" (Minor Outbreak)
    # A common heuristic is < 5% of the population infected/recovered.
    EXTINCTION_THRESHOLD = 0.05 
    
    def calc_prob(values):
        if not isinstance(values, list): return 0.0
        arr = np.array(values)
        # Count runs where final size < threshold
        count = np.sum(arr < EXTINCTION_THRESHOLD)
        return count / len(arr)

    if 'values' not in df.columns:
        print("Error: 'values' column missing in data, cannot compute extinction probability.")
        return

    df['extinction_prob'] = df['values'].apply(calc_prob)
    
    scenarios = df['scenario_name'].unique()
    iss_values = df['I_ss'].unique()

    for scenario in scenarios:
        for iss in iss_values:
            subset = df[(df['scenario_name'] == scenario) & (df['I_ss'] == iss)].copy()
            if subset.empty: continue
            
            subset['R0_rounded'] = subset['R0'].round(4)
            subset['beta_rounded'] = subset['beta_event'].round(4)
            
            pivot_table = subset.pivot_table(index='R0_rounded', columns='beta_rounded', values='extinction_prob')
            pivot_table = pivot_table.sort_index(ascending=False) # High R0 at top
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Reds_r: 0.0 (Low Prob / Major Outbreak) = Dark Red
            #         1.0 (High Prob / Extinction) = Light Red / White
            sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="Reds_r", 
                        cbar_kws={'label': 'Early Extinction Probability'}, ax=ax, vmin=0, vmax=1)
            
            ax.set_title(f'Early Extinction Probability (Final Size < {EXTINCTION_THRESHOLD:.0%})\nScenario: {scenario}, $I_{{ss}}$={iss}')
            ax.set_xlabel(r'Event Transmission Rate ($\beta_{event}$)')
            ax.set_ylabel(r'Basic Reproduction Number ($R_0$)')
            
            filename = results_dir / f'early_extinction_heatmap_{scenario}_Iss{iss}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved early extinction heatmap to '{filename}'")

def plot_extinction_vs_size(data_list, cfg):
    """
    Generates scatter plots of Extinction Time vs Cumulative Recovered (Final Size).
    Generates one plot per (R0, Beta_Event, I_ss) combination.
    """
    if not data_list:
        return

    print("Generating extinction time vs final size scatter plots...")
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    df = pd.DataFrame(data_list)
    
    # Get unique parameters
    r0_values = sorted(df['R0'].unique())
    iss_values = sorted(df['I_ss'].unique())
    
    # Identify event betas (exclude 0.0 if it's only for baseline)
    # We assume baseline has beta_event=0.0
    event_betas = sorted([b for b in df['beta_event'].unique() if b > 0])
    if not event_betas: 
        # If only baseline exists or something is odd, just plot what we have
        event_betas = [0.0]

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for r0 in r0_values:
        for iss in iss_values:
            # Get baseline data for this R0/Iss (beta_event=0)
            base_df = df[(df['scenario_name'] == 'no_event') & (df['R0'] == r0) & (df['I_ss'] == iss)]
            
            for beta in event_betas:
                # Get event data for this R0/Iss/Beta
                event_df = df[(df['scenario_name'] != 'no_event') & 
                              (df['R0'] == r0) & 
                              (df['I_ss'] == iss) & 
                              (np.isclose(df['beta_event'], beta))]
                
                if event_df.empty and base_df.empty:
                    continue
                
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Plot Baseline
                if not base_df.empty:
                    row = base_df.iloc[0]
                    ax.scatter(row['extinction_times'], row['final_sizes'], label='Baseline', marker='x', color='black', alpha=0.7, zorder=10)
                
                # Plot Events
                for idx, (_, row) in enumerate(event_df.iterrows()):
                    color = colors[idx % len(colors)]
                    marker = markers[idx % len(markers)]
                    ax.scatter(row['extinction_times'], row['final_sizes'], label=row['scenario_name'], marker=marker, color=color, alpha=0.6)
                
                ax.set_xlabel('Extinction Time (Days)', fontsize=12)
                ax.set_ylabel('Cumulative Recovered Individuals', fontsize=12)
                ax.set_title(f'Epidemic Outcomes: Extinction Time vs Final Size\n($R_0$={r0}, $I_{{ss}}$={iss}, $\\beta_{{event}}$={beta})', fontsize=14)
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6)
                
                # Calculate data ranges for padding to ensure points at 0 are visible
                all_x = []
                all_y = []
                if not base_df.empty:
                    row = base_df.iloc[0]
                    all_x.extend(row['extinction_times'])
                    all_y.extend(row['final_sizes'])
                for _, row in event_df.iterrows():
                    all_x.extend(row['extinction_times'])
                    all_y.extend(row['final_sizes'])
                
                max_x = max(all_x) if all_x else 0
                max_y = max(all_y) if all_y else 0
                
                ax.set_xlim(left=-(max_x * 0.02 if max_x > 0 else 1.0))
                ax.set_ylim(bottom=-(max_y * 0.02 if max_y > 0 else 1.0))
                
                filename = results_dir / f'extinction_vs_size_scatter_R{int(r0*100)}_betaEvent{int(beta*100)}_Iss{iss}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved extinction scatter plot to '{filename}'")

def plot_final_size_scenario_comparison_heatmap(data_list, cfg):
    """
    Generates heatmaps comparing final epidemic size across scenarios for a fixed beta_event.
    X-axis: Scenario (Baseline first)
    Y-axis: R0
    Generates one heatmap per (I_ss, beta_event) combination.
    """
    if not data_list:
        return

    print("Generating final size scenario comparison heatmaps...")
    try:
        import seaborn as sns
    except ImportError:
        print("Seaborn not found, skipping heatmaps.")
        return

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    df = pd.DataFrame(data_list)
    
    # Ensure numeric types
    df['R0'] = pd.to_numeric(df['R0'], errors='coerce')
    df['beta_event'] = pd.to_numeric(df['beta_event'], errors='coerce')
    df['final_size_mean'] = pd.to_numeric(df['final_size_mean'], errors='coerce')

    # Separate baseline and event data
    baseline_df = df[df['scenario_name'] == 'no_event'].copy()
    event_df = df[df['scenario_name'] != 'no_event'].copy()

    if event_df.empty:
        print("No event data found for comparison heatmaps.")
        return

    unique_iss = sorted(df['I_ss'].unique())
    # Get unique betas from event scenarios only
    unique_betas = sorted(event_df['beta_event'].unique())

    for iss in unique_iss:
        # Baseline for this I_ss
        base_subset = baseline_df[baseline_df['I_ss'] == iss]
        
        for beta in unique_betas:
            # Event data for this I_ss and beta
            event_subset = event_df[
                (event_df['I_ss'] == iss) & 
                (np.isclose(event_df['beta_event'], beta))
            ]
            
            if event_subset.empty:
                continue

            # Combine baseline and event data
            plot_df = pd.concat([base_subset, event_subset], ignore_index=True)
            
            # Pivot: Index=R0, Columns=Scenario, Values=Final Size
            plot_df['R0_rounded'] = plot_df['R0'].round(4)
            
            try:
                pivot_table = plot_df.pivot_table(index='R0_rounded', columns='scenario_name', values='final_size_mean')
            except Exception as e:
                print(f"Error creating pivot for I_ss={iss}, beta={beta}: {e}")
                continue
            
            # Sort index (R0) descending
            pivot_table = pivot_table.sort_index(ascending=False)
            
            # Reorder columns: Baseline first, then alphabetical
            cols = pivot_table.columns.tolist()
            if 'no_event' in cols:
                cols.remove('no_event')
                cols.sort()
                cols.insert(0, 'no_event')
                pivot_table = pivot_table[cols]
            
            # Rename 'no_event' to 'Baseline' for display
            display_pivot = pivot_table.rename(columns={'no_event': 'Baseline'})
            
            fig, ax = plt.subplots(figsize=(max(8, len(cols)*1.5), 8))
            
            sns.heatmap(display_pivot, annot=True, fmt=".2f", cmap="Blues", 
                        cbar_kws={'label': 'Final Size (Fraction Recovered)'}, ax=ax, vmin=0, vmax=1)
            
            ax.set_title(f'Final Epidemic Size Comparison\n($I_{{ss}}$={iss}, $\\beta_{{event}}$={beta})')
            ax.set_xlabel('Scenario')
            ax.set_ylabel(r'Basic Reproduction Number ($R_0$)')
            plt.xticks(rotation=45, ha='right')
            
            filename = results_dir / f'final_size_heatmap_comparison_Iss{iss}_beta{int(beta*100)}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved scenario comparison heatmap to '{filename}'")

def _extract_geojson_from_html(html_path):
    """Helper to extract GeoJSON from Folium HTML file."""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Regex to find the start of the GeoJSON object
        # Matches {"type": "FeatureCollection" with flexible spacing and quotes
        match = re.search(r'\{\s*[\'"]type[\'"]\s*:\s*[\'"]FeatureCollection[\'"]', content)
        
        if match:
            start_idx = match.start()
            brace_count = 0
            for i in range(start_idx, len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = content[start_idx : i+1]
                        return json.loads(json_str)
        
        print(f"Warning: Could not find 'FeatureCollection' JSON in {html_path}")
        return None
    except Exception as e:
        print(f"Warning: Failed to extract GeoJSON from {html_path}: {e}")
        return None

def save_district_active_cases_per_run(all_runs_results, data, cfg):
    """
    Saves the number of active cases (E+I) per district, per day, for each run to a CSV file.
    """
    print("Saving daily active cases per district for each run...")
    n_patches = data['n_patches']
    n_days = cfg.n_days
    daily_times = np.arange(n_days + 1)
    
    all_records = []
    
    for run_id, run_res in enumerate(all_runs_results):
        t = np.array(run_res['t'])
        E_j = np.array(run_res['E_j'])
        I_j = np.array(run_res['I_j'])
        Active_j = E_j + I_j # Shape: (time_steps, n_patches)
        
        # Interpolate for each patch to get daily values
        for p in range(n_patches):
            interpolated_cases = np.interp(daily_times, t, Active_j[:, p])
            record = {'run_id': run_id, 'district_id': p}
            for day, cases in zip(daily_times, interpolated_cases):
                record[f'day_{day}'] = cases
            all_records.append(record)
            
    if not all_records:
        print("No data to save for daily active cases.")
        return
    
    # Create DataFrame from list of records
    final_df = pd.DataFrame(all_records)
    
    # Save CSV
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    
    if cfg.event_model_type.startswith('no_event'):
        scenario_name = cfg.event_model_type
    else:
        scenario_name = getattr(cfg, "current_event_scenario_name", cfg.event_model_type)

    beta_event = cfg.event_base_transmission_rate
    r0 = cfg.R_0
    iss = cfg.I_ss
    
    csv_filename = results_dir / f"run_daily_active_cases_{scenario_name}_R{int(r0*100)}_beta{int(beta_event*100)}_Iss{iss}.csv" if not scenario_name.startswith('no_event') else results_dir / f"run_daily_active_cases_{scenario_name}_R{int(r0*100)}_Iss{iss}.csv"
    final_df.to_csv(csv_filename, index=False, float_format='%.2f')
    print(f"Saved per-run daily active cases CSV to {csv_filename}")


def save_arrival_time_records(all_runs_results, data, cfg):
    """
    Saves tidy per-run arrival-time records for entropy and survival-style analyses.
    E-based (exposed individuals) with _E suffix in filename.
    """
    if not getattr(cfg, 'save_arrival_time_records', True):
        return
    print("Saving arrival-time records (E-based)...")
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    if cfg.event_model_type.startswith('no_event'):
        scenario_name = cfg.event_model_type
    else:
        scenario_name = getattr(cfg, 'current_event_scenario_name', cfg.event_model_type)
    n_patches = data['n_patches']

    rows = []
    for run_id, run_res in enumerate(all_runs_results):
        arrival_times = np.array(run_res.get('arrival_times', np.full(n_patches, np.nan)), dtype=float)
        arrival_sources = np.array(run_res.get('arrival_sources', np.full(n_patches, -1)), dtype=int)
        arrival_contexts = np.array(run_res.get('arrival_contexts', np.array([""] * n_patches, dtype=object)), dtype=object)
        arrival_steps = np.array(run_res.get('arrival_step_index', np.full(n_patches, -1)), dtype=int)
        arrival_days = np.array(run_res.get('arrival_event_day', np.full(n_patches, -1)), dtype=int)

        for district_id in range(n_patches):
            arrived = np.isfinite(arrival_times[district_id])
            rows.append({
                'scenario_name': scenario_name,
                'run_id': int(run_id),
                'district_id': int(district_id),
                'arrived': int(arrived),
                'arrival_time': float(arrival_times[district_id]) if arrived else np.nan,
                'source_district_id': int(arrival_sources[district_id]) if arrival_sources[district_id] >= 0 else -1,
                'infection_context': str(arrival_contexts[district_id]) if str(arrival_contexts[district_id]) else '',
                'event_day': int(arrival_days[district_id]) if arrival_days[district_id] >= 0 else -1,
                'step_index': int(arrival_steps[district_id]) if arrival_steps[district_id] >= 0 else -1,
            })

    if not rows:
        print("No arrival-time records were available to save.")
        return

    df = pd.DataFrame(rows)
    beta_event = cfg.event_base_transmission_rate
    r0 = cfg.R_0
    iss = cfg.I_ss
    csv_filename = results_dir / f"arrival_time_records_E_{scenario_name}_R{int(r0*100)}_beta{int(beta_event*100)}_Iss{iss}.csv" if not scenario_name.startswith('no_event') else results_dir / f"arrival_time_records_E_{scenario_name}_R{int(r0*100)}_Iss{iss}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Saved arrival-time records (E-based) to {csv_filename}")


def save_arrival_time_records_I(all_runs_results, data, cfg):
    """
    Saves tidy per-run arrival-time records for entropy and survival-style analyses.
    I-based (infectious individuals) with _I suffix in filename.
    """
    if not getattr(cfg, 'save_arrival_time_records', True):
        return
    print("Saving arrival-time records (I-based)...")
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    if cfg.event_model_type.startswith('no_event'):
        scenario_name = cfg.event_model_type
    else:
        scenario_name = getattr(cfg, 'current_event_scenario_name', cfg.event_model_type)
    n_patches = data['n_patches']

    rows = []
    for run_id, run_res in enumerate(all_runs_results):
        arrival_times = np.array(run_res.get('arrival_times_I', np.full(n_patches, np.nan)), dtype=float)
        arrival_sources = np.array(run_res.get('arrival_sources_I', np.full(n_patches, -1)), dtype=int)
        arrival_contexts = np.array(run_res.get('arrival_contexts_I', np.array([""] * n_patches, dtype=object)), dtype=object)
        arrival_steps = np.array(run_res.get('arrival_step_index_I', np.full(n_patches, -1)), dtype=int)
        arrival_days = np.array(run_res.get('arrival_event_day_I', np.full(n_patches, -1)), dtype=int)

        for district_id in range(n_patches):
            arrived = np.isfinite(arrival_times[district_id])
            rows.append({
                'scenario_name': scenario_name,
                'run_id': int(run_id),
                'district_id': int(district_id),
                'arrived': int(arrived),
                'arrival_time': float(arrival_times[district_id]) if arrived else np.nan,
                'source_district_id': int(arrival_sources[district_id]) if arrival_sources[district_id] >= 0 else -1,
                'infection_context': str(arrival_contexts[district_id]) if str(arrival_contexts[district_id]) else '',
                'event_day': int(arrival_days[district_id]) if arrival_days[district_id] >= 0 else -1,
                'step_index': int(arrival_steps[district_id]) if arrival_steps[district_id] >= 0 else -1,
            })

    if not rows:
        print("No arrival-time records (I-based) were available to save.")
        return

    df = pd.DataFrame(rows)
    beta_event = cfg.event_base_transmission_rate
    r0 = cfg.R_0
    iss = cfg.I_ss
    csv_filename = results_dir / f"arrival_time_records_I_{scenario_name}_R{int(r0*100)}_beta{int(beta_event*100)}_Iss{iss}.csv" if not scenario_name.startswith('no_event') else results_dir / f"arrival_time_records_I_{scenario_name}_R{int(r0*100)}_Iss{iss}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Saved arrival-time records (I-based) to {csv_filename}")


def extract_i_based_arrival_from_existing_results(all_runs_results, data, cfg):
    """
    Extracts I-based arrival times from existing simulation results without re-running simulations.
    Processes I_j or I_ij arrays to detect I > 0 transitions and generates I-based arrival time records.
    """
    print("Extracting I-based arrival times from existing results...")
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    if cfg.event_model_type.startswith('no_event'):
        scenario_name = cfg.event_model_type
    else:
        scenario_name = getattr(cfg, 'current_event_scenario_name', cfg.event_model_type)
    n_patches = data['n_patches']

    rows = []
    for run_id, run_res in enumerate(all_runs_results):
        # Check if we have I_j (aggregated) or I_ij (full matrix)
        if 'I_j' in run_res:
            # Aggregated format: I_j is a list of arrays, each array is I per patch at a timestep
            I_history = run_res['I_j']
            t_history = run_res.get('t', [])
        elif 'I_ij' in run_res:
            # Full matrix format: I_ij is a list of N×N matrices
            I_history = [mat.sum(axis=0) for mat in run_res['I_ij']]  # Aggregate to patch level
            t_history = run_res.get('t', [])
        else:
            print(f"Warning: No I data found in run {run_id}, skipping")
            continue

        # Detect I-based arrival times
        arrival_times = np.full(n_patches, np.nan, dtype=np.float32)
        arrival_step_index = np.full(n_patches, -1, dtype=np.int32)
        arrival_event_day = np.full(n_patches, -1, dtype=np.int32)

        for step_idx, I_by_patch in enumerate(I_history):
            if step_idx == 0:
                prev_I = np.zeros(n_patches, dtype=np.float32)
            else:
                prev_I = I_history[step_idx - 1]

            curr_I = np.array(I_by_patch, dtype=np.float32)
            newly_arrived = np.where((prev_I <= 0) & (curr_I > 0))[0]

            for patch_id in newly_arrived:
                if np.isnan(arrival_times[patch_id]):
                    arrival_times[patch_id] = t_history[step_idx] if step_idx < len(t_history) else float(step_idx * 0.5)
                    arrival_step_index[patch_id] = int(step_idx)
                    arrival_event_day[patch_id] = int(arrival_times[patch_id])

        # Add to rows
        for district_id in range(n_patches):
            arrived = np.isfinite(arrival_times[district_id])
            rows.append({
                'scenario_name': scenario_name,
                'run_id': int(run_id),
                'district_id': int(district_id),
                'arrived': int(arrived),
                'arrival_time': float(arrival_times[district_id]) if arrived else np.nan,
                'source_district_id': -1,  # Source info not available from aggregated data
                'infection_context': 'extracted',
                'event_day': int(arrival_event_day[district_id]) if arrival_event_day[district_id] >= 0 else -1,
                'step_index': int(arrival_step_index[district_id]) if arrival_step_index[district_id] >= 0 else -1,
            })

    if not rows:
        print("No I-based arrival times could be extracted from existing results.")
        return

    df = pd.DataFrame(rows)
    beta_event = cfg.event_base_transmission_rate
    r0 = cfg.R_0
    iss = cfg.I_ss
    csv_filename = results_dir / f"arrival_time_records_I_extracted_{scenario_name}_R{int(r0*100)}_beta{int(beta_event*100)}_Iss{iss}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Saved extracted I-based arrival times to {csv_filename}")


def save_invasion_edge_records(all_runs_results, data, cfg):
    """
    Saves directed invasion edges observed across runs.
    """
    if not getattr(cfg, 'save_invasion_edge_records', True):
        return
    print("Saving invasion-edge records...")
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    if cfg.event_model_type.startswith('no_event'):
        scenario_name = cfg.event_model_type
    else:
        scenario_name = getattr(cfg, 'current_event_scenario_name', cfg.event_model_type)

    rows = []
    for run_id, run_res in enumerate(all_runs_results):
        edge_records = run_res.get('invasion_edge_records', []) or []
        if edge_records:
            for rec in edge_records:
                rows.append({
                    'scenario_name': rec.get('scenario_name', scenario_name),
                    'run_id': int(rec.get('run_id', run_id)),
                    'source_district_id': int(rec.get('source_district_id', -1)),
                    'target_district_id': int(rec.get('target_district_id', -1)),
                    'arrival_time': float(rec.get('arrival_time', np.nan)),
                    'source_agent_id': int(rec.get('source_agent_id', -1)),
                    'target_agent_id': int(rec.get('target_agent_id', -1)),
                    'infection_context': str(rec.get('infection_context', '')),
                    'event_day': int(rec.get('event_day', -1)),
                    'weight_hint': int(rec.get('weight_hint', 1)),
                })
            continue

        # Fallback: synthesize one edge per first-arrival record.
        arrival_times = np.array(run_res.get('arrival_times', []), dtype=float)
        arrival_sources = np.array(run_res.get('arrival_sources', []), dtype=int)
        if arrival_times.size == 0:
            continue
        for district_id, arrival_time in enumerate(arrival_times):
            if not np.isfinite(arrival_time):
                continue
            context_arr = np.array(run_res.get('arrival_contexts', np.array([""] * len(arrival_times), dtype=object)), dtype=object)
            rows.append({
                'scenario_name': scenario_name,
                'run_id': int(run_id),
                'source_district_id': int(arrival_sources[district_id]) if district_id < arrival_sources.size and arrival_sources[district_id] >= 0 else -1,
                'target_district_id': int(district_id),
                'arrival_time': float(arrival_time),
                'source_agent_id': -1,
                'target_agent_id': -1,
                'infection_context': str(context_arr[district_id]) if district_id < len(context_arr) else '',
                'event_day': -1,
                'weight_hint': 1,
            })

    if not rows:
        print("No invasion-edge records were available to save.")
        return

    df = pd.DataFrame(rows)
    beta_event = cfg.event_base_transmission_rate
    r0 = cfg.R_0
    iss = cfg.I_ss
    csv_filename = results_dir / f"invasion_edge_records_{scenario_name}_R{int(r0*100)}_beta{int(beta_event*100)}_Iss{iss}.csv" if not scenario_name.startswith('no_event') else results_dir / f"invasion_edge_records_{scenario_name}_R{int(r0*100)}_Iss{iss}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Saved invasion-edge records to {csv_filename}")


def aggregate_invasion_edges(edge_source: Union[pd.DataFrame, str, Path], n_runs: Optional[int] = None) -> pd.DataFrame:
    """
    Aggregate directed invasion edges across runs.

    Returns a DataFrame with one row per directed edge and summary weights.
    """
    if isinstance(edge_source, (str, Path)):
        df = pd.read_csv(edge_source)
    else:
        df = edge_source.copy()

    required_cols = {'source_district_id', 'target_district_id'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Invasion-edge data is missing required columns: {sorted(missing)}")

    if 'weight_hint' not in df.columns:
        df['weight_hint'] = 1.0
    if 'arrival_time' not in df.columns:
        df['arrival_time'] = np.nan
    if 'scenario_name' not in df.columns:
        df['scenario_name'] = ''
    if 'run_id' not in df.columns:
        df['run_id'] = -1

    numeric_cols = ['source_district_id', 'target_district_id', 'run_id', 'weight_hint', 'arrival_time']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['source_district_id', 'target_district_id'])
    df['source_district_id'] = df['source_district_id'].astype(int)
    df['target_district_id'] = df['target_district_id'].astype(int)
    df['run_id'] = df['run_id'].fillna(-1).astype(int)
    df['weight_hint'] = df['weight_hint'].fillna(1.0).astype(float)

    # Ignore unassigned sources when aggregating. A tree root will be inserted separately if needed.
    df = df[(df['source_district_id'] >= 0) & (df['target_district_id'] >= 0)]
    if df.empty:
        return pd.DataFrame(columns=[
            'source_district_id', 'target_district_id', 'support', 'weight_fraction',
            'n_observations', 'n_runs', 'mean_arrival_time', 'median_arrival_time'
        ])

    if n_runs is None:
        n_runs = int(df['run_id'].nunique()) if (df['run_id'] >= 0).any() else 1

    grouped = (
        df.groupby(['source_district_id', 'target_district_id'], as_index=False)
        .agg(
            support=('weight_hint', 'sum'),
            n_observations=('weight_hint', 'size'),
            n_runs=('run_id', lambda s: int(s[s >= 0].nunique()) if (s >= 0).any() else 0),
            mean_arrival_time=('arrival_time', 'mean'),
            median_arrival_time=('arrival_time', 'median'),
        )
    )

    grouped['n_runs'] = grouped['n_runs'].replace(0, n_runs).astype(int)
    grouped['weight_fraction'] = grouped['support'] / max(int(n_runs), 1)
    grouped['support_fraction'] = grouped['weight_fraction']
    grouped['mean_arrival_time'] = grouped['mean_arrival_time'].astype(float)
    grouped['median_arrival_time'] = grouped['median_arrival_time'].astype(float)
    return grouped.sort_values(['weight_fraction', 'support', 'source_district_id', 'target_district_id'], ascending=[False, False, True, True]).reset_index(drop=True)


def build_invasion_tree_from_records(
    edge_source: Union[pd.DataFrame, str, Path],
    root_patch_id: int,
    weight_col: str = 'weight_fraction',
    method: str = 'arborescence',
    n_runs: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate invasion edges and extract a rooted invasion tree.

    This is the core offline entry point for rebuilding a tree from a saved
    invasion_edge_records_*.csv file without rerunning the simulation.
    """
    agg_df = aggregate_invasion_edges(edge_source, n_runs=n_runs)
    tree_df = extract_invasion_tree(agg_df, root_patch_id=root_patch_id, weight_col=weight_col, method=method)
    return agg_df, tree_df


def _find_directed_cycle(parent_map: Dict[Any, Dict[str, Any]], root: Any) -> Optional[List[Any]]:
    """Find one directed cycle in a parent-pointer map, if present."""
    seen_global = set()
    for start in list(parent_map.keys()):
        if start == root or start in seen_global:
            continue
        path = []
        seen_local = {}
        node = start
        while node != root and node in parent_map and node not in seen_local:
            seen_local[node] = len(path)
            path.append(node)
            node = parent_map[node]['u']
        if node in seen_local:
            return path[seen_local[node]:]
        seen_global.update(path)
    return None


def _maximum_spanning_arborescence(nodes: List[Any], edges: List[Dict[str, Any]], root: Any) -> List[Dict[str, Any]]:
    """
    Compute a maximum-weight spanning arborescence rooted at `root`.

    Pure Python Chu-Liu/Edmonds implementation with a pseudo-root fallback for
    disconnected nodes. The returned edges preserve the original `u`/`v` labels.
    """
    node_set = set(nodes)
    if root not in node_set:
        node_set.add(root)
        nodes = list(nodes) + [root]

    filtered_edges = [e for e in edges if e['u'] in node_set and e['v'] in node_set and e['u'] != e['v']]

    parent_map: Dict[Any, Dict[str, Any]] = {}
    for v in nodes:
        if v == root:
            continue
        incoming = [e for e in filtered_edges if e['v'] == v]
        if incoming:
            parent_map[v] = max(incoming, key=lambda e: e['weight'])
        else:
            parent_map[v] = {
                'u': root,
                'v': v,
                'weight': 0.0,
                'data': {'pseudo': True, 'kind': 'fallback'},
            }

    cycle = _find_directed_cycle(parent_map, root)
    if not cycle:
        return list(parent_map.values())

    cycle_set = set(cycle)
    cycle_super = ('cycle', tuple(cycle))
    cycle_parent_weight = {v: parent_map[v]['weight'] for v in cycle}
    cycle_internal_edges = {v: parent_map[v] for v in cycle}

    best_enter: Dict[Any, Dict[str, Any]] = {}
    best_exit: Dict[Any, Dict[str, Any]] = {}
    contracted_edges: List[Dict[str, Any]] = []

    for e in filtered_edges:
        u, v, w = e['u'], e['v'], e['weight']
        u_in = u in cycle_set
        v_in = v in cycle_set
        if u_in and v_in:
            continue
        if v_in and not u_in:
            adjusted_weight = w - cycle_parent_weight[v]
            candidate = {
                'u': u,
                'v': cycle_super,
                'weight': adjusted_weight,
                'data': {'kind': 'enter', 'orig_edge': e, 'target_cycle_node': v},
            }
            if u not in best_enter or adjusted_weight > best_enter[u]['weight']:
                best_enter[u] = candidate
        elif u_in and not v_in:
            candidate = {
                'u': cycle_super,
                'v': v,
                'weight': w,
                'data': {'kind': 'exit', 'orig_edge': e, 'source_cycle_node': u},
            }
            if v not in best_exit or w > best_exit[v]['weight']:
                best_exit[v] = candidate
        else:
            contracted_edges.append(e)

    if not best_enter:
        fallback_node = min(cycle, key=lambda v: cycle_parent_weight[v])
        best_enter[root] = {
            'u': root,
            'v': cycle_super,
            'weight': 0.0,
            'data': {
                'kind': 'enter',
                'orig_edge': {'u': root, 'v': fallback_node, 'weight': 0.0, 'data': {'pseudo': True}},
                'target_cycle_node': fallback_node,
            },
        }

    contracted_edges.extend(best_enter.values())
    contracted_edges.extend(best_exit.values())

    contracted_nodes = [n for n in nodes if n not in cycle_set] + [cycle_super]
    contracted_tree = _maximum_spanning_arborescence(contracted_nodes, contracted_edges, root)

    expanded_tree: List[Dict[str, Any]] = []
    entering_edge = None
    for e in contracted_tree:
        if e['v'] == cycle_super:
            entering_edge = e
        elif e['u'] == cycle_super:
            expanded_tree.append(e['data']['orig_edge'])
        else:
            expanded_tree.append(e)

    entered_node = None
    if entering_edge is not None:
        entered_node = entering_edge['data']['target_cycle_node']
        expanded_tree.append(entering_edge['data']['orig_edge'])

    for v in cycle:
        if v == entered_node:
            continue
        expanded_tree.append(cycle_internal_edges[v])

    return expanded_tree


def _greedy_maximum_branching(nodes: List[Any], edges: List[Dict[str, Any]], root: Any) -> List[Dict[str, Any]]:
    """
    Greedy fallback that selects high-weight edges while avoiding cycles and multi-parent violations.
    """
    parent: Dict[Any, Dict[str, Any]] = {}
    adjacency_parent: Dict[Any, Any] = {root: None}

    def creates_cycle(u: Any, v: Any) -> bool:
        cur = u
        while cur is not None:
            if cur == v:
                return True
            cur = adjacency_parent.get(cur)
        return False

    for e in sorted(edges, key=lambda x: x['weight'], reverse=True):
        u, v = e['u'], e['v']
        if u == v or v == root:
            continue
        if v in parent:
            continue
        if creates_cycle(u, v):
            continue
        parent[v] = e
        adjacency_parent[v] = u

    for v in nodes:
        if v == root or v in parent:
            continue
        parent[v] = {'u': root, 'v': v, 'weight': 0.0, 'data': {'pseudo': True, 'kind': 'fallback'}}

    return list(parent.values())


def extract_invasion_tree(
    aggregated_edges: pd.DataFrame,
    root_patch_id: int,
    weight_col: str = 'weight_fraction',
    method: str = 'arborescence',
) -> pd.DataFrame:
    """
    Build a rooted invasion tree from aggregated directed edge weights.

    Returns a DataFrame with one parent edge per invaded district.
    """
    if aggregated_edges is None or aggregated_edges.empty:
        return pd.DataFrame(columns=[
            'source_district_id', 'target_district_id', 'weight', 'support', 'n_observations',
            'n_runs', 'mean_arrival_time', 'median_arrival_time', 'is_root_edge'
        ])

    if weight_col not in aggregated_edges.columns:
        raise ValueError(f"Weight column '{weight_col}' not found in aggregated edge table.")

    nodes = sorted(set(aggregated_edges['source_district_id'].astype(int)).union(set(aggregated_edges['target_district_id'].astype(int))))
    if root_patch_id not in nodes:
        nodes = [root_patch_id] + nodes

    edge_rows = []
    for _, row in aggregated_edges.iterrows():
        edge_rows.append({
            'u': int(row['source_district_id']),
            'v': int(row['target_district_id']),
            'weight': float(row[weight_col]),
            'data': row.to_dict(),
        })

    if method == 'arborescence':
        try:
            tree_edges = _maximum_spanning_arborescence(nodes, edge_rows, int(root_patch_id))
        except Exception:
            tree_edges = _greedy_maximum_branching(nodes, edge_rows, int(root_patch_id))
    elif method == 'greedy':
        tree_edges = _greedy_maximum_branching(nodes, edge_rows, int(root_patch_id))
    else:
        raise ValueError(f"Unknown invasion tree extraction method: '{method}'")

    tree_rows = []
    for e in tree_edges:
        data = e.get('data', {})
        tree_rows.append({
            'source_district_id': int(e['u']),
            'target_district_id': int(e['v']),
            'weight': float(e['weight']),
            'support': float(data.get('support', data.get('weight_hint', 1.0))),
            'n_observations': int(data.get('n_observations', 1)),
            'n_runs': int(data.get('n_runs', 1)),
            'weight_fraction': float(data.get('weight_fraction', e['weight'])),
            'mean_arrival_time': float(data.get('mean_arrival_time', np.nan)),
            'median_arrival_time': float(data.get('median_arrival_time', np.nan)),
            'is_root_edge': bool(data.get('pseudo', False)),
        })

    tree_df = pd.DataFrame(tree_rows)
    if not tree_df.empty:
        tree_df = tree_df.sort_values(['weight', 'source_district_id', 'target_district_id'], ascending=[False, True, True]).reset_index(drop=True)
    return tree_df


def save_invasion_tree_from_records(all_runs_results, data, cfg, method: str = 'arborescence'):
    """
    Aggregate invasion edges, extract a tree, and save both the aggregate and tree tables.
    """
    if not getattr(cfg, 'save_invasion_tree_records', True):
        return

    print("Building invasion tree...")
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    if cfg.event_model_type.startswith('no_event'):
        scenario_name = cfg.event_model_type
    else:
        scenario_name = getattr(cfg, 'current_event_scenario_name', cfg.event_model_type)
    root_patch_id = int(getattr(cfg, 'initial_infection_patch_id', 0))

    rows = []
    for run_id, run_res in enumerate(all_runs_results):
        edge_records = run_res.get('invasion_edge_records', []) or []
        for rec in edge_records:
            rows.append({
                'scenario_name': rec.get('scenario_name', scenario_name),
                'run_id': int(rec.get('run_id', run_id)),
                'source_district_id': int(rec.get('source_district_id', -1)),
                'target_district_id': int(rec.get('target_district_id', -1)),
                'arrival_time': float(rec.get('arrival_time', np.nan)),
                'source_agent_id': int(rec.get('source_agent_id', -1)),
                'target_agent_id': int(rec.get('target_agent_id', -1)),
                'infection_context': str(rec.get('infection_context', '')),
                'event_day': int(rec.get('event_day', -1)),
                'weight_hint': int(rec.get('weight_hint', 1)),
            })

    if not rows:
        print("No invasion edges available; skipping invasion tree generation.")
        return

    edge_df = pd.DataFrame(rows)
    beta_event = cfg.event_base_transmission_rate
    r0 = cfg.R_0
    iss = cfg.I_ss
    edge_csv = results_dir / f"invasion_edge_records_{scenario_name}_R{int(r0*100)}_beta{int(beta_event*100)}_Iss{iss}.csv" if not scenario_name.startswith('no_event') else results_dir / f"invasion_edge_records_{scenario_name}_R{int(r0*100)}_Iss{iss}.csv"
    if not edge_csv.exists():
        edge_df.to_csv(edge_csv, index=False)

    agg_df, tree_df = build_invasion_tree_from_records(
        edge_df,
        root_patch_id=root_patch_id,
        method=method,
        n_runs=len(all_runs_results),
    )
    agg_csv = results_dir / f"invasion_edge_aggregated_{scenario_name}_R{int(r0*100)}_beta{int(beta_event*100)}_Iss{iss}.csv" if not scenario_name.startswith('no_event') else results_dir / f"invasion_edge_aggregated_{scenario_name}_R{int(r0*100)}_Iss{iss}.csv"
    agg_df.to_csv(agg_csv, index=False)
    tree_csv = results_dir / f"invasion_tree_{scenario_name}_R{int(r0*100)}_beta{int(beta_event*100)}_Iss{iss}.csv" if not scenario_name.startswith('no_event') else results_dir / f"invasion_tree_{scenario_name}_R{int(r0*100)}_Iss{iss}.csv"
    tree_df.to_csv(tree_csv, index=False)

    print(f"Saved aggregated invasion edges to {agg_csv}")
    print(f"Saved invasion tree to {tree_csv}")


def rebuild_invasion_tree_from_csv(
    edge_csv: Union[str, Path],
    root_patch_id: int = 0,
    method: str = 'arborescence',
    output_dir: Optional[Union[str, Path]] = None,
    weight_col: str = 'weight_fraction',
    n_runs: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Rebuild and optionally save an invasion tree from a saved edge CSV.

    Parameters
    ----------
    edge_csv:
        Path to invasion_edge_records_*.csv or an equivalent DataFrame source.
    root_patch_id:
        Patch id used as the tree root. This should match the initial infection patch.
    output_dir:
        If provided, writes invasion_edge_aggregated_*.csv and invasion_tree_*.csv
        into this directory using the input filename stem.
    """
    edge_path = Path(edge_csv) if isinstance(edge_csv, (str, Path)) else None
    agg_df, tree_df = build_invasion_tree_from_records(
        edge_csv,
        root_patch_id=root_patch_id,
        weight_col=weight_col,
        method=method,
        n_runs=n_runs,
    )

    result = {
        "aggregated_edges": agg_df,
        "tree": tree_df,
    }

    if output_dir is None:
        return result

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = edge_path.stem if edge_path is not None else "invasion_edge_records"
    if stem.startswith("invasion_edge_records_"):
        stem = stem.replace("invasion_edge_records_", "", 1)
    agg_csv = out_dir / f"invasion_edge_aggregated_{stem}.csv"
    tree_csv = out_dir / f"invasion_tree_{stem}.csv"
    agg_df.to_csv(agg_csv, index=False)
    tree_df.to_csv(tree_csv, index=False)
    result["aggregated_csv"] = agg_csv
    result["tree_csv"] = tree_csv
    return result

def save_arrival_time_statistics(group_results: Dict[str, Any], data: Dict[str, Any], cfg: Any):
    """
    Saves summary statistics (mean, 25th, 75th percentiles) of epidemic arrival times
    for all scenarios in a parameter group to a single CSV file.
    """
    print(f"Saving arrival time statistics for R0={cfg.R_0}, beta_event={cfg.event_base_transmission_rate}...")
    
    n_patches = data['n_patches']
    n_iterations = cfg.n_iterations
    
    all_stats = []
    
    # Sort scenarios: Baseline first, then alphabetical
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
        
    for scenario in scenarios:
        res = group_results[scenario]
        # Use existing helper to get (n_iterations, n_patches)
        arrival_times = _get_arrival_times(res['all_runs_results'], n_patches, n_iterations)
        
        with np.errstate(all='ignore'):
            means = np.nanmean(arrival_times, axis=0)
            q25 = np.nanpercentile(arrival_times, 25, axis=0)
            q75 = np.nanpercentile(arrival_times, 75, axis=0)
            
        for p in range(n_patches):
            # Only record if it was infected at least once in some realization
            if not np.isnan(means[p]):
                all_stats.append({
                    'scenario': "Baseline" if scenario == 'no_event' else scenario,
                    'patch_id': p,
                    'mean_arrival': round(means[p], 3),
                    'q25_arrival': round(q25[p], 3),
                    'q75_arrival': round(q75[p], 3)
                })
                
    if not all_stats:
        return

    df = pd.DataFrame(all_stats)
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    
    beta_event = cfg.event_base_transmission_rate
    filename = results_dir / f'arrival_time_stats_R{int(cfg.R_0*100)}_beta{int(beta_event*100)}_Iss{cfg.I_ss}.csv'
    df.to_csv(filename, index=False)
    print(f"  -> Arrival time statistics saved to {filename}")

def _levenshtein_distance(s1: List[Any], s2: List[Any]) -> int:
    """Calculates the Levenshtein distance between two sequences."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def analyze_arrival_time_sequence(group_results: Dict[str, Any], data: Dict[str, Any], cfg: Any):
    """
    Analyzes the order of infection arrivals across districts, compares rankings,
    calculates correlations, and generates visualization (bump chart and heatmap).
    """
    print(f"Analyzing arrival time sequence for R0={cfg.R_0}, beta_event={cfg.event_base_transmission_rate}...")
    
    n_patches = data['n_patches']
    n_iterations = cfg.n_iterations
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
    else:
        return

    summary_data, rank_df_list, mean_arrivals, raw_arrival_data = [], [], {}, {}

    for sc in scenarios:
        res = group_results[sc]
        arrival_times = _get_arrival_times(res['all_runs_results'], n_patches, n_iterations)
        raw_arrival_data[sc] = arrival_times
        
        with np.errstate(all='ignore'):
            means = np.nanmean(arrival_times, axis=0)
            medians = np.nanmedian(arrival_times, axis=0)
        
        mean_arrivals[sc] = means
        valid_idx = np.where(~np.isnan(means))[0]
        if len(valid_idx) == 0:
            continue

        valid_means = means[valid_idx]
        ranks = stats.rankdata(valid_means, method='min') if stats else np.argsort(np.argsort(valid_means)) + 1
        
        sc_label = "Baseline" if sc == 'no_event' else sc
        for i, p_idx in enumerate(valid_idx):
            summary_data.append({
                'scenario': sc_label, 'patch_id': p_idx,
                'mean_arrival': round(float(means[p_idx]), 3),
                'median_arrival': round(float(medians[p_idx]), 3),
                'rank': int(ranks[i])
            })
            rank_df_list.append({'Scenario': sc_label, 'Patch': p_idx, 'Rank': int(ranks[i])})

    if not summary_data: return
    df_ranks = pd.DataFrame(rank_df_list)
    file_tag = f"R{int(cfg.R_0*100)}_beta{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}"

    # Correlation vs Baseline
    corr_results = []
    if stats and "Baseline" in df_ranks['Scenario'].values:
        base_ranks = df_ranks[df_ranks['Scenario'] == 'Baseline'].set_index('Patch')['Rank']
        base_df = df_ranks[df_ranks['Scenario'] == 'Baseline'].sort_values('Rank')
        base_ranks = base_df.set_index('Patch')['Rank']
        base_seq_full = base_df['Patch'].tolist()

        for sc in scenarios:
            if sc == 'no_event': continue
            sc_ranks = df_ranks[df_ranks['Scenario'] == sc].set_index('Patch')['Rank']
            sc_df = df_ranks[df_ranks['Scenario'] == sc].sort_values('Rank')
            sc_ranks = sc_df.set_index('Patch')['Rank']
            sc_seq_full = sc_df['Patch'].tolist()

            common = base_ranks.index.intersection(sc_ranks.index)
            if len(common) > 1:
                # Check for constant input to avoid ConstantInputWarning
                if np.ptp(base_ranks.loc[common]) == 0 or np.ptp(sc_ranks.loc[common]) == 0:
                    rho, _ = np.nan, np.nan
                else:
                    rho, _ = stats.spearmanr(base_ranks.loc[common], sc_ranks.loc[common])
                tau, _ = stats.kendalltau(base_ranks.loc[common], sc_ranks.loc[common])

                # Levenshtein distance on the intersection of infected patches
                s1 = [p for p in base_seq_full if p in common]
                s2 = [p for p in sc_seq_full if p in common]
                l_dist = _levenshtein_distance(s1, s2)
                norm_l = l_dist / len(common)

                corr_results.append({
                    'scenario': sc, 
                    'spearman_rho': round(rho, 4), 
                    'kendall_tau': round(tau, 4),
                    'norm_levenshtein': round(norm_l, 4),
                    'n_patches': len(common)
                })

    # Correlation Bar Chart
    if corr_results:
        corr_df = pd.DataFrame(corr_results)
        fig, ax = plt.subplots(figsize=(8, 5))
        
        x = np.arange(len(corr_df['scenario']))
        width = 0.35
        
        ax.bar(x - width/2, corr_df['spearman_rho'], width, label="Spearman's ρ", color='skyblue', edgecolor='black')
        ax.bar(x + width/2, corr_df['kendall_tau'], width, label="Kendall's τ", color='salmon', edgecolor='black')
        
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title(f'Arrival sequence rank correlation vs. Baseline\n{file_tag}')
        ax.set_xticks(x)
        ax.set_xticklabels(corr_df['scenario'], rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.legend(loc='lower right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add numeric labels on top of bars
        for i, val in enumerate(corr_df['spearman_rho']):
            ax.text(i - width/2, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=7)
        for i, val in enumerate(corr_df['kendall_tau']):
            ax.text(i + width/2, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=7)
            
        plt.tight_layout()
        # plt.savefig(results_dir / f"arrival_rank_correlation_{file_tag}.png", dpi=300)
        plt.close()

        # Levenshtein Distance Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_dist = plt.cm.viridis(np.linspace(0.2, 0.8, len(corr_df)))
        
        x = np.arange(len(corr_df['scenario']))
        bars = ax.bar(x, corr_df['norm_levenshtein'], color=colors_dist, edgecolor='black')
        
        ax.set_ylabel('Normalized Levenshtein Distance')
        ax.set_title(f'Arrival Sequence Edit Distance vs. Baseline\n{file_tag}')
        ax.set_xticks(x)
        ax.set_xticklabels(corr_df['scenario'], rotation=45, ha='right')
        ax.set_ylim(0, max(corr_df['norm_levenshtein'].max() * 1.2, 0.5))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add numeric labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        # plt.savefig(results_dir / f"arrival_levenshtein_distance_{file_tag}.png", dpi=300)
        plt.close()

    # Kendall's Tau Correlation Matrix Heatmap
    if stats and len(scenarios) > 1:
        try:
            import seaborn as sns
            unique_sc_labels = ["Baseline"] + [s for s in scenarios if s != 'no_event']
            tau_matrix = pd.DataFrame(index=unique_sc_labels, columns=unique_sc_labels, dtype=float)
            
            for s1 in unique_sc_labels:
                r1 = df_ranks[df_ranks['Scenario'] == s1].set_index('Patch')['Rank']
                for s2 in unique_sc_labels:
                    if s1 == s2:
                        tau_matrix.loc[s1, s2] = 1.0
                        continue
                    r2 = df_ranks[df_ranks['Scenario'] == s2].set_index('Patch')['Rank']
                    common = r1.index.intersection(r2.index)
                    if len(common) > 1:
                        tau, _ = stats.kendalltau(r1.loc[common], r2.loc[common])
                        tau_matrix.loc[s1, s2] = tau
            
            plt.figure(figsize=(5, 4))
            # Use a mask to show only the lower triangle for clarity
            mask = np.triu(np.ones_like(tau_matrix, dtype=bool))
            sns.heatmap(tau_matrix, annot=True, cmap='YlGnBu', vmin=0, vmax=1, fmt=".2f", 
                        mask=mask, square=True, linewidths=.5, cbar_kws={"shrink": .8, "label": "Kendall's tau"})
            
            # plt.title(f"Arrival sequence similarity matrix", fontsize=14)
            plt.tight_layout()
            # plt.savefig(results_dir / f"arrival_kendall_tau_heatmap_{file_tag}.png", dpi=300)
            plt.close()
            # print(f"  -> Kendall's tau heatmap saved to arrival_kendall_tau_heatmap_{file_tag}.png")
        except ImportError:
            pass

    # Levenshtein Distance Matrix Heatmap
    if len(scenarios) > 1:
        try:
            import seaborn as sns
            unique_sc_labels = ["Baseline"] + [s for s in scenarios if s != 'no_event']
            lev_matrix = pd.DataFrame(index=unique_sc_labels, columns=unique_sc_labels, dtype=float)
            
            for s1_label in unique_sc_labels:
                s1_seq_full = df_ranks[df_ranks['Scenario'] == s1_label].sort_values('Rank')['Patch'].tolist()
                for s2_label in unique_sc_labels:
                    if s1_label == s2_label:
                        lev_matrix.loc[s1_label, s2_label] = 0.0
                        continue
                    
                    s2_seq_full = df_ranks[df_ranks['Scenario'] == s2_label].sort_values('Rank')['Patch'].tolist()
                    common = set(s1_seq_full).intersection(set(s2_seq_full))
                    
                    if len(common) > 1:
                        seq1 = [p for p in s1_seq_full if p in common]
                        seq2 = [p for p in s2_seq_full if p in common]
                        l_dist = _levenshtein_distance(seq1, seq2)
                        lev_matrix.loc[s1_label, s2_label] = l_dist / len(common)
            
            plt.figure(figsize=(5, 4))
            mask = np.triu(np.ones_like(lev_matrix, dtype=bool))
            sns.heatmap(lev_matrix, annot=True, cmap='YlOrRd', vmin=0, vmax=1, fmt=".2f", 
                        mask=mask, square=True, linewidths=.5, cbar_kws={"shrink": .8, "label": "Normalized Levenshtein Distance"})
            
            # plt.title(f"Arrival sequence dissimilarity matrix", fontsize=14)
            plt.tight_layout()
            # plt.savefig(results_dir / f"arrival_levenshtein_heatmap_{file_tag}.png", dpi=300)
            plt.close()
            # print(f"  -> Levenshtein distance heatmap saved to arrival_levenshtein_heatmap_{file_tag}.png")
        except ImportError:
            pass

    # Bump Chart (All Districts)
    all_patches = df_ranks[df_ranks['Scenario'] == 'Baseline'].sort_values('Rank')['Patch'].tolist()
    bump_subset = df_ranks[df_ranks['Patch'].isin(all_patches)]
    if not bump_subset.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        pivot_bump = bump_subset.pivot(index='Patch', columns='Scenario', values='Rank')
        disp_cols = ["Baseline"] + [s for s in scenarios if s != 'no_event']
        pivot_bump = pivot_bump.reindex(columns=disp_cols)
        for p_id in all_patches:
            if p_id in pivot_bump.index:
                ax.plot(disp_cols, pivot_bump.loc[p_id], marker='o', lw=1.5, label=f"District {p_id}", alpha=0.7)
        ax.invert_yaxis()
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_title(f"District Arrival Rank Shift (All Districts)\n{file_tag}", fontsize=14)
        ax.set_ylabel("Rank (1 = Earliest Arrival)")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize='x-small', ncol=min(8, len(all_patches)))
        # plt.savefig(results_dir / f"arrival_rank_bump_{file_tag}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Difference Heatmap
    try:
        import seaborn as sns
        event_scenarios = [s for s in scenarios if s != 'no_event']
        if event_scenarios:
            diff_list = []
            base_m = mean_arrivals['no_event']
            seed_id = int(getattr(cfg, 'initial_infection_patch_id', 0))
            for sc in event_scenarios:
                diffs = mean_arrivals[sc] - base_m
                for p in range(n_patches):
                    if p == seed_id:
                        continue
                    if not np.isnan(diffs[p]):
                        diff_list.append({'Scenario': sc, 'Patch': p, 'Diff': diffs[p]})
            if diff_list:
                pivot_diff = pd.DataFrame(diff_list).pivot(index='Scenario', columns='Patch', values='Diff')
                plt.figure(figsize=(7, 2.5))
                # sns.heatmap(pivot_diff, cmap='RdBu_r', center=0, cbar_kws={'label': 'Arrival Time Shift (Days)'})
                sns.heatmap(pivot_diff, cmap='RdBu', center=0, cbar_kws={'label': 'Arrival Time Shift (Days)'})
                #plt.title(f"Arrival Time Shift (Event - Baseline)\n{file_tag}", fontsize=14)
                r0=cfg.R_0
                beta_event=cfg.event_base_transmission_rate
                iss = cfg.I_ss             
                plt.title(f"Simulation results: Arrival time shift (Event - Baseline)\n($R_0$={r0}, $\\beta_{{event}}$={beta_event}, $I_{{ss}}$={iss})", fontsize=13)
                plt.xlabel("District ID"); plt.ylabel("Scenario")
                plt.tight_layout()
                plt.savefig(results_dir / f"arrival_diff_heatmap_{file_tag}.png", dpi=300, bbox_inches='tight')
                plt.close()
    except ImportError: pass

    pd.DataFrame(summary_data).to_csv(results_dir / f"arrival_sequence_ranks_{file_tag}.csv", index=False)
    if corr_results: pd.DataFrame(corr_results).to_csv(results_dir / f"arrival_sequence_correlations_{file_tag}.csv", index=False)

    # --- Combined Rank-Shift vs. Time-Shift Scatter Plot ---
    if "Baseline" in df_ranks['Scenario'].values:
        base_data = df_ranks[df_ranks['Scenario'] == "Baseline"].set_index('Patch')
        base_means = pd.Series(mean_arrivals['no_event'], name='base_mean')
        
        # Pre-calculate global limits for consistent scaling across scenarios for this parameter set
        global_rank_min, global_rank_max = 0, 0
        global_time_min, global_time_max = 0.0, 0.0
        datasets_to_plot = {}

        for sc in scenarios:
            if sc == 'no_event': continue
            sc_data = df_ranks[df_ranks['Scenario'] == sc].set_index('Patch')
            sc_means = pd.Series(mean_arrivals[sc], name='sc_mean')
            
            # Merge data for common patches
            m = pd.concat([base_data['Rank'].rename('base_rank'), 
                                sc_data['Rank'].rename('sc_rank'),
                                base_means, sc_means], axis=1).dropna()
            
            if not m.empty:
                m['rank_shift'] = m['sc_rank'] - m['base_rank']
                m['time_shift'] = m['sc_mean'] - m['base_mean']
                datasets_to_plot[sc] = m
                
                global_rank_min = min(global_rank_min, m['rank_shift'].min())
                global_rank_max = max(global_rank_max, m['rank_shift'].max())
                global_time_min = min(global_time_min, m['time_shift'].min())
                global_time_max = max(global_time_max, m['time_shift'].max())

        # Focus colorbar on negative shifts (accelerations). Range: [min, 0]
        # Delays (positive shifts) will be clipped to the 'not significant' color (light red).
        t_vmin = min(-0.1, global_time_min)
        norm = plt.Normalize(vmin=t_vmin, vmax=0)
        # Reds_r: Dark Red at vmin (much earlier), Light Red at 0 (not significant/delayed)
        custom_cmap = plt.cm.Reds_r

        for sc, merged in datasets_to_plot.items():
            sc_label = sc
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Scatter plot: X=Baseline Time, Y=Rank Shift, Color=Temporal Shift (Signed)
            # Use uniform symmetric color scale across scenarios for this parameter set
            sc_plot = ax.scatter(merged['base_mean'], merged['rank_shift'], 
                                 c=merged['time_shift'], cmap=custom_cmap, 
                                 norm=norm,
                                 s=100, alpha=0.8, edgecolor='black', zorder=3)
            
            # Identity lines and labeling
            ax.axhline(0, color='black', linestyle='--', alpha=0.5, zorder=2)
            
            # Apply uniform Y-axis range
            ax.set_ylim(global_rank_min - 2, global_rank_max + 2)
            
            # Label ALL data points with the requested format
            texts = []
            for p_id, row in merged.iterrows():
                # Place label below the point for better visibility with increased font size.
                # We apply a vertical offset (-0.5) to ensure the text starts below the marker circle.
                texts.append(ax.text(row['base_mean'], row['rank_shift'] - 0.5, f"#{int(p_id)}", 
                                     fontsize=10, fontweight='normal', ha='center', va='top'))

            # Prevent label overlap using adjust_text if available
            try:
                from adjust_text import adjust_text
                adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.5))
            except ImportError:
                # If adjust_text is not installed, labels will remain at their data points
                pass

            ax.set_xlabel('Baseline Arrival Time (Mean Days)', fontsize=12)
            ax.set_ylabel('Rank Shift (Scenario Rank - Baseline Rank)', fontsize=12)
            
            # Add twin axis to show the "Infection order" on top
            ax_top = ax.twiny()
            ax_top.set_xlim(ax.get_xlim())
            ax_top.set_xlabel('Baseline Chronological Sequence (Days)', fontsize=10, alpha=0.7)
            
            plt.title(f'Sequence Disruption vs. Temporal Acceleration\nScenario: {sc_label} | {file_tag}', fontsize=14, pad=35)

            # Adjust horizontal size of the main plot to prevent colorbar overlap.
            # We use rect to leave room on the right (up to 0.88) for the dedicated colorbar axis.
            fig.tight_layout(rect=[0, 0, 0.88, 0.95])
            
            cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(sc_plot, cax=cbar_ax)
            cbar.set_label('Temporal Shift (Days: earlier < 0)', fontsize=12)
            
            # Annotate quadrants
            x_mid = merged['base_mean'].mean()
            ax.text(ax.get_xlim()[0]+2, ax.get_ylim()[1]-2, "Delayed / Pushed Back", 
                    fontsize=10, color='grey', fontstyle='italic', verticalalignment='top')
            ax.text(ax.get_xlim()[0]+2, ax.get_ylim()[0]+2, "Queue Jumpers", 
                    fontsize=10, color='blue', fontweight='bold', verticalalignment='bottom')

            ax.grid(True, linestyle=':', alpha=0.6)
            
            # out_fig = results_dir / f"arrival_rank_vs_time_shift_{sc_label}_{file_tag}.png"
            # plt.savefig(out_fig, dpi=300)
            plt.close()
            # print(f"  -> Combined rank-time shift plot saved to {out_fig.name}")



def plot_arrival_rank_vs_time_shift(realizations: Dict[str, np.ndarray], total_population: float, cfg: Any):
    """
    Plots the arrival rank of each individual against time shift.
    """
    if not getattr(cfg, 'plot_on_the_fly', True):
        return

    if not getattr(realizations, 'E', None):
        return

    E = realizations['E']
    I = realizations['I']
    time = E[:, 0]

    # Calculate arrival rank (order of arrival)
    arrival_order = np.argsort(E[:, 1])

    # Calculate active cases per 1000 for individual runs
    EI_runs = ((E[:, 1:] + I[:, 1:]) / total_population) * 1000

    # Plot individual runs
    fig, ax = plt.subplots(figsize=(8, 4.5))  # Change the figsize here

    for i in range(EI_runs.shape[1]):
        ax.plot(time[arrival_order], EI_runs[:, i], color='thistle', alpha=0.1, lw=0.5)

    ax.set_title('arrival rank vs time shift', fontsize=16)
    ax.set_xlabel('Time Shift (days)', fontsize=12)
    ax.set_ylabel('Arrival Rank (order of arrival)', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    scenario_name = getattr(cfg, 'current_event_scenario_name', cfg.event_model_type)
    if cfg.event_model_type.startswith('no_event'):
        scenario_name = cfg.event_model_type

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    r0_str = f"{int(cfg.R_0*100)}"
    beta_str = f"{int(cfg.event_base_transmission_rate*100)}"
    iss_str = f"{cfg.I_ss}"

    # filename = results_dir / f"arrival_rank_vs_time_shift_{scenario_name}_R{r0_str}_beta{beta_str}_Iss{iss_str}.png"
    # plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    # print(f"  -> Arrival rank vs time shift plot saved to '{filename}'")

def analyze_and_plot_arrival_times(all_runs_results, data, cfg):
    """
    Analyzes and visualizes the arrival time of the infection (first E > 0) for each patch.
    1. Prints stats on total patches affected.
    2. Plots a Boxplot of Arrival Times per Patch.
    3. Plots a Choropleth Map of Mean Arrival Times.
    """
    print(f"Analyzing epidemic arrival times for '{getattr(cfg, 'event_model_type', 'unknown')}'...")
    n_patches = data['n_patches']
    n_iterations = cfg.n_iterations
    n_days = cfg.n_days
    
    # Storage
    arrival_times = np.full((n_iterations, n_patches), np.nan)
    
    for run_id, res in enumerate(all_runs_results):
        t = np.array(res['t'])
        
        # Get E_j (Exposed present in j)
        if 'E_j' in res:
            E_j = np.array(res['E_j']) # Shape: (time_steps, n_patches)
        elif 'E_ij' in res:
             # Fallback if full results are kept
             E_ij = np.array(res['E_ij'])
             E_j = E_ij.reshape(-1, n_patches, n_patches).sum(axis=1)
        else:
            continue
            
        # Find first time E > 0 for each patch
        is_infected = (E_j > 0)
        first_indices = np.argmax(is_infected, axis=0)
        
        for p in range(n_patches):
            idx = first_indices[p]
            # Check if it was actually infected (is_infected[idx, p] is True)
            # OR if idx is 0, check if it was infected at t=0
            if is_infected[idx, p]:
                arrival_times[run_id, p] = t[idx]
            else:
                arrival_times[run_id, p] = np.nan # Never infected
        
    # --- 1. Quantify Affected Patches ---
    total_affected = np.sum(~np.isnan(arrival_times), axis=1)
    mean_affected = np.mean(total_affected)
    std_affected = np.std(total_affected)
    print(f"  -> Patches affected by end of simulation: {mean_affected:.2f} +/- {std_affected:.2f} (out of {n_patches})")
    
    # Determine scenario info for remaining plots
    if cfg.event_model_type.startswith('no_event'):
        scenario_name_for_file = cfg.event_model_type
    else:
        scenario_name_for_file = getattr(cfg, 'current_event_scenario_name', cfg.event_model_type)

    # Decouple: If plotting on the fly is disabled, return stats without generating figures.
    if not getattr(cfg, 'plot_on_the_fly', True):
        return {
            "mean_affected": mean_affected, 
            "std_affected": std_affected, 
            "affected_values": total_affected.tolist()
        }

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    beta_event = cfg.event_base_transmission_rate

    # --- 2. Arrival Time Boxplot ---
    df_arrival = pd.DataFrame(arrival_times, columns=range(n_patches))
    valid_cols = df_arrival.columns[df_arrival.notna().any()].tolist()
    
    if valid_cols:
        df_plot = df_arrival[valid_cols]
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        df_plot.boxplot(ax=ax2, rot=90, fontsize=8)
        ax2.set_xlabel('Patch ID')
        ax2.set_ylabel('Arrival Time (Days)')
        # ax2.set_title(f'Distribution of Epidemic Arrival Times per Patch\n({scenario_name_for_file})')
        ax2.set_title(f'Epidemic Arrival Time Distribution')
        
        filename2 = results_dir / f'arrival_time_boxplot_{scenario_name_for_file}_R{int(cfg.R_0*100)}_beta{int(beta_event*100)}_Iss{cfg.I_ss}.png'
        plt.savefig(filename2, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"  -> Saved arrival time boxplot to {filename2}")

    # --- 3. Arrival Time Map ---
    if HAS_GEOPANDAS:
        # Calculate mean arrival time per patch
        with np.errstate(invalid='ignore'):
            mean_arrival_times = np.nanmean(arrival_times, axis=0)

        # Load GeoJSON (Using logic similar to plot_spatial_spread_snapshots)
        mobility_dir = Path(f"data_mobility_{cfg.dataset_name}")
        geojson_filename = f"{cfg.dataset_name.lower()}-districts.geojson"
        geojson_path = mobility_dir / geojson_filename

        if not geojson_path.exists():
            if mobility_dir.exists():
                geojsons = list(mobility_dir.glob("*.geojson"))
                if geojsons:
                    geojson_path = geojsons[0]

        if geojson_path.exists():
            try:
                geodf = gpd.read_file(geojson_path)
                
                # --- Get Seed Patch Coordinates ---
                # This is duplicated from the comparison plot but necessary for the individual map
                seed_patch_id = getattr(cfg, 'initial_infection_patch_id', 0)
                seed_coords = None
                try:
                    # We need to find the row in the GeoDataFrame that corresponds to the seed patch ID
                    # This assumes the 'id' column has been processed correctly below
                    # We will apply this logic *after* ID alignment.
                    pass # Placeholder, logic moved down
                except Exception as e:
                    print(f"  -> Warning: Could not get seed patch coordinates for individual map: {e}")
                
                # Align IDs
                id_col_found = None
                id_candidates = ['id', 'patch_id', 'ID', 'cartodb_id']
                for col_name in id_candidates:
                    if col_name in geodf.columns:
                        id_col_found = col_name
                        break
                
                if id_col_found:
                    if id_col_found != 'id':
                        geodf = geodf.rename(columns={id_col_found: 'id'})
                    
                    if not pd.api.types.is_numeric_dtype(geodf['id']):
                         geodf['id'] = pd.to_numeric(geodf['id'], errors='coerce')
                         geodf.dropna(subset=['id'], inplace=True)
                         geodf['id'] = geodf['id'].astype(int)

                    if geodf['id'].min() == 1 and geodf['id'].max() == n_patches:
                         geodf['id'] = geodf['id'] - 1

                    # Now that IDs are aligned, get seed coordinates
                    try:
                        seed_patch_row = geodf[geodf['id'] == seed_patch_id]
                        if not seed_patch_row.empty:
                            # Re-project to a projected CRS (3857) to calculate centroid accurately and avoid UserWarning
                            c = seed_patch_row.to_crs(epsg=3857).centroid.to_crs(geodf.crs).iloc[0]
                            seed_coords = (c.x, c.y)
                    except Exception as e:
                        print(f"  -> Warning: Could not get seed patch coordinates for individual map: {e}")

                    # Create DataFrame for plotting
                    arrival_df = pd.DataFrame({'id': range(n_patches), 'arrival_time': mean_arrival_times})
                    merged_gdf = geodf.merge(arrival_df, on='id', how='left')

                    # Determine color scale for this specific plot
                    valid_times = mean_arrival_times[~np.isnan(mean_arrival_times)]
                    vmin = valid_times.min() if len(valid_times) > 0 else 0
                    vmax = valid_times.max() if len(valid_times) > 0 else 1
                    if vmax == vmin: vmax = vmin + 1

                    fig3, ax3 = plt.subplots(figsize=(8, 6))
                    merged_gdf.plot(column='arrival_time', cmap='Reds_r', legend=True, vmin=vmin, vmax=vmax,
                                    legend_kwds={'label': "Mean Arrival Time (Days)", 'orientation': "vertical"},
                                    missing_kwds={'color': 'lightgrey', 'label': 'Never Infected'},
                                    ax=ax3)
                    
                    if seed_coords:
                        seed_arrival_time = mean_arrival_times[seed_patch_id]
                        marker_color = 'black'
                        if not np.isnan(seed_arrival_time):
                            color_threshold = vmin + (vmax - vmin) * 0.5
                            if seed_arrival_time <= color_threshold:
                                marker_color = 'white'
                        ax3.scatter(seed_coords[0], seed_coords[1], marker='x', color=marker_color, s=50, lw=2.5, zorder=10)
                    
                    # ax3.set_title(f'Mean epidemic arrival time per district\n({scenario_name_for_file})')
                    ax3.set_title(f'Mean epidemic arrival time')
                    ax3.set_axis_off()
                    
                    filename3 = results_dir / f'arrival_time_map_{scenario_name_for_file}_R{int(cfg.R_0*100)}_beta{int(beta_event*100)}_Iss{cfg.I_ss}.png'
                    plt.savefig(filename3, dpi=300, bbox_inches='tight')
                    plt.close(fig3)
                    print(f"  -> Saved arrival time map to {filename3}")
            except Exception as e:
                print(f"  -> Failed to generate arrival time map: {e}")

    return {
        "mean_affected": mean_affected, 
        "std_affected": std_affected, 
        "affected_values": total_affected.tolist()
    }

def plot_spread_curve_comparison(group_results, data, cfg, group_name):
    """
    Plots the spread curve (Number of affected districts vs Time) for multiple scenarios.
    Visualizes Mean and 25th-75th quantile boundaries.
    """
    print(f"Generating spread curve comparison plot for group: {group_name}...")
    
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
    
    n_patches = data['n_patches']
    n_days = cfg.n_days
    common_time = np.linspace(0, n_days, n_days*2 + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    marker_interval = 20 # Show markers every 20 days to avoid clutter

    for idx, scenario in enumerate(scenarios):
        res = group_results[scenario]
        all_runs = res['all_runs_results']
        n_iterations = len(all_runs)
        
        # Calculate arrival times and spread curves
        arrival_times = _get_arrival_times(all_runs, n_patches, n_iterations)
        spread_curves = np.zeros((n_iterations, len(common_time)))
        
        for r in range(n_iterations):
            arrivals = arrival_times[r, :]
            arrivals = arrivals[~np.isnan(arrivals)]
            if len(arrivals) == 0: continue
            arrivals_sorted = np.sort(arrivals)
            spread_curves[r, :] = np.searchsorted(arrivals_sorted, common_time, side='right')
            
        median_spread = np.median(spread_curves, axis=0)
        low = np.percentile(spread_curves, 25, axis=0)
        high = np.percentile(spread_curves, 75, axis=0)
        
        label = "Baseline" if scenario == 'no_event' else scenario
        color = 'black' if scenario == 'no_event' else colors[idx % len(colors)]
        marker = 'x' if scenario == 'no_event' else markers[(idx-1) % len(markers)]
        zorder = 10 if scenario == 'no_event' else 5

        # Shaded Interquartile Range (25th-75th percentile)
        ax.fill_between(common_time, low, high, color=color, alpha=0.1, zorder=zorder-1)
        
        # Mean line with markers
        mark_indices = [np.abs(common_time - t_val).argmin() for t_val in np.arange(0, n_days + 1, marker_interval)]
        ax.plot(common_time, median_spread, label=label, color=color, marker=marker, 
                markevery=mark_indices, markersize=7, lw=2, zorder=zorder)

    ax.set_xlabel('Time (Days)', fontsize=12)
    ax.set_ylabel('Median Number of Affected Districts', fontsize=12)
    ax.set_title(f'Time to Giant Component (Spatial Percolation Growth): {group_name}\n'
                 f'($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=14)
    
    ax.set_ylim(0, n_patches * 1.05)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle='--', alpha=0.6)
    # ax.legend(loc='upper left', fontsize=10)
    ax.legend(fontsize=10)
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'spread_curve_comparison_{group_name}_R{int(cfg.R_0*100)}_beta{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Spread curve comparison plot saved to '{filename}'")

def plot_affected_districts_vs_beta(data_list, cfg, n_patches=None):
    """
    Plots the Mean Number of Affected Districts vs Beta_Event.
    Generates one plot per (R0, I_ss) combination.
    """
    if not data_list:
        return

    print("Generating affected districts vs beta plots...")
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    df = pd.DataFrame(data_list)
    
    params = df[['R0', 'I_ss']].drop_duplicates()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for _, row in params.iterrows():
        r0, iss = row['R0'], row['I_ss']
        subset = df[(df['R0'] == r0) & (df['I_ss'] == iss)]
        if subset.empty: continue

        # Get baseline for reference
        baseline_subset = subset[subset['scenario_name'] == 'no_event']
        
        scenarios = sorted(subset['scenario_name'].unique())
        fig, ax = plt.subplots(figsize=(8, 6))

        if not baseline_subset.empty:
            base_val = baseline_subset['mean_affected'].iloc[0]
            ax.axhline(base_val, color='black', linestyle='--', label='Baseline (No Event)', zorder=1)

        for idx, scenario in enumerate(scenarios):
            if scenario == 'no_event': continue
            scen_data = subset[subset['scenario_name'] == scenario].sort_values('beta_event')
            color = colors[idx % len(colors)]
            
            ax.errorbar(scen_data['beta_event'], scen_data['mean_affected'], 
                        yerr=scen_data['std_affected'],
                        label=scenario, color=color, marker='o', capsize=5, alpha=0.8, zorder=3)
            
        ax.set_xlabel(r'Event Transmission Rate ($\beta_{event}$)', fontsize=12)
        ax.set_ylabel('Steady State Affected Districts', fontsize=12)
        ax.set_title(f'Geographic Spread vs Event Risk\n($R_0$={r0}, $I_{{ss}}$={iss})', fontsize=14)
        
        if n_patches:
            ax.set_ylim(0, n_patches)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        filename = results_dir / f'affected_districts_vs_beta_R{int(r0*100)}_Iss{iss}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_affected_districts_vs_r0(data_list, cfg, n_patches=None):
    """
    Plots the Mean Number of Affected Districts vs R0.
    Generates one plot per (Beta_Event, I_ss) combination.
    """
    if not data_list:
        return

    print("Generating affected districts vs R0 plots...")
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    df = pd.DataFrame(data_list)
    
    params = df[['beta_event', 'I_ss']].drop_duplicates()
    params = params[params['beta_event'] > 0] 

    for _, row in params.iterrows():
        beta, iss = row['beta_event'], row['I_ss']
        subset = df[((np.isclose(df['beta_event'], beta)) | (df['scenario_name'] == 'no_event')) & (df['I_ss'] == iss)]
        if subset.empty: continue
            
        scenarios = sorted(subset['scenario_name'].unique())
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for idx, scenario in enumerate(scenarios):
            scen_data = subset[subset['scenario_name'] == scenario].sort_values('R0')
            ax.errorbar(scen_data['R0'], scen_data['mean_affected'], 
                        yerr=scen_data['std_affected'],
                        label="Baseline" if scenario == 'no_event' else scenario, 
                        color=colors[idx % len(colors)], marker='o', capsize=5, alpha=0.8, zorder=3)
            
        ax.set_xlabel(r'Basic Reproduction Number ($R_0$)', fontsize=12)
        ax.set_ylabel('Steady State Affected Districts', fontsize=12)
        ax.set_title(f'Geographic Spread vs $R_0$\n($\\beta_{{event}}$={beta}, $I_{{ss}}$={iss})', fontsize=14)
        
        if n_patches:
            ax.set_ylim(0, n_patches)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.savefig(results_dir / f'affected_districts_vs_R0_beta{int(beta*100)}_Iss{iss}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_affected_districts_heatmap(data_list, cfg):
    """Generates heatmaps of mean affected districts vs R0 and Beta_Event."""
    if not data_list: return
    try: import seaborn as sns
    except ImportError: return
    print("Generating affected districts heatmaps...")
    df = pd.DataFrame(data_list)
    for scenario in [s for s in df['scenario_name'].unique() if s != 'no_event']:
        for iss in df['I_ss'].unique():
            subset = df[(df['scenario_name'] == scenario) & (df['I_ss'] == iss)].copy()
            pivot = subset.pivot_table(index='R0', columns='beta_event', values='mean_affected').sort_index(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
            ax.set_title(f'Mean Affected Districts: {scenario} (Iss={iss})')
            plt.savefig(cfg.results_dir / f'affected_districts_heatmap_{scenario}_Iss{iss}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

def plot_arrival_time_variability_scatter(group_results, data, cfg, group_name):
    """
    Generates a scatter plot of Arrival Time IQR vs Mean Arrival Time.
    Each point represents a district. Different markers represent different scenarios.
    """
    print(f"Generating arrival time variability scatter plot for group: {group_name}...")
    
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')

    n_patches = data['n_patches']
    n_iterations = cfg.n_iterations
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, scenario in enumerate(scenarios):
        res = group_results[scenario]
        all_runs = res['all_runs_results']
        
        # Get arrival times (n_iterations, n_patches)
        arrival_times = _get_arrival_times(all_runs, n_patches, n_iterations)

        # Calculate Mean and IQR (75th - 25th) per patch
        with np.errstate(invalid='ignore', divide='ignore'):
            if arrival_times.size > 0 and not np.all(np.isnan(arrival_times)):
                mean_arr = np.nanmean(arrival_times, axis=0)
                std_arr = np.nanstd(arrival_times, axis=0)
            else:
                mean_arr = np.full(n_patches, np.nan)
                std_arr = np.full(n_patches, np.nan)

        # Filter out districts that were never infected (NaN means)
        valid_mask = ~np.isnan(mean_arr)
        x = mean_arr[valid_mask]
        y = std_arr[valid_mask]
        
        label = "Baseline" if scenario == 'no_event' else scenario
        color = 'black' if scenario == 'no_event' else colors[idx % len(colors)]
        marker = 'x' if scenario == 'no_event' else markers[(idx-1) % len(markers)]
        alpha = 0.5 if scenario == 'no_event' else 0.7
        zorder = 10 if scenario == 'no_event' else 5

        ax.scatter(x, y, label=label, color=color, marker=marker, alpha=alpha, s=40, zorder=zorder)

    ax.set_xlabel('Mean Arrival Time (Days)', fontsize=12)
    ax.set_ylabel('Arrival Time Standard Deviation (Days)', fontsize=12)
    ax.set_title(f'Arrival Time Variability: {group_name}\n'
                 f'($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=14)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    # ax.legend(loc='upper right', fontsize=10)
    ax.legend(fontsize=10)
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'arrival_variability_scatter_{group_name}_R{int(cfg.R_0*100)}_beta{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Arrival variability scatter plot saved to '{filename}'")


def plot_arrival_time_variability_grid(group_results, data, cfg, group_name):
    """
    Generates a grid of scatter plots (Mean vs Std Dev) for arrival times across scenarios.
    Scenarios are arranged in a grid to match the style of spatial comparison plots.
    """
    print(f"Generating arrival time variability grid plot for group: {group_name}...")
    
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')

    n_patches = data['n_patches']
    n_iterations = cfg.n_iterations
    
    # Determine grid layout (e.g., 3 columns)
    ncols = 3
    nrows = int(np.ceil(len(scenarios) / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, scenario in enumerate(scenarios):
        ax = axes_flat[idx]
        res = group_results[scenario]
        arrival_times = _get_arrival_times(res['all_runs_results'], n_patches, n_iterations)
        
        with np.errstate(all='ignore'):
            mean_arr = np.nanmean(arrival_times, axis=0)
            std_arr = np.nanstd(arrival_times, axis=0)
            
        valid_mask = ~np.isnan(mean_arr)
        x, y = mean_arr[valid_mask], std_arr[valid_mask]
        
        label = "Baseline" if scenario == 'no_event' else scenario
        color = 'black' if scenario == 'no_event' else colors[idx % len(colors)]
        marker = 'x' if scenario == 'no_event' else markers[(idx-1) % len(markers)]
        
        ax.scatter(x, y, color=color, marker=marker, alpha=0.6, s=30)
        ax.set_title(label, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        if idx >= (nrows - 1) * ncols: ax.set_xlabel('Mean Arrival Time (Days)')
        if idx % ncols == 0: ax.set_ylabel('Arrival Time Std Dev (Days)')

    # Hide unused subplots
    for i in range(len(scenarios), len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(f'Arrival Time Variability Grid: {group_name}\n'
                 f'($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'arrival_variability_grid_{group_name}_R{int(cfg.R_0*100)}_beta{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_arrival_time_variability_by_day_grid(group_results, data, cfg, group_name):
    """
    Generates a grid of scatter plots (Mean vs Std Dev) for arrival times,
    where each subplot represents a specific day.
    Districts are plotted if their mean arrival time is <= the current day.
    Different scenarios are represented by different symbols.
    """
    print(f"Generating arrival time variability by day grid plot for group: {group_name}...")
    
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')

    n_patches = data['n_patches']
    n_iterations = cfg.n_iterations
    
    time_points = getattr(cfg, 'custom_spatial_comparison_days', [100, 150, 200]) # Days to plot
    
    ncols = 3
    nrows = int(np.ceil(len(time_points) / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Pre-calculate mean and std dev for all scenarios and all districts once
    scenario_stats = {}
    for scenario in scenarios:
        res = group_results[scenario]
        all_runs = res['all_runs_results']
        arrival_times = _get_arrival_times(all_runs, n_patches, n_iterations)
        with np.errstate(all='ignore'):
            mean_arr = np.nanmean(arrival_times, axis=0)
            std_arr = np.nanstd(arrival_times, axis=0)
        scenario_stats[scenario] = {'mean': mean_arr, 'std': std_arr}

    # Determine global max for x and y axes for consistent scaling
    global_max_mean = 0
    global_max_std = 0
    for stats in scenario_stats.values():
        global_max_mean = max(global_max_mean, np.nanmax(stats['mean']) if np.any(~np.isnan(stats['mean'])) else 0)
        global_max_std = max(global_max_std, np.nanmax(stats['std']) if np.any(~np.isnan(stats['std'])) else 0)

    # Plotting loop for each day
    for day_idx, current_day in enumerate(time_points):
        ax = axes_flat[day_idx]

        for idx, scenario in enumerate(scenarios):
            stats = scenario_stats[scenario]
            mean_arr = stats['mean']
            std_arr = stats['std']

            # Filter districts: only include those whose mean arrival time is <= current_day
            valid_mask = (~np.isnan(mean_arr)) & (mean_arr <= current_day)
            
            x = mean_arr[valid_mask]
            y = std_arr[valid_mask]

            label = "Baseline" if scenario == 'no_event' else scenario
            color = 'black' if scenario == 'no_event' else colors[idx % len(colors)]
            marker = 'x' if scenario == 'no_event' else markers[(idx-1) % len(markers)]

            ax.scatter(x, y, label=label, color=color, marker=marker, alpha=0.6, s=30)

        ax.set_title(f'Day {current_day}', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.4)

        # Set common labels only for the outer plots
        if day_idx % ncols == 0: # Leftmost column
            ax.set_ylabel('Arrival Time Std Dev (Days)')
        if day_idx >= (nrows - 1) * ncols: # Bottom row
            ax.set_xlabel('Mean Arrival Time (Days)')

    # Set global x and y limits
    for ax in axes_flat:
        ax.set_xlim(0, global_max_mean * 1.1) # Add a small buffer
        ax.set_ylim(0, global_max_std * 1.1)

    # Add a single legend for all subplots
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.05, 0.95), title="Scenario")

    # Hide unused subplots
    for i in range(len(time_points), len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(f'Arrival Time Variability by Day\n($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=16)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95]) # Adjust rect to make space for legend
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'arrival_variability_by_day_grid_{group_name}_R{int(cfg.R_0*100)}_beta{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_active_cases_variability_by_day_grid(group_results, data, cfg, group_name):
    """
    Generates a grid of scatter plots (Mean vs Std Dev) for active cases,
    where each subplot represents a specific day.
    X-axis: Mean active cases (E+I).
    Y-axis: Standard Deviation of active cases (E+I).
    """
    print(f"Generating active cases variability by day grid plot for group: {group_name}...")
    
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')

    n_patches = data['n_patches']
    n_iterations = cfg.n_iterations
    n_days = cfg.n_days
    
    time_points = getattr(cfg, 'custom_spatial_comparison_days', [100, 150, 175, 200])
    
    ncols = len(time_points)
    nrows = 1
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 5 * nrows), sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Pre-calculate mean and std dev for all scenarios and all districts
    scenario_stats = {}
    for scenario in scenarios:
        res = group_results[scenario]
        _, all_runs_active = _get_daily_active_cases(res['all_runs_results'], n_patches, n_iterations, n_days)
        
        mean_active = np.mean(all_runs_active, axis=0) # (n_days+1, n_patches)
        std_active = np.std(all_runs_active, axis=0)   # (n_days+1, n_patches)
        scenario_stats[scenario] = {'mean': mean_active, 'std': std_active}

    for day_idx, current_day in enumerate(time_points):
        ax = axes_flat[day_idx]
        if current_day > n_days: continue

        for idx, scenario in enumerate(scenarios):
            stats = scenario_stats[scenario]
            # Get mean and std for all patches at the specified day index
            x = stats['mean'][current_day, :]
            y = stats['std'][current_day, :]

            label = "Baseline" if scenario == 'no_event' else scenario
            color = 'black' if scenario == 'no_event' else colors[idx % len(colors)]
            marker = 'x' if scenario == 'no_event' else markers[(idx-1) % len(markers)]

            ax.scatter(x, y, label=label, color=color, marker=marker, alpha=0.6, s=30)

        ax.set_title(f'Day {current_day}', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlabel('Mean Active Cases')
        if day_idx == 0:
            ax.set_ylabel('Std Dev of Active Cases')

    # Add a global legend for the figure below the subplots
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.12), 
               ncol=min(len(scenarios), 4), title="Scenario", frameon=True)

    fig.suptitle(f'Active Cases Variability by Day: {group_name}\n'
                 f'($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=16)
    # Use rect=[left, bottom, right, top] to leave space for the legend at the bottom
    fig.tight_layout(rect=[0, 0.15, 1, 0.95])

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'active_cases_variability_by_day_grid_{group_name}_R{int(cfg.R_0*100)}_beta{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_time_to_threshold_comparison(group_results: Dict[str, Any], data: Dict[str, Any], cfg: Any, group_name: str, threshold_per_1000: float = 1.0):
    """
    Plots a boxplot comparison of the time taken to reach a specific threshold of active cases.
    """
    print(f"Generating time to threshold ({threshold_per_1000}/1000) comparison for group: {group_name}...")
    
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
        
    total_population = data['population_df']['population'].sum()
    threshold_value = (threshold_per_1000 / 1000.0) * total_population
    
    results_to_plot = []
    labels = []
    
    # Define styles
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    for idx, scenario in enumerate(scenarios):
        realizations = group_results[scenario]['realizations']
        E = realizations['E']
        I = realizations['I']
        time = E[:, 0]
        # Active cases across all runs (skip time column)
        active_cases = E[:, 1:] + I[:, 1:]
        
        times_to_threshold = []
        for i in range(active_cases.shape[1]):
            # Find first index where active cases >= threshold
            idx_threshold = np.where(active_cases[:, i] >= threshold_value)[0]
            if len(idx_threshold) > 0:
                times_to_threshold.append(time[idx_threshold[0]])
        
        if times_to_threshold:
            results_to_plot.append(times_to_threshold)
            labels.append("Baseline" if scenario == 'no_event' else scenario)

    if not results_to_plot:
        print(f"  -> No runs reached the threshold of {threshold_per_1000}/1000.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(results_to_plot, labels=labels, patch_artist=True, medianprops={'visible': False})

    for i, line in enumerate(bp['medians']):
        median_x = np.mean(line.get_xdata())
        median_y = np.mean(line.get_ydata())
        ax.plot(median_x, median_y, 'rx', markersize=10, markeredgewidth=2.5, zorder=5)
    
    for idx, box in enumerate(bp['boxes']):
        color = 'black' if labels[idx] == "Baseline" else colors[idx % len(colors)]
        box.set_facecolor(color)
        box.set_alpha(0.6)

    ax.set_ylabel('Time to reach threshold (Days)', fontsize=12)
    ax.set_ylim(0, cfg.n_days)
    ax.set_title(f'Days to Reach {threshold_per_1000} Active Cases per 1000: {group_name}\n($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'time_to_threshold_{group_name}_R{int(cfg.R_0*100)}_beta{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Time to threshold plot saved to '{filename}'")

def plot_cumulative_cases_time_to_threshold_comparison(group_results: Dict[str, Any], data: Dict[str, Any], cfg: Any, group_name: str, threshold_per_1000: float = 1.0, show_outliers: bool = True):
    """
    Plots a boxplot comparison of the time taken to reach a specific threshold of cumulative cases.
    """
    if not getattr(cfg, 'plot_on_the_fly', True):
        return

    outlier_str = "" if show_outliers else " (no outliers)"
    print(f"Generating time to threshold ({threshold_per_1000}/1000 cumulative cases) comparison for group: {group_name}{outlier_str}...")
    
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
        
    total_population = data['population_df']['population'].sum()
    threshold_value = (threshold_per_1000 / 1000.0) * total_population
    
    results_to_plot = []
    labels = []
    
    # Define styles
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    for idx, scenario in enumerate(scenarios):
        realizations = group_results[scenario]['realizations']
        # For cumulative cases, we need E, I, and R
        if not all(k in realizations for k in ['E', 'I', 'R']):
            print(f"  -> Skipping scenario '{scenario}': Missing E, I, or R realizations.")
            continue
            
        E = realizations['E']
        I = realizations['I']
        R = realizations['R']
        time = E[:, 0]
        
        # Cumulative cases across all runs (skip time column)
        cumulative_cases = E[:, 1:] + I[:, 1:] + R[:, 1:]
        
        times_to_threshold = []
        for i in range(cumulative_cases.shape[1]): # Iterate through each simulation run
            # Find first index where cumulative cases >= threshold
            idx_threshold = np.where(cumulative_cases[:, i] >= threshold_value)[0]
            if len(idx_threshold) > 0:
                times_to_threshold.append(time[idx_threshold[0]])
            else:
                # If threshold is never reached, record n_days (or NaN, depending on desired behavior)
                times_to_threshold.append(cfg.n_days) 
        
        if times_to_threshold:
            results_to_plot.append(times_to_threshold)
            labels.append("Baseline" if scenario == 'no_event' else scenario)

    if not results_to_plot:
        print(f"  -> No runs reached the threshold of {threshold_per_1000}/1000 cumulative cases.")
        return
    print(f"  -> Plotting cumulative cases time to threshold for {len(results_to_plot)} scenarios.")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bp = ax.boxplot(results_to_plot, labels=labels, patch_artist=True, medianprops={'visible': False}, vert=False, showfliers=show_outliers) # vert=False for horizontal boxplots

    for i, line in enumerate(bp['medians']):
        median_x = np.mean(line.get_xdata())
        median_y = np.mean(line.get_ydata())
        ax.plot(median_x, median_y, 'rx', markersize=10, markeredgewidth=2.5, zorder=5)
    
    for idx, box in enumerate(bp['boxes']):
        color = 'black' if labels[idx] == "Baseline" else colors[idx % len(colors)]
        box.set_facecolor(color)
        box.set_alpha(0.6)

    ax.set_xlabel(f'Time to reach {int(threshold_per_1000)} cumulative cases per 1000 (Days)', fontsize=12)
    if show_outliers:
        ax.set_xlim(0, cfg.n_days) # X-axis is time, so it should go up to n_days
    else:
        ax.set_xlim(left=0)
    ax.set_ylabel('Scenario', fontsize=12)
    ax.set_title(f'Days to Reach {int(threshold_per_1000)} Cumulative Cases per 1000: {group_name}\n($R_0$={cfg.R_0:.2f}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate:.2f})', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6, axis='x') # Grid on x-axis for time
    plt.yticks(rotation=0, ha='right') # Keep y-labels horizontal
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    outlier_suffix = "" if show_outliers else "_no_outliers"
    filename = results_dir / f'cumulative_cases_time_to_threshold_{group_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Cumulative cases time to threshold plot saved to '{filename}'")

def plot_cumulative_active_cases_distribution_at_day(group_results: Dict[str, Any], data: Dict[str, Any], cfg: Any, group_name: str, target_day: float = 150.0, xmax: Optional[float] = None):
    """
    Plots the distribution (Density) of cumulative infections (E+I+R) per 1000 
    at a specific day across different scenarios for comparison.
    """
    if not getattr(cfg, 'plot_on_the_fly', True):
        return

    try:
        from scipy import stats
    except ImportError:
        print("Scipy not found, skipping smoothed distribution plot. Reverting to step histogram.")
        stats = None

    print(f"Generating cumulative cases distribution at day {target_day} for group: {group_name}...")
    
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
    
    total_population = data['population_df']['population'].sum()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    actual_day = target_day
    distributions = {}
    global_max_val = 0
    
    for idx, scenario in enumerate(scenarios):
        realizations = group_results[scenario]['realizations']
        if not all(k in realizations for k in ['E', 'I', 'R']):
            continue
            
        time = realizations['E'][:, 0]
        day_idx = np.abs(time - target_day).argmin()
        actual_day = time[day_idx]
        
        cum_at_day = (realizations['E'][day_idx, 1:] + 
                      realizations['I'][day_idx, 1:] + 
                      realizations['R'][day_idx, 1:])
        
        vals = (cum_at_day / total_population) * 1000
        distributions[scenario] = vals
        if len(vals) > 0:
            global_max_val = max(global_max_val, vals.max())

    # Calculate Baseline mean for reference line
    baseline_mean = np.mean(distributions['no_event']) if 'no_event' in distributions else None

    # Shared x-range for smooth curves
    plot_xmax = xmax if xmax is not None else (global_max_val * 1.2 if global_max_val > 0 else 10)
    x_range = np.linspace(0, plot_xmax, 500)

    # Plotting loop
    for idx, scenario in enumerate(scenarios):
        if scenario not in distributions: continue
        
        vals = distributions[scenario]
        label = "Baseline" if scenario == 'no_event' else scenario
        color = 'black' if scenario == 'no_event' else colors[idx % len(colors)]

        # Overlay Smoothed distributions (KDE)
        if stats and len(vals) > 1 and np.var(vals) > 0:
            kde = stats.gaussian_kde(vals)
            density = kde(x_range)
            ax.plot(x_range, density, color=color, lw=2.5, label=label)
            ax.fill_between(x_range, 0, density, color=color, alpha=0.15)
        elif len(vals) > 0:
            # Fallback to step histogram if KDE fails or not enough data
            ax.hist(vals, bins='auto', density=True, histtype='step', color=color, lw=2.5, label=label)

    # 3. Reference Line (Baseline Mean)
    if baseline_mean is not None:
        ax.axvline(baseline_mean, color='red', linestyle='--', linewidth=1.5, alpha=0.8, 
                   label=f'Baseline Mean ({baseline_mean:.2f})')

    # Formatting
    ax.set_xlabel('Cumulative Infections per 1000 individuals', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Distribution of Cumulative Cases at Day {actual_day}\n'
                 f'($R_0$={cfg.R_0:.2f}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate:.2f})', fontsize=14)
    
    ax.legend(loc='upper right', fontsize=10)
    # Ensure Y-axis values are clearly indicated and well-formatted
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='plain')

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0)
    if xmax is not None:
        ax.set_xlim(0, xmax)
    else:
        ax.set_xlim(left=0)
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'cumulative_cases_distribution_day{int(actual_day)}_{group_name}_R{int(cfg.R_0*100)}_beta{int(cfg.event_base_transmission_rate*100)}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved cumulative distribution plot to {filename}")

def plot_cumulative_active_cases_boxplot_at_day(group_results: Dict[str, Any], data: Dict[str, Any], cfg: Any, group_name: str, target_day: float = 250.0, vmax: Optional[float] = None, show_outliers: bool = False):
    """
    Generates a boxplot comparison of cumulative infections (E+I+R) per 1000 
    at a specific day across different scenarios.
    """
    if not getattr(cfg, 'plot_on_the_fly', True):
        return

    outlier_str = "" if show_outliers else " (no outliers)"
    print(f"Generating cumulative cases boxplot at day {target_day} for group: {group_name}{outlier_str}...")
    
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
    
    total_population = data['population_df']['population'].sum()
    
    plot_data = []
    labels = []
    
    actual_day = target_day
    for scenario in scenarios:
        realizations = group_results[scenario]['realizations']
        if not all(k in realizations for k in ['E', 'I', 'R']):
            continue
            
        time = realizations['E'][:, 0]
        day_idx = np.abs(time - target_day).argmin()
        actual_day = time[day_idx]
        
        cum_at_day = (realizations['E'][day_idx, 1:] + 
                      realizations['I'][day_idx, 1:] + 
                      realizations['R'][day_idx, 1:])
        
        vals = (cum_at_day / total_population) * 1000
        plot_data.append(vals)
        labels.append("Baseline" if scenario == 'no_event' else scenario)

    if not plot_data:
        return

    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    bp = ax.boxplot(plot_data, labels=labels, vert=False, patch_artist=True, medianprops={'visible': False}, showfliers=show_outliers)

    for i, line in enumerate(bp['medians']):
        median_x = np.mean(line.get_xdata())
        median_y = np.mean(line.get_ydata())
        ax.plot(median_x, median_y, 'rx', markersize=10, markeredgewidth=2.5, zorder=5)
    
    for idx, box in enumerate(bp['boxes']):
        color = 'black' if labels[idx] == "Baseline" else colors[idx % len(colors)]
        box.set_facecolor(color)
        box.set_alpha(0.6)

    ax.set_xlabel('Cumulative Infections per 1000 individuals', fontsize=12)
    ax.set_ylabel('Scenario', fontsize=12)
    if vmax is not None and show_outliers:
        ax.set_xlim(0, vmax)
    else:
        ax.set_xlim(left=0)
    ax.set_title(f'Distribution of Cumulative Cases at Day {int(actual_day)}\n'
                 f'($R_0$={cfg.R_0:.2f}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate:.2f})', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6, axis='x')
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    outlier_suffix = "" if show_outliers else "_no_outliers"
    filename = results_dir / f'cumulative_cases_boxplot_day{int(actual_day)}_{group_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved cumulative boxplot to {filename}")

def plot_active_cases_boxplot_at_day(group_results: Dict[str, Any], data: Dict[str, Any], cfg: Any, group_name: str, target_day: float = 150.0, vmax: Optional[float] = None):
    """
    Generates a boxplot comparison of active cases (E+I) per 1000 
    at a specific day across different scenarios.
    """
    if not getattr(cfg, 'plot_on_the_fly', True):
        return

    print(f"Generating active cases boxplot at day {target_day} for group: {group_name}...")
    
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
    
    total_population = data['population_df']['population'].sum()
    
    plot_data = []
    labels = []
    
    actual_day = target_day
    for scenario in scenarios:
        realizations = group_results[scenario]['realizations']
        if not all(k in realizations for k in ['E', 'I']):
            continue
            
        time = realizations['E'][:, 0]
        day_idx = np.abs(time - target_day).argmin()
        actual_day = time[day_idx]
        
        active_at_day = (realizations['E'][day_idx, 1:] + 
                        realizations['I'][day_idx, 1:])
        
        vals = (active_at_day / total_population) * 1000
        plot_data.append(vals)
        labels.append("Baseline" if scenario == 'no_event' else scenario)

    if not plot_data:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    bp = ax.boxplot(plot_data, labels=labels, vert=False, patch_artist=True, medianprops={'visible': False})

    for i, line in enumerate(bp['medians']):
        median_x = np.mean(line.get_xdata())
        median_y = np.mean(line.get_ydata())
        ax.plot(median_x, median_y, 'rx', markersize=10, markeredgewidth=2.5, zorder=5)
    
    for idx, box in enumerate(bp['boxes']):
        color = 'black' if labels[idx] == "Baseline" else colors[idx % len(colors)]
        box.set_facecolor(color)
        box.set_alpha(0.6)

    ax.set_xlabel('Active Cases per 1000 individuals', fontsize=12)
    ax.set_ylabel('Scenario', fontsize=12)
    if vmax is not None:
        ax.set_xlim(0, vmax)
    ax.set_title(f'Distribution of Active Cases at Day {actual_day}\n'
                 f'($R_0$={cfg.R_0:.2f}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate:.2f})', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6, axis='x')
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'active_cases_boxplot_day{int(actual_day)}_{group_name}_R{int(cfg.R_0*100)}_beta{int(cfg.event_base_transmission_rate*100)}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved active cases boxplot to {filename}")

def plot_cumulative_active_cases_hist_at_day(group_results: Dict[str, Any], data: Dict[str, Any], cfg: Any, group_name: str, target_day: float = 250.0, xmax: Optional[float] = None, use_log_x: bool = False):
    """
    Generates histograms of the distribution of cumulative active cases (E+I+R) per 1000
    at a specific day across different scenarios.
    Subplots are arranged in 3 columns, 2 rows.
    """
    if not getattr(cfg, 'plot_on_the_fly', True):
        return

    print(f"Generating cumulative cases histogram at day {target_day} for group: {group_name}...")
    
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
    
    total_population = data['population_df']['population'].sum()
    
    plot_data_by_scenario = {}
    actual_day = target_day
    
    # First pass: Collect data for all scenarios
    for scenario in scenarios:
        realizations = group_results[scenario]['realizations']
        if not all(k in realizations for k in ['E', 'I', 'R']):
            continue
            
        time = realizations['E'][:, 0]
        day_idx = np.abs(time - target_day).argmin()
        actual_day = time[day_idx]
        
        cum_at_day = (realizations['E'][day_idx, 1:] + 
                      realizations['I'][day_idx, 1:] + 
                      realizations['R'][day_idx, 1:])
        
        vals = (cum_at_day / total_population) * 1000
        plot_data_by_scenario[scenario] = vals

    # Determine bins and x-range globally for consistency across subplots
    all_vals_list = list(plot_data_by_scenario.values())
    flattened = np.concatenate(all_vals_list) if all_vals_list else np.array([])
    
    if use_log_x:
        # Ensure v_min is not zero for log scale
        v_min_candidate = np.min(flattened[flattened > 0]) if flattened[flattened > 0].size > 0 else 0.1
        v_min = max(v_min_candidate, 1e-3) # Ensure a reasonable minimum for log scale
        v_max = xmax if xmax else (np.max(flattened) if flattened.size > 0 else 10)
        if v_max <= v_min: v_max = v_min * 10 # Ensure v_max is greater than v_min

        if v_min > 0 and v_max > v_min:
            bins = np.logspace(np.log10(v_min), np.log10(v_max), 51)
        else:
            v_min, v_max, bins = 0.1, 10, 50 # Fallback
        
        # Calculate the maximum relative frequency across all scenarios to set a shared Y-axis
        # Calculate the shared maximum relative frequency across all scenarios
        global_hist_ymax = 0
        for scenario_vals in all_vals_list:
            if len(scenario_vals) > 0:
                # Compute histogram heights using the relative frequency weights
                # Compute histogram using the relative frequency weights
                hist_heights, _ = np.histogram(
                    scenario_vals, 
                    bins=bins, 
                    weights=np.ones_like(scenario_vals) / len(scenario_vals)
                )
                global_hist_ymax = max(global_hist_ymax, np.max(hist_heights))
        
        # Apply a 10% margin and a minimum floor for visibility
        global_hist_ymax = max(global_hist_ymax * 1.1, 0.05)
    else:
        bins = 50
        # Use a global range for linear bins and axes to ensure consistency across subplots
        v_min, v_max = 0, (xmax if xmax else (np.max(flattened) if flattened.size > 0 else 10))
        bins = np.linspace(v_min, v_max, 51)
        # For linear plots, Y-range is fixed at [0, 1.0] as previously requested
        global_hist_ymax = 1.0

    # Define the threshold for visual reference (x = 1 case per 1000)
    threshold_val = 1.0

    # Second pass: Plotting (3 columns, 2 rows)
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 5.5), squeeze=False)
    axes_flat = axes.flatten()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, scenario in enumerate(scenarios):
        if scenario not in plot_data_by_scenario: continue
        
        ax = axes_flat[idx]
        vals = plot_data_by_scenario[scenario]
        label = "Baseline" if scenario == 'no_event' else scenario
        color = 'black' if scenario == 'no_event' else colors[idx % len(colors)]
        
        weights = np.ones_like(vals) / len(vals)
        ax.hist(vals, bins=bins, weights=weights, color=color, alpha=0.7, edgecolor='black')
        ax.set_title(label, fontsize=14)
        
        if use_log_x: 
            ax.set_xscale('log')
            ax.axvline(threshold_val, color='red', linestyle='--', linewidth=1.5, alpha=0.8, zorder=5)
            
            # Calculate and display the percentage of runs below the threshold
            pct_below = (np.sum(vals < threshold_val) / len(vals)) * 100
            # ax.text(0.05, 0.7, f"below threshold\n{pct_below:.1f}%", transform=ax.transAxes, color='blue', fontsize=14) # fontweight='bold', 
            ax.text(0.05, 0.65, f"aboved\nthreshold\n{100-pct_below:.1f}%", transform=ax.transAxes, color='blue', fontsize=12) # fontweight='bold',             
        # Strictly enforce x and y ranges calculated globally for the figure
        ax.set_xlim(v_min, v_max)
        ax.set_ylim(0, global_hist_ymax)
        ax.grid(True, linestyle='--', alpha=0.6)

    fig.subplots_adjust(bottom=0.14, top=0.82, hspace=0.45, wspace=0.25, left=0.10, right=0.98)
    fig.suptitle(f'Distribution of Cumulative Cases at Day {int(actual_day)}\n'
                 f'($R_0$={cfg.R_0:.2f}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate:.2f})', fontsize=16)
    fig.supxlabel('Cumulative Cases per 1000 individuals', fontsize=12)
    fig.supylabel('Relative Frequency', fontsize=12)
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    log_suffix = "_logx" if use_log_x else ""
    filename = results_dir / f'cumulative_cases_histogram_day{int(actual_day)}_{group_name}{log_suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved cumulative histogram to {filename}")

def save_cumulative_case_proportion_below_threshold(group_results: Dict[str, Any], data: Dict[str, Any], cfg: Any, target_day: float, threshold: float = 1.0):
    """
    Calculates the percentage of simulation results below a threshold for cumulative cases per 1000
    at a specific day for all scenarios and records it in a text file.
    """
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / "cumulative_case_proportion_below_threshold.txt"
    
    total_population = data['population_df']['population'].sum()
    
    # Determine scenario order: Baseline ('no_event') first, then others alphabetically
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
    
    results_to_record = []
    scenario_labels = []
    
    for scenario in scenarios:
        realizations = group_results[scenario]['realizations']
        if not all(k in realizations for k in ['E', 'I', 'R']):
            continue
            
        time = realizations['E'][:, 0]
        day_idx = np.abs(time - target_day).argmin()
        
        cum_at_day = (realizations['E'][day_idx, 1:] + 
                      realizations['I'][day_idx, 1:] + 
                      realizations['R'][day_idx, 1:])
        
        vals = (cum_at_day / total_population) * 1000
        pct_below = (np.sum(vals < threshold) / len(vals)) * 100
        
        results_to_record.append(f"{pct_below:.2f}")
        scenario_labels.append("baseline" if scenario == 'no_event' else scenario)

    if not results_to_record:
        return

    file_exists = filename.exists() and filename.stat().st_size > 0
    with open(filename, 'a') as f:
        if not file_exists:
            header = "I_ss,R_0,beta_event," + ",".join(scenario_labels) + "\n"
            f.write(header)
        
        line = f"{cfg.I_ss},{cfg.R_0:.2f},{cfg.event_base_transmission_rate:.2f}," + ",".join(results_to_record) + "\n"
        f.write(line)

def plot_active_case_curves_only(realizations: Dict[str, np.ndarray], total_population: float, cfg: Any):
    """
    Generates a plot showing the mean active cases (E+I) per 1000 individuals,
    overlayed on individual stochastic realizations.
    """
    if not getattr(cfg, 'plot_on_the_fly', True):
        return

    if 'E' not in realizations or 'I' not in realizations:
        return
        
    E = realizations['E']
    I = realizations['I']
    time = E[:, 0]

    # Filter for successful runs: cumulative infections at the end > I_ss
    if 'R' in realizations:
        successful_runs_mask = realizations['R'][-1, 1:] > cfg.I_ss
    else:
        successful_runs_mask = np.max(E[:, 1:] + I[:, 1:], axis=0) > 0

    if not np.any(successful_runs_mask):
        print(f"  -> No successful runs found. Skipping active case curves plot.")
        return

    # Filter to include only successful runs
    E_filtered = E[:, 1:][:, successful_runs_mask]
    I_filtered = I[:, 1:][:, successful_runs_mask]
    n_runs = np.sum(successful_runs_mask)
    
    # Calculate mean E+I per 1000 (incidence per 1000)
    # Columns 1 to end are the stochastic realizations
    EI_runs = ((E_filtered + I_filtered) / total_population) * 1000
    median_EI = np.median(EI_runs, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot individual runs in a lighter color
    for i in range(EI_runs.shape[1]):
        ax.plot(time, EI_runs[:, i], color='thistle', alpha=0.1, lw=0.5)

    ax.plot(time, median_EI, color='purple', lw=3, label='Median Active Cases (E+I)')
    
    ax.set_title('active cases per 1000', fontsize=16)
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Active Cases per 1000', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    scenario_name = getattr(cfg, 'current_event_scenario_name', cfg.event_model_type)
    if cfg.event_model_type.startswith('no_event'):
        scenario_name = cfg.event_model_type
        
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    r0_str = f"{int(cfg.R_0*100)}"
    beta_str = f"{int(cfg.event_base_transmission_rate*100)}"
    iss_str = f"{cfg.I_ss}"
    
    filename = results_dir / f"active_case_curves_{scenario_name}_R{r0_str}_beta{beta_str}_Iss{iss_str}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Active case curves plot saved to '{filename}'")

def plot_arrival_time_spatial_clusters_grid(group_results: Dict[str, Any], data: Dict[str, Any], cfg: Any, group_name: str, cluster_threshold: float = 4.0):
    """
    Generates a 2x3 grid of spatial maps showing district clusters based on mean arrival time.
    Row 1: Baseline, AMS_dance, AMS_football
    Row 2: Leipzig_1, Leipzig_2, Leipzig_3
    """
    if not HAS_GEOPANDAS or not HAS_CLUSTERING:
        return

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    print(f"Generating arrival time spatial clusters grid for group: {group_name} (threshold={cluster_threshold}d)...")

    # Layout configuration
    scenarios_to_plot = [
        ['no_event', 'AMS_dance', 'AMS_football'],
        ['Leipzig_1', 'Leipzig_2', 'Leipzig_3']
    ]
    scenario_titles = {
        "no_event": "Baseline",
        "AMS_dance": "AMS Dance",
        "AMS_football": "AMS Football",
        "Leipzig_1": "Leipzig 1",
        "Leipzig_2": "Leipzig 2",
        "Leipzig_3": "Leipzig 3"
    }

    # Load GeoJSON
    mobility_dir = Path(f"data_mobility_{cfg.dataset_name}")
    geojson_path = mobility_dir / f"{cfg.dataset_name.lower()}-districts.geojson"
    if not geojson_path.exists() and mobility_dir.exists():
        geojsons = list(mobility_dir.glob("*.geojson"))
        if geojsons: geojson_path = geojsons[0]
    
    if not geojson_path.exists(): return
    
    try:
        geodf = gpd.read_file(geojson_path)
        id_col = next((c for c in ['id', 'patch_id', 'ID', 'cartodb_id'] if c in geodf.columns), None)
        if id_col:
            if id_col != 'id': geodf = geodf.rename(columns={id_col: 'id'})
            geodf['id'] = pd.to_numeric(geodf['id'], errors='coerce')
            if geodf['id'].min() == 1 and geodf['id'].max() == data['n_patches']:
                geodf['id'] = geodf['id'] - 1
    except Exception: return

    n_patches = data['n_patches']
    n_iterations = cfg.n_iterations
    seed_patch_id = getattr(cfg, 'initial_infection_patch_id', 0)
    seed_coords = None
    try:
        seed_row = geodf[geodf['id'] == seed_patch_id]
        if not seed_row.empty:
            c = seed_row.to_crs(epsg=3857).centroid.to_crs(geodf.crs).iloc[0]
            seed_coords = (c.x, c.y)
    except Exception: pass

    # Setup Colors (consistent across subplots)
    cmap_tab = plt.get_cmap('tab10')
    cluster_colors = {i: cmap_tab(i-1) for i in range(1, 11)} 

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    max_cluster_found = 0

    for r in range(2):
        for c in range(3):
            ax = axes[r, c]
            scenario = scenarios_to_plot[r][c]
            if scenario not in group_results:
                ax.axis('off'); continue

            res = group_results[scenario]
            arr_times = _get_arrival_times(res['all_runs_results'], n_patches, n_iterations)
            with np.errstate(invalid='ignore', divide='ignore'):
                if arr_times.size > 0 and not np.all(np.isnan(arr_times)):
                    mean_times = np.nanmean(arr_times, axis=0)
                else:
                    mean_times = np.full(n_patches, np.nan)
            valid_mask = ~np.isnan(mean_times)
            if not np.any(valid_mask):
                ax.axis('off'); continue
            
            # Filter for clustering: districts infected BUT NOT the seed
            cluster_mask = valid_mask.copy()
            if 0 <= seed_patch_id < n_patches:
                cluster_mask[seed_patch_id] = False

            if np.any(cluster_mask):
                cluster_patch_ids = np.where(cluster_mask)[0]
                cluster_vals = mean_times[cluster_mask].reshape(-1, 1)
                if len(cluster_vals) > 1:
                    Z = hierarchy.linkage(cluster_vals, method='single')
                    clusters = hierarchy.fcluster(Z, t=cluster_threshold, criterion='distance')
                else:
                    clusters = np.ones(len(cluster_vals), dtype=int)
                    
                # Re-map clusters chronologically based on their mean arrival time
                cluster_means = {cid: np.mean(cluster_vals[clusters == cid]) for cid in np.unique(clusters)}
                sorted_cluster_ids = sorted(cluster_means.keys(), key=lambda x: cluster_means[x])
                cluster_mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(sorted_cluster_ids)}
                mapped_clusters = np.array([cluster_mapping[cid] for cid in clusters])
                max_cluster_found = max(max_cluster_found, mapped_clusters.max())
                cluster_df = pd.DataFrame({'id': cluster_patch_ids, 'cluster': mapped_clusters})
            else:
                cluster_df = pd.DataFrame(columns=['id', 'cluster'])
                mapped_clusters = np.array([])
            merged = geodf.merge(cluster_df, on='id', how='left')
            geodf.plot(ax=ax, color='#f0f0f0', edgecolor='0.8', linewidth=0.5)
            for cluster_id in np.unique(mapped_clusters):
                merged[merged['cluster'] == cluster_id].plot(ax=ax, color=cluster_colors.get(cluster_id, 'grey'), edgecolor='0.3', linewidth=0.5)

            if seed_coords:
                ax.scatter(seed_coords[0], seed_coords[1], marker='x', color='black', s=50, lw=2, zorder=10)
            ax.set_title(scenario_titles.get(scenario, scenario), fontsize=16)
            ax.set_axis_off()

    # Unified legend
    legend_elements = [Patch(facecolor=cluster_colors[i], label=f'Cluster {i}') for i in range(1, max_cluster_found + 1)]
    legend_elements.append(Line2D([0], [0], marker='x', color='black', label='Initial Seed', markersize=8, linestyle='None'))
    fig.legend(handles=legend_elements, loc='lower center', ncol=max_cluster_found + 1, bbox_to_anchor=(0.5, 0.02), frameon=True, fontsize=14)
    
    # fig.suptitle(f'Spatial Clusters of Mean Arrival Time (Threshold={int(cluster_threshold)}d)\n($R_0$={cfg.R_0}, $\\beta_{{event}}$={cfg.event_base_transmission_rate}, $I_{{ss}}$={cfg.I_ss})', fontsize=16)
    fig.suptitle(f'Spatial Clusters of Mean Arrival Time\n($R_0$={cfg.R_0}, $\\beta_{{event}}$={cfg.event_base_transmission_rate}, $I_{{ss}}$={cfg.I_ss})', fontsize=18)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    filename = results_dir / f'arrival_time_spatial_clusters_grid_{group_name}_R{int(cfg.R_0*100)}_beta{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close(fig)

def plot_arrival_time_comparison_grid(group_results: Dict[str, Any], data: Dict[str, Any], cfg: Any, group_name: str, metric: str = 'mean'):
    """
    Generates a 2x3 grid of spatial maps for arrival time metric (mean or std).
    Row 1: Baseline, AMS_dance, AMS_football
    Row 2: Leipzig_1, Leipzig_2, Leipzig_3
    """
    if not HAS_GEOPANDAS: return
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    scenarios_to_plot = [['no_event', 'AMS_dance', 'AMS_football'], ['Leipzig_1', 'Leipzig_2', 'Leipzig_3']]
    scenario_titles = {"no_event": "Baseline", "AMS_dance": "AMS Dance", "AMS_football": "AMS Football",
                       "Leipzig_1": "Leipzig 1", "Leipzig_2": "Leipzig 2", "Leipzig_3": "Leipzig 3"}

    mobility_dir = Path(f"data_mobility_{cfg.dataset_name}")
    geojson_path = mobility_dir / f"{cfg.dataset_name.lower()}-districts.geojson"
    if not geojson_path.exists() and mobility_dir.exists():
        geojsons = list(mobility_dir.glob("*.geojson"))
        if geojsons: geojson_path = geojsons[0]
    if not geojson_path.exists(): return
    try:
        geodf = gpd.read_file(geojson_path)
        id_col = next((c for c in ['id', 'patch_id', 'ID', 'cartodb_id'] if c in geodf.columns), None)
        if id_col:
            if id_col != 'id': geodf = geodf.rename(columns={id_col: 'id'})
            geodf['id'] = pd.to_numeric(geodf['id'], errors='coerce')
            if geodf['id'].min() == 1 and geodf['id'].max() == data['n_patches']: geodf['id'] = geodf['id'] - 1
    except Exception: return

    n_patches, n_iterations = data['n_patches'], cfg.n_iterations
    seed_patch_id = getattr(cfg, 'initial_infection_patch_id', 0)
    seed_coords = None
    try:
        seed_row = geodf[geodf['id'] == seed_patch_id]
        if not seed_row.empty:
            c = seed_row.to_crs(epsg=3857).centroid.to_crs(geodf.crs).iloc[0]
            seed_coords = (c.x, c.y)
    except Exception: pass

    all_stats = {}; global_vmax = 0.0
    for row_scens in scenarios_to_plot:
        for sc in row_scens:
            if sc in group_results:
                res = group_results[sc]
                arr_times = _get_arrival_times(res['all_runs_results'], n_patches, n_iterations)
                with np.errstate(invalid='ignore', divide='ignore'):
                    if arr_times.size > 0 and not np.all(np.isnan(arr_times)):
                        vals = np.nanmean(arr_times, axis=0) if metric == 'mean' else np.nanstd(arr_times, axis=0)
                    else:
                        vals = np.full(n_patches, np.nan)
                all_stats[sc] = vals
                if np.any(~np.isnan(vals)): global_vmax = max(global_vmax, np.nanmax(vals))
    if not all_stats: return
    if global_vmax == 0: global_vmax = 1.0

    cmap = 'Reds_r' if metric == 'mean' else 'Blues'
    fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    for r in range(2):
        for c in range(3):
            ax = axes[r, c]
            sc = scenarios_to_plot[r][c]
            if sc not in all_stats: ax.axis('off'); continue
            vals = all_stats[sc]
            merged = geodf.merge(pd.DataFrame({'id': range(n_patches), 'stat': vals}), on='id', how='left')
            merged.plot(column='stat', cmap=cmap, vmin=0, vmax=global_vmax, ax=ax, edgecolor='0.3', linewidth=0.5, missing_kwds={'color': '#f0f0f0'})
            if seed_coords:
                m_color = 'black'
                if metric == 'mean' and not np.isnan(vals[seed_patch_id]) and vals[seed_patch_id] <= global_vmax * 0.5: m_color = 'white'
                ax.scatter(seed_coords[0], seed_coords[1], marker='x', color=m_color, s=40, lw=1.5, zorder=10)
            ax.set_title(scenario_titles.get(sc, sc), fontsize=12); ax.axis('off')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=global_vmax))
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    label = 'Mean Arrival Time (Days)' if metric == 'mean' else 'Arrival Time Std Dev (Days)'
    fig.colorbar(sm, cax=cbar_ax).set_label(label, fontsize=12)
    title_main = 'Mean Epidemic Arrival Time Comparison' if metric == 'mean' else 'Arrival Time Variability Comparison'
    fig.suptitle(f'{title_main}\n($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=16)
    # Note: tight_layout omitted due to externally positioned colorbar with add_axes
    fname = results_dir / f'arrival_time_{metric}_comparison_grid_{group_name}_R{int(cfg.R_0*100)}_beta{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight'); plt.close(fig)


def _calculate_hoover_index_runs(all_runs_results: List[Dict[str, Any]], populations: np.ndarray) -> np.ndarray:
    """
    Vectorized calculation of Hoover Index for all runs and time steps.
    Returns: (n_steps, n_iterations)
    """
    n_iterations = len(all_runs_results)
    if n_iterations == 0:
        return np.array([])
    
    t = all_runs_results[0]['t']
    n_steps = len(t)
    
    hoover_runs = np.zeros((n_steps, n_iterations))
    total_pop = np.sum(populations)
    if total_pop <= 0:
        return hoover_runs
        
    pop_share = populations / total_pop
    
    for run_id in range(n_iterations):
        run_res = all_runs_results[run_id]
        # Sum E_j and I_j. In post-sim reconstruction, E_j contains active cases and I_j is 0.
        active_j = np.array(run_res['E_j']) + np.array(run_res['I_j']) # (n_steps, n_patches)
        
        total_active_per_step = active_j.sum(axis=1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # Calculate share of total cases in each patch at each time step
            case_share = active_j / total_active_per_step[:, np.newaxis]
            case_share = np.nan_to_num(case_share)
            
        # Hoover Index = 0.5 * sum(|case_share_i - pop_share_i|)
        hoover_per_step = 0.5 * np.sum(np.abs(case_share - pop_share), axis=1)
        # If there are no active cases, index is 0 (uniform non-infection)
        hoover_per_step[total_active_per_step <= 0] = 0.0
        hoover_runs[:, run_id] = hoover_per_step
        
    return hoover_runs

def plot_hoover_index_comparison(group_results: Dict[str, Any], data: Dict[str, Any], cfg: Any, group_name: str):
    """
    Plots the time evolution of the mean locational Hoover Index for a group of scenarios.
    """
    print(f"Generating Hoover index comparison plot for group: {group_name}...")
    
    scenarios = list(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.sort()
        scenarios.insert(0, 'no_event')
    
    pop_df = data['population_df'].sort_values('id')
    district_pops = pop_df['population'].values.astype(float)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    marker_interval = 20

    for idx, scenario in enumerate(scenarios):
        results = group_results[scenario]
        all_runs = results['all_runs_results']
        
        t = all_runs[0]['t']
        hoover_runs = _calculate_hoover_index_runs(all_runs, district_pops)
        
        # Calculate mean across all stochastic realizations
        hoover_mean = np.mean(hoover_runs, axis=1) * 100 # Convert to percentage (0-100%)
        
        label = "Baseline" if scenario == 'no_event' else scenario
        color = 'black' if scenario == 'no_event' else colors[idx % len(colors)]
        marker = 'x' if scenario == 'no_event' else markers[(idx-1) % len(markers)]
        
        # Plot mean trend with markers at intervals
        mark_indices = [np.abs(t - t_val).argmin() for t_val in np.arange(0, t[-1] + 1, marker_interval)]
        ax.plot(t, hoover_mean, label=label, color=color, linewidth=2.5, 
                marker=marker, markevery=mark_indices, markersize=8)

    ax.set_xlabel('Time (Days)', fontsize=14)
    ax.set_ylabel('Locational Hoover Index (%)', fontsize=14)
    ax.set_title(f'Epidemic Spatial Concentration\n'
                 f'($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=16)
    
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=12)

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'hoover_index_comparison_{group_name}_R{int(cfg.R_0*100)}_beta{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved Hoover index comparison plot to '{filename}'")

def plot_district_active_curves_grid(all_runs_results: List[Dict[str, Any]], data: Dict[str, Any], cfg: Any, scenario: str, use_log_scale: bool = False):
    """
    Generates a grid of plots showing active case curves for each district.
    Ordered by district ID.
    """
    if not all_runs_results:
        return

    # all_runs_results[0]['E_j'] is (n_days, n_patches)
    _, n_districts = all_runs_results[0]['E_j'].shape
    t = all_runs_results[0]['t']

    pop_df = data['population_df'].sort_values('id')
    district_pops = pop_df['population'].values

    n_cols = 6
    n_rows = (n_districts + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 2 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    global_max_y = 0.0
    for i in range(n_districts):
        ax = axes_flat[i]
        pop_i = district_pops[i]
        trajectories = np.stack([(run['E_j'][:, i] / pop_i) * 1000 for run in all_runs_results])
        
        for traj in trajectories:
            ax.plot(t, traj, color='gray', alpha=0.2, linewidth=0.5)
        
        mean_traj = np.mean(trajectories, axis=0)
        global_max_y = max(global_max_y, np.max(trajectories))
        ax.plot(t, mean_traj, color='red', linewidth=1.5)
        
        ax.set_title(f"District {i}", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        if use_log_scale:
            ax.set_yscale('log')
        
    for j in range(n_districts, len(axes_flat)):
        axes_flat[j].axis('off')

    title_scale = " (Log Scale)" if use_log_scale else ""
    fig.suptitle(f"Active Case Curves by District - {scenario}\n"
                 f"($R_0$={cfg.R_0:.2f}, $\\beta_{{event}}$={cfg.event_base_transmission_rate:.2f}){title_scale}", fontsize=14)
    fig.supxlabel("Day", fontsize=12, y=0.06)
    fig.supylabel("Active Cases per 1000", fontsize=12)
    plt.tight_layout(rect=[0.03, 0.08, 1, 0.95])
    
    if use_log_scale:
        floor = 0.01 
        upper_lim = max(global_max_y * 1.5, floor * 10)
        for ax in axes_flat[:n_districts]: ax.set_ylim(floor, upper_lim)
    else:
        for ax in axes_flat[:n_districts]: ax.set_ylim(0, global_max_y * 1.1)
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    suffix = "_log" if use_log_scale else ""
    out_path = results_dir / f"district_active_curves_{scenario}_R{int(cfg.R_0*100)}_beta{int(cfg.event_base_transmission_rate*100)}{suffix}.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_district_active_comparison_grid(group_results: Dict[str, Any], data: Dict[str, Any], cfg: Any, group_name: str, use_log_scale: bool = False):
    """
    Generates a grid of plots comparing the mean active case ratios of all scenarios for each district.
    """
    if not group_results:
        return

    first_scen = next(iter(group_results))
    all_runs_0 = group_results[first_scen]['all_runs_results']
    _, n_districts = all_runs_0[0]['E_j'].shape
    t = all_runs_0[0]['t']

    pop_df = data['population_df'].sort_values('id')
    district_pops = pop_df['population'].values

    n_cols = 6
    n_rows = (n_districts + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 1.8 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    global_max_y = 0.0
    scenarios = sorted(group_results.keys())
    if 'no_event' in scenarios:
        scenarios.remove('no_event')
        scenarios.insert(0, 'no_event')

    for i in range(n_districts):
        ax = axes_flat[i]
        pop_i = district_pops[i]
        for sc in scenarios:
            all_runs = group_results[sc]['all_runs_results']
            trajectories = np.stack([(run['E_j'][:, i] / pop_i) * 1000 for run in all_runs])
            mean_traj = np.mean(trajectories, axis=0)
            global_max_y = max(global_max_y, np.max(mean_traj))
            label = "Baseline" if sc == 'no_event' else sc
            ax.plot(t, mean_traj, label=label, linewidth=1.5)
        ax.set_title(f"District {i}", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        if use_log_scale: ax.set_yscale('log')

    for j in range(n_districts, len(axes_flat)): axes_flat[j].axis('off')

    title_scale = " (Log Scale)" if use_log_scale else ""
    fig.suptitle(f"District Mean Active Cases per 1000 Comparison: {group_name}\n"
                 f"($R_0$={cfg.R_0:.2f}, $\\beta_{{event}}$={cfg.event_base_transmission_rate:.2f}){title_scale}", fontsize=16)
    fig.supxlabel("Day", fontsize=12, y=0.11)
    fig.supylabel("Mean Active Cases per 1000", fontsize=12)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(scenarios), 5), fontsize=10, bbox_to_anchor=(0.5, 0.01))
    
    if use_log_scale:
        floor = 0.01
        upper_lim = max(global_max_y * 1.5, floor * 10)
        for ax in axes_flat[:n_districts]: ax.set_ylim(floor, upper_lim)
    else:
        for ax in axes_flat[:n_districts]: ax.set_ylim(0, global_max_y * 1.1)

    plt.tight_layout(rect=[0.03, 0.14, 1, 0.93])
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    suffix = "_log" if use_log_scale else ""
    out_path = results_dir / f"district_active_comparison_{group_name}_R{int(cfg.R_0*100)}_beta{int(cfg.event_base_transmission_rate*100)}{suffix}.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()




def plot_mge_recruitment_distributions(all_runs_results: List[Dict[str, Any]], data: Dict[str, Any], cfg: Any):
    """
    Plots the distribution of total attendees and initially infected attendees 
    recruited from each district across all simulation runs.
    """
    if not getattr(cfg, 'plot_on_the_fly', True):
        return

    n_patches = data['n_patches']
    attendee_data = np.stack([run.get('recruitment_data', np.zeros(n_patches)) for run in all_runs_results])
    infected_data = np.stack([run.get('infected_attendee_counts', np.zeros(n_patches)) for run in all_runs_results])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    df_att = pd.DataFrame(attendee_data, columns=range(n_patches))
    df_inf = pd.DataFrame(infected_data, columns=range(n_patches))

    # Plot Total Attendees
    valid_att = df_att.columns[df_att.any()].tolist()
    if valid_att:
        df_att[valid_att].boxplot(ax=ax1, grid=True, patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.6))
    
    # Plot Infected Attendees
    valid_inf = df_inf.columns[df_inf.any()].tolist()
    if valid_inf:
        df_inf[valid_inf].boxplot(ax=ax2, grid=True, patch_artist=True, boxprops=dict(facecolor='salmon', alpha=0.6))

    scenario_label = getattr(cfg, "current_event_scenario_name", cfg.event_model_type)
    ax1.set_title(f'Distribution of Event Attendees per District\n(Scenario: {scenario_label})', fontsize=14)
    ax2.set_title(f'Distribution of Initially Infected Attendees per District', fontsize=14)
    ax2.set_xlabel('District ID', fontsize=12)
    ax1.set_ylabel('Total Attendees', fontsize=12)
    ax2.set_ylabel('Infected Attendees (Seeds)', fontsize=12)
    
    plt.setp(ax2.get_xticklabels(), rotation=45)
    plt.tight_layout()

    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'mge_recruitment_distributions_{scenario_label}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> MGE recruitment distributions saved to '{filename.name}'")





def plot_arrival_vs_effective_distance_baseline(mean_arrival: np.ndarray, sp_dist: np.ndarray, cfg: Any):
    """
    Visualizes epidemic arrival times against SPD effective distances for baseline.
    """
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    file_tag = f"R{int(cfg.R_0*100)}_beta{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}"
    filename = results_dir / f'arrival_vs_effective_dist_Baseline_{file_tag}.png'
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Filter valid points, excluding seed districts (effective distance ~ 0)
    mask = (~np.isnan(mean_arrival)) & (np.isfinite(sp_dist)) & (sp_dist > 1e-6)
    y = mean_arrival[mask]
    x_sp = sp_dist[mask]

    if len(y) == 0:
        plt.close(fig)
        return

    # Plot SPD (Circles)
    ax.scatter(x_sp, y, color='royalblue', marker='o', alpha=0.7, s=70, label='Shortest-Path Distance ($D_{eff}$)')

    # Indicate the seed source (0,0)
    seed_points = (x_sp <= 1e-6)
    if np.any(seed_points):
        ax.scatter(x_sp[seed_points], y[seed_points], s=250, facecolors='none', 
                   edgecolors='black', linewidths=2, zorder=10, label="Seed District")

    # Add regression lines to compare predictive power (R^2)
    if stats:
        if np.ptp(x_sp) > 0:
            # Force trendline to pass through (0,0) by modeling y = m * x
            m = np.sum(x_sp * y) / np.sum(x_sp**2)
            
            y_pred = m * x_sp
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            line_x = np.array([0, x_sp.max()])
            ax.plot(line_x, m * line_x, color='royalblue', linestyle='--', alpha=0.4, 
                    label=f'SPD Fit ($R^2$={r_squared:.2f})')

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Shortest Effective Distance ($D_{eff}$)', fontsize=12)
    ax.set_ylabel('Mean Epidemic Arrival Time (Days)', fontsize=12)
    ax.set_title(f'Simulation Arrival Time vs. Percolation Front (Effective Distance)\n(Baseline | $R_0$={cfg.R_0:.2f}, $I_{{ss}}$={cfg.I_ss})', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved arrival time vs effective distance scatter plot to '{filename.name}'")

def plot_mge_arrival_correlation(all_runs_results: List[Dict[str, Any]], data: Dict[str, Any], cfg: Any):
    """
    Performs correlation analysis between recruitment metrics and epidemic arrival times.
    """
    if not getattr(cfg, 'plot_on_the_fly', True) or stats is None:
        return

    n_patches = data['n_patches']
    n_iterations = cfg.n_iterations
    
    # 1. Get Mean Arrival Times
    arrival_times = _get_arrival_times(all_runs_results, n_patches, n_iterations)
    with np.errstate(all='ignore'):
        mean_arrival = np.nanmean(arrival_times, axis=0)
    
    # 2. Get Mean Recruitment Counts
    mean_att = np.mean(np.stack([run.get('recruitment_data', np.zeros(n_patches)) for run in all_runs_results]), axis=0)
    mean_inf = np.mean(np.stack([run.get('infected_attendee_counts', np.zeros(n_patches)) for run in all_runs_results]), axis=0)

    # Filter for districts that were actually infected
    valid_mask = ~np.isnan(mean_arrival)
    if not np.any(valid_mask): return

    y = mean_arrival[valid_mask]
    x_metrics = [('Total Attendees', mean_att[valid_mask]), ('Infected Attendees (Seeds)', mean_inf[valid_mask])]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (label, x) in enumerate(x_metrics):
        ax = axes[idx]
        # Calculate Spearman correlation
        # Check for constant input to avoid ConstantInputWarning
        if np.ptp(x) == 0 or np.ptp(y) == 0:
            rho, p_val = np.nan, np.nan
        else:
            rho, p_val = stats.spearmanr(x, y)
        
        ax.scatter(x, y, alpha=0.6, edgecolors='w', s=60, color='darkcyan')
        
        # Add trend line if variation exists
        if np.ptp(x) > 0:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(np.sort(x), p(np.sort(x)), "r--", alpha=0.8, lw=2)

        ax.set_title(f'Arrival Time vs {label}\nSpearman $\\rho$: {rho:.3f} (p={p_val:.4f})', fontsize=14)
        ax.set_xlabel(f'Mean {label} per District', fontsize=12)
        ax.set_ylabel('Mean Arrival Time (Days)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if idx == 1: # Specifically for infected seeds
             ax.set_xscale('symlog', linthresh=1) # Handle many zeros and small counts

    scenario_label = getattr(cfg, "current_event_scenario_name", cfg.event_model_type)
    if cfg.event_model_type.startswith('no_event'):
        scenario_label = cfg.event_model_type
    
    fig.suptitle(f"MGE Seeding Impact Analysis: {scenario_label} ($R_0$={cfg.R_0:.2f})", fontsize=16, y=1.05)
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'mge_arrival_correlation_{scenario_label}_R{int(cfg.R_0*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> MGE arrival correlation analysis saved to '{filename.name}'")

"""
def plot_arrival_rank_map(all_runs_results: List[Dict[str, Any]], data: Dict[str, Any], cfg: Any):
    # Categorizes districts into groups of 4 based on their epidemic arrival rank
    # and visualizes them on a discrete choropleth map.

    print(f"Generating arrival rank map for '{getattr(cfg, 'current_event_scenario_name', 'unknown')}'...")
    if not HAS_GEOPANDAS or not getattr(cfg, 'plot_on_the_fly', True):
        return

    n_patches = data['n_patches']
    n_iterations = cfg.n_iterations
    
    # 1. Get Mean Arrival Times
    arrival_times = _get_arrival_times(all_runs_results, n_patches, n_iterations)
    with np.errstate(all='ignore'):
        mean_arrival = np.nanmean(arrival_times, axis=0)

    # 2. Identify the Seed District for this scenario
    if cfg.event_model_type == 'large_venue':
        # For LGE, the seed is typically the designated attendee patch
        seed_id = getattr(cfg, 'large_venue_ini_infected_attendee_patch', 
                         getattr(cfg, 'initial_infection_patch_id', 0))
    elif cfg.event_model_type == 'multi_venue':
        # For multi-venue, seeds are distributed; we mark the first venue as a reference
        venue_ids = list(getattr(cfg, 'multi_venue_group_venues', {}).values())
        seed_id = venue_ids[0] if venue_ids else 0
    else:
        seed_id = getattr(cfg, 'initial_infection_patch_id', 0)

    # Ensure seed_id is an integer
    seed_id = int(seed_id) if str(seed_id).isdigit() else 0

    # Identify infected districts that are NOT the seed to calculate ranks
    mask_to_rank = ~np.isnan(mean_arrival)
    mask_to_rank[seed_id] = False
    
    ids_to_rank = np.where(mask_to_rank)[0]
    if len(ids_to_rank) == 0: return

    # 3. Calculate Ranks and Grouping
    arrival_vals = mean_arrival[ids_to_rank]
    if stats:
        ranks = stats.rankdata(arrival_vals, method='min')
    else:
        ranks = np.argsort(np.argsort(arrival_vals)) + 1
        
    groups = np.zeros(n_patches)
    district_ranks = np.zeros(n_patches)
    for i, p_id in enumerate(ids_to_rank):
        # Group 1: rank 1-4, Group 2: rank 5-8, etc.
        groups[p_id] = int(np.ceil(ranks[i] / 4.0))
        district_ranks[p_id] = int(ranks[i])

    # 4. Load GeoJSON
    mobility_dir = Path(f"data_mobility_{cfg.dataset_name}")
    geojson_path = mobility_dir / f"{cfg.dataset_name.lower()}-districts.geojson"
    if not geojson_path.exists(): return

    try:
        geodf = gpd.read_file(geojson_path)
        # Robust ID discovery
        id_candidates = ['id', 'patch_id', 'ID', 'cartodb_id', 'OBJECTID', 'id_0']
        id_col = next((c for c in id_candidates if c in geodf.columns), None)

        if id_col:
            if id_col != 'id': geodf = geodf.rename(columns={id_col: 'id'})
            geodf['id'] = pd.to_numeric(geodf['id'], errors='coerce')
            if geodf['id'].min() == 1 and geodf['id'].max() == n_patches:
                geodf['id'] = geodf['id'] - 1
        
        # Create a mapping for the legend
        group_df = pd.DataFrame({'id': range(n_patches), 'rank_group': groups, 'rank': district_ranks})

        # Create labels and categories in numerical order to fix legend sorting
        unique_g_ids = sorted([g for g in np.unique(groups) if g > 0])
        ordered_labels = [f"Ranks {int((g-1)*4+1)}-{int(g*4)}" for g in unique_g_ids]
        group_labels = {g: label for g, label in zip(unique_g_ids, ordered_labels)}
        group_df['Group'] = group_df['rank_group'].map(group_labels).fillna("Seed/Not Infected")
        # Explicitly set Categorical type to enforce legend order
        group_df['Group'] = pd.Categorical(group_df['Group'], categories=ordered_labels, ordered=True)
        
        merged = geodf.merge(group_df, on='id', how='left')

        # Calculate areas for adaptive font sizing
        # Re-project to a projected CRS (3857) to calculate area accurately and avoid UserWarning
        areas = merged.to_crs(epsg=3857).geometry.area
        min_label_font_size = 8  # Minimum font size for labels
        max_label_font_size = 16 # Maximum font size for labels

        # Filter areas only for districts that will have labels (rank_group > 0)
        labeled_areas = areas[merged['rank_group'] > 0]

        min_area = labeled_areas.min() if not labeled_areas.empty else 0
        max_area = labeled_areas.max() if not labeled_areas.empty else 1

        # Avoid division by zero if all labeled areas are the same
        area_range = max_area - min_area
        if area_range == 0:
            area_range = 1.0 # Use a default scaling if all areas are identical

        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot districts by Group (excluding the seed group from the color map)
        merged[merged['rank_group'] > 0].plot(column='Group', categorical=True, legend=True, 
                                              cmap='viridis', ax=ax, edgecolor='0.3', linewidth=0.5,
                                              legend_kwds={'title': "Arrival Rank Groups", 'loc': 'upper left', 'bbox_to_anchor': (1.05, 1)})
        
        # Plot seed and uninfected in light grey
        merged[merged['rank_group'] == 0].plot(ax=ax, color='#f0f0f0', edgecolor='0.6', linewidth=0.5)
        
        # Add Rank Labels for non-seed infected districts
        # Calculate centroids accurately in projected CRS to avoid UserWarning
        centroids = merged.to_crs(epsg=3857).centroid.to_crs(merged.crs)
        max_g = max(unique_g_ids) if unique_g_ids else 1
        for i, row in merged.iterrows():
            if row['rank_group'] > 0:
                c = centroids.loc[i]
                
                # Calculate adaptive font size based on district area
                current_area = areas.loc[i]
                scaled_font_size = min_label_font_size + (current_area - min_area) / area_range * (max_label_font_size - min_label_font_size) * 1.2
                adaptive_font_size = np.clip(scaled_font_size, min_label_font_size, max_label_font_size)

                # Determine text color based on background color (viridis cmap)
                # Use black text for yellow/bright districts (approx. top 20% of range), otherwise white
                text_color = 'black' if (row['rank_group'] >= 0.8 * max_g) else 'white'
                # Determine text color based on perceived luminance of the background color
                cmap = plt.cm.viridis
                color_rgba = cmap(row['rank_group'] / max_g) # Normalize rank_group to [0, 1] for colormap
                # Calculate luminance (perceived brightness)
                # Formula: L = 0.299*R + 0.587*G + 0.114*B
                luminance = 0.299 * color_rgba[0] + 0.587 * color_rgba[1] + 0.114 * color_rgba[2]
                text_color = 'black' if luminance > 0.7 else 'white' # Use black for bright colors (e.g., yellow), white for dark
                ax.text(c.x, c.y, f"#{int(row['rank'])}", 
                        fontsize=adaptive_font_size, ha='center', va='center', fontweight='bold', color=text_color)

        # 5. Mark the Seed with 'X'
        seed_row = geodf[geodf['id'] == seed_id]
        if not seed_row.empty:
            centroid = seed_row.to_crs(epsg=3857).centroid.to_crs(geodf.crs).iloc[0]
            ax.scatter(centroid.x, centroid.y, marker='x', color='red', s=120, lw=3, label='Seed District', zorder=10)

        scenario_label = getattr(cfg, "current_event_scenario_name", cfg.event_model_type)
        ax.set_title(f'Spatial Distribution of Arrival Ranks: {scenario_label}\n($R_0$={cfg.R_0:.2f}, $\\beta_{{event}}$={cfg.event_base_transmission_rate:.2f})', fontsize=14)
        ax.set_axis_off()

        results_dir = getattr(cfg, 'results_dir', Path("results"))
        beta_event = cfg.event_base_transmission_rate
        filename = results_dir / f'arrival_rank_groups_{scenario_label}_R{int(cfg.R_0*100)}_beta{int(beta_event*100)}_Iss{cfg.I_ss}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Arrival rank groups map saved to {filename.name}")
    except Exception as e:
        print(f"  -> Failed to generate arrival rank map: {e}")
"""
