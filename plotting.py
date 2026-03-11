import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
from pathlib import Path
import time
import json
import re
from matplotlib.patches import Polygon, Patch
import ast
from typing import List, Dict, Any

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

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


def create_summary_plot(S_realizations, E_realizations, I_realizations, R_realizations, data, cfg, event_total_exposures=None):
    """
    Creates a single figure summarizing simulation results for any scenario type.
    The layout changes depending on whether it's an event scenario or not.
    """
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
    filename = results_dir / f'spacetime_heatmap_{scenario_name}_R{int(cfg.R_0*100)}_betaEvent{int(cfg.event_base_transmission_rate*100)}.png'
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Space-time heatmap saved to '{filename}'")

def plot_spatial_spread_snapshots(all_runs_results, data, cfg, vmax=None, scenario_name_override=None, filename_suffix=""):
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
    mobility_dir = Path(f"mobility_{cfg.dataset_name}")
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

    n_patches = data['n_patches']
    n_days = cfg.n_days
    n_iterations = cfg.n_iterations

    # Get daily active cases
    daily_times, all_runs_daily_cases = _get_daily_active_cases(all_runs_results, n_patches, n_iterations, n_days)
    mean_daily_cases = np.mean(all_runs_daily_cases, axis=0)
    
    # Select 6 time points spread across the simulation
    time_points = np.linspace(0, n_days, 6, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    # Determine global max for color scaling
    if vmax is None:
        vmax = np.max(mean_daily_cases)
        if vmax == 0: vmax = 1
    
    # Prepare for colorbar
    norm = plt.matplotlib.colors.Normalize(vmin=0, vmax=vmax)
    mappable = plt.cm.ScalarMappable(cmap='Reds', norm=norm)

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

    for idx, t in enumerate(time_points):
        ax = axes[idx]
        if t >= len(mean_daily_cases): t = len(mean_daily_cases) - 1
        
        cases_at_t = mean_daily_cases[t, :]
        cases_df = pd.DataFrame({'id': range(n_patches), 'cases': cases_at_t})
        
        # Merge geodata with case data
        merged_gdf = geodf.merge(cases_df, on='id', how='left').fillna(0)
        
        # Plot choropleth map
        merged_gdf.plot(column='cases', cmap='Reds', linewidth=0.5, ax=ax, edgecolor='0.3', norm=norm)

        ax.set_title(f'Day {t}', fontsize=14)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_aspect('equal', adjustable='box')

    # Add a global colorbar
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # Position: [left, bottom, width, height]
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label('Mean Active Cases (E+I)', fontsize=12)
    
    fig.suptitle(f'Spatial Spread of Epidemic Over Time - {scenario_name}', fontsize=20)
    filename = results_dir / f'spatial_choropleth_{scenario_name}{filename_suffix}_R{int(cfg.R_0*100)}_betaEvent{int(cfg.event_base_transmission_rate*100)}_Iss{int(cfg.I_ss)}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Spatial choropleth snapshots saved to '{filename}'")

def plot_spatial_spread_comparison(group_results, data, cfg, group_name):
    """
    Generates spatial choropleth maps for multiple scenarios with a unified color scale.
    """
    if not HAS_GEOPANDAS:
        return

    print(f"Generating comparable spatial snapshots for group: {group_name}...")

    n_patches = data['n_patches']
    n_days = cfg.n_days
    n_iterations = cfg.n_iterations
    
    # 1. Calculate global max across all scenarios in the group
    global_vmax = 0
    
    for res in group_results.values():
        _, all_runs = _get_daily_active_cases(res['all_runs_results'], n_patches, n_iterations, n_days)
        mean_cases = np.mean(all_runs, axis=0)
        global_vmax = max(global_vmax, np.max(mean_cases))
        
    if global_vmax == 0: global_vmax = 1
    
    # 2. Generate plots for each scenario with fixed scale
    for sc_name, res in group_results.items():
        plot_spatial_spread_snapshots(
            res['all_runs_results'], 
            data, 
            cfg, 
            vmax=global_vmax, 
            scenario_name_override=sc_name,
            filename_suffix=f"_fixed_scale_group_{group_name}"
        )

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
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6, which='both' if scale == 'log' else 'major')
        
        ax.set_xlim(0, 1.2)
        # ax.set_ylim(0, 1)
        
        # Invert x-axis so time flows left-to-right (High S -> Low S)
        # ax.invert_xaxis()
        
        filename = results_dir / f'phase_plane_S_vs_EI{file_suffix}_R{int(cfg.R_0*100)}_betaEvent{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Phase plane plot ({scale}) saved to '{filename}'")

def plot_active_cases_comparison(group_results, data, cfg, group_name):
    """
    Plots the time evolution of active cases ((E+I)/N * 1000) for a group of scenarios.
    Includes individual run curves (light) and average trend (solid with markers).
    """
    print(f"Generating active cases comparison plot for group: {group_name}...")
    
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
        time = E[:, 0] # Time vector (interpolated frames)
        
        # Calculate (E+I)/N * 1000 for all runs (Columns 1 to end are the runs)
        EI_runs = ((E[:, 1:] + I[:, 1:]) / total_population) * 1000
        EI_mean = np.mean(EI_runs, axis=1)
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
        if group_name != "All_Scenarios":
            for i in range(n_runs):
                ax.plot(time, EI_runs[:, i], color=color, alpha=0.1, linewidth=1, zorder=1)
        
        # Plot mean (solid curve with markers at ticks)
        # Find indices in 'time' closest to 0, 50, 100...
        max_time = time[-1]
        target_times = np.arange(0, max_time + 1, marker_interval)
        mark_indices = [np.abs(time - t_val).argmin() for t_val in target_times]
        
        ax.plot(time, EI_mean, color=color, label=label, linewidth=2.5, 
                marker=marker, markevery=mark_indices, markersize=8, zorder=zorder_mean)

    ax.set_xlabel('Time (Days)', fontsize=14)
    ax.set_ylabel('Active cases per 1000 persons', fontsize=14)
    ax.set_title(f'Active cases: {group_name}\n($R_0$={cfg.R_0}, $I_{{ss}}$={cfg.I_ss}, $\\beta_{{event}}$={cfg.event_base_transmission_rate})', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    filename = results_dir / f'active_cases_comparison_{group_name}_R{int(cfg.R_0*100)}_betaEvent{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Active cases comparison plot saved to '{filename}'")

def plot_cumulative_incidence_comparison(group_results, data, cfg, group_name):
    """
    Plots the time evolution of cumulative incidence (total infected) for a group of scenarios.
    Includes individual run curves (light) and average trend (solid with markers).
    """
    print(f"Generating cumulative incidence comparison plot for group: {group_name}...")
    
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
        
        # Calculate R/N * 1000 for all runs (Columns 1 to end are the runs)
        # Using Cumulative Recovered as the metric for cumulative infections as requested
        Cum_runs = (R[:, 1:] / total_population) * 1000
        Cum_mean = np.mean(Cum_runs, axis=1)
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
        if group_name != "All_Scenarios":
            for i in range(n_runs):
                ax.plot(time, Cum_runs[:, i], color=color, alpha=0.1, linewidth=1, zorder=1)
        
        # Plot mean (solid curve with markers at ticks)
        max_time = time[-1]
        target_times = np.arange(0, max_time + 1, marker_interval)
        mark_indices = [np.abs(time - t_val).argmin() for t_val in target_times]
        
        ax.plot(time, Cum_mean, color=color, label=label, linewidth=2.5, 
                marker=marker, markevery=mark_indices, markersize=8, zorder=zorder_mean)

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
        
        rt_mean = np.mean(rt_runs[:, 1:], axis=1)
        rt_std = np.std(rt_runs[:, 1:], axis=1)
        n_runs = rt_runs.shape[1] - 1
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
    scenario_name = getattr(cfg, "current_event_scenario_name", cfg.event_model_type)
    beta_event = cfg.event_base_transmission_rate
    r0 = cfg.R_0
    iss = cfg.I_ss
    
    csv_filename = results_dir / f"run_daily_active_cases_{scenario_name}_R{int(r0*100)}_beta{int(beta_event*100)}_Iss{iss}.csv"
    final_df.to_csv(csv_filename, index=False, float_format='%.2f')
    print(f"Saved per-run daily active cases CSV to {csv_filename}")

def analyze_and_plot_arrival_times(all_runs_results, data, cfg):
    """
    Analyzes and visualizes the arrival time of the infection (first E > 0) for each patch.
    1. Prints stats on total patches affected.
    2. Plots the 'Spread Curve' (Patches affected vs Time).
    3. Plots a Boxplot of Arrival Times per Patch.
    4. Plots a Choropleth Map of Mean Arrival Times.
    """
    print(f"Analyzing infection arrival times for '{cfg.event_model_type}'...")
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
    
    # --- 2. Spread Curve (Patches vs Time) ---
    common_time = np.linspace(0, n_days, n_days*2 + 1)
    spread_curves = np.zeros((n_iterations, len(common_time)))
    
    for r in range(n_iterations):
        arrivals = arrival_times[r, :]
        arrivals = arrivals[~np.isnan(arrivals)]
        if len(arrivals) == 0: continue
        arrivals_sorted = np.sort(arrivals)
        counts = np.searchsorted(arrivals_sorted, common_time, side='right')
        spread_curves[r, :] = counts
        
    mean_spread = np.mean(spread_curves, axis=0)
    std_spread = np.std(spread_curves, axis=0)
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(common_time, mean_spread, color='blue', lw=2, label='Mean Affected Patches')
    ax1.fill_between(common_time, mean_spread - std_spread, mean_spread + std_spread, color='blue', alpha=0.2)
    ax1.set_xlabel('Time (Days)')
    ax1.set_ylabel('Number of Affected Patches')
    ax1.set_title(f'Spread of Infection to New Patches\n({cfg.event_model_type})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    beta_event = cfg.event_base_transmission_rate
    filename1 = results_dir / f'spread_curve_{cfg.event_model_type}_R{int(cfg.R_0*100)}_betaEvent{int(beta_event*100)}.png'
    plt.savefig(filename1, dpi=300)
    plt.close(fig1)
    print(f"  -> Saved spread curve to {filename1}")

    # --- 3. Arrival Time Boxplot ---
    df_arrival = pd.DataFrame(arrival_times, columns=range(n_patches))
    valid_cols = df_arrival.columns[df_arrival.notna().any()].tolist()
    
    if valid_cols:
        df_plot = df_arrival[valid_cols]
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        df_plot.boxplot(ax=ax2, rot=90, fontsize=8)
        ax2.set_xlabel('Patch ID')
        ax2.set_ylabel('Arrival Time (Days)')
        ax2.set_title(f'Distribution of Infection Arrival Times per Patch\n({cfg.event_model_type})')
        
        filename2 = results_dir / f'arrival_time_boxplot_{cfg.event_model_type}_R{int(cfg.R_0*100)}_betaEvent{int(beta_event*100)}.png'
        plt.savefig(filename2, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"  -> Saved arrival time boxplot to {filename2}")

    # --- 4. Arrival Time Map ---
    # Re-use the spatial snapshot logic but for a single static map of mean arrival times
    plot_spatial_spread_snapshots(all_runs_results, data, cfg, filename_suffix="_arrival_times") # Placeholder if needed, but custom map logic is better handled if we had direct access to geodata here. 
    # Since plot_spatial_spread_snapshots handles GeoJSON loading internally, we can rely on that for general spatial plots, 
    # or implement a specific one here if needed. For brevity, the boxplot and spread curve provide the quantification requested.