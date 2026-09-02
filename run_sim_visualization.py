#!/usr/bin/env python
"""
Post-hoc visualization script for Hybrid Epidemic Simulation results.
Generates spatial choropleth maps, spacetime heatmaps, and comparison plots.
Uses run_daily_active_cases files which have per-district data.

Automatically discovers all available parameter combinations (R0, beta_event, I_ss)
in the results directory and generates figures for each.

Generates the following figures:
  - Spatial spread comparison plots (from daily active cases CSVs)
  - Active cases comparison curves (from realizations CSVs)
  - Peak summary rectangles (from realizations CSVs)
  - Arrival time comparison maps (from all_runs_results reconstruction)
  - Mean arrival time clustered plots (from all_runs_results reconstruction)
  - Arrival time difference heatmaps (from all_runs_results reconstruction)
  - Cumulative cases time-to-threshold (excluding outliers)
  - Cumulative cases boxplot at day (renamed filenames)
  - Cumulative cases distribution at day (renamed filenames)
  - MGE seeding boxplots (renamed filenames)
  - Infection arrival interval boxplots (renamed filenames)
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from code_config_models import create_default_config, load_config_from_json
from code_simulation.data import DataLoader
import code_plotting as plotting

cfg = create_default_config()


def load_daily_active_cases(results_dir: Path, scenario: str, r0: float, beta: float, iss: int):
    """Load daily active cases data for a specific scenario and parameter combination."""
    results_dir = Path(results_dir)
    
    if scenario.lower() == 'no_event':
        filename = results_dir / f"run_daily_active_cases_{scenario}_R{int(r0*100)}_Iss{iss}.csv"
    else:
        filename = results_dir / f"run_daily_active_cases_{scenario}_R{int(r0*100)}_beta{int(beta*100)}_Iss{iss}.csv"
    
    if not filename.exists():
        print(f"Warning: File not found: {filename}")
        return None
    
    df = pd.read_csv(filename)
    
    day_cols = [col for col in df.columns if col.startswith('day_')]
    if not day_cols:
        return None
    
    daily_by_district = df.groupby('district_id')[day_cols].mean()
    mean_daily_cases = daily_by_district.values.T
    
    return mean_daily_cases


def load_realizations(results_dir: Path, scenario: str, r0: float, beta: float, iss: int):
    """
    Load SEIR realizations (S, E, I, R, rt) from CSV files for a scenario.
    Returns a dict with numpy arrays, or None if no realizations found.
    """
    results_dir = Path(results_dir)

    if scenario.lower() == 'no_event':
        prefix = f"realizations_{scenario}_R{int(r0*100)}_Iss{iss}"
    else:
        prefix = f"realizations_{scenario}_R{int(r0*100)}_beta{int(beta*100)}_Iss{iss}"

    comps = {}
    for comp in ['S', 'E', 'I', 'R', 'rt']:
        fname = results_dir / f"{prefix}_{comp}.csv"
        if fname.exists():
            df = pd.read_csv(fname)
            comps[comp] = df.to_numpy()

    return comps if comps else None


def reconstruct_all_runs_results(results_dir: Path, scenario: str, r0: float, beta: float,
                                 iss: int, n_iterations: int, n_patches: int):
    """
    Reconstruct a minimal all_runs_results list (one dict per run) from saved
    I-based arrival time records.  Each element has 't' and
    'precomputed_arrival_times' (per-district arrival times in days).
    This ensures the same I-based arrival times as computed during the
    simulation, not a proxy from E+I daily active cases.
    """
    results_dir = Path(results_dir)

    if scenario.lower() == 'no_event':
        prefix = f"arrival_time_records_I_{scenario}_R{int(r0*100)}_Iss{iss}"
    else:
        prefix = f"arrival_time_records_I_{scenario}_R{int(r0*100)}_beta{int(beta*100)}_Iss{iss}"

    fname = results_dir / f"{prefix}.csv"

    # Try E-based records if I-based not available
    if not fname.exists():
        if scenario.lower() == 'no_event':
            prefix = f"arrival_time_records_E_{scenario}_R{int(r0*100)}_Iss{iss}"
        else:
            prefix = f"arrival_time_records_E_{scenario}_R{int(r0*100)}_beta{int(beta*100)}_Iss{iss}"
        fname = results_dir / f"{prefix}.csv"

    if not fname.exists():
        # Fallback: reconstruct from daily active cases (E+I as proxy)
        if scenario.lower() == 'no_event':
            daily_fname = results_dir / f"run_daily_active_cases_{scenario}_R{int(r0*100)}_Iss{iss}.csv"
        else:
            daily_fname = results_dir / f"run_daily_active_cases_{scenario}_R{int(r0*100)}_beta{int(beta*100)}_Iss{iss}.csv"

        if not daily_fname.exists():
            return None, n_iterations

        df = pd.read_csv(daily_fname)
        day_cols = [col for col in df.columns if col.startswith('day_')]
        if not day_cols:
            return None, n_iterations

        max_run_id = int(df['run_id'].max())
        actual_n_iterations = max_run_id + 1

        n_days_file = int(day_cols[-1].replace('day_', ''))
        t = np.arange(n_days_file + 1, dtype=float)

        all_runs_results = []
        for run_id in range(actual_n_iterations):
            run_data = df[df['run_id'] == run_id]
            if run_data.empty:
                continue
            # E+I as proxy for I_j — not ideal but fallback
            I_j = run_data[day_cols].values.T
            all_runs_results.append({'t': t, 'I_j': I_j})

        return all_runs_results, actual_n_iterations

    # Load arrival time records and reconstruct per-run arrival times
    df = pd.read_csv(fname)
    max_run_id = int(df['run_id'].max())
    actual_n_iterations = max_run_id + 1

    arrival_lookup = {}  # (run_id, district_id) -> arrival_time
    for _, row in df.iterrows():
        run_id = int(row['run_id'])
        district_id = int(row['district_id'])
        arrived = int(row.get('arrived', 0))
        if arrived and district_id < n_patches:
            arrival_lookup[(run_id, district_id)] = float(row['arrival_time'])

    all_runs_results = []
    for run_id in range(actual_n_iterations):
        arrivals = np.full(n_patches, np.nan)
        for p in range(n_patches):
            if (run_id, p) in arrival_lookup:
                arrivals[p] = arrival_lookup[(run_id, p)]
        # Provide a dummy t array (not used when precomputed_arrival_times is set)
        t = np.arange(1, dtype=float)
        all_runs_results.append({'t': t, 'precomputed_arrival_times': arrivals})

    # Update n_iterations to match actual data
    n_iterations = actual_n_iterations

    return all_runs_results, n_iterations


def build_group_results(results_dir: Path, scenarios, r0, beta, iss, n_iterations, n_patches):
    """
    Build a group_results dict compatible with code_plotting functions.
    Each entry has 'realizations' (loaded from CSVs) and 'all_runs_results'
    (reconstructed from daily active cases CSVs).
    """
    group_results = {}
    actual_n_iterations = n_iterations
    for scenario in scenarios:
        beta_val = beta if scenario != 'no_event' else 0.0
        reals = load_realizations(results_dir, scenario, r0, beta_val, iss)
        result = reconstruct_all_runs_results(
            results_dir, scenario, r0, beta_val, iss, n_iterations, n_patches
        )
        # Handle both old and new return types (tuple vs single value)
        if isinstance(result, tuple):
            all_runs, detected_n = result
        else:
            all_runs = result
            detected_n = n_iterations

        if all_runs is not None and len(all_runs) > actual_n_iterations:
            actual_n_iterations = len(all_runs)

        entry = {}
        if reals is not None:
            entry['realizations'] = reals
        if all_runs is not None:
            entry['all_runs_results'] = all_runs
        if entry:
            group_results[scenario] = entry

    # Update cfg.n_iterations to match the actual number of runs in the data
    if actual_n_iterations > n_iterations:
        cfg.n_iterations = actual_n_iterations
        print(f"  Detected {actual_n_iterations} runs in data (cfg.n_iterations updated from {n_iterations} to {actual_n_iterations})")

    return group_results


def discover_parameter_combinations(results_dir: Path):
    """
    Scan the results directory for all run_daily_active_cases files and extract
    the unique parameter combinations (scenario, R0, beta, I_ss).

    Returns a list of dicts, each with keys: scenario, r0, beta, iss.
    For 'no_event' scenarios, beta is set to None.
    """
    results_dir = Path(results_dir)
    pattern = re.compile(
        r'^run_daily_active_cases_(.+)_R(\d+)(?:_beta(\d+))?_Iss(\d+)\.csv$'
    )
    
    combinations = []
    seen = set()
    
    for filepath in sorted(results_dir.glob("run_daily_active_cases_*.csv")):
        match = pattern.match(filepath.name)
        if not match:
            continue
        
        scenario = match.group(1)
        r0 = int(match.group(2)) / 100.0
        beta_str = match.group(3)
        beta = int(beta_str) / 100.0 if beta_str else None
        iss = int(match.group(4))
        
        key = (scenario, r0, beta, iss)
        if key not in seen:
            seen.add(key)
            combinations.append({
                'scenario': scenario,
                'r0': r0,
                'beta': beta,
                'iss': iss,
            })
    
    return combinations


def group_combinations_by_params(combinations):
    """
    Group discovered combinations by (r0, beta, iss) so that all scenarios
    for a given parameter set can be plotted together in comparison plots.
    The 'no_event' scenario (which has no beta in its filename) is included
    in every group for the same (r0, iss) so it can serve as a baseline.
    Returns a list of dicts: {'r0': ..., 'beta': ..., 'iss': ..., 'scenarios': [...]}
    """
    # Separate no_event combinations (beta is None) from event combinations
    no_event_combos = [c for c in combinations if c['scenario'] == 'no_event']
    event_combos = [c for c in combinations if c['scenario'] != 'no_event']
    
    # Build a lookup: (r0, iss) -> True if no_event data exists
    no_event_lookup = {(c['r0'], c['iss']) for c in no_event_combos}
    
    # Group event scenarios by (r0, beta, iss)
    groups = {}
    for combo in event_combos:
        key = (combo['r0'], combo['beta'], combo['iss'])
        if key not in groups:
            groups[key] = {
                'r0': combo['r0'],
                'beta': combo['beta'],
                'iss': combo['iss'],
                'scenarios': [],
            }
        groups[key]['scenarios'].append(combo['scenario'])
    
    # Add 'no_event' to each group if data exists for that (r0, iss)
    for group in groups.values():
        if (group['r0'], group['iss']) in no_event_lookup:
            group['scenarios'].insert(0, 'no_event')
    
    return list(groups.values())


def plot_spacetime_heatmap_from_data(mean_daily_cases: np.ndarray, n_days: int, title: str, output_path: Path):
    """Plot spacetime heatmap from mean daily cases array."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    heatmap_data = mean_daily_cases.T
    peak_times = np.argmax(heatmap_data, axis=1)
    sorted_indices = np.argsort(peak_times)
    sorted_data = heatmap_data[sorted_indices, :]
    
    if np.max(sorted_data) > 100:
        norm = plt.matplotlib.colors.LogNorm(vmin=max(1, np.min(sorted_data[sorted_data>0])), vmax=np.max(sorted_data))
    else:
        norm = plt.Normalize(vmin=0, vmax=np.max(sorted_data))
    
    n_patches = mean_daily_cases.shape[1]
    im = ax.imshow(sorted_data, aspect='auto', cmap='Reds', interpolation='nearest',
                   origin='lower', extent=[0, n_days, 0, n_patches], norm=norm)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Active Cases (E+I)')
    
    ax.set_title(f'Space-Time Heatmap of Epidemic Spread\n(Sorted by Peak Time) - {title}', fontsize=16)
    ax.set_xlabel('Time (Days)', fontsize=12)
    ax.set_ylabel('District ID (Sorted)', fontsize=12)
    
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Space-time heatmap saved to '{output_path}'")


def _prepare_shared(sim_data=None, geodf=None, seed_centroid=None):
    """Load shared, seed-independent data once.

    Computes sim_data (population + flow matrix + O(n^2) distance matrix) and,
    if geopandas is available, the GeoJSON-derived geodf and seed centroid.
    These are independent of the simulation seed, so computing them once and
    reusing across seeds avoids redundant O(n^2) work. Returns
    (sim_data, geodf, seed_centroid); geodf/seed_centroid are None when
    geopandas or GeoJSON is unavailable.
    """
    if sim_data is None:
        loader = DataLoader(cfg.dataset_name)
        sim_data = loader.prepare_all()

    if geodf is None:
        try:
            import geopandas as gpd
        except ImportError:
            print("Warning: 'geopandas' required for spatial plots")
            return sim_data, None, None

        mobility_dir = Path(f"data_mobility_{cfg.dataset_name}")
        geojson_files = list(mobility_dir.glob("*.geojson"))
        geojson_path = geojson_files[0] if geojson_files else None

        if not geojson_path:
            print("No GeoJSON found for spatial plotting.")
            return sim_data, None, None

        geodf = gpd.read_file(geojson_path)

        n_patches = sim_data['n_patches']
        id_col_found = None
        for col_name in ['id', 'patch_id', 'ID', 'cartodb_id']:
            if col_name in geodf.columns:
                id_col_found = col_name
                break

        if id_col_found and id_col_found != 'id':
            geodf = geodf.rename(columns={id_col_found: 'id'})

        if geodf['id'].min() == 1 and geodf['id'].max() >= n_patches:
            geodf['id'] = geodf['id'] - 1

        seed_district_id = getattr(cfg, 'initial_infection_patch_id', 0)
        seed_geom = geodf[geodf['id'] == seed_district_id]['geometry']
        if len(seed_geom) > 0:
            try:
                proj_crs = geodf.estimate_utm_crs()
                seed_centroid = seed_geom.to_crs(proj_crs).centroid.to_crs(geodf.crs)
            except Exception:
                seed_centroid = seed_geom.centroid
        else:
            seed_centroid = None

    return sim_data, geodf, seed_centroid


def run_figures(results_dir, output_dir, sim_data=None, geodf=None, seed_centroid=None):
    """Generate figures for the parameter combinations found in results_dir.

    Figures are written to output_dir. Shared seed-independent data may be
    supplied to avoid recomputing it; when omitted, _prepare_shared() loads it.
    """
    if sim_data is None:
        sim_data, geodf, seed_centroid = _prepare_shared()
    if geodf is None:
        print("No GeoJSON / geopandas available; cannot generate spatial figures.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    n_patches = sim_data['n_patches']

    scenario_titles = getattr(cfg, 'custom_spatial_comparison_titles', {})
    time_points = getattr(cfg, 'custom_spatial_comparison_days', [100, 150, 175, 200])

    # --- Discover all available parameter combinations ---
    print("Discovering available result files...")
    combinations = discover_parameter_combinations(results_dir)

    if not combinations:
        print(f"No 'run_daily_active_cases' files found in '{results_dir}'.")
        return

    param_groups = group_combinations_by_params(combinations)
    print(f"Found {len(combinations)} scenario-parameter combinations across {len(param_groups)} parameter groups.")

    
    # --- Process each parameter group ---
    for group in param_groups:
        r0 = group['r0']
        beta = group['beta']
        iss = group['iss']
        scenarios = group['scenarios']
        
        # Sort scenarios so 'no_event' comes first
        if 'no_event' in scenarios:
            scenarios.remove('no_event')
            scenarios.sort()
            scenarios.insert(0, 'no_event')
        
        beta_label = f"beta{int(beta*100)}" if beta is not None else "noBeta"
        r0_label = int(r0 * 100)
        
        print(f"\n{'='*60}")
        print(f"Processing: R0={r0}, beta_event={beta}, I_ss={iss}")
        print(f"  Scenarios: {scenarios}")
        print(f"{'='*60}")
        
        cfg.R_0 = r0
        cfg.I_ss = iss
        if beta is not None:
            cfg.event_base_transmission_rate = beta
        
        all_mean_cases = {}
        for scenario in scenarios:
            beta_val = beta if scenario != 'no_event' else 0.1
            mean_cases = load_daily_active_cases(results_dir, scenario, r0, beta_val, iss)
            if mean_cases is not None:
                all_mean_cases[scenario] = mean_cases
        
        if not all_mean_cases:
            print(f"  No data loaded for this parameter group. Skipping.")
            continue
        
        global_vmax = max(np.max(cases) for cases in all_mean_cases.values()) if all_mean_cases else 1
        if global_vmax == 0:
            global_vmax = 1
        
        # Default scenario titles (merged with any custom overrides from config)
        default_titles = {
            'no_event': 'Baseline',
            'AMS_dance': 'AMS Dance',
            'AMS_football': 'AMS Football',
            'Leipzig_1': 'Leipzig 1',
            'Leipzig_2': 'Leipzig 2',
            'Leipzig_3': 'Leipzig 3',
        }
        scenario_titles = {**default_titles, **scenario_titles}
        
        def generate_spatial_plot(scenarios_subset, prefix):
            subset_cases = {sc: all_mean_cases[sc] for sc in scenarios_subset if sc in all_mean_cases}
            if not subset_cases:
                print(f"  No data for {prefix} subset. Skipping.")
                return
            
            vmax = max(np.max(cases) for cases in subset_cases.values()) if subset_cases else 1
            if vmax == 0:
                vmax = 1
            
            nrows = len(subset_cases)
            ncols = len(time_points)
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows), squeeze=False)
            norm = plt.matplotlib.colors.Normalize(vmin=0, vmax=vmax)
            mappable = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
            
            for row_idx, sc_name in enumerate(subset_cases.keys()):
                mean_cases = subset_cases[sc_name]
                for col_idx, t in enumerate(time_points):
                    ax = axes[row_idx, col_idx]
                    t_adj = min(t, len(mean_cases) - 1)
                    cases_at_t = mean_cases[t_adj, :]
                    cases_df = pd.DataFrame({'id': range(n_patches), 'cases': cases_at_t})
                    merged_gdf = geodf.merge(cases_df, on='id', how='left').fillna(0)
                    merged_gdf.plot(column='cases', cmap='Reds', linewidth=0.5, ax=ax, edgecolor='0.3', norm=norm)
                    
                    if seed_centroid is not None:
                        ax.plot(seed_centroid.x, seed_centroid.y, marker='x', color='black', markersize=7, markeredgewidth=2, zorder=10)
                    
                    if col_idx == 0:
                        ax.set_ylabel(f"{scenario_titles.get(sc_name, sc_name)}", fontsize=16)
                    if row_idx == 0:
                        ax.set_title(f'Day {t_adj}', fontsize=16)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_aspect('equal', adjustable='box')
            
            beta_title = f"\\beta_{{event}}={beta}" if beta is not None else "\\beta_{event}=N/A"
            fig.suptitle(f'Spatial Spread Comparison (Mean)\n($R_0$={r0}, $I_{{ss}}$={iss}, ${beta_title}$)', fontsize=20, y=0.97)
            
            fig.subplots_adjust(bottom=0.08, top=0.83, hspace=0.3, wspace=0.1, left=0.08, right=0.82)
            cbar_ax = fig.add_axes([0.84, 0.10, 0.03, 0.80])
            cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='vertical')
            
            output_filename = output_dir / f'{prefix}_R{r0_label}_{beta_label}_Iss{int(iss)}.png'
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  Spatial spread plot saved to '{output_filename}'")
        
        # 1. All scenarios (default)
        print(f"\n  Generating spatial comparison plot (Mean)...")
        generate_spatial_plot(scenarios, 'spatial_spread_mean')
        
        # 2. AMS subset (baseline + AMS_dance + AMS_football)
        ams_scenarios = [sc for sc in scenarios if sc in ('no_event', 'AMS_dance', 'AMS_football')]
        if len(ams_scenarios) > 1:
            generate_spatial_plot(ams_scenarios, 'spatial_spread_mean_AMS')
        
        # 3. Leipzig subset (baseline + Leipzig_1/2/3)
        leipzig_scenarios = [sc for sc in scenarios if sc in ('no_event', 'Leipzig_1', 'Leipzig_2', 'Leipzig_3')]
        if len(leipzig_scenarios) > 1:
            generate_spatial_plot(leipzig_scenarios, 'spatial_spread_mean_Leipzig')
        
        # 4. Selected subset (baseline + AMS_dance + Leipzig_3)
        selected_scenarios = [sc for sc in scenarios if sc in ('no_event', 'AMS_dance', 'Leipzig_3')]
        if len(selected_scenarios) > 1:
            generate_spatial_plot(selected_scenarios, 'spatial_spread_mean_selected')

        # --- Generate additional comparison figures via code_plotting ---
        print(f"\n  Generating additional comparison figures...")
        cfg.results_dir = output_dir
        cfg.plot_on_the_fly = True

        # Build group_results dict from saved CSVs
        group_results = build_group_results(
            results_dir, scenarios, r0, beta, iss, cfg.n_iterations, n_patches
        )

        if group_results:
            group_name = f"R{r0_label}_{beta_label}_Iss{int(iss)}"

            # 1. Active cases comparison (from realizations)
            try:
                plotting.plot_active_cases_comparison(
                    group_results, sim_data, cfg, group_name, show_variation=False
                )
            except Exception as e:
                print(f"    Warning: plot_active_cases_comparison failed: {e}")

            # 2. Peak summary rectangles (from realizations)
            try:
                plotting.plot_peak_summary_rectangles(
                    group_results, sim_data, cfg, group_name
                )
            except Exception as e:
                print(f"    Warning: plot_peak_summary_rectangles failed: {e}")

            # 3. Arrival time comparison maps (mean and std, requires geopandas)
            try:
                plotting.plot_arrival_time_comparison(
                    group_results, sim_data, cfg, group_name, metric='mean'
                )
            except Exception as e:
                print(f"    Warning: plot_arrival_time_comparison (mean) failed: {e}")
            try:
                plotting.plot_arrival_time_comparison(
                    group_results, sim_data, cfg, group_name, metric='std'
                )
            except Exception as e:
                print(f"    Warning: plot_arrival_time_comparison (std) failed: {e}")

            # 4. Mean arrival time by district (clustered)
            try:
                plotting.plot_mean_arrival_time_by_district_and_scenario_clustered(
                    group_results, sim_data, cfg, group_name, cluster_threshold=4.0
                )
            except Exception as e:
                print(f"    Warning: plot_mean_arrival_time_clustered failed: {e}")

            # 5. Arrival time difference heatmap (via analyze_arrival_time_sequence)
            _prev_results_dir = cfg.results_dir
            try:
                cfg.results_dir = results_dir  # CSVs go to original results dir
                plotting.analyze_arrival_time_sequence(
                    group_results, sim_data, cfg
                )
                # Move the heatmap PNG to output_dir (CSV data stays in results_dir)
                _heatmap_files = list(results_dir.glob(f"arrival_diff_heatmap_R{int(r0*100)}*Iss{int(iss)}.png"))
                for _hf in _heatmap_files:
                    _dest = output_dir / _hf.name
                    if _dest.exists(): _dest.unlink()
                    _hf.rename(_dest)
            except Exception as e:
                print(f"    Warning: analyze_arrival_time_sequence failed: {e}")
            finally:
                cfg.results_dir = _prev_results_dir

            # --- "In addition" figures ---
            # 6. Cumulative cases time-to-threshold (excluding outliers)
            try:
                plotting.plot_cumulative_cases_time_to_threshold_comparison(
                    group_results, sim_data, cfg, group_name, show_outliers=False
                )
            except Exception as e:
                print(f"    Warning: plot_cumulative_cases_time_to_threshold_comparison failed: {e}")

            # 7. Cumulative cases boxplot at day 250 (no outliers)
            try:
                plotting.plot_cumulative_active_cases_boxplot_at_day(
                    group_results, sim_data, cfg, group_name, target_day=250.0, show_outliers=False
                )
            except Exception as e:
                print(f"    Warning: plot_cumulative_active_cases_boxplot_at_day failed: {e}")

            # 8. MGE seeding boxplot (no outliers)
            try:
                plotting.plot_initial_exposed_infected_distribution_boxplot(
                    group_results, sim_data, cfg, group_name, show_outliers=False
                )
            except Exception as e:
                print(f"    Warning: plot_initial_exposed_infected_distribution_boxplot failed: {e}")

            # 9. Infection arrival interval boxplot (no outliers)
            try:
                plotting.plot_infection_arrival_interval_distribution(
                    group_results, sim_data, cfg, group_name, show_outliers=False
                )
            except Exception as e:
                print(f"    Warning: plot_infection_arrival_interval_distribution failed: {e}")

            # 10. Cumulative cases histogram at day 250
            try:
                plotting.plot_cumulative_active_cases_hist_at_day(
                    group_results, sim_data, cfg, group_name, target_day=250.0, use_log_x=True
                )
            except Exception as e:
                print(f"    Warning: plot_cumulative_active_cases_hist_at_day failed: {e}")

    print("\nAll figures generated successfully!")


def _process_seeds(results_dir, output_dir, sim_data, geodf, seed_centroid,
                   seed_prefix, seed_filter=None, n_iterations_baseline=None):
    """Discover seed_* subfolders in results_dir and run figure generation per seed.

    Shared by --per-seed mode and the auto-detection fallback so the two
    code paths stay in sync.
    """
    seed_dirs = sorted(
        (d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith(seed_prefix)),
        key=lambda d: int(d.name[len(seed_prefix):]) if d.name[len(seed_prefix):].isdigit() else 0
    )
    if seed_filter is not None:
        target_name = f"{seed_prefix}{seed_filter}"
        seed_dirs = [d for d in seed_dirs if d.name == target_name]
        if not seed_dirs:
            print(f"No '{target_name}' subfolder found in '{results_dir}'.")
            return
    if not seed_dirs:
        print(f"No '{seed_prefix}' subfolders found in '{results_dir}'.")
        return

    for seed_dir in seed_dirs:
        if n_iterations_baseline is not None:
            cfg.n_iterations = n_iterations_baseline
        seed_output = output_dir / seed_dir.name
        print(f"\n{'='*60}")
        print(f"### Processing {seed_dir.name} ###")
        print(f"{'='*60}")
        run_figures(seed_dir, seed_output, sim_data, geodf, seed_centroid)


def main():
    """CLI entry point: shared setup once, then figure generation.

    Default (flat) mode scans --results-dir root. With --per-seed, each seed_*
    subfolder is processed into its own output subdir.
    """
    parser = argparse.ArgumentParser(
        description="Generate hybrid-simulation figures. Default scans the "
                    "results dir root; --per-seed processes each seed_* subfolder."
    )
    parser.add_argument("--results-dir", default="results_hybrid_sim",
                        help="Results directory containing run CSVs (default: results_hybrid_sim)")
    parser.add_argument("--output-dir", default="results_sim_visualization",
                        help="Directory for generated figures (default: results_sim_visualization)")
    parser.add_argument("--per-seed", "-p", action="store_true",
                        help="Process each seed_* subfolder in --results-dir separately, "
                             "writing one output subdir per seed")
    parser.add_argument("--seed", type=int, default=None,
                        help="With --per-seed, process only seed_<N>")
    parser.add_argument("--seed-prefix", default="seed_",
                        help="Prefix identifying seed subfolders (default: seed_)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file (overrides module-level cfg)")
    args = parser.parse_args()

    if args.config:
        global cfg
        cfg = load_config_from_json(args.config)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    n_iterations_baseline = cfg.n_iterations

    sim_data, geodf, seed_centroid = _prepare_shared()
    if geodf is None:
        # geopandas/GeoJSON unavailable: matches original early-exit behaviour.
        return

    if args.per_seed:
        _process_seeds(results_dir, output_dir, sim_data, geodf, seed_centroid,
                       args.seed_prefix, args.seed, n_iterations_baseline)
    else:
        root_combos = discover_parameter_combinations(results_dir)
        if not root_combos:
            seed_dirs = sorted(
                (d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith(args.seed_prefix)),
                key=lambda d: int(d.name[len(args.seed_prefix):]) if d.name[len(args.seed_prefix):].isdigit() else 0
            )
            if seed_dirs:
                print(f"No 'run_daily_active_cases' files at root '{results_dir}'.")
                print(f"Auto-detected {len(seed_dirs)} '{args.seed_prefix}' subfolders. Processing per seed...")
                _process_seeds(results_dir, output_dir, sim_data, geodf, seed_centroid,
                               args.seed_prefix, args.seed, n_iterations_baseline)
            else:
                print(f"\nNo 'run_daily_active_cases' files found in '{results_dir}'.")
                print("Neither flat results nor 'seed_*' subfolders were discovered. Nothing to plot.")
        else:
            run_figures(results_dir, output_dir, sim_data, geodf, seed_centroid)


if __name__ == "__main__":
    main()