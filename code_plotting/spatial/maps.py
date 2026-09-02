"""Spatial plotting functions for choropleth maps and heatmaps."""
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

try:
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except Exception:
    plt = None


def _get_daily_active_cases(all_runs_results, n_patches, n_iterations, n_days):
    """Helper to extract daily active cases for each district across runs."""
    daily_times = np.arange(n_days + 1)
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
        
        Active_j = E_in_j + I_in_j
        
        for p in range(n_patches):
            interpolated_cases = np.interp(daily_times, time_points, Active_j[:, p])
            all_runs_daily_cases[run_id, :, p] = interpolated_cases
    
    return daily_times, all_runs_daily_cases


def plot_spacetime_heatmap(all_runs_results, data, cfg):
    """Plot heatmap of active cases (E+I) for all districts over time."""
    if cfg.event_model_type.startswith('no_event'):
        scenario_name = cfg.event_model_type
    else:
        scenario_name = getattr(cfg, "current_event_scenario_name", cfg.event_model_type)
    
    n_patches = data['n_patches']
    n_iterations = cfg.n_iterations
    n_days = cfg.n_days
    
    daily_times, all_runs_daily_cases = _get_daily_active_cases(all_runs_results, n_patches, n_iterations, n_days)
    mean_daily_cases = np.mean(all_runs_daily_cases, axis=0)
    heatmap_data = mean_daily_cases.T
    
    # Sort districts by peak time
    peak_times = np.argmax(heatmap_data, axis=1)
    sorted_indices = np.argsort(peak_times)
    sorted_data = heatmap_data[sorted_indices, :]
    
    if HAS_PLOTTING:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if np.max(sorted_data) > 100:
            norm = plt.matplotlib.colors.LogNorm(vmin=max(1, np.min(sorted_data[sorted_data>0])), vmax=np.max(sorted_data))
        else:
            norm = plt.Normalize(vmin=0, vmax=np.max(sorted_data))
        
        im = ax.imshow(sorted_data, aspect='auto', cmap='Reds', interpolation='nearest',
                      origin='lower', extent=[0, n_days, 0, n_patches], norm=norm)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean Active Cases (E+I)')
        
        ax.set_title(f'Space-Time Heatmap of Epidemic Spread (Sorted by Peak Time) - {scenario_name}', fontsize=16)
        ax.set_xlabel('Time (Days)', fontsize=12)
        ax.set_ylabel('District ID (Sorted)', fontsize=12)
        
        results_dir = getattr(cfg, 'results_dir', Path("results"))
        # plt.savefig(results_dir / f'spacetime_heatmap_{scenario_name}_R{int(cfg.R_0*100)}_betaEvent{int(cfg.event_base_transmission_rate*100)}.png', dpi=300)
        plt.close(fig)


def plot_spatial_spread_snapshots(all_runs_results, data, cfg, vmax=None, scenario_name_override=None):
    """Plot choropleth map snapshots of epidemic at different time points."""
    if not HAS_GEOPANDAS:
        return
    
    results_dir = getattr(cfg, 'results_dir', Path("results"))
    mobility_dir = Path(f"mobility_{cfg.dataset_name}")
    geojson_path = mobility_dir / f"{cfg.dataset_name.lower()}-districts.geojson"
    
    if not geojson_path.exists():
        geojsons = list(mobility_dir.glob("*.geojson"))
        if geojsons:
            geojson_path = geojsons[0]
    
    if not geojson_path.exists():
        return
    
    geodf = gpd.read_file(geojson_path)
    
    # ID handling
    id_col_found = None
    for col_name in ['id', 'patch_id', 'ID', 'cartodb_id']:
        if col_name in geodf.columns:
            id_col_found = col_name
            break
    
    if id_col_found and id_col_found != 'id':
        geodf = geodf.rename(columns={id_col_found: 'id'})
    
    if not pd.api.types.is_numeric_dtype(geodf['id']):
        geodf['id'] = pd.to_numeric(geodf['id'], errors='coerce')
        geodf.dropna(subset=['id'], inplace=True)
        geodf['id'] = geodf['id'].astype(int)
    
    n_patches = data['n_patches']
    if geodf['id'].min() == 1 and geodf['id'].max() == n_patches:
        geodf['id'] = geodf['id'] - 1
    
    # Get daily cases
    n_days = cfg.n_days
    n_iterations = cfg.n_iterations
    _, all_runs_daily_cases = _get_daily_active_cases(all_runs_results, n_patches, n_iterations, n_days)
    mean_daily_cases = np.mean(all_runs_daily_cases, axis=0)
    
    time_points = getattr(cfg, 'custom_spatial_comparison_days', [0, 50, 100, 150, 200, 250])
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    if vmax is None:
        vmax = np.max(mean_daily_cases)
        if vmax == 0: vmax = 1
    
    norm = plt.matplotlib.colors.Normalize(vmin=0, vmax=vmax)
    mappable = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
    
    seed_patch_id = getattr(cfg, 'initial_infection_patch_id', 0)
    seed_coords = None
    try:
        seed_patch_row = geodf[geodf['id'] == seed_patch_id]
        if not seed_patch_row.empty:
            centroid_geom = seed_patch_row.to_crs(epsg=3857).centroid.to_crs(geodf.crs).iloc[0]
            seed_coords = (centroid_geom.x, centroid_geom.y)
    except Exception:
        pass
    
    for idx, t in enumerate(time_points):
        if idx >= len(axes):
            break
        ax = axes[idx]
        if t >= len(mean_daily_cases):
            t = len(mean_daily_cases) - 1
        
        cases_at_t = mean_daily_cases[t, :]
        cases_df = pd.DataFrame({'id': range(n_patches), 'cases': cases_at_t})
        merged_gdf = geodf.merge(cases_df, on='id', how='left').fillna(0)
        merged_gdf.plot(column='cases', cmap='Reds', linewidth=0.5, ax=ax, edgecolor='0.3', norm=norm)
        
        ax.set_title(f'Day {t}', fontsize=14)
        ax.set_aspect('equal', adjustable='box')
    
    plt.close(fig)