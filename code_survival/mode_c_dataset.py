"""
Mode C dataset builder: loads arrival-time records from seed-specific result
directories and assembles a combined survival-analysis dataset that reuses
the same column structure as Mode A/B.

Each ``results_hybrid_sim/seed_{X}/`` directory contains simulation outputs
for a specific metapopulation seed location (district X).  The dataset builder
loads I-based arrival-time records, daily active-case time series, and
per-run seed-size (Z_seed) from each seed subdirectory, computes pre-invasion
pressure covariates, and produces a DataFrame suitable for Cox regression
and KM analysis — identical in schema to Mode A/B with an additional
``seed_location`` column.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from .dataset_builder import (
    _build_records_for_config,
    compute_mobility_matrices_day_night,
    load_population_data,
    load_daily_active_cases,
    _resolve_daily_cases_path,
    load_Z_seed,
    load_arrival_times,
    compute_invasion_intensity,
    compute_pre_invasion_pressure,
)
from .discovery import (
    discover_seed_directories,
    discover_seed_param_combinations,
    create_param_combo_label,
)
from .seed_distance import (
    compute_effective_distance_matrix,
    compute_shortest_effective_path,
)


DEFAULT_SCENARIOS = ['AMS_dance', 'AMS_football', 'Leipzig_1', 'Leipzig_2', 'Leipzig_3', 'no_event']


def build_mode_c_survival_dataset(
    seed_locations: List[int],
    results_dir: Union[str, Path] = "results_hybrid_sim",
    r0: float = 1.5,
    beta: float = 0.5,
    iss: int = 1,
    scenarios: Optional[List[str]] = None,
    max_time: Optional[float] = None,
) -> pd.DataFrame:
    """Build the full Mode C survival dataset from seed subdirectories.

    Combines I-based arrival-time records, daily active cases, and per-run
    seed-size (Z_seed) from all specified seed directories.  Produces the
    **same column structure** as :func:`build_survival_dataset` (Mode A/B)
    plus a ``seed_location`` column identifying which metapopulation seed
    district the simulation started from.

    The ``simulation_run_id`` is prefixed with the seed location so that
    cluster-robust standard errors correctly account for within-seed-run
    correlation: ``f"seed_{X}_{scenario}_{run_id}"``.

    Args:
        seed_locations: List of seed district IDs.
        results_dir: Base results directory containing ``seed_{X}/`` subdirs.
        r0, beta, iss: Baseline epidemiological parameters.
        scenarios: Scenario names to include.  Defaults to
            ``['AMS_dance', 'AMS_football', 'Leipzig_1', 'Leipzig_2',
            'Leipzig_3', 'no_event']``.
        max_time: Censoring time for non-arrived districts.

    Returns:
        DataFrame with columns: ``district_id, run_id, scenario,
        seed_location, gathering_event, arrival_time, event, seed_size,
        log_seed_size, pre_pressure_3d, pre_pressure_7d, pre_pressure_14d,
        pre_pressure_sum_7d, R0, beta, simulation_run_id``.
    """
    results_dir = Path(results_dir)
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS

    configs: List[Dict[str, Any]] = []
    for scenario in scenarios:
        if scenario.lower() == 'no_event':
            configs.append({'scenario': scenario, 'R0': r0, 'beta': 0.1, 'Iss': iss})
        else:
            configs.append({'scenario': scenario, 'R0': r0, 'beta': beta, 'Iss': iss})

    M_day, _ = compute_mobility_matrices_day_night(results_dir)
    M = M_day.values
    N = load_population_data(results_dir).values

    all_records: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for seed in seed_locations:
        seed_dir = results_dir / f"seed_{seed}"
        if not seed_dir.exists():
            warnings.warn(f"Seed directory not found: {seed_dir}")
            continue

        for config in configs:
            if _resolve_daily_cases_path(
                seed_dir, config['scenario'], config['R0'],
                config['beta'], config['Iss'],
            ) is None:
                skipped.append(f"seed_{seed}/{config['scenario']}")
                continue
            records = _build_records_for_config(
                seed_dir, config, max_time, window_days=7, M=M, N=N
            )
            for r in records:
                r['seed_location'] = seed
            all_records.extend(records)

    if skipped:
        print(
            f"  Skipped {len(skipped)} scenario(s) missing daily-active-cases "
            f"files: {', '.join(skipped)}"
        )

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # simulation_run_id includes seed_location for cluster-robust SEs
    df['simulation_run_id'] = (
        f"seed_" + df['seed_location'].astype(str)
        + "_" + df['scenario'].astype(str)
        + "_" + df['run_id'].astype(str)
    )

    return df


def build_mode_c_dataset_for_params(
    seed_locations: List[int],
    results_dir: Union[str, Path] = "results_hybrid_sim",
    r0: float = 1.5,
    beta: float = 0.5,
    iss: int = 1,
    scenarios: Optional[List[str]] = None,
    max_time: Optional[float] = None,
) -> pd.DataFrame:
    """Alias for :func:`build_mode_c_survival_dataset` with explicit parameter names.

    Used by C2-A sensitivity analysis to build datasets for different
    transmission conditions.
    """
    return build_mode_c_survival_dataset(
        seed_locations=seed_locations,
        results_dir=results_dir,
        r0=r0,
        beta=beta,
        iss=iss,
        scenarios=scenarios,
        max_time=max_time,
    )


def compute_mean_arrival(
    seed_dir: Union[str, Path],
    scenario: str = "AMS_dance",
    r0: float = 1.5,
    beta: float = 0.5,
    iss: int = 1,
) -> pd.DataFrame:
    """Load I-based arrival-time records from a single seed directory and
    compute mean/median/std arrival time per district.

    Non-arrived districts (``arrived=0``) are censored at ``inferred_max_time``
    (derived from the daily-active-cases time grid, default 250 days).

    Returns DataFrame with columns:
        district, mean_arrival_time, median_arrival_time, std_arrival_time,
        n_arrived, n_runs, seed_location
    """
    seed_dir = Path(seed_dir)
    seed_location = int(seed_dir.name.split("_")[1])

    df = load_mode_c_arrival_times(seed_dir, scenario, r0, beta, iss)
    if df.empty:
        return pd.DataFrame()

    r0_enc = int(r0 * 100)
    beta_enc = int(beta * 100)
    if scenario.startswith("no_event"):
        daily_file = seed_dir / f"run_daily_active_cases_{scenario}_R{r0_enc}_Iss{iss}.csv"
    else:
        daily_file = seed_dir / f"run_daily_active_cases_{scenario}_R{r0_enc}_beta{beta_enc}_Iss{iss}.csv"
    if daily_file.exists():
        daily = pd.read_csv(daily_file, nrows=2)
        day_cols = [c for c in daily.columns if c.startswith("day_")]
        if day_cols:
            inferred_max_time = float(day_cols[-1].split("_")[1])
        else:
            inferred_max_time = 250.0
    else:
        inferred_max_time = 250.0

    df["arrival_time"] = df.apply(
        lambda row: row["arrival_time"] if row["arrived"] == 1
        else inferred_max_time,
        axis=1,
    )

    records = []
    for district in sorted(df["district_id"].unique()):
        dist_df = df[df["district_id"] == district]
        arrived = dist_df[dist_df["arrived"] == 1]
        records.append({
            "district": int(district),
            "mean_arrival_time": float(arrived["arrival_time"].mean()) if len(arrived) > 0 else float(inferred_max_time),
            "median_arrival_time": float(arrived["arrival_time"].median()) if len(arrived) > 0 else float(inferred_max_time),
            "std_arrival_time": float(arrived["arrival_time"].std()) if len(arrived) > 1 else 0.0,
            "n_arrived": int(len(arrived)),
            "n_runs": int(len(dist_df)),
            "seed_location": seed_location,
        })

    return pd.DataFrame(records)


def load_mode_c_arrival_times(
    seed_dir: Union[str, Path],
    scenario: str = "AMS_dance",
    r0: float = 1.5,
    beta: float = 0.5,
    iss: int = 1,
) -> pd.DataFrame:
    """Load I-based arrival-time records from a single seed directory.

    File naming convention:
    ``arrival_time_records_I_{scenario}_R{r0*100}_beta{beta*100}_Iss{iss}.csv``

    Returns DataFrame with columns:
        scenario_name, run_id, district_id, arrived, arrival_time,
        source_district_id, infection_context, event_day, step_index,
        seed_location (added)
    """
    seed_dir = Path(seed_dir)
    seed_location = int(seed_dir.name.split("_")[1])

    r0_encoded = int(r0 * 100)
    beta_encoded = int(beta * 100)

    file_path = seed_dir / f"arrival_time_records_I_{scenario}_R{r0_encoded}_beta{beta_encoded}_Iss{iss}.csv"

    if not file_path.exists():
        if scenario.lower() == "no_event":
            file_path = seed_dir / f"arrival_time_records_I_{scenario}_R{r0_encoded}_Iss{iss}.csv"
        if not file_path.exists():
            warnings.warn(f"Arrival time file not found: {file_path}")
            return pd.DataFrame()

    df = pd.read_csv(file_path)
    df["seed_location"] = seed_location
    return df


def compute_mean_arrival_multi(
    seed_locations: List[int],
    results_dir: Union[str, Path],
    scenario: str = "AMS_dance",
    r0: float = 1.5,
    beta: float = 0.5,
    iss: int = 1,
) -> pd.DataFrame:
    """Compute mean arrival-time table across multiple seed locations.

    Iterates over each seed directory, loads arrival-time records, and
    compiles a long-format DataFrame with one row per (seed, destination).
    """
    results_dir = Path(results_dir)
    all_dfs = []

    for seed in seed_locations:
        seed_dir = results_dir / f"seed_{seed}"
        if not seed_dir.exists():
            print(f"  (Skipping seed {seed}: directory not found)")
            continue

        df = compute_mean_arrival(
            seed_dir=str(seed_dir),
            scenario=scenario,
            r0=r0,
            beta=beta,
            iss=iss,
        )
        if df.empty:
            print(f"  WARNING: No arrival data for seed {seed}")
            continue

        all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


def get_effective_distances_from_seed(
    seed_district: int,
    output_dir: Union[str, Path],
    results_dir: Union[str, Path] = "results_hybrid_sim",
    r0: float = 1.5,
    iss: int = 1,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame of effective distances from *seed_district* to all
    destination districts.

    Columns: ``district, effective_distance``.

    This is an optional exploratory/descriptive helper.  It is NOT used
    as a covariate in the primary Mode C Cox models.
    """
    csv_path = Path(output_dir) / "effective_distance_Madrid.csv"

    if csv_path.exists() and not force_recompute:
        df = pd.read_csv(csv_path)
        seed_df = df[df["origin_district"] == seed_district].copy()
        seed_df = seed_df.rename(columns={"destination_district": "district"})
        seed_df = seed_df[["district", "effective_distance"]].reset_index(drop=True)
        return seed_df

    d_matrix = compute_effective_distance_matrix(results_dir, r0=r0, iss=iss)
    dists = compute_shortest_effective_path(d_matrix, seed_district)

    result = pd.DataFrame({
        "district": range(len(dists)),
        "effective_distance": dists,
    })
    return result
