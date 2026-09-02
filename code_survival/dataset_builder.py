"""
Dataset builder for survival analysis.

Builds event-level survival analysis dataset with pre-invasion pressure covariates.
Supports both scenario comparison (Mode A) and parameter sensitivity (Mode B) analyses.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings

import numpy as np
import pandas as pd

from .discovery import (
    get_scenario_names,
    get_param_combinations,
    get_no_event_baseline,
    create_param_combo_label,
    discover_seed_directories,
)


def load_mobility_matrix(results_dir: Union[str, Path] = "results_hybrid_sim") -> pd.DataFrame:
    """Load the commuter flow matrix M_ij from mobility data."""
    results_dir = Path(results_dir)
    flow_file = results_dir.parent / "data_mobility_Madrid" / "flow_matrix_Madrid.csv"
    if not flow_file.exists():
        flow_file = Path("data_mobility_Madrid") / "flow_matrix_Madrid.csv"
    
    df = pd.read_csv(flow_file, index_col=0)
    df.columns = [int(col.replace('zone_', '')) for col in df.columns]
    df.index = [int(idx.replace('zone_', '')) for idx in df.index]
    return df


def load_population_data(results_dir: Union[str, Path] = "results_hybrid_sim") -> pd.Series:
    """Load district populations N_j."""
    results_dir = Path(results_dir)
    pop_file = results_dir.parent / "data_mobility_Madrid" / "population_Madrid.csv"
    if not pop_file.exists():
        pop_file = Path("data_mobility_Madrid") / "population_Madrid.csv"
    
    df = pd.read_csv(pop_file)
    return df.set_index('id')['population']


def _resolve_daily_cases_path(
    results_dir: Union[str, Path],
    scenario: str,
    r0: float,
    beta: float,
    iss: int,
) -> Optional[Path]:
    """Resolve the daily-active-cases CSV path for a scenario.

    Returns the :class:`~pathlib.Path` if it exists (falling back to the
    beta-suffixed name for ``no_event`` scenarios), or ``None`` if no
    matching file is present.
    """
    results_dir = Path(results_dir)
    r0_encoded = int(r0 * 100)
    iss_encoded = iss
    beta_encoded = int(beta * 100)

    if scenario.lower() == 'no_event':
        file_path = results_dir / f"run_daily_active_cases_no_event_R{r0_encoded}_Iss{iss_encoded}.csv"
        if not file_path.exists():
            file_path = results_dir / f"run_daily_active_cases_no_event_R{r0_encoded}_beta{beta_encoded}_Iss{iss_encoded}.csv"
    else:
        file_path = results_dir / f"run_daily_active_cases_{scenario}_R{r0_encoded}_beta{beta_encoded}_Iss{iss_encoded}.csv"

    return file_path if file_path.exists() else None


def load_daily_active_cases(results_dir: Union[str, Path], scenario: str, r0: float, beta: float, iss: int) -> pd.DataFrame:
    """Load daily active cases (E+I) per district per run.

    Returns an empty DataFrame (without warning) when the file is absent.
    Callers that iterate scenarios should use
    :func:`_resolve_daily_cases_path` to detect and summarize missing inputs
    rather than relying on a per-call warning.
    """
    file_path = _resolve_daily_cases_path(results_dir, scenario, r0, beta, iss)
    if file_path is None:
        return pd.DataFrame()
    return pd.read_csv(file_path)


def load_Z_seed(results_dir: Union[str, Path], scenario: str, r0: float, beta: float, iss: int) -> pd.DataFrame:
    """Load Z_seed values per run."""
    results_dir = Path(results_dir)
    
    if scenario and scenario.lower() != 'no_event':
        r0_encoded = int(r0 * 100)
        beta_encoded = int(beta * 100)
        
        pattern = f"n_new_infections_R{r0_encoded}_betaEvent{beta_encoded}_Iss{iss}.txt"
        file_path = results_dir / pattern
        
        if not file_path.exists():
            warnings.warn(f"n_new_infections file not found: {file_path}")
            return pd.DataFrame({'run_id': [], 'Z_seed': []})
        
        df = pd.read_csv(file_path)
        df = df[df['scenario_name'] == scenario].copy()
        if df.empty:
            return pd.DataFrame({'run_id': [], 'Z_seed': []})
        return df[['run_id', 'n_all']].rename(columns={'n_all': 'Z_seed'}).reset_index(drop=True)
    else:
        return pd.DataFrame({'run_id': [], 'Z_seed': []})


def load_arrival_times(results_dir: Union[str, Path], scenario: str, r0: float, beta: float, iss: int, use_E: bool = False) -> pd.DataFrame:
    """Load arrival time records for I-based (or E-based) arrivals."""
    results_dir = Path(results_dir)
    
    r0_encoded = int(r0 * 100)
    iss_encoded = iss
    
    if scenario.lower() == 'no_event' or use_E:
        file_path = results_dir / f"arrival_time_records_E_{scenario}_R{r0_encoded}_Iss{iss_encoded}.csv"
        if not file_path.exists():
            beta_encoded = int(beta * 100)
            file_path = results_dir / f"arrival_time_records_E_{scenario}_R{r0_encoded}_beta{beta_encoded}_Iss{iss_encoded}.csv"
    else:
        beta_encoded = int(beta * 100)
        file_path = results_dir / f"arrival_time_records_I_{scenario}_R{r0_encoded}_beta{beta_encoded}_Iss{iss_encoded}.csv"
    
    if not file_path.exists():
        warnings.warn(f"Arrival time file not found: {file_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    
    if 'district_id' in df.columns:
        df = df.rename(columns={'district_id': 'district'})
    
    return df[['run_id', 'district', 'arrived', 'arrival_time']].copy()


def load_E_by_run(results_dir: Union[str, Path], scenario: str, r0: float, beta: float, iss: int) -> pd.DataFrame:
    """Load run-aggregated exposed (E) counts over time."""
    results_dir = Path(results_dir)
    
    r0_encoded = int(r0 * 100)
    iss_encoded = iss
    
    if scenario and scenario.lower() != 'no_event':
        beta_encoded = int(beta * 100)
        file_path = results_dir / f'realizations_{scenario}_R{r0_encoded}_beta{beta_encoded}_Iss{iss_encoded}_E.csv'
    else:
        file_path = results_dir / f'realizations_no_event_R{r0_encoded}_Iss{iss_encoded}_E.csv'
        if not file_path.exists():
            beta_encoded = int(beta * 100)
            file_path = results_dir / f'realizations_no_event_R{r0_encoded}_beta{beta_encoded}_Iss{iss_encoded}_E.csv'
    
    if not file_path.exists():
        warnings.warn(f'E totals file not found: {file_path}')
        return pd.DataFrame({'run_id': [], 'time': [], 'E_total': []})
    
    df = pd.read_csv(file_path)
    time_col = df['time'].values
    run_cols = [c for c in df.columns if c.startswith('run_')]
    
    records = []
    for col in run_cols:
        run_id = int(col.split('_')[1])
        for t_idx, t in enumerate(time_col):
            records.append({'run_id': run_id, 'time': t, 'E_total': df[col].iloc[t_idx]})
    
    return pd.DataFrame(records)


def compute_mobility_matrices_day_night(results_dir: Union[str, Path] = 'results_hybrid_sim') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load separate day and night mobility matrices."""
    results_dir = Path(results_dir)
    flow_file = results_dir.parent / 'data_mobility_Madrid' / 'flow_matrix_Madrid.csv'
    if not flow_file.exists():
        flow_file = Path('data_mobility_Madrid') / 'flow_matrix_Madrid.csv'
    
    df = pd.read_csv(flow_file, index_col=0)
    df.columns = [int(col.replace('zone_', '')) for col in df.columns]
    df.index = [int(idx.replace('zone_', '')) for idx in df.index]
    
    M_day = df.copy()
    M_night = df.copy()
    return M_day, M_night


def compute_invasion_intensity(I_districts_t: np.ndarray, M: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Compute invasion intensity phi_d(t) for all districts at time t.
    
    Args:
        I_districts_t: Array of infected individuals per district at time t, shape (n_patches,)
        M: Mobility matrix, shape (n_patches, n_patches)
        N: Population per district, shape (n_patches,)
        
    Returns:
        Array of invasion intensity values for each district, shape (n_patches,)
    """
    N_safe = np.where(N > 0, N, 1.0)
    I_ratio = I_districts_t / N_safe
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = (M.T * I_ratio).sum(axis=0)
    return np.nan_to_num(phi)


def compute_pre_invasion_pressure(phi_matrix: np.ndarray, times: np.ndarray, arrival_time: float,
                                  district: int, window_days: float = 7, sum_pressure: bool = False) -> float:
    """Compute mean pressure over window before arrival time.
    
    Args:
        phi_matrix: Invasion intensity time series, shape (n_time_steps, n_districts)
        times: Array of time points, shape (n_time_steps,)
        arrival_time: Time of arrival for the district
        district: District index
        window_days: Size of window before arrival to average over
        sum_pressure: If True, sum values instead of averaging
        
    Returns:
        Mean (or sum) pre-invasion pressure value
    """
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    window_steps = int(window_days / dt)
    
    if arrival_time == np.inf or arrival_time >= times[-1]:
        end_time = times[-1]
    else:
        end_time = arrival_time
    
    end_idx = np.argmin(np.abs(times - end_time))
    
    if end_idx < window_steps:
        return 0.0
    
    window_values = phi_matrix[end_idx - window_steps:end_idx, district]
    
    if sum_pressure:
        return float(window_values.sum())
    else:
        return float(window_values.mean())


def _build_records_for_config(
    results_dir: Union[str, Path],
    config: Dict[str, Any],
    max_time: Optional[float],
    window_days: float,
    M: Optional[np.ndarray] = None,
    N: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Build survival-analysis records for a single (scenario, R0, beta, Iss) config.

    This helper encapsulates the per-config loading logic shared by
    :func:`build_survival_dataset` (Mode A/B) and
    :func:`build_mode_c_survival_dataset` (Mode C).

    Reads I-based or E-based arrival-time records, daily active-case time
    series, and the per-run seed-size (Z_seed) from *results_dir*, then
    computes pre-invasion pressure covariates via the invasion-intensity
    time series.

    Args:
        results_dir: Directory containing the simulation output files.
        config: Dict with keys ``'scenario'``, ``'R0'``, ``'beta'``, ``'Iss'``.
        max_time: Censoring time for non-arrived districts (infers from data
            when *None*).
        window_days: Window size (days) for pre-invasion pressure.
        M: Pre-loaded mobility matrix (n×n).  Computed lazily if *None*.
        N: Pre-loaded population vector (n,).  Computed lazily if *None*.

    Returns:
        List of record dicts, each with keys:
        ``district_id, run_id, scenario, gathering_event, arrival_time,
        event, seed_size, log_seed_size, pre_pressure_3d, pre_pressure_7d,
        pre_pressure_14d, pre_pressure_sum_7d, R0, beta``.
    """
    scenario = config['scenario']
    r0_val = config['R0']
    beta_val = config['beta']
    iss_val = config['Iss']

    beta_to_load = 0.1 if scenario.lower() == 'no_event' else beta_val

    records: List[Dict[str, Any]] = []

    try:
        daily_cases = load_daily_active_cases(results_dir, scenario, r0_val, beta_to_load, iss_val)
        if daily_cases.empty:
            return records

        Z_df = load_Z_seed(results_dir, scenario, r0_val, beta_to_load, iss_val)

        use_e_arrivals = (scenario.lower() == 'no_event')
        arrival_df = load_arrival_times(results_dir, scenario, r0_val, beta_to_load, iss_val, use_E=use_e_arrivals)
        if arrival_df.empty:
            return records

        if M is None or N is None:
            M_day, _ = compute_mobility_matrices_day_night(results_dir)
            M = M_day.values
            N = load_population_data(results_dir).values

        n_districts = int(daily_cases['district_id'].nunique())
        day_cols = [col for col in daily_cases.columns if col.startswith('day_')]
        times = np.array([float(col.split('_')[1]) for col in day_cols])

        inferred_max_time = float(times[-1]) if max_time is None else max_time

        arrival_lookup = {}
        for _, row in arrival_df.iterrows():
            key = (int(row['run_id']), int(row['district']))
            arrival_lookup[key] = {'arrived': row['arrived'], 'arrival_time': row['arrival_time']}

        z_lookup = {int(r['run_id']): r['Z_seed'] for _, r in Z_df.iterrows()} if not Z_df.empty else {}

        for run_id in daily_cases['run_id'].unique():
            run_cases = daily_cases[daily_cases['run_id'] == run_id]

            I_matrix = run_cases[day_cols].values.T
            phi_timeseries = np.zeros((len(times), n_districts))
            for t_idx, t in enumerate(times):
                I_t = I_matrix[t_idx, :]
                phi_timeseries[t_idx, :] = compute_invasion_intensity(I_t, M, N)

            Z_seed = z_lookup.get(int(run_id), 0.0)
            log_Z_seed = np.log1p(Z_seed)
            gathering_event = 0 if scenario.lower() == 'no_event' else 1

            for district in range(n_districts):
                arrival_info = arrival_lookup.get((int(run_id), district), {})

                if arrival_info.get('arrived') == 1:
                    at = arrival_info['arrival_time']
                    event = 1
                else:
                    at = inferred_max_time
                    event = 0

                pre_p_3d = compute_pre_invasion_pressure(phi_timeseries, times, at, district, window_days=3)
                pre_p_7d = compute_pre_invasion_pressure(phi_timeseries, times, at, district, window_days=7)
                pre_p_14d = compute_pre_invasion_pressure(phi_timeseries, times, at, district, window_days=14)
                pre_p_sum = compute_pre_invasion_pressure(phi_timeseries, times, at, district, window_days=7, sum_pressure=True)

                record = {
                    'district_id': int(district),
                    'run_id': int(run_id),
                    'scenario': scenario,
                    'gathering_event': gathering_event,
                    'arrival_time': at,
                    'event': event,
                    'seed_size': Z_seed,
                    'log_seed_size': log_Z_seed,
                    'pre_pressure_3d': pre_p_3d,
                    'pre_pressure_7d': pre_p_7d,
                    'pre_pressure_14d': pre_p_14d,
                    'pre_pressure_sum_7d': pre_p_sum,
                    'R0': r0_val,
                    'beta': beta_val
                }

                records.append(record)

    except Exception as e:
        warnings.warn(f"Error processing {scenario} R{r0_val} beta{beta_val}: {e}")

    return records


def build_survival_dataset(results_dir: Union[str, Path] = "results_hybrid_sim",
                           configs: Optional[List[Dict[str, Any]]] = None,
                           r0: float = 1.5,
                           beta: float = 0.5,
                           iss: int = 1,
                           max_time: Optional[float] = None,
                           window_days: float = 7,
                           analysis_mode: str = "scenario",
                           seed_locations: Optional[List[int]] = None) -> pd.DataFrame:
    """Build event-level survival analysis dataset with pre-invasion pressure covariates.
    
    Args:
        results_dir: Path to results directory
        configs: List of dicts with 'scenario', 'R0', 'beta', 'Iss' keys
        r0, beta, iss: Default parameters for Mode A (scenario comparison)
        max_time: Maximum time for censoring
        window_days: Window size for pre-pressure computation
        analysis_mode: "scenario" for comparing scenarios, "parameter" for sensitivity analysis
        seed_locations: If provided, restrict (or enable) iteration over
            ``results_dir/seed_{X}/`` subdirectories.  When ``None``,
            auto-discovers all seed subdirectories.  When an empty list is
            given ``[]``, forces flat-directory mode.  If no seed
            subdirectories exist, falls back to flat mode automatically.
        
    Returns:
        DataFrame with columns including district_id, scenario, arrival_time, event,
        seed_size, pre_pressure_*, R0, beta
    """
    results_dir = Path(results_dir)
    
    # Determine whether to use seed subdirectory mode
    if seed_locations is None:
        auto_seeds = discover_seed_directories(results_dir)
        if auto_seeds:
            seed_locations = auto_seeds

    use_seeds = seed_locations is not None and len(seed_locations) > 0
    
    if analysis_mode == "scenario" and configs is None:
        scenarios = ['AMS_dance', 'AMS_football', 'Leipzig_1', 'Leipzig_2', 'Leipzig_3', 'no_event']
        configs = []
        for scenario in scenarios:
            if scenario.lower() == 'no_event':
                configs.append({'scenario': scenario, 'R0': r0, 'beta': 0.1, 'Iss': iss})
            else:
                configs.append({'scenario': scenario, 'R0': r0, 'beta': beta, 'Iss': iss})
    
    # Load mobility and population data once for all configs
    M_day, _ = compute_mobility_matrices_day_night(results_dir)
    M = M_day.values
    N = load_population_data(results_dir).values
    
    all_records: List[Dict[str, Any]] = []
    skipped: List[str] = []
    
    if use_seeds:
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
                    seed_dir, config, max_time, window_days, M=M, N=N
                )
                for r in records:
                    r['seed_location'] = seed
                all_records.extend(records)
    else:
        for config in configs:
            if _resolve_daily_cases_path(
                results_dir, config['scenario'], config['R0'],
                config['beta'], config['Iss'],
            ) is None:
                skipped.append(f"{config['scenario']}")
                continue
            records = _build_records_for_config(results_dir, config, max_time, window_days, M=M, N=N)
            all_records.extend(records)
    
    if skipped:
        print(
            f"  Skipped {len(skipped)} scenario(s) missing daily-active-cases "
            f"files: {', '.join(skipped)}"
        )
    
    df = pd.DataFrame(all_records)
    
    if df.empty:
        return pd.DataFrame()
    
    if analysis_mode == "parameter":
        df['param_combo'] = df.apply(
            lambda row: create_param_combo_label(row['scenario'], row['R0'], row['beta'], row.get('Iss', iss)), 
            axis=1
        )
    
    # Create simulation_run_id for cluster-robust standard errors.
    # Districts within the same stochastic simulation run share the same epidemic
    # realization, commuter mobility, and transmission history, so they are not
    # statistically independent. The cluster ID uniquely identifies each run.
    if use_seeds:
        seed_prefix = "seed_" + df['seed_location'].astype(str) + "_"
    else:
        seed_prefix = ""
    
    if 'param_combo' in df.columns:
        df['simulation_run_id'] = seed_prefix + df['param_combo'].astype(str) + '_' + df['run_id'].astype(str)
    else:
        df['simulation_run_id'] = seed_prefix + df['scenario'].astype(str) + '_' + df['run_id'].astype(str)
    
    return df