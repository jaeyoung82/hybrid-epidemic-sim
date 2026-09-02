"""
Automatic discovery of simulation outputs for survival analysis.

Scans simulation result directories to identify available
(scenario, R0, beta, Iss) combinations from file naming conventions.
"""

import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any


def discover_seed_directories(
    results_dir: Union[str, Path] = "results_hybrid_sim",
) -> List[int]:
    """Auto-discover seed directories and return sorted seed district IDs.

    Looks for ``results_hybrid_sim/seed_{X}/`` directories.
    Returns a list of integer seed IDs, e.g. ``[0, 7, 18]``.
    """
    results_path = Path(results_dir)
    seed_ids = []
    for d in sorted(results_path.glob("seed_*")):
        if d.is_dir():
            parts = d.name.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                seed_ids.append(int(parts[1]))
    return sorted(seed_ids)


def _scan_directory_for_configs(
    scan_dir: Path, seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Scan a single directory for simulation config files.

    Parses filenames to extract scenario, R0, beta, and Iss values.
    When *seed* is not ``None``, a ``seed`` key is added to each config dict.

    Args:
        scan_dir: Directory to scan for ``*.csv`` files.
        seed: Seed district ID to attach to discovered configs, or ``None``
            for flat (non-seeded) mode.

    Returns:
        List of config dicts with keys: ``scenario``, ``R0``, ``beta``,
        ``Iss``, ``has_arrival_E``, ``has_arrival_I``, ``has_daily``,
        ``has_E_totals``, and optionally ``seed``.
    """
    configs: Dict[Tuple, Dict[str, Any]] = {}

    pattern = re.compile(
        r'(?:arrival_time_records_[EI]|realization)_([a-zA-Z0-9_]+)_R(\d+)_beta(\d+)_Iss(\d+)'
    )

    for file_path in scan_dir.glob("*.csv"):
        match = pattern.search(file_path.name)
        if match:
            scenario, r0_encoded, beta_encoded, iss_encoded = match.groups()

            r0 = int(r0_encoded) / 100.0
            beta = int(beta_encoded) / 100.0
            iss = int(iss_encoded)

            key = (scenario, r0, beta, iss, seed)

            if key not in configs:
                configs[key] = {
                    'scenario': scenario,
                    'R0': r0,
                    'beta': beta,
                    'Iss': iss,
                    'has_arrival_E': False,
                    'has_arrival_I': False,
                    'has_daily': False,
                    'has_E_totals': False,
                }
                if seed is not None:
                    configs[key]['seed'] = seed

            if 'arrival_time_records_E' in file_path.name:
                configs[key]['has_arrival_E'] = True
            elif 'arrival_time_records_I' in file_path.name:
                configs[key]['has_arrival_I'] = True
            elif file_path.name.startswith('realization_') and '_E.csv' in file_path.name:
                configs[key]['has_E_totals'] = True

    for file_path in scan_dir.glob("run_daily_active_cases_*.csv"):
        match = re.search(r'run_daily_active_cases_([a-zA-Z0-9_]+)_R(\d+)(?:_beta(\d+))?_Iss(\d+)', file_path.name)
        if match:
            scenario = match.group(1)
            r0_encoded = match.group(2)
            beta_encoded = match.group(3) if match.group(3) else "0"
            iss_encoded = match.group(4)

            r0 = int(r0_encoded) / 100.0
            beta = int(beta_encoded) / 100.0
            iss = int(iss_encoded)

            key = (scenario, r0, beta, iss, seed)

            if key not in configs:
                configs[key] = {
                    'scenario': scenario,
                    'R0': r0,
                    'beta': beta,
                    'Iss': iss,
                    'has_arrival_E': False,
                    'has_arrival_I': False,
                    'has_daily': False,
                    'has_E_totals': False,
                }
                if seed is not None:
                    configs[key]['seed'] = seed

            configs[key]['has_daily'] = True

    return list(configs.values())


def discover_simulations(
    results_dir: Union[str, Path] = "results_hybrid_sim",
    seed_locations: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Scan results directory and discover all available simulation configurations.
    
    Parses filenames to extract scenario, R0, beta_event, and Iss values.
    
    When *seed_locations* is provided, scans ``results_dir/seed_{X}/``
    subdirectories for each seed and adds a ``seed`` column to each config.
    When *seed_locations* is ``None``, auto-discovers all ``seed_*``
    subdirectories.  When no seed subdirectories exist, falls back to
    scanning *results_dir* directly (flat mode, backward compatible).
    
    Filename patterns:
        arrival_time_records_E_{scenario}_R{r0*100}_beta{beta*100}_Iss{iss}.csv
        realization_{scenario}_R{r0*100}_beta{beta*100}_Iss{iss}_E.csv
    
    Returns DataFrame with columns: scenario, R0, beta, Iss, has_arrival_E,
    has_arrival_I, has_daily, has_E_totals, and optionally ``seed``.
    """
    results_path = Path(results_dir)

    # Determine which directories to scan and their associated seed values
    if seed_locations is not None:
        scan_specs = [(results_path / f"seed_{s}", s) for s in sorted(seed_locations)]
    else:
        auto_seeds = discover_seed_directories(results_path)
        if auto_seeds:
            scan_specs = [(results_path / f"seed_{s}", s) for s in auto_seeds]
        else:
            scan_specs = [(results_path, None)]

    all_configs: List[Dict[str, Any]] = []
    for scan_dir, seed_val in scan_specs:
        if not scan_dir.exists():
            continue
        all_configs.extend(_scan_directory_for_configs(scan_dir, seed_val))

    if not all_configs:
        return pd.DataFrame(columns=[
            'scenario', 'R0', 'beta', 'Iss',
            'has_arrival_E', 'has_arrival_I', 'has_daily', 'has_E_totals',
        ])

    return pd.DataFrame(all_configs)


def get_scenarios(results_dir: Union[str, Path] = "results_hybrid_sim") -> List[str]:
    """
    Return list of unique scenarios with available simulation results.
    
    Returns scenarios sorted alphabetically, excluding 'no_event' if it exists.
    """
    df = discover_simulations(results_dir)
    if df.empty:
        return []
    return sorted(df['scenario'].unique().tolist())


def get_scenario_names(results_dir: Union[str, Path] = "results_hybrid_sim") -> List[str]:
    """
    Return all scenario names including 'no_event' for analysis.
    
    This is the complete list used in dataset building.
    """
    df = discover_simulations(results_dir)
    if df.empty:
        return ['no_event']
    return sorted(df['scenario'].unique().tolist())


def get_param_combinations(results_dir: Union[str, Path] = "results_hybrid_sim",
                           seed_locations: Optional[List[int]] = None,
                           scenario: Optional[str] = None,
                           r0: Optional[float] = None,
                           iss: int = 1) -> List[Dict[str, Any]]:
    """Return list of parameter combinations for analysis.
    
    Args:
        results_dir: Path to results directory
        seed_locations: Seed district IDs to scan.  When ``None``,
            auto-discovers all ``seed_*`` subdirectories.  Configs are
            identical across seeds (same parameter grid), so the ``seed``
            column is not included in the returned dicts.
        scenario: If provided, filter to specific scenario
        r0: If provided, filter to specific R0 value
        iss: Initial seed size filter
        
    Returns:
        List of dicts with keys: 'scenario', 'R0', 'beta', 'Iss'
    """
    df = discover_simulations(results_dir, seed_locations=seed_locations)
    
    if df.empty:
        return []
    
    filters = [df['Iss'] == iss]
    
    if scenario:
        filters.append(df['scenario'] == scenario)
    if r0 is not None:
        filters.append(df['R0'] == r0)
    
    filtered = df
    for f in filters:
        filtered = filtered[f]
    
    required_cols = ['has_arrival_E', 'has_daily']
    for col in required_cols:
        if col in filtered.columns:
            filtered = filtered[filtered[col]]
    
    # Deduplicate by (scenario, R0, beta, Iss) — configs are the same across seeds
    unique_cols = ['scenario', 'R0', 'beta', 'Iss']
    config_cols = [c for c in unique_cols if c in filtered.columns]
    if config_cols:
        filtered = filtered.drop_duplicates(subset=config_cols)
    
    result = []
    for _, row in filtered.iterrows():
        result.append({
            'scenario': row['scenario'],
            'R0': row['R0'],
            'beta': row['beta'],
            'Iss': int(row['Iss'])
        })
    
    return sorted(result, key=lambda x: (x['R0'], x['beta']))


def create_param_combo_label(scenario: str, r0: float, beta: float, iss: int = 1) -> str:
    """
    Create a standardized label for a parameter combination.
    
    Format: "{scenario}_R{r0}_beta{beta}" e.g., "AMS_dance_R1.5_beta0.5"
    """
    return f"{scenario}_R{r0}_beta{beta}"


def get_no_event_baseline(results_dir: Union[str, Path] = "results_hybrid_sim", 
                           r0: float = 1.5, iss: int = 1) -> Dict:
    """
    Get the no_event baseline configuration for Mode A analysis.
    
    For no_event scenarios, uses beta=0.1 internally as defined in build_survival_dataset.
    """
    return {
        'scenario': 'no_event',
        'R0': r0,
        'beta': 0.1,
        'Iss': iss
    }


def discover_seed_param_combinations(
    seed_dir: Union[str, Path],
    scenario: str = "AMS_dance",
    iss: int = 1,
) -> List[Dict[str, Any]]:
    """Discover available (R0, beta) parameter combinations within a seed directory.

    Scans ``seed_dir`` for arrival-time record files matching the pattern
    ``arrival_time_records_I_{scenario}_R{r0*100}_beta{beta*100}_Iss{iss}.csv``
    and also the no_event baseline files.

    Args:
        seed_dir: Path to a ``seed_{X}`` directory.
        scenario: Scenario name to search for (default ``"AMS_dance"``).
        iss: Initial seed size to filter on.

    Returns:
        List of dicts with keys ``'scenario'``, ``'R0'``, ``'beta'``, ``'Iss'``
        sorted by R0 then beta.
    """
    seed_dir = Path(seed_dir)
    if not seed_dir.is_dir():
        return []

    configs = {}
    pattern = re.compile(
        r'arrival_time_records_I_([a-zA-Z0-9_]+)_R(\d+)_beta(\d+)_Iss(\d+)'
    )

    for file_path in seed_dir.glob("arrival_time_records_I_*.csv"):
        match = pattern.search(file_path.name)
        if match:
            scenario_name, r0_enc, beta_enc, iss_enc = match.groups()
            r0 = int(r0_enc) / 100.0
            beta = int(beta_enc) / 100.0
            iss_val = int(iss_enc)

            if scenario_name != scenario:
                continue
            if iss_val != iss:
                continue

            key = (scenario_name, r0, beta, iss_val)
            configs[key] = {
                'scenario': scenario_name,
                'R0': r0,
                'beta': beta,
                'Iss': iss_val,
            }

    if not configs:
        return []

    return sorted(configs.values(), key=lambda x: (x['R0'], x['beta']))


def get_c2a_transmission_conditions(results_dir: Union[str, Path] = "results_hybrid_sim",
                                     seed: int = 0,
                                     scenario: str = "AMS_dance",
                                     iss: int = 1) -> List[Dict[str, Any]]:
    """Determine low, baseline, and high transmission conditions for C2-A.

    Uses the existing Mode B parameter sweep ranges. The baseline condition
    matches the primary Mode C analysis parameters (R0=1.5, beta=0.5).
    Lower and higher conditions are selected from the available sweep grid.

    Falls back to hardcoded defaults if discovery fails.

    Args:
        results_dir: Base results directory.
        seed: Seed directory to inspect (default 0).
        scenario: Scenario name.
        iss: Initial seed size.

    Returns:
        List of three dicts: [{'condition': 'lower', ...},
                              {'condition': 'baseline', ...},
                              {'condition': 'higher', ...}]
    """
    seed_dir = Path(results_dir) / f"seed_{seed}"
    available = discover_seed_param_combinations(seed_dir, scenario=scenario, iss=iss)

    if not available:
        return [
            {'condition': 'lower', 'scenario': scenario, 'R0': 1.0, 'beta': 0.1, 'Iss': iss},
            {'condition': 'baseline', 'scenario': scenario, 'R0': 1.5, 'beta': 0.5, 'Iss': iss},
            {'condition': 'higher', 'scenario': scenario, 'R0': 3.0, 'beta': 0.5, 'Iss': iss},
        ]

    r0_values = sorted(set(c['R0'] for c in available))
    beta_values = sorted(set(c['beta'] for c in available))

    def _find_condition(target_r0, target_beta):
        for c in available:
            if c['R0'] == target_r0 and c['beta'] == target_beta:
                return c
        for c in available:
            if c['R0'] == target_r0 and c['beta'] == beta_values[-1]:
                return c
        return available[len(available) // 2]

    lower = _find_condition(r0_values[0], beta_values[0])
    baseline = _find_condition(1.5, beta_values[-1])
    higher = _find_condition(r0_values[-1], beta_values[-1])

    return [
        {'condition': 'lower', **lower},
        {'condition': 'baseline', **baseline},
        {'condition': 'higher', **higher},
    ]

