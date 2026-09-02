"""
Core SEIR simulation logic for the Hybrid Epidemic Simulation Model.

Backward-compatibility layer that delegates to the new modular structure.
New code should import from simulation.models directly.
"""
from typing import Dict, Any, Optional
import numpy as np

# Re-export from new modular structure for backward compatibility
from .models import compute_lambda, compute_SEIR
from .models.next_generation import compute_rt_from_state, compute_structural_ngm
from .models.gillespie import _infer_metapop_infection_source


def analyze_metapopulation_percolation_risk(data: Dict[str, Any], cfg: Any) -> tuple:
    """Analyzes the percolation risk of the metapopulation network."""
    n_patches = data['n_patches']
    N_ij = data['N_ij_initial']
    
    if N_ij.ndim == 1:
        N_ij = N_ij.reshape(n_patches, n_patches)
    
    N_i = N_ij.sum(axis=1)
    N_j_day = N_ij.sum(axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        P_ij = N_ij / N_i[:, np.newaxis]
    P_ij = np.nan_to_num(P_ij)
    
    beta = cfg.beta
    gamma = 1.0 / cfg.period_infectious
    T_day = 0.5
    T_night = 0.5
    
    with np.errstate(divide='ignore', invalid='ignore'):
        M_uj = N_ij / N_j_day[np.newaxis, :]
    M_uj = np.nan_to_num(M_uj)
    
    K_day = (T_day * beta) * (P_ij @ M_uj.T)
    K_night = np.zeros((n_patches, n_patches))
    np.fill_diagonal(K_night, T_night * beta)
    
    K_total = (K_day + K_night) * (1.0 / gamma)
    
    p_vu = 1 - np.exp(-K_total)
    
    rng = np.random.default_rng(cfg.rnd_seed_0)
    rand_vals = rng.random((n_patches, n_patches))
    
    adj = (p_vu > rand_vals).astype(int)
    
    seed_patch = getattr(cfg, 'initial_infection_patch_id', 0)
    visited = set()
    stack = [seed_patch]
    visited.add(seed_patch)
    while stack:
        curr = stack.pop()
        neighbors = np.where(adj[curr, :] == 1)[0]
        for n in neighbors:
            if n not in visited:
                visited.add(n)
                stack.append(n)
                
    reachable_count = len(visited)
    reachable_fraction = reachable_count / n_patches
    
    return reachable_fraction, reachable_count


def compute_percolation_metrics(data: Dict[str, Any], cfg: Any, I_ss: int) -> tuple:
    """Computes effective distance matrix based on percolation theory."""
    import pandas as pd
    S_mat = compute_structural_ngm(data)
    R0 = getattr(cfg, 'R_0', 1.0)
    K = R0 * S_mat
    
    p_matrix = 1 - np.exp(-np.maximum(I_ss, 1e-10) * K)
    
    with np.errstate(divide='ignore'):
        d_matrix = 1.0 - np.log(np.clip(p_matrix, 1e-10, 1 - 1e-10))
        
    return p_matrix, d_matrix


def compute_shortest_effective_paths(d_matrix: np.ndarray, seed_indices: Any) -> np.ndarray:
    """Computes shortest path distances from seed patches to all other patches."""
    try:
        from scipy.sparse.csgraph import dijkstra
        shortest_paths_all = dijkstra(d_matrix, directed=True, indices=seed_indices)
        
        if shortest_paths_all.ndim > 1:
            shortest_paths = np.min(shortest_paths_all, axis=0)
        else:
            shortest_paths = shortest_paths_all
            
        return shortest_paths
    except ImportError:
        print("Warning: scipy.sparse.csgraph not found. Shortest path calculation skipped.")
        return np.array([])


def _seed_initial_infections(cfg, S, I, n_patches, run_id, initial_infections_data):
    """Seeds the initial infections based on configuration and data."""
    import pandas as pd
    if getattr(cfg, 'no_event_seeding_mode', 'single_patch') == 'from_file' and initial_infections_data is not None and isinstance(initial_infections_data, pd.DataFrame):
        day0_infections_df = initial_infections_data[
            (initial_infections_data['run_id'] == run_id) &
            (initial_infections_data['event_day'] == 0)
        ]
        S_reshaped = S.reshape((n_patches, n_patches))
        I_reshaped = I.reshape((n_patches, n_patches))
        for _, row in day0_infections_df.iterrows():
            home_patch = int(row['home_patch_id'])
            comm_patch = int(row['commuting_district_id'])
            if S_reshaped[home_patch, comm_patch] > 0:
                S_reshaped[home_patch, comm_patch] -= 1
                I_reshaped[home_patch, comm_patch] += 1
        return S_reshaped.ravel(), I_reshaped.ravel()
    
    elif getattr(cfg, 'no_event_seeding_mode', 'single_patch') == 'single_patch' and getattr(cfg, 'I_ss', 0) > 0:
        if getattr(cfg, 'event_model_type', '') == 'large_venue' and getattr(cfg, 'large_venue_ini_infected_attendee_seed', None) == 'single_patch':
            seed_patch_idx = getattr(cfg, 'large_venue_ini_infected_attendee_patch', getattr(cfg, 'initial_infection_patch_id', 0))
        else:
            seed_patch_idx = getattr(cfg, 'initial_infection_patch_id', 0)
        
        seed_subpopulation_idx = seed_patch_idx * n_patches + seed_patch_idx
        
        if getattr(cfg, 'seeding_method', 'recruit') == 'recruit':
            num_to_infect = min(getattr(cfg, 'I_ss', 0), S[seed_subpopulation_idx])
            S[seed_subpopulation_idx] -= num_to_infect
            I[seed_subpopulation_idx] += num_to_infect
        else:
            I[seed_subpopulation_idx] += getattr(cfg, 'I_ss', 0)
        
    return S, I


def precompute_event_outcomes(data: Dict[str, Any], cfg: Any, rng: np.random.Generator, run_id: int) -> Dict[str, Any]:
    """Pre-calculates the micro-scale event outcomes (recruitment + transmission) for t=0."""
    # Delegates to simulation.events module
    from .events import _recruit_and_transmit_large_venue
    n_patches = data['n_patches']
    N_ij_initial_arr = data['N_ij_initial'].ravel().astype(np.int32)
    S = N_ij_initial_arr.copy()
    E = np.zeros_like(N_ij_initial_arr)
    I_in = np.zeros_like(N_ij_initial_arr)
    R = np.zeros_like(N_ij_initial_arr)
    
    S, I_out = _seed_initial_infections(cfg, S, I_in, n_patches, run_id, None)
    
    micro_results = _recruit_and_transmit_large_venue(data, cfg, rng, S, E, I_out, R, run_id)
    
    return micro_results