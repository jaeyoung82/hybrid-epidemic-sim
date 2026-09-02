"""
Effective distance computation and caching for Mode C (seed location analysis).

Computes effective distances from each seed district to all other districts
using the Next Generation Matrix (NGM) and Dijkstra's shortest-path algorithm
on the directed effective-distance graph.

The effective distance from district *i* to district *j* is:

    d(i, j) = 1 - ln(1 - exp(-I_ss * R0 * S_mat[i, j]))

where S_mat is the structural NGM component (computed from the mobility
matrix M_ij and population N) and the effective distance matrix is asymmetric
because the NGM is directed.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union

from code_simulation.models.next_generation import compute_structural_ngm
from .dataset_builder import load_mobility_matrix, load_population_data


# ============================================================================
# Core computation
# ============================================================================

def compute_effective_distance_matrix(
    results_dir: Union[str, Path] = "results_hybrid_sim",
    r0: float = 1.5,
    iss: int = 1,
) -> np.ndarray:
    """Compute the full effective distance matrix d_matrix.

    Steps:
    1. Load mobility matrix M_ij (21x21) and population vector N.
    2. Derive N_ij = (M_ij / row_sum) * N_i  (the population-weighted
       mobility matrix used by the simulation's DataLoader).
    3. Compute structural NGM S_mat = compute_structural_ngm(N_ij).
    4. Compute p_matrix = 1 - exp(-I_ss * R0 * S_mat).
    5. Compute d_matrix = 1 - ln(1 - p_matrix).

    The resulting d_matrix is asymmetric (directed graph).

    Args:
        results_dir: Path to results directory (for locating data_mobility_Madrid).
        r0: Basic reproduction number.
        iss: Initial infectious seed size (I_ss).

    Returns:
        2D ndarray of shape (n_districts, n_districts) — effective distances.
    """
    M = load_mobility_matrix(results_dir)
    M_vals = M.to_numpy(dtype=float)

    N = load_population_data(results_dir)
    N_vals = N.reindex(M.index).to_numpy(dtype=float)

    # Row-normalise M and multiply by population to get N_ij
    row_sums = M_vals.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        D_ij = np.nan_to_num(M_vals / row_sums)
    N_ij = (D_ij * N_vals[:, np.newaxis]).round().astype(int)

    S_mat = compute_structural_ngm(N_ij)

    K = r0 * S_mat
    p_matrix = 1.0 - np.exp(-max(iss, 1e-10) * K)

    p_clipped = np.clip(p_matrix, 1e-10, 1.0 - 1e-10)
    d_matrix = 1.0 - np.log(p_clipped)

    return d_matrix


def compute_shortest_effective_path(
    d_matrix: np.ndarray,
    seed_district: int,
) -> np.ndarray:
    """Compute shortest effective-distance paths from *seed_district* to all
    destination districts.

    Uses Dijkstra on the directed d_matrix.  When called with a scalar
    index, ``scipy.sparse.csgraph.dijkstra`` returns a 1-D array.

    Args:
        d_matrix: Effective distance matrix (asymmetric).
        seed_district: Origin district index.

    Returns:
        1-D array of length n_districts with effective distances from the
        seed to each district.
    """
    from scipy.sparse.csgraph import dijkstra

    shortest = dijkstra(
        d_matrix, directed=True, indices=seed_district
    )

    if shortest.ndim > 1:
        shortest = np.min(shortest, axis=0)

    return shortest


# ============================================================================
# CSV loading / caching
# ============================================================================

def get_effective_distance_csv_path(output_dir: Union[str, Path]) -> Path:
    """Return the path where the effective-distance CSV is cached."""
    return Path(output_dir) / "effective_distance_Madrid.csv"


def load_or_compute_effective_distance(
    seed_locations: List[int],
    output_dir: Union[str, Path],
    results_dir: Union[str, Path] = "results_hybrid_sim",
    r0: float = 1.5,
    iss: int = 1,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Load effective-distance CSV if it exists, otherwise compute and save it.

    The CSV has a long format with columns:
        origin_district, destination_district, effective_distance
    plus a wide format with columns for each seed.

    Args:
        seed_locations: List of seed district IDs to compute distances for.
        output_dir: Base output directory for Mode C results.
        results_dir: Path to simulation results (for mobility data).
        r0: Basic reproduction number.
        iss: Initial infectious seed size.
        force_recompute: If True, recompute even if CSV exists.

    Returns:
        DataFrame with long-format effective distances.
    """
    csv_path = get_effective_distance_csv_path(output_dir)

    if csv_path.exists() and not force_recompute:
        df = pd.read_csv(csv_path)
        long_df = df[["origin_district", "destination_district", "effective_distance"]].copy()
        return long_df

    long_records = []
    wide_data = {}

    for seed in seed_locations:
        d_matrix = compute_effective_distance_matrix(results_dir, r0=r0, iss=iss)
        dists = compute_shortest_effective_path(d_matrix, seed)

        for dest in range(len(dists)):
            long_records.append({
                "origin_district": int(seed),
                "destination_district": int(dest),
                "effective_distance": float(dists[dest]),
            })

        col_name = f"district_{seed}"
        wide_data[col_name] = dists

    long_df = pd.DataFrame(long_records)

    wide_df = pd.DataFrame(wide_data)
    wide_df.insert(0, "destination_district", range(len(wide_df)))

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    long_df.to_csv(csv_path, index=False)
    wide_df.to_csv(output_dir_path / "effective_distance_wide.csv", index=False)

    return long_df


# ============================================================================
# Convenience accessor
# ============================================================================

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

    The seed district itself has an effective distance of 0 by construction
    (shortest path from a node to itself is zero).
    """
    csv_path = get_effective_distance_csv_path(output_dir)

    if csv_path.exists() and not force_recompute:
        df = pd.read_csv(csv_path)
    else:
        df = load_or_compute_effective_distance(
            [seed_district], output_dir, results_dir,
            r0=r0, iss=iss, force_recompute=force_recompute,
        )

    seed_df = df[df["origin_district"] == seed_district].copy()
    seed_df = seed_df.rename(columns={"destination_district": "district"})
    seed_df = seed_df[["district", "effective_distance"]].reset_index(drop=True)
    return seed_df
