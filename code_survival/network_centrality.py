"""
Network centrality computation for the Madrid district network.

Loads the mobility matrix and population data, builds a directed weighted
graph, and computes all centrality measures required for Part C of the
Mode C analysis (seed location impact analysis).

Centrality measures computed:
    InStrength, OutStrength, TotalStrength, Betweenness, Closeness,
    Eigenvector, PageRank, IntraDistrictRatio, InterDistrictRatio,
    WeightedClusteringCoefficient, OutgoingEntropy, MeanEffectiveDistance,
    AverageShortestPathLength

The results are cached to ``network_centrality_Madrid.csv`` in the
output directory.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Tuple

import networkx as nx

from .dataset_builder import load_mobility_matrix, load_population_data
from .seed_distance import compute_effective_distance_matrix


def build_transportation_graph(
    results_dir: Union[str, Path] = "results_hybrid_sim",
) -> Tuple[nx.DiGraph, np.ndarray, np.ndarray, pd.Index]:
    """Build the directed weighted transportation graph from mobility data.

    Steps:
        1. Load mobility matrix M_ij and population vector N.
        2. Derive N_ij = row_normalise(M_ij) * N_i  (population-weighted flows).
        3. Construct a directed graph where edge weight = N_ij flow.

    Args:
        results_dir: Path to simulation results (for locating data files).

    Returns:
        Tuple of (graph, N_ij matrix, population vector, district index).
    """
    results_dir = Path(results_dir)
    M = load_mobility_matrix(results_dir)
    N = load_population_data(results_dir)
    N_vals = N.reindex(M.index).to_numpy(dtype=float)
    M_vals = M.to_numpy(dtype=float)

    row_sums = M_vals.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        D_ij = np.nan_to_num(M_vals / row_sums)
    N_ij = (D_ij * N_vals[:, np.newaxis]).round().astype(int)

    n = N_ij.shape[0]
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i, population=N_vals[i])
        for j in range(n):
            if N_ij[i, j] > 0 and i != j:
                G.add_edge(i, j, weight=float(N_ij[i, j]))

    return G, N_ij, N_vals, M.index


def _compute_outgoing_entropy(N_ij: np.ndarray) -> np.ndarray:
    """Compute the outgoing-flow entropy for each district.

    H_i = -sum_j p_ij * ln(p_ij),  where p_ij = N_ij[i,j] / sum_j N_ij[i,j].

    A district with all outgoing flow concentrated to one neighbour has
    H_i = 0; a district with uniformly distributed outgoing flows has
    H_i = ln(n_destinations).
    """
    row_sums = N_ij.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.nan_to_num(N_ij / row_sums)
    n = N_ij.shape[0]
    entropies = np.zeros(n)
    for i in range(n):
        p_row = P[i]
        mask = p_row > 0
        if mask.sum() > 0:
            entropies[i] = -np.sum(p_row[mask] * np.log(p_row[mask]))
    return entropies


def _compute_intra_inter_ratio(N_ij: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute intra- and inter-district ratios.

    intra_ratio_i  = N_ij[i, i] / sum_j N_ij[i, j]
    inter_ratio_i  = (sum_{j != i} N_ij[i, j]) / sum_j N_ij[i, j]
    """
    row_sums = N_ij.sum(axis=1, keepdims=True)
    diag = np.diag(N_ij).reshape(-1, 1)
    off_diag = row_sums - diag
    with np.errstate(divide="ignore", invalid="ignore"):
        intra = np.nan_to_num(diag / np.where(row_sums > 0, row_sums, 1)).flatten()
        inter = np.nan_to_num(off_diag / np.where(row_sums > 0, row_sums, 1)).flatten()
    return intra, inter


def compute_all_centralities(
    results_dir: Union[str, Path] = "results_hybrid_sim",
    r0: float = 1.5,
    iss: int = 1,
) -> pd.DataFrame:
    """Compute all network centrality measures for the Madrid district network.

    Args:
        results_dir: Path to simulation results (for locating data files).
        r0: Basic reproduction number (for effective-distance computation).
        iss: Initial infectious seed size (I_ss, for effective distance).

    Returns:
        DataFrame with one row per district (0..20) and columns:
            district, InStrength, OutStrength, TotalStrength,
            Betweenness, Closeness, Eigenvector, PageRank,
            IntraDistrictRatio, InterDistrictRatio,
            WeightedClusteringCoefficient, OutgoingEntropy,
            MeanEffectiveDistance, AverageShortestPathLength
    """
    G, N_ij, _N_vals, _district_index = build_transportation_graph(results_dir)
    n = N_ij.shape[0]
    d_matrix = compute_effective_distance_matrix(results_dir, r0=r0, iss=iss)

    results = {
        "district": list(range(n)),
        "InStrength": N_ij.sum(axis=0),
        "OutStrength": N_ij.sum(axis=1),
        "TotalStrength": N_ij.sum(axis=0) + N_ij.sum(axis=1),
    }

    results["IntraDistrictRatio"], results["InterDistrictRatio"] = (
        _compute_intra_inter_ratio(N_ij)
    )

    results["OutgoingEntropy"] = _compute_outgoing_entropy(N_ij)

    try:
        betweenness = nx.betweenness_centrality(G, normalized=True, weight="weight")
        results["Betweenness"] = [betweenness.get(i, 0.0) for i in range(n)]
    except (nx.NetworkXError, np.linalg.LinAlgError, ValueError):
        results["Betweenness"] = [0.0] * n

    try:
        closeness = nx.closeness_centrality(G, distance="weight")
        results["Closeness"] = [closeness.get(i, 0.0) for i in range(n)]
    except (nx.NetworkXError, np.linalg.LinAlgError, ValueError):
        results["Closeness"] = [0.0] * n

    try:
        eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
        results["Eigenvector"] = [eigenvector.get(i, 0.0) for i in range(n)]
    except (nx.NetworkXError, np.linalg.LinAlgError, ValueError):
        results["Eigenvector"] = [0.0] * n

    try:
        pagerank = nx.pagerank(G, weight="weight")
        results["PageRank"] = [pagerank.get(i, 0.0) for i in range(n)]
    except (nx.NetworkXError, np.linalg.LinAlgError, ValueError):
        results["PageRank"] = [0.0] * n

    try:
        clustering = nx.clustering(G, weight="weight")
        results["WeightedClusteringCoefficient"] = [
            clustering.get(i, 0.0) for i in range(n)
        ]
    except (nx.NetworkXError, np.linalg.LinAlgError, ValueError):
        results["WeightedClusteringCoefficient"] = [0.0] * n

    results["MeanEffectiveDistance"] = d_matrix.mean(axis=1)

    try:
        G_ed = nx.DiGraph()
        for i in range(n):
            for j in range(n):
                if i != j:
                    G_ed.add_edge(i, j, weight=d_matrix[i, j])
        mean_sp_per_node = [
            float(np.mean(list(nx.shortest_path_length(G_ed, source=i, weight="weight").values())))
            for i in range(n)
        ]
        results["AverageShortestPathLength"] = mean_sp_per_node
    except (nx.NetworkXError, np.linalg.LinAlgError, ValueError):
        results["AverageShortestPathLength"] = [float(d_matrix[i].mean()) for i in range(n)]

    df = pd.DataFrame(results)
    return df


def load_or_compute_centralities(
    output_dir: Union[str, Path],
    results_dir: Union[str, Path] = "results_hybrid_sim",
    r0: float = 1.5,
    iss: int = 1,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Load cached centrality CSV or compute and cache it.

    Args:
        output_dir: Base output directory for Mode C results.
        results_dir: Path to simulation results.
        r0, iss: Epidemiological parameters for effective-distance computation.
        force_recompute: If True, recompute even if CSV exists.

    Returns:
        DataFrame with centrality measures (one row per district).
    """
    csv_path = Path(output_dir) / "network_centrality_Madrid.csv"

    if csv_path.exists() and not force_recompute:
        return pd.read_csv(csv_path)

    df = compute_all_centralities(results_dir, r0=r0, iss=iss)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df
