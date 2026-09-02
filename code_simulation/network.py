"""Percolation and network risk analysis for event contact networks."""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .recruitment.recruitment import _resolve_column


def analyze_event_percolation_risk(contact_data_file, beta, fps, header_map=None):
    """
    Performs a static bond percolation analysis on the contact network.
    Calculates the size of the Giant Connected Component (GCC) assuming
    edges exist with probability p = 1 - exp(-beta * duration).

    Also calculates First-Generation Risk metrics (Expected Secondary Infections).
    """
    try:
        contacts_df = pd.read_csv(contact_data_file)
    except FileNotFoundError:
        return 0.0, 0, 0.0, 0.0

    # Resolve columns
    cfg_col_i = header_map.get("agent_i") if header_map else None
    col_i = _resolve_column(contacts_df, cfg_col_i, ["agent_i", "ped1_id", "id_1", "u", "p1"])
    cfg_col_j = header_map.get("agent_j") if header_map else None
    col_j = _resolve_column(contacts_df, cfg_col_j, ["agent_j", "ped2_id", "id_2", "v", "p2"])
    cfg_col_dur = header_map.get("duration_frames") or header_map.get("tc_s") or header_map.get("frames") if header_map else None
    col_dur = _resolve_column(contacts_df, cfg_col_dur, ["total_duration_frames", "tc_s", "duration", "frames"])

    if not col_i or not col_j or not col_dur:
        return 0.0, 0, 0.0, 0.0

    # Aggregate durations between pairs
    contacts_df['duration_sec'] = pd.to_numeric(contacts_df[col_dur], errors='coerce') / fps
    pair_durations = contacts_df.groupby([col_i, col_j])['duration_sec'].sum().reset_index()

    # Calculate Bond Probabilities (Edge weights)
    # Convert duration to hours for consistency with simulation units if beta is per hour
    # Assuming beta is per hour based on simulation.py usage
    pair_durations['prob'] = 1 - np.exp(-beta * (pair_durations['duration_sec'] / 3600.0))

    # --- First-Generation Risk Analysis (Expected Secondary Infections) ---
    # Sum of probabilities connected to each node = Expected Degree
    node_probs = pd.concat([
        pair_durations[[col_i, 'prob']].rename(columns={col_i: 'node'}),
        pair_durations[[col_j, 'prob']].rename(columns={col_j: 'node'})
    ])
    # Sum probabilities per node
    node_strengths = node_probs.groupby('node')['prob'].sum()
    avg_expected_infections = node_strengths.mean() if not node_strengths.empty else 0.0
    max_expected_infections = node_strengths.max() if not node_strengths.empty else 0.0

    # Build Adjacency List for edges that "percolate" (exist)
    # Since this is a probabilistic check, we run one realization of the static network structure
    # For a robust metric, we treat edges with p > 0.5 as 'structural' connections,
    # or we can sample. Here we sample to match the stochastic nature.
    rng = np.random.default_rng()
    random_vals = rng.random(len(pair_durations))
    active_edges = pair_durations[pair_durations['prob'] > random_vals]

    adj = {}
    all_nodes = set(contacts_df[col_i].unique()) | set(contacts_df[col_j].unique())

    for _, row in active_edges.iterrows():
        u, v = int(row[col_i]), int(row[col_j])
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append(v)
        adj[v].append(u)

    # BFS to find Connected Components
    visited = set()
    max_component_size = 0

    for node in all_nodes:
        if node not in visited:
            component_size = 0
            stack = [node]
            visited.add(node)
            while stack:
                curr = stack.pop()
                component_size += 1
                for neighbor in adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            if component_size > max_component_size:
                max_component_size = component_size

    total_nodes = len(all_nodes)
    gcc_fraction = max_component_size / total_nodes if total_nodes > 0 else 0.0

    return gcc_fraction, total_nodes, avg_expected_infections, max_expected_infections


def analyze_event_reachability_from_all_seeds(contact_data_file, beta, fps, header_map=None, rng=None):
    """
    Performs a static bond percolation analysis on the contact network.
    For a single realization of the network, it calculates the size of the
    connected component for every node if it were the seed.
    """
    try:
        contacts_df = pd.read_csv(contact_data_file)
    except FileNotFoundError:
        return np.array([]), []

    # Resolve columns
    cfg_col_i = header_map.get("agent_i") if header_map else None
    col_i = _resolve_column(contacts_df, cfg_col_i, ["agent_i", "ped1_id", "id_1", "u", "p1"])
    cfg_col_j = header_map.get("agent_j") if header_map else None
    col_j = _resolve_column(contacts_df, cfg_col_j, ["agent_j", "ped2_id", "id_2", "v", "p2"])
    cfg_col_dur = header_map.get("duration_frames") or header_map.get("tc_s") or header_map.get("frames") if header_map else None
    col_dur = _resolve_column(contacts_df, cfg_col_dur, ["total_duration_frames", "tc_s", "duration", "frames"])

    if not col_i or not col_j or not col_dur:
        return np.array([]), []

    # Aggregate durations between pairs
    contacts_df['duration_sec'] = pd.to_numeric(contacts_df[col_dur], errors='coerce') / fps
    pair_durations = contacts_df.groupby([col_i, col_j])['duration_sec'].sum().reset_index()

    # Calculate Bond Probabilities
    pair_durations['prob'] = 1 - np.exp(-beta * (pair_durations['duration_sec'] / 3600.0))

    # Use provided RNG or create a new one
    if rng is None:
        rng = np.random.default_rng()

    # Sample one realization of the network
    random_vals = rng.random(len(pair_durations))
    active_edges = pair_durations[pair_durations['prob'] > random_vals]

    adj = {}
    all_nodes_set = set(contacts_df[col_i].unique()) | set(contacts_df[col_j].unique())
    all_nodes = sorted(list(all_nodes_set))

    for _, row in active_edges.iterrows():
        u, v = int(row[col_i]), int(row[col_j])
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append(v)
        adj[v].append(u)

    # --- Find all connected components and map nodes to their component size ---
    visited = set()
    component_sizes = {} # map node_id -> size of its component

    for node in all_nodes:
        if node not in visited:
            component_nodes = []
            stack = [node]
            visited.add(node)
            while stack:
                curr = stack.pop()
                component_nodes.append(curr)
                for neighbor in adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

            size = len(component_nodes)
            for comp_node in component_nodes:
                component_sizes[comp_node] = size

    # Create an array of reachability values, one for each node, in a consistent order
    reachability_values = np.array([component_sizes.get(node, 1) for node in all_nodes])

    return reachability_values, all_nodes


def extract_gcc_structure(contact_data_file, beta, fps, header_map=None, rng=None):
    """
    Constructs the probabilistic contact network and extracts the Giant Connected Component.
    Returns:
        gcc_nodes (set): Set of node IDs in the GCC.
        gcc_edges (list): List of (u, v) tuples representing edges in the GCC.
    """
    try:
        contacts_df = pd.read_csv(contact_data_file)
    except FileNotFoundError:
        return set(), []

    # Resolve columns
    cfg_col_i = header_map.get("agent_i") if header_map else None
    col_i = _resolve_column(contacts_df, cfg_col_i, ["agent_i", "ped1_id", "id_1", "u", "p1"])
    cfg_col_j = header_map.get("agent_j") if header_map else None
    col_j = _resolve_column(contacts_df, cfg_col_j, ["agent_j", "ped2_id", "id_2", "v", "p2"])
    cfg_col_dur = header_map.get("duration_frames") or header_map.get("tc_s") or header_map.get("frames") if header_map else None
    col_dur = _resolve_column(contacts_df, cfg_col_dur, ["total_duration_frames", "tc_s", "duration", "frames"])

    if not col_i or not col_j or not col_dur:
        return set(), []

    # Aggregate durations between pairs
    contacts_df['duration_sec'] = pd.to_numeric(contacts_df[col_dur], errors='coerce') / fps
    pair_durations = contacts_df.groupby([col_i, col_j])['duration_sec'].sum().reset_index()

    # Calculate Bond Probabilities
    pair_durations['prob'] = 1 - np.exp(-beta * (pair_durations['duration_sec'] / 3600.0))

    if rng is None:
        rng = np.random.default_rng()

    random_vals = rng.random(len(pair_durations))
    active_edges_df = pair_durations[pair_durations['prob'] > random_vals]

    adj = {}
    all_nodes = set(contacts_df[col_i].unique()) | set(contacts_df[col_j].unique())

    # Build adjacency for traversal
    for _, row in active_edges_df.iterrows():
        u, v = int(row[col_i]), int(row[col_j])
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append(v)
        adj[v].append(u)

    # Find GCC
    visited = set()
    max_component_nodes = set()

    for node in all_nodes:
        if node not in visited:
            component_nodes = set()
            stack = [node]
            visited.add(node)
            component_nodes.add(node)
            while stack:
                curr = stack.pop()
                for neighbor in adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        component_nodes.add(neighbor)
                        stack.append(neighbor)

            if len(component_nodes) > len(max_component_nodes):
                max_component_nodes = component_nodes

    # Extract edges for GCC
    gcc_edges = []
    if max_component_nodes:
        for _, row in active_edges_df.iterrows():
            u, v = int(row[col_i]), int(row[col_j])
            if u in max_component_nodes and v in max_component_nodes:
                gcc_edges.append((u, v))

    return max_component_nodes, gcc_edges
