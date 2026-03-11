import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import config as cfg
import event_modeling

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: 'networkx' library not found. Cannot visualize graph structure.")

def main():
    if not HAS_NETWORKX:
        return

    results_dir = getattr(cfg, 'results_dir', Path("results_hybrid_sim"))
    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)

    beta = cfg.event_base_transmission_rate
    print(f"Visualizing GCC for scenarios with beta_event={beta}...")

    for scenario_name in cfg.current_event_scenario_names:
        # Update config for this scenario to get correct file paths
        cfg.set_active_scenario(cfg, scenario_name)
        
        print(f"Processing {scenario_name}...")
        
        # Use a fixed seed for reproducibility of the percolation realization
        rng = np.random.default_rng(cfg.rnd_seed_0)
        
        nodes, edges = event_modeling.extract_gcc_structure(
            cfg.large_venue_contact_file,
            beta,
            cfg.event_contact_fps,
            cfg.event_contact_header_map,
            rng
        )
        
        if not nodes:
            print(f"  -> No GCC found (or empty network).")
            continue
            
        print(f"  -> GCC Size: {len(nodes)} nodes, {len(edges)} edges")
        
        # Build NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        # Plot
        plt.figure(figsize=(12, 12))
        
        # Use spring layout. For very large graphs, this can be slow.
        print("  -> Computing layout...")
        pos = nx.spring_layout(G, seed=42, k=0.15, iterations=50)
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(G, pos, node_size=20, node_color='tab:blue', alpha=0.7)
        nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray')
        
        plt.title(f"Giant Connected Component: {scenario_name}\nNodes: {len(nodes)}, Edges: {len(edges)} ($\\beta_{{event}}$={beta})")
        plt.axis('off')
        
        filename = results_dir / f"gcc_network_{scenario_name}_beta{int(beta*100)}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved plot to {filename}")

if __name__ == "__main__":
    main()