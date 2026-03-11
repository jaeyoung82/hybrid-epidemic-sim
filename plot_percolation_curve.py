import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import config as cfg
import event_modeling
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

def main():
    # Setup results directory
    results_dir = Path("results_percolation")
    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)
    
    # Define Beta range for sweep
    betas = np.linspace(0.0, 1.0, 51) 
    
    # Iterate over configured scenarios
    scenarios = getattr(cfg, 'current_event_scenario_names', [])
    if not scenarios:
        print("No scenarios found in config.current_event_scenario_names.")
        return

    for scenario_name in scenarios:
        print(f"\n{'='*40}")
        print(f"Analyzing percolation curve for all individuals in scenario: {scenario_name}")
        
        # Load scenario config
        try:
            cfg.set_active_scenario(cfg, scenario_name)
        except ValueError as e:
            print(f"Skipping {scenario_name}: {e}")
            continue

        contact_file = cfg.large_venue_contact_file
        fps = cfg.event_contact_fps
        header_map = cfg.event_contact_header_map
        
        if not Path(contact_file).exists():
            print(f"Contact file not found: {contact_file}")
            continue

        # --- Perform the sweep and analysis ---
        # Use a fixed RNG for reproducibility of the percolation realizations
        rng = np.random.default_rng(cfg.rnd_seed_0)
        
        # We need the node list first to initialize the results matrix
        # So we do a dummy run with beta=0
        _, node_list = event_modeling.analyze_event_reachability_from_all_seeds(
            contact_file, 0, fps, header_map, rng
        )
        total_nodes = len(node_list)
        if total_nodes == 0:
            print(f"No nodes found in contact file for {scenario_name}. Skipping.")
            continue
            
        # Storage: [node_id, beta_index]
        results_matrix = np.zeros((total_nodes, len(betas)))
        
        print(f"Sweeping {len(betas)} beta values for {total_nodes} individuals...")
        
        for b_idx, beta in enumerate(betas):
            n_realizations = 100
            # Store reachability vectors for each realization
            realization_reachabilities = []
            
            for _ in range(n_realizations):
                reach_vector, _ = event_modeling.analyze_event_reachability_from_all_seeds(
                    contact_file, beta, fps, header_map, rng
                )
                if len(reach_vector) == total_nodes:
                    realization_reachabilities.append(reach_vector)
            
            if realization_reachabilities:
                # Average over realizations for this beta
                avg_reach_vector = np.mean(np.array(realization_reachabilities), axis=0)
                results_matrix[:, b_idx] = avg_reach_vector / total_nodes # Store as fraction

        # --- Plot 1: Spaghetti Plot (All Individuals) ---
        fig, ax = plt.subplots(figsize=(
        # Plot all individuals in gray
        for i in range(total_nodes):
            ax.plot(betas, results_matrix[i, :], color='gray', alpha=0.1, linewidth=1)
            
        # Plot mean
        mean_curve = np.mean(results_matrix, axis=0)
        ax.plot(betas, mean_curve, color='black', linestyle='--', linewidth=2.5, label='Mean reachability (all individuals)')
        
        ax.set_xlabel(r'Transmission Rate ($\beta$) [1/hr]')
        ax.set_ylabel('Fraction of individuals reachable')
        ax.set_title(f'Event Percolation Analysis for All Individuals\nScenario: {scenario_name} (Nodes: {total_nodes})')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        
        filename_spaghetti = results_dir / f"percolation_spaghetti_{scenario_name}.png"
        plt.savefig(filename_spaghetti, dpi=300)
        plt.close(fig)
        print(f"Saved spaghetti plot to {filename_spaghetti}")

        # --- Plot 2: Heatmap (Individuals vs Beta) ---
        if HAS_SEABORN:
            fig, ax = p
            # Create DataFrame for easier labeling, without sorting by risk
            df_heatmap = pd.DataFrame(results_matrix, columns=np.round(betas, 2))
            df_heatmap.index = node_list
            
            # Subsample x-axis labels if too many
            xticklabels = 10 if len(betas) > 20 else True
            
            sns.heatmap(df_heatmap, ax=ax, cmap="Reds", vmin=0, vmax=1, xticklabels=xticklabels, cbar_kws={'label': 'Reachable fraction'})
            
            ax.set_title(f'Percolation Dynamics by Seed Individual (Unsorted)\nScenario: {scenario_name}')
            ax.set_xlabel(r'Transmission Rate ($\beta$) [1/hr]')
            ax.set_ylabel('Seed Individual ID')
            
            # Subsample y-axis labels if too many nodes
            if total_nodes > 50:
                step = total_nodes // 20
                ax.set_yticks(np.arange(0, total_nodes, step))
                # Use the index directly since it's not sorted
                ax.set_yticklabels(df_heatmap.index[::step].astype(int))
            
            filename_hm = results_dir / f"percolation_heatmap_{scenario_name}.png"
            plt.savefig(filename_hm, dpi=300)
            plt.close(fig)
            print(f"Saved heatmap to {filename_hm}")

if __name__ == "__main__":
    main()