import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import config as cfg
import data_loader
import simulation
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
    
    print(f"Loading data for dataset: {cfg.dataset_name}...")
    sim_data = data_loader.load_and_prepare_data(cfg)
    
    # Define R0 range for sweep
    # Typically R0=1 is the threshold for simple models, but spatial structure changes this.
    r0_values = np.linspace(0.0, 4.0, 81) # 0.05 steps
    """
    r0_set = set()
    if hasattr(cfg, 'parameter_sweep_config') and cfg.parameter_sweep_config:
        for _, params in cfg.parameter_sweep_config.items():
            # Logic matches config.py to include endpoint
            vals = np.arange(params['R0_start'], params['R0_end'] + params['R0_step'], params['R0_step'])
            r0_set.update(np.round(vals, 6))
        r0_values = np.array(sorted(list(r0_set)))
    
    if len(r0_values) == 0:
        r0_values = np.linspace(0.0, 4.0, 81) # 0.05 steps
    """
    
    n_patches = sim_data['n_patches']
    
    print(f"Sweeping {len(r0_values)} R0 values (0.0 to 4.0)...")
    # print(f"Sweeping {len(r0_values)} R0 values ({r0_values[0]} to {r0_values[-1]})...")
    print(f"Seed Patch ID: {cfg.initial_infection_patch_id}")
    
    # Store original config to restore later
    orig_R0 = cfg.R_0
    orig_beta = cfg.beta
    orig_seed = cfg.rnd_seed_0
    
    # --- Optimization: Precompute Structural Matrix ---
    # The mobility structure is constant. K_total = R0 * Structural_Matrix.
    # This avoids re-calculating matrix products inside the loop.
    S_mat = simulation.compute_structural_ngm(sim_data)
    
    # --- Invasion Threshold Calculation ---
    # The invasion threshold R0_crit is 1 / spectral_radius(S_mat)
    # where S_mat is the structural component of the Next Generation Matrix.
    eigenvalues = np.linalg.eigvals(S_mat)
    spectral_radius = np.max(np.abs(eigenvalues))
    invasion_threshold_R0 = 1 / spectral_radius if spectral_radius > 0 else np.inf


    # Storage: [patch_id, r0_index]
    results_matrix = np.zeros((n_patches, len(r0_values)))
    
    # Use a fixed RNG for reproducibility of the percolation realizations
    rng = np.random.default_rng(orig_seed)

    for r_idx, r0 in enumerate(r0_values):
        # Calculate Bond Probabilities for this R0
        # p_vu = 1 - exp(-K_total[v, u])
        K_total = r0 * S_mat
        p_vu = 1 - np.exp(-K_total)
        
        n_realizations = 100
        reachability_sum = np.zeros(n_patches)
        
        for _ in range(n_realizations):
            # Generate one realization of the network
            rand_vals = rng.random((n_patches, n_patches))
            adj = (p_vu > rand_vals).astype(int)
            
            # Calculate reachability for ALL patches as seeds
            # Since N is small (~100-200), we can run BFS from each node
            for seed in range(n_patches):
                # Simple BFS
                visited = set()
                stack = [seed]
                visited.add(seed)
                count = 0
                while stack:
                    curr = stack.pop()
                    count += 1
                    # Find neighbors
                    neighbors = np.where(adj[curr, :] == 1)[0]
                    for n in neighbors:
                        if n not in visited:
                            visited.add(n)
                            stack.append(n)
                reachability_sum[seed] += count
        
        # Average fraction of districts reachable
        results_matrix[:, r_idx] = (reachability_sum / n_realizations) / n_patches

    # Extract the curve for the configured seed patch for specific reporting
    frac_arr = results_matrix[cfg.initial_infection_patch_id, :]
    
    # Threshold > 50%
    idx_50 = np.argmax(frac_arr > 0.5)
    # Check if it actually crossed 0.5
    if frac_arr[idx_50] > 0.5:
        r0_50 = r0_values[idx_50]
        # Linear interpolation for more precise threshold
        if idx_50 > 0:
            y1 = frac_arr[idx_50 - 1]
            y2 = frac_arr[idx_50]
            x1 = r0_values[idx_50 - 1]
            x2 = r0_values[idx_50]
            # y = mx + c => 0.5 = m * x_50 + c
            # (0.5 - y1) / (y2 - y1) = (x_50 - x1) / (x2 - x1)
            r0_50_interp = x1 + (x2 - x1) * (0.5 - y1) / (y2 - y1)
        else:
            r0_50_interp = r0_50
    else:
        r0_50 = None
        r0_50_interp = None
    
    # Threshold > 5% (Onset)
    idx_start = np.argmax(frac_arr > 0.05)
    r0_start = r0_values[idx_start] if frac_arr[idx_start] > 0.05 else None

    print("\n" + "="*40)
    print("Metapopulation Percolation Analysis Results")
    print("="*40)
    if r0_start is not None:
        print(f"  -> Theoretical Invasion Threshold (R* > 1) at R0 = {invasion_threshold_R0:.3f}")
        print(f"  -> Epidemic Onset (>5% districts) at R0 ~ {r0_start:.2f}")
    if r0_50_interp is not None:
        print(f"  -> Critical Threshold (>50% districts) at R0 ~ {r0_50_interp:.3f}")
    else:
        print("  -> Did not reach 50% spread in this R0 range.")

    # --- Plot 1: Spaghetti Plot (All Patches) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot all patches in gray
    for i in range(n_patches):
        ax.plot(r0_values, results_matrix[i, :], color='gray', alpha=0.1, linewidth=1)
        
    # Plot mean
    mean_curve = np.mean(results_matrix, axis=0)
    ax.plot(r0_values, mean_curve, color='black', linestyle='--', linewidth=2, label='Mean (All Patches)')
    
    # Plot configured seed
    seed_curve = results_matrix[cfg.initial_infection_patch_id, :]
    ax.plot(r0_values, seed_curve, color='tab:red', linewidth=2.5, label=f'Seed Patch {cfg.initial_infection_patch_id}')
    
    # ax.plot(r0_values, reachable_fractions, marker='o', markersize=3, color=color, label='Reachable Fraction', alpha=0.8)
    
    ax.set_xlabel('$R_0$')
    ax.set_ylabel('Fraction of Districts Reachable')
    ax.set_title(f'Metapopulation Percolation Analysis\n(Seed Patch: {cfg.initial_infection_patch_id}, Dataset: {cfg.dataset_name})')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    if r0_50_interp:
        # Add theoretical threshold line
        ax.axvline(invasion_threshold_R0, color='green', linestyle=':', lw=2.5, label=f'Invasion Threshold ($R_0 \\approx {invasion_threshold_R0:.2f}$)')
        # Add stochastic threshold line
        ax.axvline(r0_50_interp, color='red', linestyle='--', label=f'50% Threshold ($R_0 \\approx {r0_50_interp:.2f}$)')
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    
    ax.legend(fontsize=12)
    
    filename = results_dir / f"metapopulation_percolation_curve_{cfg.dataset_name}.png"
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {filename}")
    
    # --- Plot 2: Heatmap (Patches vs R0) ---
    if HAS_SEABORN:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create DataFrame for easier labeling
        # Do not sort by risk, keep original patch ID order
        df_heatmap = pd.DataFrame(results_matrix, columns=np.round(r0_values, 2))
        
        # Subsample x-axis labels if too many
        xticklabels = 5 if len(r0_values) > 20 else True
        
        hm = sns.heatmap(df_heatmap, ax=ax, cmap="Reds", vmin=0, vmax=1, xticklabels=xticklabels, cbar_kws={'label': 'Reachable Fraction'})
        
        cbar = hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        
        ax.set_title(f'Percolation Dynamics by Seed Patch\nDataset: {cfg.dataset_name}', fontsize=14)
        ax.set_xlabel('$R_0$', fontsize=10)
        ax.set_ylabel('Seed Patch ID', fontsize=10)
        ax.tick_params(axis='both', labelsize=10)
        
        filename_hm = results_dir / f"metapopulation_percolation_heatmap_{cfg.dataset_name}.png"
        plt.savefig(filename_hm, dpi=300)
        plt.close(fig)
        print(f"Saved heatmap to {filename_hm}")

if __name__ == "__main__":
    main()