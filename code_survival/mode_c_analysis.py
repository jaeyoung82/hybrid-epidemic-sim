"""
Mode C: Seed Location Dependence of Epidemic Arrival and Predictability.

Refactored to reuse the Mode A/B primary analysis infrastructure
(``build_survival_dataset`` -> ``run_primary_survival_analysis``) operating
on seed-directory data.

Mode C answers:
    "Does the initial epidemic seed location affect the relationship between
     mass gathering events, initial seed size, and epidemic arrival timing,
     and are these effects sensitive to transmission conditions?"

Primary analysis (Mode C):
    - Builds a survival dataset from ``results_hybrid_sim/seed_{X}/``
      subdirectories — same column schema as Mode A/B, plus ``seed_location``.
    - Runs the same M0/M1/M2 cluster-robust Cox hierarchy:
        M0: hazard ~ gathering_event
        M1: hazard ~ gathering_event + log_seed_size
        M2: hazard ~ gathering_event + log_seed_size + pre_pressure_7d
    - Produces KM curves, forest plots, model comparison tables, and
      interpretation via the shared ``run_primary_survival_analysis`` pipeline.

C2-A (sensitivity to transmission conditions):
    - For each representative seed × (lower/baseline/higher) transmission
      condition, computes correlation statistics only (no separate Cox models).
    - Reports Spearman rho, Pearson r, OLS R^2 for effective-distance vs
      arrival-time relationship, plus seed-size effect proxy.

Exploratory / supplementary:
    - Seed × destination heatmap (Figure C3)
    - Effective-distance scatter plots (Figure C1, C2)
    - Network centrality associations

effective_distance is intentionally **excluded** from Cox regression models.
It is retained only for scatter plots and supplementary diagnostics.

Usage:
    python run_survival_analysis.py mode_c
    python run_survival_analysis.py mode_c --seed-locations 0 7 18
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .discovery import get_c2a_transmission_conditions
from .cox_models import validate_survival_data


REPRESENTATIVE_SEEDS = [0, 7, 18]
EXPECTED_DISTRICTS = 21


def _save_figure(fig, base_path, dpi=600):
    """Save figure as PNG and PDF."""
    import matplotlib.pyplot as plt
    fig.savefig(str(base_path) + ".png", dpi=dpi, bbox_inches="tight", format="png")
    fig.savefig(str(base_path) + ".pdf", bbox_inches="tight", format="pdf")
    plt.close(fig)


def discover_seed_locations(results_dir, seed_locations=None):
    """Auto-discover seed directories if no explicit list is provided.

    Returns sorted list of seed IDs.  If *seed_locations* is provided,
    that list is returned as-is (sorted).
    """
    results_dir = Path(results_dir)
    if seed_locations is not None and len(seed_locations) > 0:
        return sorted(seed_locations)

    from .discovery import discover_seed_directories
    discovered = discover_seed_directories(results_dir)
    if not discovered:
        return []
    return sorted(discovered)


def save_seed_discovery(seed_locations, results_dir, output_dir):
    """Save a table describing discovered seed directories."""
    records = []
    for seed in seed_locations:
        seed_path = Path(results_dir) / "seed_{}".format(seed)
        records.append({
            "seed_location": seed,
            "directory": str(seed_path),
            "exists": seed_path.exists(),
        })
    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "seed_discovery.csv", index=False)
    return df


def _validate_n_seeds(seed_locations):
    """Validate that we have the expected number of seed districts."""
    n_seeds = len(seed_locations)
    if n_seeds != EXPECTED_DISTRICTS:
        print("  (Using {} of {} expected seed locations.)".format(n_seeds, EXPECTED_DISTRICTS))
    return n_seeds


# ============================================================================
# C1.1: Seed × Destination Heatmap (Exploratory)
# ============================================================================

def compute_seed_destination_arrival_matrix(
    mean_arrival_df, seed_locations, destination_districts=None
):
    """Build the S×D mean epidemic arrival-time matrix.

    Each cell T[s,d] is the mean arrival time at destination district *d*
    averaged across all simulation runs for seed district *s*.

    The diagonal (seed == destination) is masked to NaN.
    """
    if destination_districts is None:
        all_dests = sorted(mean_arrival_df["district"].unique())
        destination_districts = all_dests

    matrix = mean_arrival_df.pivot_table(
        index="seed_location",
        columns="district",
        values="mean_arrival_time",
    )
    matrix = matrix.reindex(index=seed_locations, columns=destination_districts)

    for s in seed_locations:
        if s in matrix.index and s in matrix.columns:
            matrix.loc[s, s] = np.nan

    matrix.index.name = "seed_district"
    matrix.columns.name = "destination_district"
    return matrix


def plot_seed_destination_arrival_heatmap(arrival_matrix, seed_locations, output_path):
    """Generate Figure C3: seed × destination mean arrival-time heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        has_sns = True
    except ImportError:
        has_sns = False

    matrix = arrival_matrix.to_numpy(dtype=float)

    off_diag_mask = ~np.isnan(matrix)
    valid_vals = matrix[off_diag_mask]
    if len(valid_vals) > 0:
        vmin = 0.0
        vmax = float(np.nanmax(valid_vals))
    else:
        vmin, vmax = 0.0, 1.0

    labels = [str(s) for s in seed_locations]
    fig, ax = plt.subplots(figsize=(12, 10))

    if has_sns:
        sns.heatmap(
            matrix, annot=True, fmt=".1f",
            cmap="viridis",
            vmin=vmin, vmax=vmax,
            xticklabels=labels, yticklabels=labels,
            ax=ax,
            cbar_kws={"label": "Mean Arrival Time (days)"},
            linewidths=0.5, linecolor="gray",
            mask=np.isnan(matrix),
        )
        for i, s in enumerate(seed_locations):
            ax.add_patch(plt.Rectangle(
                (i, i), 1, 1, fill=True, color="#d9d9d9",
                zorder=0, linewidth=0,
            ))
    else:
        masked_matrix = np.ma.masked_invalid(matrix)
        im = ax.imshow(masked_matrix, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(seed_locations)))
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_yticks(range(len(seed_locations)))
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax, label="Mean Arrival Time (days)")

    ax.set_xlabel("Destination District", fontsize=12)
    ax.set_ylabel("Seed District", fontsize=12)
    ax.set_title(
        "Figure C3: Mean Epidemic Arrival Time by Seed and Destination District\n"
        "(MSER event scenario, R0=1.5, beta=0.5, I_ss=1)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save_figure(fig, output_path)
    print("  Saved Figure C3 to {}".format(output_path))


def validate_c1_data(mean_arrival_df, seed_locations, expected_runs=1000):
    """Validate the data used to build the C1 seed × destination arrival-time matrix."""
    n_seeds = len(seed_locations)
    destination_districts = sorted(mean_arrival_df["district"].unique())
    n_dests = len(destination_districts)
    total_cells = n_seeds * n_dests
    diagonal_cells = min(n_seeds, n_dests)
    off_diagonal_cells = total_cells - diagonal_cells

    run_counts = {}
    for seed in seed_locations:
        seed_data = mean_arrival_df[mean_arrival_df["seed_location"] == seed]
        if len(seed_data) > 0:
            run_counts[seed] = int(seed_data["n_runs"].max())
        else:
            run_counts[seed] = 0

    print("  Validation Summary:")
    print("    Seed locations: {}".format(n_seeds))
    print("    Destination districts: {}".format(n_dests))
    print("    Expected simulation runs per seed: {}".format(expected_runs))
    print("    Total seed-destination cells: {}".format(total_cells))
    print("    Valid off-diagonal cells: {}".format(off_diagonal_cells))
    print("    Masked diagonal cells: {}".format(diagonal_cells))

    return {
        "n_seeds": n_seeds,
        "n_destinations": n_dests,
        "expected_runs": expected_runs,
        "run_counts": run_counts,
        "total_cells": total_cells,
        "off_diagonal_cells": off_diagonal_cells,
        "diagonal_cells": diagonal_cells,
    }


def compute_c1_seed_destination_analysis(
    mean_arrival_df, seed_locations, output_path,
    results_dir="results_hybrid_sim", r0=1.5, beta=0.5, iss=1
):
    """C1.1: Build the seed × destination mean arrival-time matrix and heatmap."""
    print("  C1.1: Seed × Destination Mean Arrival-Time Matrix")

    validation = validate_c1_data(mean_arrival_df, seed_locations, expected_runs=1000)

    arrival_matrix = compute_seed_destination_arrival_matrix(
        mean_arrival_df, seed_locations,
    )

    arrival_matrix.to_csv(
        output_path / "mode_c_seed_destination_mean_arrival_time.csv",
    )
    print("  Saved mode_c_seed_destination_mean_arrival_time.csv")

    plot_seed_destination_arrival_heatmap(
        arrival_matrix, seed_locations,
        str(output_path / "Figure_C3_seed_destination_mean_arrival_time"),
    )

    return {
        "arrival_matrix": arrival_matrix,
        "validation": validation,
    }


# ============================================================================
# C1.3/1.4: Effective Distance Correlations & Scatter Plots (Exploratory)
# ============================================================================

def scipy_spearman(x, y):
    """Helper to compute Spearman correlation safely."""
    from scipy import stats as scipy_stats
    if len(x) >= 3 and len(np.unique(x)) >= 3 and len(np.unique(y)) >= 3:
        return float(scipy_stats.spearmanr(x, y)[0])
    return np.nan


def _merge_ed_for_seed(mean_arrival_df, effective_distance_df, seed):
    """Join effective distance for the given seed into the mean-arrival records.

    The seed district itself is excluded so that scatter plots and regression
    are only fit on off-diagonal (non-seed) districts.
    """
    seed_arr = mean_arrival_df[mean_arrival_df["seed_location"] == seed].copy()
    ed_seed = effective_distance_df[
        effective_distance_df["origin_district"] == seed
    ][["destination_district", "effective_distance"]].copy()
    ed_seed = ed_seed.rename(columns={"destination_district": "district"})
    merged = seed_arr.merge(ed_seed, on="district", how="inner")
    merged = merged[merged["district"] != seed]
    return merged


def run_effective_distance_correlations(
    mean_arrival_df, effective_distance_df, seed_locations, output_path
):
    """C1.3: Compute Pearson/Spearman correlations, OLS regression per seed.

    Exploratory only — effective_distance is NOT used as a Cox covariate.
    """
    from scipy import stats as scipy_stats
    from scipy.stats import linregress

    print("  C1.3: Effective Distance vs Arrival-Time Correlations (Exploratory)")

    stats_records = []
    regression_records = []

    for seed in seed_locations:
        merged = _merge_ed_for_seed(mean_arrival_df, effective_distance_df, seed)
        x = merged["effective_distance"].to_numpy(dtype=float)
        y = merged["mean_arrival_time"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        pearson_r, pearson_p = np.nan, np.nan
        spearman_r, spearman_p = np.nan, np.nan

        if len(x) >= 3:
            pearson_r, pearson_p = scipy_stats.pearsonr(x, y)
            if len(np.unique(x)) >= 3 and len(np.unique(y)) >= 3:
                spearman_r, spearman_p = scipy_stats.spearmanr(x, y)

        slope = np.nan
        intercept = np.nan
        r_squared = np.nan
        ols_p = np.nan
        rmse = np.nan
        n_obs = len(x)

        if len(x) >= 2 and np.std(x) > 0:
            lr = linregress(x, y)
            slope = float(lr.slope)
            intercept = float(lr.intercept)
            r_squared = float(lr.rvalue ** 2)
            ols_p = float(lr.pvalue)
            y_pred = lr.slope * x + lr.intercept
            residuals = y - y_pred
            rmse = float(np.sqrt(np.mean(residuals ** 2)))

        stats_records.append({
            "seed_location": seed,
            "n_districts": n_obs,
            "pearson_r": float(pearson_r) if not np.isnan(pearson_r) else np.nan,
            "pearson_p": float(pearson_p) if not np.isnan(pearson_p) else np.nan,
            "spearman_rho": float(spearman_r) if not np.isnan(spearman_r) else np.nan,
            "spearman_p": float(spearman_p) if not np.isnan(spearman_p) else np.nan,
        })

        regression_records.append({
            "seed_location": seed,
            "n_districts": n_obs,
            "ols_slope": slope,
            "ols_intercept": intercept,
            "ols_r_squared": r_squared,
            "ols_p": ols_p,
            "ols_rmse": rmse,
        })

        if n_obs > 0:
            print("    Seed {}: Pearson r={:.4f}, Spearman rho={:.4f}, "
                  "slope={:.4f}, R^2={:.4f}, p={:.4f}".format(
                      seed, pearson_r, spearman_r, slope, r_squared, ols_p))

    stats_df = pd.DataFrame(stats_records)
    stats_df.to_csv(output_path / "correlation_stats.csv", index=False)
    print("  Saved correlation_stats.csv")

    reg_df = pd.DataFrame(regression_records)
    reg_df.to_csv(output_path / "linear_regression.csv", index=False)
    print("  Saved linear_regression.csv")

    plot_regression_comparison({r["seed_location"]: r for r in regression_records}, output_path)

    return {"stats": stats_df, "regression": reg_df,
            "regression_results": {r["seed_location"]: r for r in regression_records}}


def plot_seed_scatter(mean_arrival_df, effective_distance_df, output_path,
                      seed_locations, representative_seeds=None):
    """Generate Figure C1: scatter of effective distance vs mean arrival time.

    Exploratory only.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import linregress

    if representative_seeds is not None:
        seeds_to_plot = representative_seeds
        title_suffix = " Representative Seeds"
    else:
        seeds_to_plot = seed_locations
        title_suffix = ""

    n_seeds = len(seeds_to_plot)
    n_cols = min(n_seeds, 3)
    n_rows = (n_seeds + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows),
                             squeeze=False)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, seed in enumerate(seeds_to_plot):
        ax = axes[idx // n_cols, idx % n_cols]
        merged = _merge_ed_for_seed(mean_arrival_df, effective_distance_df, seed)
        x = merged["effective_distance"].to_numpy(dtype=float)
        y = merged["mean_arrival_time"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        ax.scatter(x, y, color=colors[idx % len(colors)], alpha=0.7, s=80,
                   edgecolors="black", linewidths=0.5, zorder=5)

        if len(x) >= 2 and np.std(x) > 0:
            lr = linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = lr.slope * x_line + lr.intercept
            ax.plot(x_line, y_line, color="black", linewidth=1.5, linestyle="-", zorder=3)
            r2 = lr.rvalue ** 2
            spearman_rho = scipy_spearman(x, y) if len(x) >= 3 else np.nan
            ax.text(0.05, 0.95, "R^2={:.3f}\nrho={:.3f}".format(r2, spearman_rho),
                    transform=ax.transAxes, fontsize=10, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_xlabel("Effective Distance", fontsize=11)
        ax.set_ylabel("Mean Arrival Time (days)", fontsize=11)
        ax.set_title("Seed {}".format(seed), fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    for idx in range(n_seeds, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle("Figure C1: Effective Distance vs Mean Arrival Time{}".format(title_suffix),
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_figure(fig, output_path)
    print("  Saved Figure C1 to {}".format(output_path))


def plot_regression_comparison(regression_results, output_path):
    """Generate Figure C2: regression comparison across seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = sorted(regression_results.keys())
    slopes = [regression_results[s]["ols_slope"] for s in seeds]
    r_squared = [regression_results[s]["ols_r_squared"] for s in seeds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x_pos = np.arange(len(seeds))

    ax1.bar(x_pos, slopes, color="#1f77b4", edgecolor="black", linewidth=1, alpha=0.8)
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(["Seed {}".format(s) for s in seeds], fontsize=9, rotation=45, ha="right")
    ax1.set_ylabel("Slope (days per effective distance unit)", fontsize=11)
    ax1.set_title("OLS Regression Slope by Seed Location", fontsize=12, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3)
    for i, (slope, r2) in enumerate(zip(slopes, r_squared)):
        if not np.isnan(slope) and not np.isnan(r2):
            ax1.text(i, slope + 0.3, "R^2={:.3f}".format(r2), ha="center", va="bottom",
                     fontsize=7, fontweight="bold")

    ax2.bar(x_pos, r_squared, color="#2ca02c", edgecolor="black", linewidth=1, alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(["Seed {}".format(s) for s in seeds], fontsize=9, rotation=45, ha="right")
    ax2.set_ylabel("R^2", fontsize=11)
    ax2.set_title("OLS R^2 by Seed Location", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.suptitle("Figure C2: OLS Regression Comparison Across Seed Locations\n"
                 "Arrival Time ~ Effective Distance",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_figure(fig, output_path)
    print("  Saved Figure C2 to {}".format(output_path))


# ============================================================================
# C2-A: Sensitivity to Transmission Conditions (Correlation Stats Only)
# ============================================================================

def run_c2a_sensitivity(
    seed_locations, results_dir, output_path,
    r0=1.5, beta=0.5, iss=1, scenario="AMS_dance",
    effective_distance_df=None,
):
    """C2-A: Sensitivity of seed-location effects to transmission conditions.

    For each representative seed × transmission condition, computes
    **correlation statistics only** (no separate Cox models):

        - Spearman rho (primary): effective distance vs mean arrival time
        - Pearson r (secondary)
        - OLS R^2 (secondary)
        - Seed-size effect proxy: Spearman rho between log_seed_size and
          arrival_time within the pooled dataset
        - Sample size (n_districts, n_runs)

    effective_distance is used ONLY for the correlation statistics — it is
    NOT included in any Cox regression model.

    Saves: ``mode_c_c2a_sensitivity.csv`` and ``Figure_C6``.
    """
    from scipy import stats as scipy_stats
    from scipy.stats import linregress

    print("  C2-A: Sensitivity to Transmission Conditions (Correlation Stats)")
    print("  " + "-" * 50)

    c2a_seeds = [s for s in REPRESENTATIVE_SEEDS if s in seed_locations]

    transmission_conditions = get_c2a_transmission_conditions(
        results_dir=results_dir,
        seed=c2a_seeds[0] if c2a_seeds else 0,
        scenario=scenario,
        iss=iss,
    )

    print("  Transmission conditions for C2-A:")
    for tc in transmission_conditions:
        print("    {}: R0={}, beta={}, Iss={}".format(
            tc['condition'], tc['R0'], tc['beta'], tc['Iss']))

    records = []

    for tc in transmission_conditions:
        cond_name = tc['condition']
        tc_r0 = tc['R0']
        tc_beta = tc['beta']
        tc_iss = tc['Iss']
        tc_scenario = tc['scenario']

        print("    Condition: {} (R0={}, beta={}, Iss={})".format(
            cond_name, tc_r0, tc_beta, tc_iss))

        for seed in c2a_seeds:
            seed_dir = Path(results_dir) / "seed_{}".format(seed)
            if not seed_dir.exists():
                print("      Seed {}: directory not found, skipping".format(seed))
                continue

            from .mode_c_dataset import compute_mean_arrival
            mean_arrival = compute_mean_arrival(
                seed_dir, tc_scenario, tc_r0, tc_beta, tc_iss
            )
            if mean_arrival.empty:
                print("      Seed {}: no arrival data, skipping".format(seed))
                continue

            ed_dist = None
            if effective_distance_df is not None:
                ed_dist = effective_distance_df[
                    effective_distance_df["origin_district"] == seed
                ][["destination_district", "effective_distance"]].copy()
                ed_dist = ed_dist.rename(columns={"destination_district": "district"})
            else:
                from .seed_distance import compute_effective_distance_matrix, compute_shortest_effective_path
                d_matrix = compute_effective_distance_matrix(results_dir, r0=tc_r0, iss=tc_iss)
                dists = compute_shortest_effective_path(d_matrix, seed)
                ed_dist = pd.DataFrame({
                    "district": range(len(dists)),
                    "effective_distance": dists,
                })

            merged = mean_arrival.merge(ed_dist, on="district", how="inner")
            merged = merged[merged["district"] != seed]

            x = merged["effective_distance"].to_numpy(dtype=float)
            y = merged["mean_arrival_time"].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]

            n = len(x)
            spearman_rho = np.nan
            spearman_p = np.nan
            pearson_r = np.nan
            pearson_p = np.nan
            ols_r2 = np.nan
            ols_p = np.nan

            if n >= 3:
                pearson_r, pearson_p = scipy_stats.pearsonr(x, y)
                if len(np.unique(x)) >= 3 and len(np.unique(y)) >= 3:
                    spearman_rho, spearman_p = scipy_stats.spearmanr(x, y)

            if n >= 2 and np.std(x) > 0:
                lr = linregress(x, y)
                ols_r2 = float(lr.rvalue ** 2)
                ols_p = float(lr.pvalue)

            # Seed-size effect proxy: correlation between log_seed_size and arrival_time
            # Load the survival dataset for this (seed, condition) to get seed_size
            seed_size_rho = np.nan
            seed_size_p = np.nan
            try:
                from .mode_c_dataset import build_mode_c_survival_dataset
                cond_df = build_mode_c_survival_dataset(
                    seed_locations=[seed],
                    results_dir=results_dir,
                    r0=tc_r0,
                    beta=tc_beta,
                    iss=tc_iss,
                    scenarios=[tc_scenario],
                    max_time=None,
                )
                if not cond_df.empty and 'log_seed_size' in cond_df.columns:
                    sx = cond_df['log_seed_size'].to_numpy(dtype=float)
                    sy = cond_df['arrival_time'].to_numpy(dtype=float)
                    s_mask = np.isfinite(sx) & np.isfinite(sy) & (sx > 0)
                    sx, sy = sx[s_mask], sy[s_mask]
                    if len(sx) >= 3 and len(np.unique(sx)) >= 3 and len(np.unique(sy)) >= 3:
                        seed_size_rho, seed_size_p = scipy_stats.spearmanr(sx, sy)
            except Exception:
                pass

            n_runs = int(mean_arrival["n_runs"].max()) if len(mean_arrival) > 0 else 0

            print("      Seed {}: rho={:.4f}, r={:.4f}, R^2={:.4f}, seed_size_rho={:.4f}, n={}".format(
                seed, spearman_rho, pearson_r, ols_r2, seed_size_rho, n))

            records.append({
                "transmission_condition": cond_name,
                "R0": tc_r0,
                "beta_event": tc_beta,
                "I_ss": tc_iss,
                "seed_district": seed,
                "spearman_rho": float(spearman_rho) if not np.isnan(spearman_rho) else np.nan,
                "spearman_p": float(spearman_p) if not np.isnan(spearman_p) else np.nan,
                "pearson_r": float(pearson_r) if not np.isnan(pearson_r) else np.nan,
                "pearson_p": float(pearson_p) if not np.isnan(pearson_p) else np.nan,
                "ols_r_squared": float(ols_r2) if not np.isnan(ols_r2) else np.nan,
                "ols_p": float(ols_p) if not np.isnan(ols_p) else np.nan,
                "seed_size_spearman_rho": float(seed_size_rho) if not np.isnan(seed_size_rho) else np.nan,
                "seed_size_spearman_p": float(seed_size_p) if not np.isnan(seed_size_p) else np.nan,
                "n_districts": n,
                "n_runs": n_runs,
            })

    c2a_df = pd.DataFrame(records)
    c2a_df.to_csv(output_path / "mode_c_c2a_sensitivity.csv", index=False)
    print("  Saved mode_c_c2a_sensitivity.csv")

    plot_c2a_sensitivity_heatmap(c2a_df, c2a_seeds, transmission_conditions, output_path)

    return {"c2a_sensitivity": c2a_df, "transmission_conditions": transmission_conditions}


def plot_c2a_sensitivity_heatmap(c2a_df, c2a_seeds, transmission_conditions, output_path):
    """Generate Figure C6: sensitivity of effective-distance predictability
    to transmission conditions for different seed locations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        has_sns = True
    except ImportError:
        has_sns = False

    conditions = [tc['condition'] for tc in transmission_conditions]

    rho_matrix = np.full((len(c2a_seeds), len(conditions)), np.nan)
    r2_matrix = np.full((len(c2a_seeds), len(conditions)), np.nan)

    for i, seed in enumerate(c2a_seeds):
        for j, cond in enumerate(conditions):
            row = c2a_df[
                (c2a_df["seed_district"] == seed) &
                (c2a_df["transmission_condition"] == cond)
            ]
            if len(row) > 0:
                rho_matrix[i, j] = row["spearman_rho"].iloc[0]
                r2_matrix[i, j] = row["ols_r_squared"].iloc[0]

    seed_labels = ["Seed {}".format(s) for s in c2a_seeds]
    cond_labels = ["Lower\n({})".format(conditions[0]),
                   "Baseline\n({})".format(conditions[1]) if len(conditions) > 1 else "",
                   "Higher\n({})".format(conditions[2]) if len(conditions) > 2 else ""]
    cond_labels = [l for l in cond_labels if l]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, matrix, title, label in [
        (axes[0], rho_matrix, "Spearman rho", "Spearman rho"),
        (axes[1], r2_matrix, "OLS R^2", "OLS R\u00b2"),
    ]:
        if has_sns:
            sns.heatmap(
                matrix, annot=True, fmt=".3f",
                cmap="YlOrRd", vmin=0, vmax=1,
                xticklabels=cond_labels, yticklabels=seed_labels,
                ax=ax,
                cbar_kws={"label": label},
                linewidths=0.5, linecolor="gray",
            )
        else:
            masked = np.ma.masked_invalid(matrix)
            im = ax.imshow(masked, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(cond_labels, rotation=0)
            ax.set_yticks(range(len(c2a_seeds)))
            ax.set_yticklabels(seed_labels)
            fig.colorbar(im, ax=ax, label=label)

        ax.set_xlabel("Transmission Condition", fontsize=11)
        ax.set_ylabel("Seed District", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")

    fig.suptitle("Figure C6: Sensitivity of Effective-Distance Predictability\n"
                 "to Transmission Conditions (C2-A)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_figure(fig, str(output_path / "Figure_C6_c2a_sensitivity"))
    print("  Saved Figure C6 to {}".format(output_path / "Figure_C6_c2a_sensitivity"))


# ============================================================================
# Supplementary: Network Centrality (Exploratory)
# ============================================================================

def compute_network_predictability_supplementary(
    centrality_df, seed_summary_df, output_path,
):
    """SUPPLEMENTARY / EXPLORATORY: Relate seed network properties to predictability.

    Retained for exploratory interpretation of why seed locations differ
    in predictability.  Makes NO causal claims.
    """
    from scipy import stats as scipy_stats
    from scipy.stats import linregress

    print("  [SUPPLEMENTARY] Network Centrality vs Predictability (Exploratory)")

    if "SeedDistrict" not in seed_summary_df.columns:
        print("    WARNING: 'SeedDistrict' not found in summary; attempting alias mapping.")
        alias_map = {"Exposure_HR": "Gathering_HR", "Concordance_Index": "Cox_Cindex"}
        seed_summary_df = seed_summary_df.rename(columns=alias_map)

    if "SeedDistrict" not in seed_summary_df.columns:
        print("    WARNING: No 'SeedDistrict' key available for merge; skipping supplementary analysis.")
        return {"pearson_corr": pd.DataFrame(), "spearman_corr": pd.DataFrame(),
                "regression": pd.DataFrame()}

    merged = seed_summary_df.merge(
        centrality_df.rename(columns={"district": "SeedDistrict"}),
        left_on="SeedDistrict", right_on="SeedDistrict",
        how="inner",
    )

    if merged.empty:
        print("    WARNING: No overlap between seed districts and centrality data.")
        return {"pearson_corr": pd.DataFrame(), "spearman_corr": pd.DataFrame(),
                "regression": pd.DataFrame()}

    centrality_cols = [
        "InStrength", "OutStrength", "TotalStrength",
        "Betweenness", "Closeness", "Eigenvector", "PageRank",
        "IntraDistrictRatio", "InterDistrictRatio",
        "WeightedClusteringCoefficient", "OutgoingEntropy",
        "MeanEffectiveDistance", "AverageShortestPathLength",
    ]

    predictability_cols = [
        "Gathering_HR", "Seed_HR", "Cox_Cindex",
    ]

    available_centrality = [c for c in centrality_cols if c in merged.columns]
    available_predictability = [c for c in predictability_cols if c in merged.columns]

    pearson_records = []
    spearman_records = []
    regression_records = []

    for cent in available_centrality:
        for pred in available_predictability:
            x = merged[cent].to_numpy(dtype=float)
            y = merged[pred].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            r, p = np.nan, np.nan
            rho, pp = np.nan, np.nan
            reg = {"CentralityMeasure": cent, "PredictabilityMetric": pred,
                   "slope": np.nan, "intercept": np.nan,
                   "r_squared": np.nan, "p_value": np.nan,
                   "std_error": np.nan, "n_obs": len(x)}

            if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
                r, p = scipy_stats.pearsonr(x, y)
                rho, pp = scipy_stats.spearmanr(x, y)
                lr = linregress(x, y)
                reg["slope"] = float(lr.slope)
                reg["intercept"] = float(lr.intercept)
                reg["r_squared"] = float(lr.rvalue ** 2)
                reg["p_value"] = float(lr.pvalue)
                reg["std_error"] = float(lr.stderr)

            pearson_records.append({
                "CentralityMeasure": cent,
                "PredictabilityMetric": pred,
                "Pearson_r": float(r) if not np.isnan(r) else np.nan,
                "p_value": float(p) if not np.isnan(p) else np.nan,
                "n_seeds": len(x),
            })
            spearman_records.append({
                "CentralityMeasure": cent,
                "PredictabilityMetric": pred,
                "Spearman_rho": float(rho) if not np.isnan(rho) else np.nan,
                "p_value": float(pp) if not np.isnan(pp) else np.nan,
                "n_seeds": len(x),
            })
            regression_records.append(reg)

    pearson_df = pd.DataFrame(pearson_records)
    spearman_df = pd.DataFrame(spearman_records)
    regression_df = pd.DataFrame(regression_records)

    pearson_df.to_csv(output_path / "centrality_vs_predictability.csv", index=False)
    spearman_df.to_csv(output_path / "centrality_vs_predictability_spearman.csv", index=False)
    regression_df.to_csv(output_path / "centrality_predictability_regression.csv", index=False)

    print("  Top exploratory associations:")
    top_pearson = pearson_df.dropna(subset=["Pearson_r"]).copy()
    top_pearson["abs_r"] = top_pearson["Pearson_r"].abs()
    top_pearson = top_pearson.sort_values("abs_r", ascending=False).head(5)
    for _, row in top_pearson.iterrows():
        print("    {} -> {}: r={:.4f} (p={:.4f})".format(
            row["CentralityMeasure"], row["PredictabilityMetric"],
            row["Pearson_r"], row["p_value"]))

    return {"pearson_corr": pearson_df, "spearman_corr": spearman_df,
            "regression": regression_df, "merged": merged}


def compute_trajectory_similarity_exploratory(mean_arrival_df, seed_locations):
    """EXPLORATORY: Compute trajectory similarity between seed locations."""
    from scipy import stats as scipy_stats

    trajectories = {}
    for seed in seed_locations:
        seed_data = mean_arrival_df[mean_arrival_df["seed_location"] == seed]
        traj = seed_data.set_index("district")["mean_arrival_time"].to_dict()
        trajectories[seed] = traj

    records = []
    for s_a in seed_locations:
        for s_b in seed_locations:
            if s_a >= s_b:
                continue
            common_dists = sorted(
                set(trajectories[s_a].keys()) & set(trajectories[s_b].keys())
            )
            if len(common_dists) < 3:
                continue

            x = np.array([trajectories[s_a][d] for d in common_dists])
            y = np.array([trajectories[s_b][d] for d in common_dists])

            pearson_r, pearson_p = np.nan, np.nan
            spearman_r, spearman_p = np.nan, np.nan

            if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
                pearson_r, pearson_p = scipy_stats.pearsonr(x, y)
                if len(np.unique(x)) >= 3 and len(np.unique(y)) >= 3:
                    spearman_r, spearman_p = scipy_stats.spearmanr(x, y)

            records.append({
                "seed_a": s_a, "seed_b": s_b,
                "pearson_r": float(pearson_r) if not np.isnan(pearson_r) else np.nan,
                "pearson_p": float(pearson_p) if not np.isnan(pearson_p) else np.nan,
                "spearman_rho": float(spearman_r) if not np.isnan(spearman_r) else np.nan,
                "spearman_p": float(spearman_p) if not np.isnan(spearman_p) else np.nan,
                "n_common": len(common_dists),
            })

    return pd.DataFrame(records)


def generate_c1_summary(seed_locations, c1_stats_df, c1_corr, arrival_matrix):
    """Generate concise textual summary answering the C1 scientific questions."""
    print()
    print("=" * 70)
    print("C1 SUMMARY: Seed-Location Dependence")
    print("=" * 70)

    print()
    print("1. Do seed locations produce different epidemic arrival patterns?")
    n_seeds = len(seed_locations)
    if n_seeds >= 2:
        seed_vals = seed_locations[:min(n_seeds, 5)]
        diffs = []
        for i in range(len(seed_vals)):
            for j in range(i + 1, len(seed_vals)):
                s_a, s_b = seed_vals[i], seed_vals[j]
                if s_a in arrival_matrix.index and s_b in arrival_matrix.index:
                    row_a = arrival_matrix.loc[s_a].dropna().values
                    row_b = arrival_matrix.loc[s_b].dropna().values
                    common_n = min(len(row_a), len(row_b))
                    if common_n > 0:
                        diff = np.mean(np.abs(row_a[:common_n] - row_b[:common_n]))
                        diffs.append(diff)
        if diffs:
            mean_diff = np.mean(diffs)
            max_diff = np.max(diffs)
            print("   Mean absolute arrival-time difference across seed pairs: {:.2f} days".format(mean_diff))
            print("   Maximum difference: {:.2f} days".format(max_diff))
            if max_diff > 0:
                print("   -> Yes, seed locations produce different arrival patterns.")
            else:
                print("   -> No, seed locations produce identical arrival patterns.")

    print()
    print("2. How strong is effective-distance predictability for each seed? (Exploratory)")
    print("   (Spearman rho is the primary measure)")
    if not c1_stats_df.empty:
        for _, row in c1_stats_df.iterrows():
            seed = int(row["seed_location"])
            rho = row.get("spearman_rho", np.nan)
            r = row.get("pearson_r", np.nan)
            if np.isnan(rho):
                print("   Seed {}: rho=NA, r={:.4f}".format(seed, r))
            else:
                print("   Seed {}: rho={:.4f} (p={:.4f}), r={:.4f} (p={:.4f})".format(
                    seed, rho, row.get("spearman_p", np.nan), r, row.get("pearson_p", np.nan)))

    print()
    print("3. Representative seeds illustrating predictability:")
    available_rep = [s for s in REPRESENTATIVE_SEEDS if s in seed_locations]
    for seed in available_rep:
        row = c1_stats_df[c1_stats_df["seed_location"] == seed]
        if len(row) > 0:
            rho = row["spearman_rho"].iloc[0]
            if not np.isnan(rho):
                if rho >= 0.9:
                    label = "strong"
                elif rho >= 0.5:
                    label = "intermediate"
                else:
                    label = "weak"
            else:
                label = "not computable"
            print("   Seed {}: rho={:.4f} -> {}".format(seed, rho, label))


# ============================================================================
# Primary Analysis Entry Point
# ============================================================================

def run_mode_c_analysis(
    seed_locations: Optional[List[int]] = None,
    output_dir: str = "results_survival_analysis_c",
    results_dir: str = "results_hybrid_sim",
    scenario: str = "AMS_dance",
    r0: float = 1.5,
    beta: float = 0.5,
    iss: int = 1,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """Run the complete Mode C analysis pipeline.

    Mode C reuses the Mode A/B primary analysis infrastructure:
        1. Build survival dataset from seed subdirectories
        2. Run cluster-robust Cox models (M0/M1/M2 hierarchy)
        3. Kaplan-Meier curves, forest plots, model comparison
        4. C2-A sensitivity (correlation stats only)
        5. Exploratory scatter plots and heatmaps

    effective_distance is NOT used as a Cox covariate.

    Args:
        seed_locations: List of seed district IDs. Auto-discovered if None.
        output_dir: Output directory for results.
        results_dir: Directory containing ``seed_{X}/`` subdirectories.
        scenario: Baseline scenario name (default AMS_dance).
        r0, beta, iss: Baseline epidemiological parameters.
        force_recompute: Whether to force recomputation of effective distances.

    Returns:
        Dictionary with primary results, C2-A sensitivity, and exploratory outputs.
    """
    from .mode_c_dataset import (
        build_mode_c_survival_dataset,
        compute_mean_arrival_multi,
    )
    from .seed_distance import load_or_compute_effective_distance
    from .reporting import run_primary_survival_analysis

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Mode C: Seed Location Dependence of Epidemic Arrival and Predictability")
    print("=" * 70)
    t_start = time.time()

    seed_locations = discover_seed_locations(results_dir, seed_locations)
    if not seed_locations:
        print("ERROR: No seed directories discovered in {}".format(results_dir))
        return {}

    n_seeds = _validate_n_seeds(seed_locations)
    print("Seed locations: {}".format(seed_locations))
    print("Total seeds: {}".format(n_seeds))

    save_seed_discovery(seed_locations, results_dir, output_path)

    effective_distance_df = load_or_compute_effective_distance(
        seed_locations=seed_locations,
        output_dir=output_path,
        results_dir=results_dir,
        r0=r0,
        iss=iss,
        force_recompute=force_recompute,
    )
    print("Effective distance records: {}".format(len(effective_distance_df)))

    print()
    print("=" * 70)
    print("Building Mode C Survival Dataset (Baseline Parameters)")
    print("=" * 70)
    survival_df = build_mode_c_survival_dataset(
        seed_locations=seed_locations,
        results_dir=results_dir,
        r0=r0,
        beta=beta,
        iss=iss,
    )
    if survival_df.empty:
        print("ERROR: Survival dataset is empty")
        return {}
    survival_df.to_csv(output_path / "survival_dataset_mode_c.csv", index=False)
    print("Survival dataset records: {}".format(len(survival_df)))

    validate_survival_data(survival_df, cluster_col="simulation_run_id")

    print()
    print("=" * 70)
    print("PRIMARY ANALYSIS: Cluster-Robust Cox Models (Mode C)")
    print("=" * 70)

    primary_results = run_primary_survival_analysis(
        survival_df=survival_df,
        analysis_mode="seed",
        r0=r0,
        beta=beta,
        iss=iss,
        fixed_scenario=None,
        output_dir=str(output_path),
        cluster_col="simulation_run_id",
        expected_districts=EXPECTED_DISTRICTS,
    )

    print()
    print("=" * 70)
    print("C1 (Exploratory): Seed-Destination Patterns and Effective Distance")
    print("=" * 70)

    mean_arrival_df = compute_mean_arrival_multi(
        seed_locations, results_dir=results_dir,
        scenario=scenario, r0=r0, beta=beta, iss=iss,
    )
    if not mean_arrival_df.empty:
        mean_arrival_df.to_csv(output_path / "mean_arrival_times.csv", index=False)
        print("Mean arrival time records: {}".format(len(mean_arrival_df)))

        print()
        print("  C1.1: Seed × Destination Mean Arrival-Time Matrix")
        c1_matrix = compute_c1_seed_destination_analysis(
            mean_arrival_df, seed_locations, output_path,
            results_dir=results_dir, r0=r0, beta=beta, iss=iss,
        )

        print()
        print("  C1.3: Effective Distance vs Arrival-Time Correlations (Exploratory)")
        c1_corr = run_effective_distance_correlations(
            mean_arrival_df, effective_distance_df,
            seed_locations, output_path,
        )

        print()
        c1_corr_result = c1_corr["stats"] if isinstance(c1_corr, dict) else c1_corr

        print()
        print("  C1.4: Representative Scatter Plots (Seeds 0, 7, 18)")
        plot_seed_scatter(
            mean_arrival_df, effective_distance_df,
            str(output_path / "Figure_C1_scatter_effective_distance_vs_arrival"),
            seed_locations,
            representative_seeds=REPRESENTATIVE_SEEDS,
        )

        generate_c1_summary(
            seed_locations,
            c1_corr["stats"] if isinstance(c1_corr, dict) else pd.DataFrame(),
            c1_corr["stats"] if isinstance(c1_corr, dict) else pd.DataFrame(),
            c1_matrix["arrival_matrix"],
        )
    else:
        print("  WARNING: No arrival-time data for exploratory analysis")
        c1_matrix = {}
        c1_corr = {"stats": pd.DataFrame(), "regression": pd.DataFrame()}

    print()
    print("=" * 70)
    print("C2-A: Sensitivity to Transmission Conditions (Correlation Stats Only)")
    print("=" * 70)
    c2a_results = run_c2a_sensitivity(
        seed_locations, results_dir, output_path,
        r0=r0, beta=beta, iss=iss,
        scenario=scenario,
        effective_distance_df=effective_distance_df,
    )

    print()
    print("=" * 70)
    print("SUPPLEMENTARY: Network Centrality (Exploratory)")
    print("=" * 70)
    centrality_df = None
    try:
        from .network_centrality import load_or_compute_centralities
        centrality_df = load_or_compute_centralities(
            output_dir=output_path,
            results_dir=results_dir,
            r0=r0,
            iss=iss,
            force_recompute=force_recompute,
        )
        centrality_df.to_csv(output_path / "network_centrality_Madrid.csv", index=False)
        print("Centrality measures: {} districts x {} measures".format(
            len(centrality_df), len(centrality_df.columns) - 1))
    except Exception as e:
        print("  Centrality computation skipped: {}".format(e))

    if centrality_df is not None:
        seed_summary_for_central = primary_results.get("comparison")
        if seed_summary_for_central is not None:
            compute_network_predictability_supplementary(
                centrality_df, seed_summary_for_central, output_path,
            )

    traj_sim = compute_trajectory_similarity_exploratory(
        mean_arrival_df, seed_locations
    )
    traj_sim.to_csv(output_path / "pairwise_seed_similarity.csv", index=False)

    t_elapsed = time.time() - t_start
    print()
    print("=" * 70)
    print("Mode C analysis complete in {:.1f}s".format(t_elapsed))
    print("Output directory: {}".format(output_path))
    print("=" * 70)

    return {
        "seed_locations": seed_locations,
        "survival_df": primary_results.get("survival_df"),
        "model_results": primary_results.get("model_results", {}),
        "comparison": primary_results.get("comparison"),
        "naive_vs_clustered": primary_results.get("naive_vs_clustered"),
        "mediation_metrics": primary_results.get("mediation_metrics"),
        "interpretation": primary_results.get("interpretation", ""),
        "c1_matrix": c1_matrix,
        "c1_correlations": c1_corr,
        "c2a_sensitivity": c2a_results,
        "centrality": centrality_df,
        "t_elapsed": t_elapsed,
    }
