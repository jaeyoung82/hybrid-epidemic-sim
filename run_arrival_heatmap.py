#!/usr/bin/env python
"""
Standalone script to generate seed x destination epidemic arrival time heatmaps.

Generates three SxD matrices (seed districts x destination districts):
1. Baseline (no_event) mean arrival time
2. Event scenario mean arrival time
3. Difference (baseline - event)

Reads I-state (infected) arrival time records from result directories.
Not part of the survival analysis pipeline.

Usage:
    python run_arrival_heatmap.py
    python run_arrival_heatmap.py --scenario AMS_dance --r0 1.0 --beta 0.2
    python run_arrival_heatmap.py --scenario Leipzig_1 --r0 1.5 --beta 0.5 --seed-locations 0 7 18
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_DISTRICTS = 21
DEFAULT_MAX_TIME = 250.0
ROW_HEIGHT = 0.55
COL_WIDTH = 0.55
MARGIN_WIDTH = 3.0
MARGIN_HEIGHT = 2.5
MIN_WIDTH = 9.0
MIN_HEIGHT = 6.0


def discover_seed_directories(results_dir: Path) -> list[int]:
    """Scan results_dir for seed_* subdirectories, return sorted seed IDs."""
    results_path = Path(results_dir)
    seed_ids = []
    for d in sorted(results_path.glob("seed_*")):
        if d.is_dir():
            parts = d.name.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                seed_ids.append(int(parts[1]))
    return sorted(seed_ids)


def get_inferred_max_time(
    seed_dir: Path, scenario: str, r0: float, beta: float, iss: int
) -> float:
    """Derive the censoring time from the daily active cases file.

    Looks for run_daily_active_cases_{scenario}_R{R0_enc}[_beta{beta_enc}]_Iss{iss}.csv,
    reads the header, finds day_* columns, returns the max day value.
    Falls back to DEFAULT_MAX_TIME (250.0) if the file is unavailable.
    """
    r0_enc = int(r0 * 100)
    beta_enc = int(beta * 100)
    if scenario.lower().startswith("no_event"):
        daily_file = (
            seed_dir / f"run_daily_active_cases_no_event_R{r0_enc}_Iss{iss}.csv"
        )
    else:
        daily_file = (
            seed_dir
            / f"run_daily_active_cases_{scenario}_R{r0_enc}_beta{beta_enc}_Iss{iss}.csv"
        )
    if not daily_file.exists():
        return DEFAULT_MAX_TIME
    daily = pd.read_csv(daily_file, nrows=2)
    day_cols = [c for c in daily.columns if c.startswith("day_")]
    if day_cols:
        return float(day_cols[-1].split("_")[1])
    return DEFAULT_MAX_TIME


def load_arrival_records(
    seed_dir: Path, scenario: str, r0: float, beta: float, iss: int
) -> pd.DataFrame:
    """Load I-state arrival time records for a given scenario.

    File patterns:
    - no_event: arrival_time_records_I_no_event_R{R0_enc}_Iss{iss}.csv
                (fallback: arrival_time_records_I_no_event_R{R0_enc}_beta{beta_enc}_Iss{iss}.csv)
    - event:    arrival_time_records_I_{scenario}_R{R0_enc}_beta{beta_enc}_Iss{iss}.csv

    Returns the raw DataFrame. Empty if the file is not found.
    """
    r0_enc = int(r0 * 100)
    beta_enc = int(beta * 100)
    if scenario.lower().startswith("no_event"):
        file_path = seed_dir / f"arrival_time_records_I_no_event_R{r0_enc}_Iss{iss}.csv"
        if not file_path.exists():
            file_path = (
                seed_dir
                / f"arrival_time_records_I_no_event_R{r0_enc}_beta{beta_enc}_Iss{iss}.csv"
            )
    else:
        file_path = (
            seed_dir
            / f"arrival_time_records_I_{scenario}_R{r0_enc}_beta{beta_enc}_Iss{iss}.csv"
        )
    if not file_path.exists():
        warnings.warn(f"Arrival time file not found: {file_path}")
        return pd.DataFrame()
    return pd.read_csv(file_path)


def compute_mean_arrival_per_district(
    records: pd.DataFrame, max_time: float
) -> pd.DataFrame:
    """Compute mean arrival time per district from raw records.

    Mirrors mode_c's compute_mean_arrival(): districts with at least one
    arrival use the mean over arrived runs; districts with no arrivals are
    censored at max_time.

    Returns a DataFrame with columns: district, mean_arrival_time, n_runs, n_arrived.
    """
    if records.empty:
        return pd.DataFrame(
            columns=["district", "mean_arrival_time", "n_runs", "n_arrived"]
        )
    records = records.copy()
    records["arrival_time"] = records.apply(
        lambda row: row["arrival_time"] if row["arrived"] == 1 else max_time,
        axis=1,
    )
    rows = []
    for district in sorted(records["district_id"].unique()):
        sub = records[records["district_id"] == district]
        arrived = sub[sub["arrived"] == 1]
        if len(arrived) > 0:
            mean_arr = float(arrived["arrival_time"].mean())
        else:
            mean_arr = float(max_time)
        rows.append(
            {
                "district": int(district),
                "mean_arrival_time": mean_arr,
                "n_runs": len(sub),
                "n_arrived": len(arrived),
            }
        )
    return pd.DataFrame(rows)


_COLS = [
    "seed_location",
    "district",
    "mean_arrival_time",
    "n_runs",
    "n_arrived",
    "max_time",
]


def build_mean_arrival_df(
    seed_locations: list[int],
    results_dir: Path,
    scenario: str,
    r0: float,
    beta: float,
    iss: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build long-format DataFrames with mean arrival times for baseline and event.

    For each seed, loads the no_event (baseline) and event scenario arrival records,
    computes per-district mean arrival times. A seed is skipped if either the
    baseline or event records are missing, keeping the two matrices aligned.

    Returns (baseline_df, event_df), each with columns:
        seed_location, district, mean_arrival_time, n_runs, n_arrived, max_time
    """
    results_dir = Path(results_dir)
    baseline_rows = []
    event_rows = []

    for seed in seed_locations:
        seed_dir = results_dir / f"seed_{seed}"
        if not seed_dir.exists():
            print(f"  WARNING: Seed directory not found: {seed_dir}")
            continue

        b_max = get_inferred_max_time(seed_dir, "no_event", r0, beta, iss)
        b_records = load_arrival_records(seed_dir, "no_event", r0, beta, iss)
        e_max = get_inferred_max_time(seed_dir, scenario, r0, beta, iss)
        e_records = load_arrival_records(seed_dir, scenario, r0, beta, iss)

        if b_records.empty:
            print(
                f"  WARNING: No baseline (no_event) arrival records for seed {seed}; skipping."
            )
            continue
        if e_records.empty:
            print(
                f"  WARNING: No {scenario} event arrival records for seed {seed}; skipping."
            )
            continue

        b_means = compute_mean_arrival_per_district(b_records, b_max)
        b_means = b_means.copy()
        b_means["seed_location"] = seed
        b_means["max_time"] = float(b_max)
        baseline_rows.append(b_means[_COLS])

        e_means = compute_mean_arrival_per_district(e_records, e_max)
        e_means = e_means.copy()
        e_means["seed_location"] = seed
        e_means["max_time"] = float(e_max)
        event_rows.append(e_means[_COLS])

    baseline_df = (
        pd.concat(baseline_rows, ignore_index=True)
        if baseline_rows
        else pd.DataFrame(columns=_COLS)
    )
    event_df = (
        pd.concat(event_rows, ignore_index=True)
        if event_rows
        else pd.DataFrame(columns=_COLS)
    )
    return baseline_df, event_df


def build_seed_destination_matrix(
    mean_df: pd.DataFrame,
    seed_locations: list[int],
    all_districts: list[int],
) -> pd.DataFrame:
    """Pivot long-format arrival data into a seed x destination matrix.

    Masks the diagonal (seed == destination) to NaN.
    """
    if mean_df.empty:
        return pd.DataFrame(index=seed_locations, columns=all_districts, dtype=float)
    matrix = mean_df.pivot_table(
        index="seed_location",
        columns="district",
        values="mean_arrival_time",
    )
    matrix = matrix.reindex(index=seed_locations, columns=all_districts)

    for s in seed_locations:
        if s in matrix.index and s in matrix.columns:
            matrix.loc[s, s] = np.nan

    matrix.index.name = "seed_district"
    matrix.columns.name = "destination_district"
    return matrix


def plot_arrival_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_path: Path,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "Mean Arrival Time (days)",
    dpi: int = 300,
) -> None:
    """Generate and save a seed x destination arrival-time heatmap.

    Uses seaborn if available, falling back to matplotlib imshow.
    Masks NaN cells (the diagonal). Annotates cells with .1f values.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        has_sns = True
    except ImportError:
        has_sns = False

    seed_labels = [str(s) for s in matrix.index]
    dest_labels = [str(d) for d in matrix.columns]
    data = matrix.to_numpy(dtype=float)
    mask = np.isnan(data)

    valid = data[~mask]
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = float(np.max(valid)) if valid.size else 1.0

    n_s = len(matrix.index)
    n_d = len(matrix.columns)
    width = max(MIN_WIDTH, n_d * COL_WIDTH + MARGIN_WIDTH)
    height = max(MIN_HEIGHT, n_s * ROW_HEIGHT + MARGIN_HEIGHT)

    fig, ax = plt.subplots(figsize=(width, height))

    if has_sns:
        sns.heatmap(
            data,
            annot=True,
            fmt=".1f",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            xticklabels=dest_labels,
            yticklabels=seed_labels,
            ax=ax,
            cbar_kws={
                "orientation": "horizontal",
                "label": cbar_label,
                "pad": 0.10,
                "shrink": 0.9,
                "aspect": 30,
            },
            linewidths=0.5,
            linecolor="gray",
            mask=mask,
        )
        for i, s in enumerate(matrix.index):
            if s in matrix.columns:
                j = list(matrix.columns).index(s)
                ax.add_patch(
                    plt.Rectangle(
                        (j, i),
                        1,
                        1,
                        fill=True,
                        color="#d9d9d9",
                        zorder=0,
                        linewidth=0,
                    )
                )
    else:
        masked_matrix = np.ma.masked_invalid(data)
        im = ax.imshow(masked_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(dest_labels)))
        ax.set_xticklabels(dest_labels, rotation=0, ha="center")
        ax.set_yticks(range(len(seed_labels)))
        ax.set_yticklabels(seed_labels)
        fig.colorbar(
            im,
            ax=ax,
            orientation="horizontal",
            label=cbar_label,
            pad=0.25,
        )

    ax.set_xlabel("Destination District", fontsize=12)
    ax.set_ylabel("Seed District", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    if has_sns:
        fig.subplots_adjust(bottom=0.18, top=0.93, left=0.08, right=0.96)
    else:
        fig.subplots_adjust(bottom=0.30, left=0.08, right=0.96, top=0.93)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight", format="png")
    plt.close(fig)


def generate_heatmaps(
    seed_locations: list[int],
    results_dir: Path,
    scenario: str,
    r0: float,
    beta: float,
    iss: int,
    output_dir: Path,
    dpi: int,
    save_csv: bool,
) -> None:
    """Main pipeline: discover seeds, compute matrices, render heatmaps + CSVs."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    r0_enc = int(r0 * 100)
    beta_enc = int(beta * 100)
    tag = f"R{r0_enc}_beta{beta_enc}_Iss{iss}"
    all_districts = list(range(EXPECTED_DISTRICTS))

    if not seed_locations:
        seed_locations = discover_seed_directories(results_dir)
    seed_locations = sorted(seed_locations)

    print(
        f"Generating arrival heatmaps for scenario={scenario}, "
        f"R0={r0}, beta={beta}, Iss={iss}"
    )
    print(f"  Seeds: {seed_locations}")

    baseline_df, event_df = build_mean_arrival_df(
        seed_locations, results_dir, scenario, r0, beta, iss
    )

    if baseline_df.empty or event_df.empty:
        print("  ERROR: No data could be loaded for baseline or event. Aborting.")
        return

    available_seeds = sorted(
        set(baseline_df["seed_location"].unique())
        & set(event_df["seed_location"].unique())
    )
    if not available_seeds:
        print("  ERROR: No seeds have both baseline and event data. Aborting.")
        return

    baseline_matrix = build_seed_destination_matrix(
        baseline_df, available_seeds, all_districts
    )
    event_matrix = build_seed_destination_matrix(
        event_df, available_seeds, all_districts
    )
    diff_matrix = baseline_matrix - event_matrix

    if save_csv:
        baseline_matrix.to_csv(output_dir / f"baseline_arrival_matrix_{tag}.csv")
        event_matrix.to_csv(output_dir / f"{scenario}_arrival_matrix_{tag}.csv")
        diff_matrix.to_csv(output_dir / f"arrival_diff_matrix_{tag}.csv")
        print(f"  Saved CSV matrices to {output_dir}")

    b_valid = baseline_matrix.to_numpy(dtype=float)[
        ~np.isnan(baseline_matrix.to_numpy(dtype=float))
    ]
    b_vmax = float(np.max(b_valid)) if b_valid.size else 1.0

    plot_arrival_heatmap(
        baseline_matrix,
        title=f"Baseline (no_event) Mean Epidemic Arrival Time\n"
        f"R0={r0}, beta={beta}, Iss={iss}",
        output_path=output_dir / f"baseline_arrival_heatmap_{tag}.png",
        cmap="viridis",
        vmin=0.0,
        vmax=b_vmax,
        cbar_label="Mean Arrival Time (days)",
        dpi=dpi,
    )
    print("  Saved baseline_arrival_heatmap.png")

    e_valid = event_matrix.to_numpy(dtype=float)[
        ~np.isnan(event_matrix.to_numpy(dtype=float))
    ]
    e_vmax = float(np.max(e_valid)) if e_valid.size else 1.0

    plot_arrival_heatmap(
        event_matrix,
        title=f"{scenario} Mean Epidemic Arrival Time\nR0={r0}, beta={beta}, Iss={iss}",
        output_path=output_dir / f"{scenario}_arrival_heatmap_{tag}.png",
        cmap="viridis",
        vmin=0.0,
        vmax=e_vmax,
        cbar_label="Mean Arrival Time (days)",
        dpi=dpi,
    )
    print(f"  Saved {scenario}_arrival_heatmap.png")

    d_valid = diff_matrix.to_numpy(dtype=float)[
        ~np.isnan(diff_matrix.to_numpy(dtype=float))
    ]
    abs_max = float(np.max(np.abs(d_valid))) if d_valid.size else 1.0

    plot_arrival_heatmap(
        diff_matrix,
        title=f"Arrival Time Difference (Baseline - {scenario})\n"
        f"R0={r0}, beta={beta}, Iss={iss}",
        output_path=output_dir / f"arrival_diff_heatmap_{tag}.png",
        cmap="RdBu_r",
        vmin=-abs_max,
        vmax=abs_max,
        cbar_label="Arrival Time Shift (days)",
        dpi=dpi,
    )
    print("  Saved arrival_diff_heatmap.png")
    print(f"Done. Outputs written to {output_dir}")


def main() -> None:
    """CLI entry point with argparse."""
    parser = argparse.ArgumentParser(
        description="Generate seed x destination epidemic arrival time heatmaps "
        "(baseline, event, and difference)."
    )
    parser.add_argument(
        "--scenario",
        default="AMS_dance",
        help="Event scenario name (e.g. AMS_dance, AMS_football, Leipzig_1, Leipzig_2, Leipzig_3)",
    )
    parser.add_argument(
        "--r0",
        type=float,
        default=1.0,
        help="Base reproduction number R0 (default: 1.0)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.2,
        help="Event transmission rate beta_event (default: 0.2)",
    )
    parser.add_argument(
        "--iss",
        type=int,
        default=1,
        help="Initial seed size (default: 1)",
    )
    parser.add_argument(
        "--results-dir",
        default="results_hybrid_sim",
        help="Path to results directory (default: results_hybrid_sim)",
    )
    parser.add_argument(
        "--output-dir",
        default="results_arrival_heatmap",
        help="Output directory for heatmaps (default: results_arrival_heatmap)",
    )
    parser.add_argument(
        "--seed-locations",
        type=int,
        nargs="*",
        default=None,
        help="Seed district IDs to include (default: auto-discover all)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved PNG files (default: 300)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip saving CSV matrices",
    )
    args = parser.parse_args()

    generate_heatmaps(
        seed_locations=args.seed_locations or [],
        results_dir=args.results_dir,
        scenario=args.scenario,
        r0=args.r0,
        beta=args.beta,
        iss=args.iss,
        output_dir=args.output_dir,
        dpi=args.dpi,
        save_csv=not args.no_csv,
    )


if __name__ == "__main__":
    main()
