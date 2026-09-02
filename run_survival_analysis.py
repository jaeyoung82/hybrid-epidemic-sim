#!/usr/bin/env python
"""
Causal Survival Analysis for Mass Gathering Effects on Epidemic Invasion Timing.

This script provides three analysis modes:
- mode_a (scenario): Compare gathering scenarios with fixed epidemiological parameters
- mode_b (parameter): Analyze parameter sensitivity across R0/beta/I_ss combinations
- mode_robustness: Test robustness of M0/M1 findings to seed location (per-seed)

Usage:
    python run_survival_analysis.py mode_a                                    # Mode A (auto-discover seeds)
    python run_survival_analysis.py mode_a --seed-locations 0 7 18          # Mode A (specific seeds)
    python run_survival_analysis.py mode_a --seed-locations 0 7 18 --per-seed # Mode A (per-seed output)
    python run_survival_analysis.py mode_a --results-dir /custom/path         # Mode A (custom results dir)
    python run_survival_analysis.py mode_a --results-dir results_hybrid_sim --per-seed  # Mode A (custom dir + per-seed; --results-dir before --per-seed)
    python run_survival_analysis.py mode_b AMS_dance                          # Mode B (auto-discover seeds)
    python run_survival_analysis.py mode_b AMS_dance --seed-locations 0 7     # Mode B (specific seeds)
    python run_survival_analysis.py mode_b AMS_dance --results-dir results_hybrid_sim --per-seed  # Mode B (custom dir + per-seed; --results-dir before --per-seed)
    python run_survival_analysis.py mode_a --supplementary frailty            # Mode A + frailty
    python run_survival_analysis.py mode_b AMS_dance --supplementary frailty  # Mode B + frailty
    python run_survival_analysis.py mode_a --supplementary all                # All supplementary analyses
    python run_survival_analysis.py mode_robustness                            # Robustness (per-seed M0/M1, default seeds)
    python run_survival_analysis.py mode_robustness --selected-seeds 0 7 18   # Robustness (specified seeds)

The cluster-robust Cox proportional hazards model is the primary analysis.
The shared frailty (mixed-effects Cox) model is available as an optional
supplementary analysis via the --supplementary flag.
"""

import sys
import argparse
from code_survival.reporting import run_analysis


def main():
    """Main entry point with argparse for command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Causal Survival Analysis for Mass Gathering Effects on Epidemic Invasion Timing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_survival_analysis.py mode_a                                       # Mode A (auto-discover seeds)
    python run_survival_analysis.py mode_a --seed-locations 0 7 18               # Mode A (specific seeds)
    python run_survival_analysis.py mode_a --seed-locations 0 7 18 --per-seed    # Mode A (per-seed output)
    python run_survival_analysis.py mode_a --results-dir /path/to/results        # Mode A (custom results dir)
    python run_survival_analysis.py mode_a --results-dir results_hybrid_sim --per-seed   # Mode A (custom dir + per-seed)
    python run_survival_analysis.py mode_b AMS_dance                             # Mode B (auto-discover seeds)
    python run_survival_analysis.py mode_b AMS_dance --seed-locations 0 7        # Mode B (specific seeds)
    python run_survival_analysis.py mode_b AMS_dance --results-dir results_hybrid_sim --per-seed  # Mode B (custom dir + per-seed)
    python run_survival_analysis.py mode_a --supplementary frailty               # Mode A + frailty
    python run_survival_analysis.py mode_b AMS_dance --supplementary frailty     # Mode B + frailty
    python run_survival_analysis.py mode_a --supplementary all                   # All supplementary analyses
    python run_survival_analysis.py mode_robustness                             # Robustness (per-seed M0/M1)
    python run_survival_analysis.py mode_robustness --selected-seeds 0 7 18     # Robustness (specified seeds)
        """,
    )
    parser.add_argument(
        "mode",
        choices=["mode_a", "mode_b", "mode_robustness", "a", "b", "robustness", "scenario", "parameter"],
        help="Analysis mode: mode_a (scenario comparison), mode_b (parameter sensitivity), "
             "or mode_robustness (per-seed robustness).",
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default=None,
         help="Scenario name for Mode B (e.g., AMS_dance, AMS_football, Leipzig_1, Leipzig_2, Leipzig_3).",
    )
    parser.add_argument(
        "seed_args",
        nargs="*",
         help="Unused for current modes.",
    )
    parser.add_argument(
        "--supplementary",
        nargs="?",
        choices=["frailty", "all"],
        default=None,
        const="frailty",
        help="Run supplementary analysis: 'frailty' (shared frailty model) or 'all'. "
             "Without a value, defaults to 'frailty'.",
    )
    parser.add_argument(
        "--seed-locations",
        type=int,
        nargs="+",
        default=None,
         help="Seed district IDs to compare (e.g. 0 7 18). Auto-discovered if omitted. "
              "Used with mode_a and mode_b.",
    )
    parser.add_argument(
        "--iss",
        type=int,
        default=1,
        help="Initial susceptible import rate (default: 1)",
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results_hybrid_sim',
        help='Path to simulation results directory (default: results_hybrid_sim)',
    )
    parser.add_argument(
        '--per-seed',
        action='store_true',
        default=False,
        help='Run analysis independently for each seed location (separate output folders per seed). '
             'Without this flag, all seeds are combined into a single analysis.',
    )
    parser.add_argument(
        '--selected-seeds',
        type=int,
        nargs='+',
        default=None,
        help='Seed district IDs for the main-text robustness figure '
             '(e.g. 0 7 18). Supplying this flag restricts the run to only '
             'those seeds (skips all-seed discovery and the appendix figure); '
             'omitting it discovers all seed locations and produces both the '
             'selected-seeds and all-seeds figures.',
    )

    args = parser.parse_args()

    mode = args.mode.lower()

    if mode in ["mode_a", "a", "scenario"]:
        print("Running Mode A (scenario comparison)")
        run_analysis(
            analysis_mode="scenario",
            results_dir=args.results_dir,
            per_seed=args.per_seed,
            r0=1.5,
            beta=0.5,
            iss=args.iss,
            skip_bootstrap=True,
            output_dir="results_survival_analysis_a",
            supplementary=[args.supplementary] if args.supplementary else None,
            seed_locations=args.seed_locations,
        )
    elif mode in ["mode_b", "b", "parameter"]:
        fixed_scenario = args.scenario
        if not fixed_scenario:
            print("ERROR: Mode B requires a scenario name. Usage:")
            print("  python run_survival_analysis.py mode_b AMS_dance")
            print("  python run_survival_analysis.py mode_b AMS_dance --supplementary frailty")
            print("Available scenarios: AMS_dance, AMS_football, Leipzig_1, Leipzig_2, Leipzig_3")
            sys.exit(1)
        print(f"Running Mode B (parameter sensitivity) for {fixed_scenario}")
        run_analysis(
            analysis_mode="parameter",
            results_dir=args.results_dir,
            per_seed=args.per_seed,
            fixed_scenario=fixed_scenario,
            iss=args.iss,
            skip_bootstrap=True,
            output_dir="results_survival_analysis_b",
            supplementary=[args.supplementary] if args.supplementary else None,
            seed_locations=args.seed_locations,
        )
    elif mode in ["mode_robustness", "robustness"]:
        from code_survival.seed_location_robustness import run_seed_location_robustness

        print("Running Robustness (per-seed M0/M1 robustness)")
        run_seed_location_robustness(
            results_dir=args.results_dir,
            output_dir="results_survival_analysis_robustness",
            r0=1.5,
            beta=0.5,
            iss=args.iss,
            selected_seeds=args.selected_seeds,
        )
    else:
        print(f"ERROR: Unknown mode '{mode}'. Use 'mode_a', 'mode_b', or 'mode_robustness'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
