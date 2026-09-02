# Hybrid Epidemic Simulation

A hybrid metapopulation–agent-based epidemic simulation with large-gathering event
modeling and survival analysis. The model combines a metapopulation SEIR compartmental
model (district-level mobility) with agent-based contact simulations of large events
(dances, football matches, etc.) to evaluate how mass-gathering events influence
epidemic dynamics.

## Requirements

- Python >= 3.10
- Dependencies (managed via `pyproject.toml`): numpy, pandas, scipy, matplotlib,
  seaborn, geopandas, pydantic, typer, lifelines, networkx

## Installation

Install the package in editable mode (this also installs all dependencies and
creates the `hybrid-sim` console script):

```bash
pip install -e .
```

## Quick Start

The simulation is driven by a JSON configuration file. Two ready-made configs are
provided in the repository root:

| Config file | Seeds | R0 values | Event betas | Scenarios | Iterations | Intended use |
|---|---|---|---|---|---|---|
| `simulation_config_test.json` | 1 | 1 | 5 | 5 | 1,000 | Quick test|
| `simulation_config.json` | 21 | 5 | 5 | 5 | 1,000 | Full parameter sweep|

Run a quick test:

```bash
python run_hybrid_simulation.py --config simulation_config_test.json
```

Run the full simulation:

```bash
python run_hybrid_simulation.py --config simulation_config.json
```

> **Note:** The `run_hybrid_simulation.py` script is the primary entry point for
> running the simulation. The `hybrid-sim` CLI console script provides config
> generation, scenario listing, and analysis helpers (see [CLI Reference](#cli-reference)
> below), but its `run` subcommand is currently a **stub** and does not execute the
> simulation pipeline.

## Configuration

Simulation parameters are defined in a JSON config file. Copy an existing config to
get started:

```bash
cp simulation_config_test.json my_config.json
# Edit my_config.json, then run:
python run_hybrid_simulation.py --config my_config.json
```

Fields prefixed with `_` (e.g. `_comment`) are ignored. Omitting any field falls back
to the default defined in `code_config_models/models.py`.

### Key Parameters

| Parameter | Description | Default |
|---|---|---|
| `R_0` | Basic reproduction number | 1.5 |
| `I_ss` | Initial seed size (imported infectioud individuals at t=0) | 1 |
| `n_iterations` | Monte Carlo runs per parameter combination | 1000 |
| `n_days` | Simulation duration in days (half-day resolution) | 250 |
| `period_infectious` | Infectious period in days (beta is derived from R0 / this) | 5.0 |
| `seed_patch_ids` | List of seed district IDs (integer patch indices) | 0–20 |
| `parameter_sweep_config` | Maps each I_ss to an R0 range (`R0_start`, `R0_end`, `R0_step`); `0.0` step = single value | `{"1": {"R0_start": 1.0, "R0_end": 3.0, "R0_step": 0.5}}` |
| `event_base_transmission_rate_values` | Event transmission rates (beta_event) to sweep | [0.1, 0.2, 0.3, 0.4, 0.5] |
| `current_event_scenario_names` | Event scenarios to evaluate (defined in `DEFAULT_EVENT_CONFIGURATIONS`) | `["AMS_dance"]` |
| `event_simulation_modes` | Simulation modes to run (`large_venue`, `multi_venue`, `no_event`) | `["large_venue", "no_event"]` |
| `dataset_name` | Mobility dataset / city name | `"Madrid"` |
| `results_dir` | Output directory for all results | `results_hybrid_sim` |
| `rnd_seed_0` | Base RNG seed for reproducibility | 1234567 |
| `attendee_recruitment_model` | Strategy for recruiting event attendees (`gravity`, `population`, `single_patch`, `commuter`) | `"gravity"` |
| `event_days` | Day(s) on which events occur | `[0]` |

Available built-in event scenarios (defined in
`code_config_models/models.py` → `DEFAULT_EVENT_CONFIGURATIONS`):

| Scenario | Contact file | Attendees |
|---|---|---|
| `AMS_dance` | `contacts_aggregated_AMS_dance.csv` | 1,048 |
| `AMS_football` | `contacts_aggregated_AMS_football.csv` | 362 |
| `Leipzig_1` | `contacts_aggregated_Leipzig_1.csv` | 1,194 |
| `Leipzig_2` | `contacts_aggregated_Leipzig_2.csv` | 1,158 |
| `Leipzig_3` | `contacts_aggregated_Leipzig_3.csv` | 1,054 |

## How It Works

The simulation runs in two phases, iterating over each seed district
(`seed_patch_ids`):

### Phase 1 — Event Micro-Simulation Pre-Computation

For each seed district, the agent-based event contact network is simulated once per
unique `(scenario, I_ss, beta_event)` parameter combination. The resulting event
outcomes (recruitment records, initial infections, exposures) are cached as `.pkl` files
in a temporary `temp_event_cache_seed_<id>/` directory. This avoids re-running the
expensive agent-based model for every R0 value in Phase 2.

**Progress output** (in-place, updated every ~10% of `n_iterations`):

```
  Pre-computing AMS_dance (I_ss=1, Beta=0.1): 100/1000 (10%)
  Pre-computing AMS_dance (I_ss=1, Beta=0.1): 1000/1000 (100%)
  -> I_ss=1, Beta=0.1 ... Saved to disk.
```

### Phase 2 — Metapopulation Sweep

For each seed district, a parameter sweep iterates over all `(R0, I_ss)` groups.
Each group runs:

1. A **baseline** (no event) simulation.
2. Each event scenario **with** the pre-computed event cache from Phase 1.

Within each run, the per-parameter-combination progress is shown (representative
output from the full config, which has 5 R0 groups):

```
  Phase 2: scenario-batch 1/5 (R0=1.0, I_ss=1)
  >>> Starting Scenario: large_venue <<<
    R0: 1.00 | Beta_Event: 0.10 | Iss: 1
  large_venue: 500/1000 (50%)
  large_venue: 1000/1000 (100%)
```

Progress lines use carriage-return (`\r`) in-place updates. When output is piped
(non-TTY), each update appears on its own line.

## Results

All outputs are written under `<results_dir>/seed_<seed_id>/`. Summary CSVs
(aggregating across all seeds) are written to `<results_dir>/`.

### Per-Seed Outputs (`<results_dir>/seed_<id>/`)

| File pattern | Contents |
|---|---|
| `realizations_<scenario>_R<rr>_beta<bb>_Iss<n>_{S,E,I,R,rt}.csv` | Aggregated SEIR(Rt) time series (time × runs) |
| `spatial_records_<scenario>_beta<bb>_Iss<n>.csv` | Per-district attendee counts and infected attendees |
| `summary_<mode>_R<rr>_betaEvent<bb>_Iss<n>.png` | SEIR + peak + event exposure summary plots |
| `peak-time-distribution.txt` | Peak time statistics (appended across runs) |
| `max_infectious_ratio_summary.csv` | Max infectious ratio per scenario/parameter |
| `final_size_summary.csv` | Final epidemic size per scenario/parameter |
| `affected_districts_summary.csv` | Geographic spread statistics |
| `attendee_distribution_seedXX.csv` | (Large venue only) Attendee origin distribution |
| `event_generated_seed_seedXX.csv` | (Large venue only) Event-generated infection seed |
| `day0_district_state_seedXX.csv` | District-level SEIR state before dynamics |
| `day1_district_state_seedXX.csv` | District-level SEIR state after day 1 |

Where `<scenario>` is the event scenario name (or `no_event` for baseline),
`<rr>` is `int(R_0 * 100)`, and `<bb>` is `int(beta_event * 100)`.

### Cross-Seed Summary Files (`<results_dir>/`)

| File | Contents |
|---|---|
| `max_infectious_ratio_summary.csv` | Mean/std of peak infectious ratio per parameter set |
| `final_size_summary.csv` | Mean/std of final epidemic size per parameter set |
| `affected_districts_summary.csv` | Mean/std of districts affected per parameter set |

## CLI Reference

The `hybrid-sim` console script (installed by `pip install -e .`) wraps the Typer-based
CLI in `cli.py`. It provides utilities but **does not** run the simulation (the `run`
subcommand is a stub).

```bash
hybrid-sim --help
```

### Subcommands

| Command | Description |
|---|---|
| `hybrid-sim init-config` | Generate a full JSON config template from simulation defaults. Use `--output` / `-o` to set the path (default: `simulation_config.json`). |
| `hybrid-sim list-scenarios` | List available event scenarios and their contact-file/attendee details. Use `--config` / `-c` to specify a config (defaults to built-in). |
| `hybrid-sim analyze` | Run survival analysis on simulation results. Modes: `--mode scenario` (per-scenario) or `--mode parameter` (across parameters). Options: `--results-dir`, `--r0`, `--beta`, `--iss`, `--fixed-scenario`, `--seed-locations`, `--per-seed`. |
| `hybrid-sim run` | **Stub** — loads config and overrides parameters but does not execute the simulation pipeline. Use `python run_hybrid_simulation.py --config <file>` instead. |

## Analysis (Post-Simulation)

After a simulation completes, run survival analysis on the results:

```bash
hybrid-sim analyze --mode scenario --results-dir results_hybrid_sim
```

This invokes `code_survival.run_analysis`, which performs Cox proportional-hazards
and Kaplan-Meier analyses of infection arrival and extinction times. See
`run_survival_analysis.py` for an alternative standalone entry point.

## Visualization (Post-Simulation)

Post-hoc figure generation is handled by:

```bash
python run_sim_visualization.py
python run_arrival_heatmap.py
```

These scripts produce spacetime heatmaps, spatial spread snapshots, and comparison
plots from the saved realization CSVs. During simulation, on-the-fly plotting is
disabled by default (`plot_on_the_fly: false` in config).
