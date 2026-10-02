# hybrid-epidemic-sim

A hybrid metapopulation–agent-based epidemic simulator for evaluating how
large-gathering mass events influence epidemic dynamics and invasion timing.
The model couples a **district-level metapopulation SEIR** compartmental model
(Madrid demographic and mobility data) with an **agent-based contact
simulation of large events** (dances, football matches, synthetic high-attendance
venues) and a **survival-analysis** layer (cluster-robust Cox proportional
hazards, Kaplan–Meier, and optional shared-frailty models) for causal inference
on mass-gathering effects.

You may refer to our manuscript "Hybrid epidemic simulation framework coupling equation-based and individual-based models" from https://arxiv.org/abs/2609.35162 for more details.

> This is **research software**. It is not optimized for production deployment.

---

## Features

- **Two-phase architecture** — a computationally expensive agent-based event
  micro-simulation is run once per `(scenario, seed size, event-β)` combination
  and cached to disk; a much cheaper metapopulation SEIR sweep then reuses that
  cache across `R0` values, avoiding redundant micro-simulation.
- **Empirical contact networks** for five mass-gathering events
  (`AMS_dance`, `AMS_football`, `Leipzig_1/2/3`) defined in configuration.
- **Configurable parameter sweeps** over `R0`, event transmission rate `β_event`,
  initial seed size `I_ss`, and multiple event scenarios, with deterministic,
  seed-derived RNG streams for reproducibility.
- **Survival analysis** of epidemic *invasion timing* (arrival of infection to
  each district) comparing event vs. baseline scenarios, including cluster-robust
  and shared-frailty Cox models.

---

## Installation

### Requirements

- Python **>= 3.10**
- Native dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`,
  `geopandas`, `pydantic`, `typer`, `lifelines`, `networkx` (declared in
  `pyproject.toml`).

### Install (editable)

From the repository root:

```bash
pip install -e .
```

This installs all dependencies and registers the **`hybrid-sim`** console
script (the Typer CLI defined in `cli.py`). The simulation runner
(`run_hybrid_simulation.py`) and analysis scripts are also installed as
top-level modules.

---

## Quick Start

The simulation is driven by a JSON configuration file. Two ready-made configs
ship in the repository root:

| Config file | Seed districts | R0 values | Event β values | Scenarios | Iterations | Estimated runs | Intended use |
|---|---|---|---|---|---|---|---|
| `simulation_config_test.json` | 1 | 1 | 5 | 5 | 1,000 | 51,000 | Quick smoke / development test |
| `simulation_config.json` | 21 | 5 | 5 | 5 | 1,000 | 3,255,000 | Full parameter sweep |

Estimated runs are the grand total across both phases (Phase-1 event
micro-simulation cache builds + Phase-2 metapopulation sweep), computed from the
config via `_enumerate_sweep_space` in `run_hybrid_simulation.py`:

- **Test config** — Phase 1: 25 combos (25k micro-sim runs), Phase 2: 26 combos
  (26k metapopulation runs) → **51,000 runs**.
- **Full config** — Phase 1: 525 combos (525k micro-sim runs), Phase 2: 2,730
  combos (2.73M metapopulation runs) → **3,255,000 runs**.

Run the quick test:

```bash
python run_hybrid_simulation.py --config simulation_config_test.json
```

Run the full sweep:

```bash
python run_hybrid_simulation.py --config simulation_config.json
```

> **Entry point.** `run_hybrid_simulation.py` (argparse, `--config` only) is the
> **primary** way to execute the simulation pipeline. The `hybrid-sim` console
> script provides config generation, scenario listing, and analysis helpers, but
> its `hybrid-sim run` subcommand is a **stub** that does not execute the
> pipeline — use `run_hybrid_simulation.py` instead. (The test and full configs
> above are a quick start; a more detailed walkthrough lives in
> [`QUICKSTART.md`](QUICKSTART.md).)

---

## Configuration

Copy a config to customize it, then point the runner at it:

```bash
cp simulation_config_test.json my_config.json
# edit my_config.json, then:
python run_hybrid_simulation.py --config my_config.json
```

Fields prefixed with `_` (e.g. `_comment`) are ignored. Omitting any field falls
back to the default defined in
[`code_config_models/models.py`](code_config_models/models.py).

### Key parameters

| Parameter | Description | Default |
|---|---|---|
| `R_0` | Basic reproduction number | `1.5` |
| `I_ss` | Initial seed size (imported infectious at t=0) | `1` |
| `n_iterations` | Monte Carlo realizations per parameter combination | `1000` |
| `n_days` | Simulation duration in days (half-day resolution) | `250` |
| `period_infectious` | Infectious period in days (β = R0 / this) | `5.0` |
| `period_incubation` | Incubation period in days | `4.0` |
| `TauLeaping_threshold` | Tau-leaping event threshold | `100` |
| `seed_patch_ids` | Seed district IDs (integer patch indices) | `0–20` |
| `parameter_sweep_config` | Maps each `I_ss` to an `R0` range (`R0_start`, `R0_end`, `R0_step`); step `0.0` = single value | `{"1": {"R0_start": 1.0, "R0_end": 3.0, "R0_step": 0.5}}` |
| `event_base_transmission_rate_values` | Event transmission rates (β_event) swept | `[0.1, 0.2, 0.3, 0.4, 0.5]` |
| `current_event_scenario_names` | Event scenarios to evaluate (see table below) | `["AMS_dance"]` |
| `event_simulation_modes` | Modes to run (`large_venue`, `multi_venue`, `no_event`) | `["large_venue", "multi_venue", "no_event"]` |
| `no_event_seeding_mode` | Seeding strategy for the baseline | `"single_patch"` |
| `attendee_recruitment_model` | Attendee recruitment (`gravity`, `population`, `single_patch`, `commuter`) | `"gravity"` |
| `event_days` | Day(s) on which events occur | `[0]` |
| `dataset_name` | Mobility dataset / city name | `"Madrid"` |
| `results_dir` | Output directory for all results | `results_hybrid_sim` |
| `rnd_seed_0` | Base RNG seed for reproducibility | `1234567` |
| `plot_on_the_fly` | Generate plots during simulation | `false` |
| `write_full_results` | Persist full per-run arrays | `false` |

### Available event scenarios

The five scenarios below ship with empirical contact networks under
[`data_crowd-contacts/`](data_crowd-contacts/).

| Scenario | Contact file | Attendees |
|---|---|---|
| `AMS_dance` | `contacts_aggregated_AMS_dance.csv` | 1,048 |
| `AMS_football` | `contacts_aggregated_AMS_football.csv` | 362 |
| `Leipzig_1` | `contacts_aggregated_Leipzig_1.csv` | 1,194 |
| `Leipzig_2` | `contacts_aggregated_Leipzig_2.csv` | 1,158 |
| `Leipzig_3` | `contacts_aggregated_Leipzig_3.csv` | 1,054 |

---

## Architecture

The simulation runs per seed district (`seed_patch_ids`). Within each seed, two
phases execute.

### Phase 1 — Event micro-simulation pre-computation

For each `(scenario, β_event, I_ss)` combination, the agent-based event contact
network is simulated `n_iterations` times. The resulting event outcomes
(recruitment records, initial infections, exposures) are cached as `.pkl` files
in a temporary `temp_event_cache_seed_<id>/` directory. Phase 1 is the
computationally dominant phase; caching its output lets Phase 2 vary `R0`
freely without re-running the agent-based model.

Progress is updated in place every ~10%:

```
  Pre-computing AMS_dance (I_ss=1, Beta=0.1): 100/1000 (10%)
  Pre-computing AMS_dance (I_ss=1, Beta=0.1): 1000/1000 (100%)
  -> I_ss=1, Beta=0.1 ... Saved to disk.
```

### Phase 2 — Metapopulation SEIR sweep

For each `(I_ss, R0)` group, the metapopulation SEIR model is swept. Each group
runs:

1. A **baseline** (`no_event`) simulation.
2. Each event scenario **with** the Phase-1 cache reused from disk.

```
  Phase 2: scenario-batch 1/5 (R0=1.0, I_ss=1)
  >>> Starting Scenario: large_venue <<<
    R0: 1.00 | Beta_Event: 0.10 | Iss: 1
  large_venue: 500/1000 (50%)
  large_venue: 1000/1000 (100%)
```

Progress lines use carriage-return (`\r`) in-place updates; when piped to a
non-TTY each update appears on its own line.

### Data

- **Mobility / population** — Madrid district shapefile, population, and
  origin–destination flow matrices
  (`data_mobility_Madrid/`).
- **Event contact networks** — empirical, frame-aggregated pedestrian contact
  networks for each scenario (`data_crowd-contacts/`).

---

## Results

All outputs for a given seed are written under `<results_dir>/seed_<id>/`.
Aggregate summary CSVs (across seeds) are written to `<results_dir>/`.

### Per-seed outputs (`<results_dir>/seed_<id>/`)

| File pattern | Contents |
|---|---|
| `realizations_<scenario>_R<rr>_beta<bb>_Iss<n>_{S,E,I,R,rt}.csv` | Aggregated SEIR(Rt) time series (time × runs); `rt` present when tracked |
| `spatial_records_<scenario>_beta<bb>_Iss<n>.csv` | Per-district attendee counts and infected attendees |
| `summary_<mode>_R<rr>_betaEvent<bb>_Iss<n>.png` | SEIR + peak + event-exposure summary plot |
| `peak-time-distribution.txt` | Peak-time statistics (appended across runs) |
| `max_infectious_ratio_summary.csv` / `final_size_summary.csv` / `affected_districts_summary.csv` | Per-parameter-set summary (written during the run) |
| `seedXX_attendee_distribution.csv` *(large venue only)* | Attendee origin distribution |
| `event_generated_seed_seedXX.csv` *(large venue only)* | Event-generated infection seed (E, I per district) |
| `day0_district_state_seedXX.csv` | District-level SEIR state before dynamics |
| `day1_district_state_seedXX.csv` | District-level SEIR state after day 1 |

Where `<scenario>` is the event scenario name (or `no_event` for baseline),
`<rr>` is `int(R_0 × 100)`, and `<bb>` is `int(β_event × 100)`.

### Cross-seed summary files (`<results_dir>/`)

| File | Contents |
|---|---|
| `max_infectious_ratio_summary.csv` | Mean / std of peak infectious ratio per parameter set |
| `final_size_summary.csv` | Mean / std of final epidemic size per parameter set |
| `affected_districts_summary.csv` | Mean / std of districts affected per parameter set |

---

## CLI Reference

The `hybrid-sim` console script (Typer-based, in `cli.py`) provides utilities
but **does not** run the simulation (`run` is a stub).

```bash
hybrid-sim --help
```

| Command | Description |
|---|---|
| `hybrid-sim init-config` | Generate a full JSON config template from defaults (`-o/--output`, default `simulation_config.json`). |
| `hybrid-sim list-scenarios` | List available event scenarios (`-c/--config` to specify a config; defaults to built-ins). |
| `hybrid-sim analyze` | Run survival analysis. `--mode scenario` (Mode A) or `--mode parameter` (Mode B); options: `--results-dir`, `--r0`, `--beta`, `--iss`, `--fixed-scenario`, `--seed-locations`, `--per-seed`. |
| `hybrid-sim run` | **Stub** — loads config and applies overrides but does not execute the pipeline. Use `python run_hybrid_simulation.py --config <file>` instead. |

---

## Analysis & Visualization

Post-simulation analysis and figures are produced by standalone scripts and the
CLI:

```bash
# Survival analysis of invasion timing (Cox PH + KM; see code_survival/README.md)
hybrid-sim analyze --mode scenario --results-dir results_hybrid_sim
python run_survival_analysis.py mode_a                       # Mode A: scenario comparison
python run_survival_analysis.py mode_b AMS_dance             # Mode B: parameter sensitivity
python run_survival_analysis.py mode_robustness              # Robustness across seed locations

# Post-hoc figure generation
python run_sim_visualization.py
python run_arrival_heatmap.py
```

`hybrid-sim analyze` invokes `code_survival.reporting.run_analysis`, which
performs **Kaplan–Meier** analysis (with pairwise log-rank tests) and **cluster-robust
Cox proportional hazards** models (Huber–White sandwich SEs clustered on the
simulation run), reporting sequential model comparisons, concordance indices, and
forest plots. The **shared-frailty** (mixed-effects Cox) model is an opt-in
supplementary sensitivity check (`--supplementary frailty`) — see
[`code_survival/README.md`](code_survival/README.md) for full methodology.

On-the-fly plotting during simulation is disabled by default (`plot_on_the_fly:
false` in config); figures are regenerated from the saved realization CSVs after
the run.

---

## Reproducibility

- **Determinism.** Each Monte Carlo realization is driven by a seed derived from
  `rnd_seed_0 × (seed_patch_id + 1) × (run_id + 1)`; re-running with the same
  config and the same Python/dependency versions reproduces identical trajectories.
- Two reference configs are provided — `simulation_config_test.json` (51k runs,
  suitable as a smoke test) and `simulation_config.json` (≈3.26M runs, the full
  sweep). Copy either, edit as needed, and run with `--config`.
- **Python >= 3.10** required. Pinning the dependency versions in
  `pyproject.toml` is recommended for archival reproducibility.

---

## Project layout

```
.
├── cli.py                         # Typer CLI (init-config, list-scenarios, analyze; run is a stub)
├── run_hybrid_simulation.py       # Primary simulation entry point (--config)
├── run_sim_visualization.py       # Post-hoc SEIR / spacetime / spatial figures
├── run_arrival_heatmap.py         # Epidemic arrival-time heatmaps
├── run_survival_analysis.py       # Survival analysis CLI (Modes A/B/robustness)
├── simulation_config_test.json    # Quick test config (51,000 runs)
├── simulation_config.json         # Full parameter sweep config (~3.26M runs)
├── QUICKSTART.md                  # Detailed technical walkthrough
├── code_config_models/            # Pydantic configuration models (source of defaults)
├── code_simulation/               # Metapopulation SEIR engine + event micro-simulation
├── code_plotting/                 # Result persistence & visualization (results.py)
├── code_survival/                 # Cox / KM / frailty survival analysis framework
├── data_mobility_Madrid/          # Madrid population & mobility matrices
└── data_crowd-contacts/         # Empirical event contact networks
```
