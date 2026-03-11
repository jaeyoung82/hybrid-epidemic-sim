# Hybrid Epidemic Simulation Framework

This repository contains a specialized simulation framework, `hybrid-epidemic-sim`, designed to model the spread of infectious diseases. Its core innovation is a hybrid approach that combines a computationally efficient metapopulation model for community-level transmission with a data-driven, high-resolution agent-based model for simulating specific mass-gathering events.

The framework is built to run parameter sweeps, compare different event scenarios against a baseline, and generate a rich set of analytical and visual outputs to assess the impact of mass gatherings on an epidemic's trajectory.

## Key Features

*   **Hybrid Modeling:** Seamlessly switches between a stochastic metapopulation SEIR model (for regional spread) and an agent-based model (for event-based transmission).
*   **Data-Driven:** Utilizes real-world datasets for population distribution, and inter-district mobility (e.g., Madrid, NE-England). Event simulations are driven by empirical, high-resolution contact network data.
*   **Flexible Scenarios:** Easily configure and compare multiple event scenarios (e.g., concerts, sports matches with different attendance sizes and contact patterns) against baseline (no-event) simulations.
*   **Stochastic Simulation:** Employs Gillespie's Direct Method for exact stochastic simulation at low infection counts and automatically switches to the more efficient Tau-leaping method for larger outbreaks.
*   **Advanced Recruitment Models:** Implements multiple strategies (`gravity`, `commuter`, `population`, `single_patch`) to realistically model how attendees are drawn from the general population to an event.
*   **Performance Optimization:** Features a two-phase execution model that pre-computes and caches the results of computationally expensive agent-based event simulations, allowing for rapid sweeping of community transmission parameters (`R0`).
*   **Rich Analysis & Visualization:** Generates a wide array of plots out-of-the-box, including:
    *   SEIR and active case evolution curves.
    *   Spatial-temporal heatmaps and choropleth maps of epidemic spread.
    *   Phase diagrams showing how outcomes (peak size, extinction) vary with `R0` and event transmission rates.
    *   Relative impact heatmaps (amplification factor) to quantify the event's effect.
    *   Percolation and network analysis to assess structural risk in both the metapopulation and event contact networks.

## Modeling Approach

The "hybrid" nature of the simulation is its central feature, operating on two distinct scales:

### 1. Macro-Scale: Metapopulation Model

For general community transmission, the simulation uses a **stochastic metapopulation SEIR model**.

*   The population is divided into geographic patches (e.g., city districts).
*   Within each patch, individuals are compartmentalized into Susceptible (S), Exposed (E), Infected (I), and Recovered (R) states.
*   Individuals move between their "home" patch and "work/day" patch based on a day/night cycle and a pre-defined mobility matrix (`N_ij`).
*   Transmission occurs within the population present at a given location (home patch at night, work patch during the day).
*   The simulation proceeds in stochastic time steps determined by the Gillespie algorithm or fixed Tau-leaping steps.

### 2. Micro-Scale: Agent-Based Event Model

On specific days defined in the configuration (e.g., `event_days = [0]`), the simulation pauses the metapopulation model to run a detailed micro-simulation of a mass gathering.

*   **Recruitment:** A specified number of individuals are "recruited" from the general SEIR population to attend the event. The recruitment strategy determines which patches they come from.
*   **Transmission:** Inside the event, transmission is no longer based on mass-action. Instead, it is simulated at the individual agent level using a real-world contact network dataset. The probability of infection between two specific individuals is a function of their contact duration and the event transmission rate (`beta_event`).
*   **Reintegration:** After the event simulation, attendees (some of whom may have newly become 'Exposed') are returned to their home patches in the metapopulation model, where they can then contribute to community spread.

## Core Modules

The codebase is organized into several key modules:

*   `run_hybrid_simulation.py`: The main entry point and orchestrator. It manages the parameter sweeps (e.g., for `R0`, `beta_event`, `I_ss`), handles the pre-computation phase, and calls the scenario runners.
*   `simulation.py`: The core simulation engine. It contains the main time loop (`run_simulation`), the logic for the metapopulation SEIR/Gillespie steps (`_run_metapopulation_step`), and the functions to handle and dispatch event days (`_handle_event_day`).
*   `event_modeling.py`: This module contains all logic specific to mass gatherings. It implements the various `RecruitmentStrategy` classes, simulates transmission on the contact network (`simulate_event_transmission`), and handles the removal and reintegration of attendees from the main population.
*   `config.py`: A centralized file for all simulation parameters, including epidemiological values, dataset paths, event scenario definitions, and parameter sweep ranges.
*   `data_loader.py`: Responsible for loading and preprocessing all input data, such as population counts, mobility matrices, and geographic information.
*   `plotting.py`: A comprehensive library for generating the standard analytical plots (e.g., summary plots, heatmaps, boxplots).
*   `plot_*.py` Scripts: A collection of specialized scripts for generating high-level analyses from the simulation output logs, such as:
    *   `plot_phase_diagram.py`: Creates heatmaps of peak epidemic size.
    *   `plot_extinction_phase_diagram.py`: Creates heatmaps of epidemic extinction probability.
    *   `plot_RCI.py` / `plot_relative_peak.py`: Generate heatmaps showing the amplification of the epidemic caused by an event relative to the baseline.
    *   `plot_metapopulation_percolation.py`: Analyzes the connectivity of the metapopulation network.

## Workflow

A typical simulation run follows these steps:

1.  **Configuration:** Define all parameters in `config.py`. This includes:
    *   The dataset to use (`dataset_name`).
    *   The event scenarios to run (`current_event_scenario_names`).
    *   The parameter sweep ranges (`parameter_sweep_config`, `event_base_transmission_rate_values`).

2.  **Execution:** Run the main script from the command line:
    ```bash
    python run_hybrid_simulation.py
    ```

3.  **Phase 1: Pre-computation:** The script first identifies all unique combinations of event scenarios, initial seed counts (`I_ss`), and event transmission rates (`beta_event`). For each combination, it runs the expensive agent-based event simulation and caches the results (e.g., who got infected) to a temporary directory (`temp_event_cache/`). This is done for each of the `n_iterations`.

4.  **Phase 2: Main Simulation Loop:** The script then iterates through the community `R0` values. For each `R0`:
    *   It runs a full baseline ("no event") simulation for `n_iterations`.
    *   For each event scenario, it runs a full hybrid simulation for `n_iterations`. On event days, instead of re-running the agent-based model, it loads the corresponding pre-computed results from the cache. This makes sweeping `R0` extremely fast.

5.  **Analysis & Output:** After each scenario completes, `plotting.py` and other analysis scripts are used to generate plots and save summary data. All results, logs, and figures are saved to the `results_hybrid_sim/` directory.

## Configuration (`config.py`)

The `config.py` file is the control center for the simulation. Key parameters include:

*   `n_iterations`: Number of Monte Carlo runs for each parameter set.
*   `n_days`: Duration of each simulation.
*   `dataset_name`: The mobility/population dataset to use (e.g., "Madrid").
*   `parameter_sweep_config`: A dictionary defining the `R0` ranges to test for different initial seed sizes (`I_ss`).
    ```python
    parameter_sweep_config = {
        # I_ss: {'R0_start', 'R0_end', 'R0_step'}
        1: {'R0_start': 1.0, 'R0_end': 3.0, 'R0_step': 0.2},
    }
    ```
*   `event_base_transmission_rate_values`: A list of event transmission rates (`beta_event`) to sweep over.
*   `event_configurations`: A dictionary defining each unique event scenario. This is where you specify the number of attendees, the contact network file, and other event-specific details.
    ```python
    "AMS_dance": {
        "patch_id": 0,
        "total_attendees": 1048,
        "contact_file": "data_crowd-contacts/contacts_aggregated_AMS_dance.csv",
        "contact_fps": 1,
        "initially_infected": I_ss,
        # ... and more
    },
    ```
*   `attendee_recruitment_model`: The global strategy for recruiting attendees (e.g., `'gravity'`).

## Dependencies

This codebase requires the following Python libraries:

*   `numpy`
*   `pandas`
*   `matplotlib`
*   `seaborn`

For full functionality, including certain visualizations, the following are also recommended:

*   `geopandas`: For generating choropleth map plots of spatial spread.
*   `networkx`: For visualizing the Giant Connected Component (GCC) of contact networks.

You can typically install these with pip:
```bash
pip install numpy pandas matplotlib seaborn geopandas networkx
```

