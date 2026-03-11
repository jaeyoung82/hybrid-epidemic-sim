import numpy as np
import datetime
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, TextIO
import pickle
import shutil

import config as cfg
import data_loader
import simulation
import plotting

class ScenarioRunner:
    """
    Orchestrates the epidemic simulation for a given scenario.
    """
    def __init__(self, event_model_type: str, sim_data: Dict[str, Any], initial_infections_data: Optional[Dict[int, Dict[int, np.ndarray]]] = None, event_cache: Optional[Dict[int, Any]] = None):
        self.event_model_type = event_model_type
        self.sim_data = sim_data
        self.initial_infections_data = initial_infections_data
        self.event_cache = event_cache
        
        # Set global config for this scenario
        cfg.event_model_type = self.event_model_type
        if self.event_model_type == 'no_event_distributed':
            cfg.no_event_seeding_mode = 'from_file'
        else:
            cfg.no_event_seeding_mode = 'single_patch'

        # CRITICAL FIX: Recalculate beta from R0 at the start of every scenario to ensure clean state
        cfg.gamma = 1.0 / cfg.period_infectious
        cfg.beta = cfg.R_0 * cfg.gamma

    def run(self):
        """Executes the full simulation scenario."""
        time_start = datetime.datetime.now().replace(microsecond=0)
        print("\n" + "="*60)
        if self.event_model_type == 'no_event':
            print(f"Starting Scenario: '{self.event_model_type}' (Baseline)")
            print(f"  -> Seeding {cfg.I_ss} initially INFECTIOUS individuals in patch {cfg.initial_infection_patch_id}.")
        else:
            print(f"Starting Scenario: '{self.event_model_type}' with '{cfg.attendee_recruitment_model}' recruitment.")
        print("="*60)

        # Handle string IDs in contact files (e.g., Leipzig scenarios)
        if self.event_model_type == 'large_venue' and getattr(cfg, 'large_venue_ids_are_strings', False):
            processed_contact_file, n_attendees, header_map = simulation.events.preprocess_contact_file(cfg)
            
            # CRITICAL FIX: Update total attendees count in config to match the actual data
            if n_attendees != cfg.large_venue_total_attendees:
                print(f"  -> Note: Updating total_attendees from {cfg.large_venue_total_attendees} to {n_attendees} based on contact file.")
                cfg.large_venue_total_attendees = n_attendees
                cfg.event_configurations[cfg.current_event_scenario_name]["total_attendees"] = n_attendees

            # Update config for the duration of this scenario run
            cfg.large_venue_contact_file = processed_contact_file
            cfg.large_venue_total_attendees = n_attendees
            cfg.event_contact_header_map = header_map
            cfg.event_configurations[cfg.current_event_scenario_name]["total_attendees"] = n_attendees

        # --- Optimization: Pre-load Contact Data ---
        # Load contact files once into memory to avoid repeated I/O during simulation steps
        self.sim_data['contact_cache'] = {}
        if self.event_model_type == 'large_venue':
            print(f"  -> Pre-loading contact file: {cfg.large_venue_contact_file}")
            try:
                self.sim_data['contact_cache']['large_venue'] = pd.read_csv(cfg.large_venue_contact_file)
            except Exception as e:
                print(f"Warning: Could not pre-load contact file: {e}")
        elif self.event_model_type == 'multi_venue':
            print(f"  -> Pre-loading multi-venue contact files...")
            for group_id, venue_id in cfg.multi_venue_group_venues.items():
                fname = Path("data_crowd-contacts") / f"contacts_aggregated_N{cfg.multi_venue_total_attendees}_T36000_{group_id}.csv"
                if fname.exists():
                    try:
                        self.sim_data['contact_cache'][group_id] = pd.read_csv(fname)
                    except Exception as e:
                        print(f"Warning: Could not load {fname}: {e}")

        # --- Event Percolation / Network Risk Analysis ---
        risk_metrics = {}
        if self.event_model_type == 'large_venue':
            print("  -> [Event Risk] Analyzing Contact Network Structure (Percolation)...")
            gcc_frac, total_nodes, avg_R, max_R = simulation.events.analyze_event_percolation_risk(
                cfg.large_venue_contact_file, cfg.event_base_transmission_rate, cfg.event_contact_fps, cfg.event_contact_header_map
            )
            print(f"     [Event Risk] First-Gen Expected Infections (R_event): Avg={avg_R:.3f}, Max={max_R:.3f}")
            print(f"     [Event Risk] Giant Connected Component (GCC): {gcc_frac:.1%} ({int(gcc_frac*total_nodes)}/{total_nodes} nodes)")
            print(f"     (Interpretation: 'R_event' is the expected direct infections per seed. 'GCC' is the theoretical max reach if transmission continued.)")
            risk_metrics = {
                'gcc': gcc_frac,
                'nodes': total_nodes,
                'R_avg': avg_R,
                'R_max': max_R
            }

        n_frames = cfg.n_days * 2 + 1
        frame_times = np.arange(0, cfg.n_days + 0.5, 0.5)
        realizations = self._initialize_realizations(n_frames)
        all_runs_results = []

        # Use InfectionLogger as a context manager
        with simulation.InfectionLogger(self.event_model_type, self.sim_data['n_patches'], cfg) as logger:
            for run_id in range(cfg.n_iterations):
                run_seed = cfg.rnd_seed_0 * (run_id + 1)
                rng = np.random.default_rng(run_seed)
                
                if self.event_model_type == 'large_venue':
                    msg = f"{self.event_model_type}, {getattr(cfg, 'current_event_scenario_name', 'N/A')} : Iteration {run_id + 1}/{cfg.n_iterations}"
                elif self.event_model_type == 'no_event':
                    msg = f"Baseline (no_event) : Iteration {run_id + 1}/{cfg.n_iterations}"
                else:
                    msg = f"{getattr(cfg, 'current_event_scenario_name', 'N/A')} : Iteration {run_id + 1}/{cfg.n_iterations}"
                print(f"\r{msg}", end="", flush=True)

                # Correctly handle the initial infections data (which can be a DataFrame or a Dict)
                initial_data_for_run = None
                if self.event_model_type == 'multi_venue':
                    # For multi-venue, pass the entire DataFrame down. The simulation logic will filter it.
                    initial_data_for_run = self.initial_infections_data
                elif self.initial_infections_data is not None:
                    # For other scenarios that might use this, get the specific run's data.
                    initial_data_for_run = self.initial_infections_data.get(run_id)
                
                # Get cached event data for this run if available
                precomputed_data = None
                if self.event_cache and run_id in self.event_cache:
                    precomputed_data = self.event_cache[run_id]

                results = simulation.run_simulation(self.sim_data, cfg, rng, run_id, logger, initial_data_for_run, precomputed_data)

                self._aggregate_run_results(realizations, frame_times, results, run_id)

                # Memory Optimization: Remove large spatial-temporal arrays not needed for final plotting.
                # We aggregate 'I_ij' and 'E_ij' to 'I_j' and 'E_j' (sum over home patches) to save memory.
                n_patches = self.sim_data['n_patches']
                if 'E_ij' in results:
                    results['E_j'] = [np.sum(mat.reshape(n_patches, n_patches), axis=0) for mat in results['E_ij']]
                    del results['E_ij']
                if 'I_ij' in results:
                    results['I_j'] = [np.sum(mat.reshape(n_patches, n_patches), axis=0) for mat in results['I_ij']]
                    del results['I_ij']

                keys_to_drop = ['S_ij', 'R_ij', 'N_ij']
                for key in keys_to_drop:
                    if key in results:
                        del results[key]

                # NEW: Drop aggregated keys that are now in realizations to save memory
                keys_to_drop_redundant = ['S_all', 'E_all', 'I_all', 'R_all', 'rt', 'new_exposed', 'new_infected']
                for key in keys_to_drop_redundant:
                    if key in results:
                        del results[key]

                all_runs_results.append(results)

        print("") # Ensure newline after the loop finishes
        self._analyze_and_plot_results(realizations, all_runs_results)

        time_end = datetime.datetime.now().replace(microsecond=0)
        print(f"\nScenario '{self.event_model_type}' finished. Total run time: {time_end - time_start}")
        
        return {
            "realizations": realizations,
            "all_runs_results": all_runs_results,
            "attendee_log_filename": logger.attendee_log_filename,
            "risk_metrics": risk_metrics
        }

    def _initialize_realizations(self, n_frames: int) -> Dict[str, np.ndarray]:
        """Prepares arrays to store results from all runs."""
        frame_times = np.arange(0, cfg.n_days + 0.5, 0.5) # type: ignore
        realizations = {}
        for comp in ['S', 'E', 'I', 'R']:
            realizations[comp] = np.zeros((n_frames, cfg.n_iterations + 1))
            realizations[comp][:, 0] = frame_times
        return realizations

    def _aggregate_run_results(self, realizations: Dict[str, np.ndarray], frame_times: np.ndarray, results: Dict[str, Any], run_id: int):
        """Interpolates and stores results from a single run."""
        for comp in ['S', 'E', 'I', 'R', 'rt']:
            source_key = 'rt' if comp == 'rt' else f'{comp}_all'
            
            if source_key not in results:
                continue

            if comp not in realizations:
                realizations[comp] = np.zeros((len(frame_times), cfg.n_iterations + 1))
                realizations[comp][:, 0] = frame_times
            
            realizations[comp][:, run_id + 1] = np.interp(frame_times, results['t'], results[source_key])

        if cfg.write_full_results:
            plotting.save_full_results(run_id, results, cfg)

    def _analyze_and_plot_results(self, realizations: Dict[str, np.ndarray], all_runs_results: List[Dict[str, Any]]):
        """Generates all plots and analysis for the completed scenario."""
        print("\n--- All simulations complete. Generating analysis and plots. ---")

        # Extract event and recruitment data for plotting
        event_exposures_all_runs = [r['event_exposures_by_day'] for r in all_runs_results]
        infected_recruited_all_runs = [r['infected_attendees_recruited_by_day'] for r in all_runs_results]
        recruitment_map_data = all_runs_results[0]['recruitment_data']
        infected_attendee_map_data = all_runs_results[0]['infected_attendee_counts']

        event_exposures_df = pd.DataFrame(event_exposures_all_runs).sort_index(axis=1)
        infected_attendees_df = pd.DataFrame(infected_recruited_all_runs).sort_index(axis=1)

        if not self.event_model_type.startswith('no_event'):
            # plotting.plot_event_exposure_distribution_boxplot(event_exposures_df, cfg)
            # This plot is only meaningful for multi-venue as it shows the distributed seeding.
            if self.event_model_type == 'multi_venue':
                plotting.plot_infected_attendees_recruited_boxplot(infected_attendees_df, cfg)

            # # Generate the new boxplot for initial infections by district
            # if logger.ini_filename and Path(logger.ini_filename).exists():
            #     plotting.plot_initial_infection_distribution_by_district(
            #         logger.ini_filename,
            #         self.sim_data['population_df'],
            #         cfg
            #     )

            # plotting.plot_attendee_origins_on_map(recruitment_map_data, self.sim_data, cfg)
            # plotting.plot_infected_attendees_from_specific_districts(infected_attendee_map_data, self.sim_data['population_df'], cfg)

        # plot_args = (realizations['S'], realizations['E'], realizations['I'], realizations['R'], self.sim_data, cfg)
        # if self.event_model_type in ['large_venue', 'multi_venue']:
        #     event_total_exposures = [int(sum(d.values())) for d in event_exposures_all_runs]
        #     plotting.create_summary_plot(*plot_args, event_total_exposures=event_total_exposures)
        # else: # no_event
        #     plotting.create_summary_plot(*plot_args)

        # Generate district-level plots for all scenarios
        plotting.plot_peak_time_distribution_by_district(all_runs_results, self.sim_data, cfg)
        plotting.plot_spacetime_heatmap(all_runs_results, self.sim_data, cfg)
        # plotting.plot_spatial_spread_snapshots(all_runs_results, self.sim_data, cfg) # This is now handled by plot_spatial_spread_comparison
        plotting.analyze_and_save_peak_time(realizations['E'], realizations['I'], cfg)
        plotting.analyze_and_plot_arrival_times(all_runs_results, self.sim_data, cfg)
        plotting.save_district_active_cases_per_run(all_runs_results, self.sim_data, cfg)


class ScenarioManager:
    """
    Manages and runs a sequence of simulation scenarios.
    """    
    def __init__(self, sim_data, current_R_0, current_I_ss, current_beta_event, event_cache=None):
        print(f"\n{'='*15} Starting Simulation for R0 = {current_R_0}, I_ss = {current_I_ss}, Beta_Event = {current_beta_event} {'='*15}")
        self.sim_data = sim_data
        self.all_scenario_results: Dict[str, Any] = {}
        self.event_cache = event_cache

        # Set the current parameters for this entire manager instance
        cfg.R_0 = current_R_0
        cfg.beta = current_R_0 * (1.0 / cfg.period_infectious)
        cfg.I_ss = current_I_ss
        # This is the key. Update the event-specific parameter with the current sweep value.
        cfg.large_venue_initially_infected = current_I_ss
        cfg.event_base_transmission_rate = current_beta_event

    def run_all_scenarios(self, baseline_results=None, plot_combined_summary=True):
        """Runs all four predefined scenarios in sequence."""
        # Store the original R0 to restore it later
        original_R0 = cfg.R_0
        modes = cfg.event_simulation_modes

        # 1. Large-venue scenario (must run first to generate attendee file)
        if 'large_venue' in modes:
            large_venue_runner = ScenarioRunner('large_venue', self.sim_data, event_cache=self.event_cache)
            lv_results = large_venue_runner.run()
            self.all_scenario_results['large_venue'] = {
                "realizations": lv_results["realizations"],
                "all_runs_results": lv_results["all_runs_results"],
                "risk_metrics": lv_results.get("risk_metrics", {})
            }
            attendee_log_filename = lv_results.get("attendee_log_filename")
        else:
            # If not running large_venue, we might need to find a pre-existing log file
            # This maintains functionality for running only multi-venue if the file exists.
            results_dir = getattr(cfg, 'results_dir', Path("results"))
            filename = results_dir / f"ini_infectious_attendees_R{int(cfg.R_0*100)}_betaEvent{int(cfg.event_base_transmission_rate*100)}_Iss{cfg.I_ss}.txt"
            attendee_log_filename = str(filename) if filename.exists() else None


        if attendee_log_filename and Path(attendee_log_filename).exists():
            initial_attendees_df = simulation.InfectionLogger.load_attendee_details(
                attendee_log_filename
            )
            
            # --- Print the requested summary of attendee distribution ---
            if initial_attendees_df is not None and 'multi_venue' in modes:
                print("[Multi-Venue] Planned distribution of initially infected attendees per venue:")
                # Create a map from district ID to group ID
                district_to_group_map = self.sim_data['population_df'].set_index('id')['group'].to_dict()
                # Assign a group to each attendee in the DataFrame
                initial_attendees_df['group'] = initial_attendees_df['district_id'].map(district_to_group_map)
                # Count how many infected attendees are in each group
                infected_counts_by_group = initial_attendees_df.groupby('group').size()

                for group_id, venue_id in cfg.multi_venue_group_venues.items():
                    count = infected_counts_by_group.get(group_id, 0)
                    print(f"  -> Venue at District {venue_id} (for Group '{group_id}'): {count} attendees")
            # ----------------------------------------------------------------

            # 2. Multi-venue scenario
            if 'multi_venue' in modes:
                if initial_attendees_df is not None:
                    cfg.R_0 = original_R0
                    cfg.beta = original_R0 * (1.0 / cfg.period_infectious)
                    mv_runner = ScenarioRunner('multi_venue', self.sim_data, initial_attendees_df)
                    mv_results = mv_runner.run()
                    self.all_scenario_results['multi_venue'] = {
                        "realizations": mv_results["realizations"],
                        "all_runs_results": mv_results["all_runs_results"]
                    }
                else:
                    print("\nSkipping 'multi_venue' scenario because initial attendee data is missing.")

            # 3. No-event scenario (baseline)
            if 'no_event' in modes:
                if baseline_results is not None:
                    print("Using cached baseline (no_event) results.")
                    self.all_scenario_results['no_event'] = baseline_results
                else:
                    # Run it if no cache provided
                    cfg.R_0 = original_R0
                    cfg.beta = original_R0 * (1.0 / cfg.period_infectious)
                    ne_runner = ScenarioRunner('no_event', self.sim_data)
                    ne_results = ne_runner.run()
                    self.all_scenario_results['no_event'] = {
                        "realizations": ne_results["realizations"],
                        "all_runs_results": ne_results["all_runs_results"]
                    } # No post-event threshold for baseline

        # After all scenarios are run, generate the combined plot
        if len(self.all_scenario_results) > 0:
            if plot_combined_summary:
                plotting.create_combined_summary_plot(self.all_scenario_results, self.sim_data, cfg)
            # plotting.plot_peak_scatter_comparison(self.all_scenario_results, cfg)
            # plotting.plot_patch_peak_comparison(self.all_scenario_results, self.sim_data, cfg)

def main():
    """Main entry point for the simulation."""
    
    # 1. Load data once
    sim_data = data_loader.load_and_prepare_data(cfg)
    total_population = sim_data['population_df']['population'].sum()

    # 2. Setup unified results directory
    results_dir = Path("results_hybrid_sim")
    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created unified results directory: '{results_dir}'")
    cfg.results_dir = results_dir

    # Cache for baseline results: Key = (R0, Iss)
    baseline_cache = {}

    # Collector for max infectious ratio summary
    max_infectious_ratio_data = []

    # Collector for final size summary
    final_size_data = []

    # Collector for extinction time vs size scatter plots
    extinction_vs_size_data = []

    # --- Define summary file paths for continuous saving ---
    max_ratio_summary_file = results_dir / "max_infectious_ratio_summary.csv"
    final_size_summary_file = results_dir / "final_size_summary.csv"

    # --- PRE-COMPUTATION PHASE ---
    # Pre-calculate event outcomes for unique (Scenario, I_ss, Beta_Event) combinations.
    # This avoids re-running the expensive agent-based model for every R0 value.
    print("\n" + "="*60)
    print("PHASE 1: Pre-computing Event Micro-Simulations")
    print("="*60)
    
    # Setup temporary cache directory
    cache_dir = Path("temp_event_cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Identify all unique combinations needed
    unique_iss_values = list(cfg.parameter_sweep_config.keys())
    unique_betas = cfg.event_base_transmission_rate_values
    
    for scenario_name in cfg.current_event_scenario_names:
        print(f"Pre-computing for scenario: {scenario_name}")
        cfg.set_active_scenario(cfg, scenario_name)
        
        # Set event_model_type explicitly for the pre-computation phase
        cfg.event_model_type = 'large_venue'
        # Set time parameters required by simulation logic (event is at t=0)
        cfg.current_day = 0
        cfg.current_time = 0.0
        
        # Handle string IDs / Preprocessing once here if needed
        if getattr(cfg, 'large_venue_ids_are_strings', False):
             processed_contact_file, n_attendees, header_map = simulation.events.preprocess_contact_file(cfg)
             cfg.large_venue_contact_file = processed_contact_file
             cfg.large_venue_total_attendees = n_attendees
             cfg.event_contact_header_map = header_map

        for iss in unique_iss_values:
            cfg.large_venue_initially_infected = iss # Ensure seeding is correct for precompute
            
            for beta in unique_betas:
                print(f"  -> I_ss={iss}, Beta={beta} ... ", end="", flush=True)
                cfg.event_base_transmission_rate = beta
                
                for run_id in range(cfg.n_iterations):
                    rng = np.random.default_rng(cfg.rnd_seed_0 * (run_id + 1))
                    res = simulation.precompute_event_outcomes(sim_data, cfg, rng, run_id)
                    
                    # Save to disk instead of memory
                    cache_file = cache_dir / f"{scenario_name}_Iss{iss}_Beta{beta}_Run{run_id}.pkl"
                    with open(cache_file, 'wb') as f:
                        pickle.dump(res, f)
                print("Saved to disk.")

    # 3. Loop over parameters
    print("\n" + "="*60)
    print("PHASE 2: Main Simulation Loop (Baseline & Event Scenarios)")
    print("="*60)
    for i_ss_value, r0_params in cfg.parameter_sweep_config.items():
        if r0_params['R0_step'] > 0:
            num_steps = int(round((r0_params['R0_end'] - r0_params['R0_start']) / r0_params['R0_step'])) + 1
            r0_values = np.linspace(r0_params['R0_start'], r0_params['R0_end'], num_steps)
            r0_values = np.round(r0_values, 6)
        else:
            r0_values = np.array([r0_params['R0_start']])
        for i_r0, r_0_value in enumerate(r0_values):

            # --- Run Baseline (No Event) Once per (R0, I_ss) ---
            baseline_key = (r_0_value, i_ss_value)
            
            if baseline_key not in baseline_cache:
                print(f"\n>>> Running Baseline (No Event) for R0={r_0_value}, Iss={i_ss_value} <<<")
                cfg.R_0 = r_0_value
                cfg.I_ss = i_ss_value
                cfg.beta = r_0_value * (1.0 / cfg.period_infectious)
                # Use the first beta value for filename generation purposes
                cfg.event_base_transmission_rate = cfg.event_base_transmission_rate_values[0]

                # --- Metapopulation Percolation Check ---
                mp_frac, mp_count = simulation.analyze_metapopulation_percolation_risk(sim_data, cfg)
                print(f"  -> [Metapopulation Risk] R0={r_0_value:.2f}: Reachable Patches from Seed {cfg.initial_infection_patch_id}: {mp_count}/{sim_data['n_patches']} ({mp_frac:.1%})")

                baseline_runner = ScenarioRunner('no_event', sim_data)
                res = baseline_runner.run()
                baseline_cache[baseline_key] = {
                    "realizations": res["realizations"],
                    "all_runs_results": res["all_runs_results"]
                }

            baseline_results = baseline_cache[baseline_key]
            
            # Calculate baseline max infectious ratio for summary plots
            res_base = baseline_results
            E_base = res_base['realizations']['E']
            I_base = res_base['realizations']['I']
            R_base = res_base['realizations']['R']
            active_cases_base = E_base[:, 1:] + I_base[:, 1:]
            max_active_base = np.max(active_cases_base, axis=0)
            max_ratios_base = max_active_base / total_population
            
            max_infectious_ratio_data.append({
                "scenario_name": "no_event",
                "R0": r_0_value,
                "beta_event": 0.0, # Placeholder
                "I_ss": i_ss_value,
                "max_infectious_ratio_mean": np.mean(max_ratios_base),
                "max_infectious_ratio_std": np.std(max_ratios_base),
                "values": max_ratios_base.tolist()
            })

            # --- Continuously update summary files ---
            plotting.update_summary_csv(max_ratio_summary_file, [{
                "scenario_name": "no_event", "R0": r_0_value, "beta_event": 0.0, "I_ss": i_ss_value,
                "max_infectious_ratio_mean": np.mean(max_ratios_base), "max_infectious_ratio_std": np.std(max_ratios_base),
                "values": max_ratios_base.tolist()
            }])

            # Calculate final size for baseline
            final_R_base = R_base[-1, 1:]
            final_frac_base = final_R_base / total_population
            final_size_data.append({
                "scenario_name": "no_event",
                "R0": r_0_value,
                "beta_event": 0.0,
                "I_ss": i_ss_value,
                "final_size_mean": np.mean(final_frac_base),
                "final_size_std": np.std(final_frac_base),
                "values": final_frac_base.tolist()
            })

            plotting.update_summary_csv(final_size_summary_file, [{
                "scenario_name": "no_event", "R0": r_0_value, "beta_event": 0.0, "I_ss": i_ss_value,
                "final_size_mean": np.mean(final_frac_base), "final_size_std": np.std(final_frac_base),
                "values": final_frac_base.tolist()
            }])

            # Collect extinction times and final sizes (counts) for baseline
            # Use raw run results to get exact extinction times
            extinction_times_base = [run_res['t'][-1] for run_res in res_base['all_runs_results']]
            # R_all is deleted to save memory, use realizations (last value corresponds to final state)
            final_R_counts_base = res_base['realizations']['R'][-1, 1:].tolist()
            extinction_vs_size_data.append({
                "scenario_name": "no_event",
                "R0": r_0_value,
                "beta_event": 0.0,
                "I_ss": i_ss_value,
                "extinction_times": extinction_times_base,
                "final_sizes": final_R_counts_base
            })

            # Ensure betas is a list
            betas = cfg.event_base_transmission_rate_values
            if not isinstance(betas, (list, np.ndarray)):
                betas = [betas]

            for beta_event_value in betas:
                # --- Collectors for this parameter set ---
                all_results_this_beta = {'no_event': baseline_results}
                risk_comparison_data = []
                all_exposure_data_this_beta = {}

                # --- Temporary collectors for continuous saving ---
                new_max_ratio_data_this_beta = []
                new_final_size_data_this_beta = []

                for scenario_name in cfg.current_event_scenario_names:
                    # Update the global configuration to match the current scenario
                    cfg.set_active_scenario(cfg, scenario_name)

                    # Load pre-computed cache from disk for this specific combination
                    current_cache = {}
                    for run_id in range(cfg.n_iterations):
                        cache_file = cache_dir / f"{scenario_name}_Iss{i_ss_value}_Beta{beta_event_value}_Run{run_id}.pkl"
                        if cache_file.exists():
                            with open(cache_file, 'rb') as f:
                                current_cache[run_id] = pickle.load(f)
                    
                    manager = ScenarioManager(sim_data, current_R_0=r_0_value, current_I_ss=i_ss_value, current_beta_event=beta_event_value, event_cache=current_cache)
                    # Run individual plots, but suppress the old combined plot
                    manager.run_all_scenarios(baseline_results=baseline_results, plot_combined_summary=False)

                    # Collect results, re-keying from generic 'large_venue' to specific scenario name
                    if 'large_venue' in manager.all_scenario_results:
                        all_results_this_beta[scenario_name] = manager.all_scenario_results['large_venue']
                        
                        # Also collect exposure data for the other comparison plot
                        all_runs = manager.all_scenario_results['large_venue']['all_runs_results']
                        event_exposures_all_runs = [r['event_exposures_by_day'] for r in all_runs]
                        first_event_day = cfg.event_days[0] if cfg.event_days else 0
                        first_day_exposures = [d.get(first_event_day, 0) for d in event_exposures_all_runs]
                        all_exposure_data_this_beta[scenario_name] = first_day_exposures
                        
                        # Collect risk metrics
                        if manager.all_scenario_results['large_venue'].get('risk_metrics'):
                            metrics = manager.all_scenario_results['large_venue']['risk_metrics'].copy()
                            metrics['scenario'] = scenario_name
                            risk_comparison_data.append(metrics)
                        
                        # Collect max infectious ratio data
                        res = manager.all_scenario_results['large_venue']
                        E = res['realizations']['E']
                        I = res['realizations']['I']
                        R = res['realizations']['R']
                        # Calculate max (E+I)/N for each run
                        # E[:, 1:] + I[:, 1:] gives (time_steps, n_iterations)
                        active_cases = E[:, 1:] + I[:, 1:]
                        max_active_cases = np.max(active_cases, axis=0) # Max over time for each run
                        max_ratios = max_active_cases / total_population
                        
                        mean_max_ratio = np.mean(max_ratios)
                        std_max_ratio = np.std(max_ratios)
                        
                        event_max_ratio_item = {
                            "scenario_name": scenario_name,
                            "R0": r_0_value,
                            "beta_event": beta_event_value,
                            "I_ss": i_ss_value,
                            "max_infectious_ratio_mean": mean_max_ratio,
                            "max_infectious_ratio_std": std_max_ratio,
                            "values": max_ratios.tolist()
                        }
                        max_infectious_ratio_data.append(event_max_ratio_item)
                        new_max_ratio_data_this_beta.append(event_max_ratio_item)
                        
                        # Calculate final size
                        final_R = R[-1, 1:]
                        final_frac = final_R / total_population
                        event_final_size_item = {
                            "scenario_name": scenario_name,
                            "R0": r_0_value,
                            "beta_event": beta_event_value,
                            "I_ss": i_ss_value,
                            "final_size_mean": np.mean(final_frac),
                            "final_size_std": np.std(final_frac),
                            "values": final_frac.tolist()
                        }
                        final_size_data.append(event_final_size_item)
                        new_final_size_data_this_beta.append(event_final_size_item)
                        
                        # Collect extinction times and final sizes (counts) for event scenario
                        all_runs = manager.all_scenario_results['large_venue']['all_runs_results']
                        ext_times = [run_res['t'][-1] for run_res in all_runs]
                        # R_all is deleted to save memory, use realizations
                        final_counts = manager.all_scenario_results['large_venue']['realizations']['R'][-1, 1:].tolist()
                        extinction_vs_size_data.append({
                            "scenario_name": scenario_name,
                            "R0": r_0_value,
                            "beta_event": beta_event_value,
                            "I_ss": i_ss_value,
                            "extinction_times": ext_times,
                            "final_sizes": final_counts
                        })

                # --- Continuously update summary files ---
                if new_max_ratio_data_this_beta:
                    plotting.update_summary_csv(max_ratio_summary_file, new_max_ratio_data_this_beta)
                if new_final_size_data_this_beta:
                    plotting.update_summary_csv(final_size_summary_file, new_final_size_data_this_beta)
                print(f"Updated summary files for R0={r_0_value}, I_ss={i_ss_value}, Beta_Event={beta_event_value}")

                # --- Generate Grouped Summary Plots ---
                scenario_groups = {"AMS": cfg.empirical_scenario_set_1, "Leipzig": cfg.empirical_scenario_set_2}
                for group_name, scenarios_in_group in scenario_groups.items():
                    # Create a dictionary with results for only the scenarios in this group + baseline
                    group_results = {s: all_results_this_beta[s] for s in scenarios_in_group if s in all_results_this_beta}
                    group_results['no_event'] = baseline_results
                    
                    if len(group_results) > 1: # Only plot if there are event scenarios to compare
                        # plotting.create_combined_summary_plot(group_results, sim_data, cfg, group_name=group_name)
                        # plotting.plot_peak_scatter_comparison(group_results, cfg, group_name=group_name)
                        # plotting.plot_grouped_peak_time_distributions(group_results, sim_data, cfg, group_name=group_name)
                        plotting.plot_active_cases_comparison(group_results, sim_data, cfg, group_name=group_name)
                        plotting.plot_cumulative_incidence_comparison(group_results, sim_data, cfg, group_name=group_name)
                        plotting.plot_spatial_spread_comparison(group_results, sim_data, cfg, group_name=group_name)
                        # plotting.plot_rt_comparison(group_results, sim_data, cfg, group_name=group_name)
                
                # Plot comparison for ALL scenarios together
                plotting.plot_active_cases_comparison(all_results_this_beta, sim_data, cfg, group_name="All_Scenarios")
                plotting.plot_cumulative_incidence_comparison(all_results_this_beta, sim_data, cfg, group_name="All_Scenarios")
                plotting.plot_spatial_spread_comparison(all_results_this_beta, sim_data, cfg, group_name="All_Scenarios")
                
                # plotting.plot_early_cumulative_cases(all_results_this_beta, sim_data, cfg, days=60)

                # Only plot exposure comparison and risk metrics for the first R0, as they are R0-independent
                if i_r0 == 0:
                    if all_exposure_data_this_beta:
                        plotting.plot_scenario_exposure_comparison_boxplot(all_exposure_data_this_beta, cfg)
                    
                    if risk_comparison_data:
                        plotting.plot_event_risk_metrics(risk_comparison_data, cfg)
                
                # plotting.plot_phase_plane_comparison(all_results_this_beta, sim_data, cfg)

    print("\n" + "="*60)
    print("PHASE 3: Global Analysis & Plotting")
    print("="*60)

    plotting.plot_max_infectious_ratio_vs_beta(max_infectious_ratio_data, cfg) # This now only plots
    plotting.plot_infectious_ratio_difference_vs_beta(max_infectious_ratio_data, cfg)
    # plotting.plot_peak_ratio_vs_beta(max_infectious_ratio_data, cfg)
    plotting.plot_final_size_vs_beta(final_size_data, cfg) # This now only plots
    plotting.plot_final_size_heatmap(final_size_data, cfg)
    plotting.plot_final_size_scenario_comparison_heatmap(final_size_data, cfg)
    plotting.plot_early_extinction_heatmap(final_size_data, cfg)
    plotting.plot_extinction_vs_size(extinction_vs_size_data, cfg)

    print(f"\n{'='*20} All simulation experiments complete. {'='*20}")
    
    # Cleanup temporary cache
    if cache_dir.exists():
        print("Cleaning up temporary event cache...")
        shutil.rmtree(cache_dir)

if __name__ == "__main__":
    main()