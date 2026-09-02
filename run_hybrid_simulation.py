import numpy as np
import datetime
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, TextIO
import pickle
import shutil
import argparse
import sys

from code_config_models import create_default_config, load_config_from_json
from code_simulation.data import DataLoader
import simulation_orchestrator as simulation
import code_plotting as plotting

class ScenarioRunner:
    """
    Orchestrates the epidemic simulation for a given scenario.
    """
    def __init__(self, event_model_type: str, sim_data: Dict[str, Any], config: Any, initial_infections_data: Optional[Dict[int, Dict[int, np.ndarray]]] = None, event_cache: Optional[Dict[int, Any]] = None):
        self.event_model_type = event_model_type
        self.sim_data = sim_data
        # Create a shallow copy of the config to ensure isolation for this scenario context
        self.cfg = config.model_copy()
        self.initial_infections_data = initial_infections_data
        self.event_cache = event_cache
        
        # Set config for this scenario
        self.cfg.event_model_type = self.event_model_type
        if self.event_model_type.startswith('no_event'):
            self.cfg.current_event_scenario_name = self.event_model_type
        if self.event_model_type == 'no_event_distributed':
            self.cfg.no_event_seeding_mode = 'from_file'
        else:
            self.cfg.no_event_seeding_mode = 'single_patch'

        # CRITICAL FIX: Recalculate beta from R0 at the start of every scenario to ensure clean state
        self.cfg.gamma = 1.0 / self.cfg.period_infectious
        self.cfg.beta = self.cfg.R_0 * self.cfg.gamma

    def _log_scenario_start(self):
        print(f"\n" + "-"*60)
        print(f">>> Starting Scenario: {self.event_model_type} <<<")
        print(f"    R0: {self.cfg.R_0:.2f} | Beta_Event: {self.cfg.event_base_transmission_rate:.2f} | Iss: {self.cfg.I_ss}")
        print("-"*60)

    def _preprocess_large_venue_data(self):
        """Handles string ID conversion and attendee count updates for empirical contact networks."""
        processed_file, n_attendees, header_map = simulation.events.preprocess_contact_file(self.cfg)
        self.cfg.large_venue_contact_file = processed_file
        self.cfg.large_venue_total_attendees = n_attendees
        self.cfg.event_contact_header_map = header_map

    def run(self):
        """Orchestrates simulation execution and result aggregation."""
        time_start = datetime.datetime.now().replace(microsecond=0)
        self._log_scenario_start()

        if self.event_model_type == 'large_venue' and getattr(self.cfg, 'large_venue_ids_are_strings', False):
            self._preprocess_large_venue_data()

        self._preload_contacts()
        risk_metrics = self._analyze_risk()

        n_frames = self.cfg.n_days * 2 + 1
        frame_times = np.arange(0, self.cfg.n_days + 0.5, 0.5)
        realizations = self._initialize_realizations(n_frames)
        all_runs_results = self._execute_iterations(realizations, frame_times)

        # --- Post-Simulation ---
        plotting.save_realizations(realizations, self.cfg)
        if not self.event_model_type.startswith('no_event'):
             plotting.save_event_recruitment_records(all_runs_results[0]['recruitment_data'], all_runs_results[0]['infected_attendee_counts'], self.cfg)

        arrival_stats = self._analyze_and_save_results(realizations, all_runs_results)

        # Determine the attendee log filename for the current scenario configuration
        temp_logger = simulation.InfectionLogger(self.event_model_type, self.sim_data['n_patches'], self.cfg)
        attendee_log_filename = temp_logger.attendee_log_filename

        time_end = datetime.datetime.now().replace(microsecond=0)
        print(f"\nScenario '{self.event_model_type}' finished. Total run time: {time_end - time_start}")
        
        return {
            "realizations": realizations, "all_runs_results": all_runs_results,
            "attendee_log_filename": attendee_log_filename, "risk_metrics": risk_metrics, "arrival_stats": arrival_stats
        }

    def _preload_contacts(self):
        # --- Optimization: Pre-load Contact Data ---
        # Load contact files once into memory to avoid repeated I/O during simulation steps
        self.sim_data['contact_cache'] = {}
        if self.event_model_type == 'large_venue':
            print(f"  -> Pre-loading contact file: {self.cfg.large_venue_contact_file}")
            try:
                self.sim_data['contact_cache']['large_venue'] = pd.read_csv(self.cfg.large_venue_contact_file)
            except Exception as e:
                print(f"Warning: Could not pre-load contact file: {e}")
        elif self.event_model_type == 'multi_venue':
            print(f"  -> Pre-loading multi-venue contact files...")
            for group_id, venue_id in self.cfg.multi_venue_group_venues.items():
                fname = Path("data_crowd-contacts") / f"contacts_aggregated_N{self.cfg.multi_venue_total_attendees}_T36000_{group_id}.csv"
                if fname.exists():
                    try:
                        self.sim_data['contact_cache'][group_id] = pd.read_csv(fname)
                    except Exception as e:
                        print(f"Warning: Could not load {fname}: {e}")

    def _analyze_risk(self):
        # --- Event Percolation / Network Risk Analysis ---
        risk_metrics = {}
        if self.event_model_type == 'large_venue':
            print("  -> [Event Risk] Analyzing Contact Network Structure (Percolation)...")
            gcc_frac, total_nodes, avg_R, max_R = simulation.events.analyze_event_percolation_risk(
                self.cfg.large_venue_contact_file, self.cfg.event_base_transmission_rate, self.cfg.event_contact_fps, self.cfg.event_contact_header_map
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
        return risk_metrics

    def _execute_iterations(self, realizations, frame_times):
        all_runs_results = []

        # Use InfectionLogger as a context manager
        with simulation.InfectionLogger(self.event_model_type, self.sim_data['n_patches'], self.cfg) as logger:
            for run_id in range(self.cfg.n_iterations):
                run_seed = self.cfg.rnd_seed_0 * (self.cfg.initial_infection_patch_id + 1) * (run_id + 1)
                rng = np.random.default_rng(run_seed)

                self._print_progress(run_id)
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

                results = simulation.run_simulation(self.sim_data, self.cfg, rng, run_id, logger, initial_data_for_run, precomputed_data)

                self._aggregate_run_results(realizations, frame_times, results, run_id)

                # Save diagnostics for run 0 only (once per scenario)
                self._save_diagnostics(run_id=run_id)
                if run_id == 0 and self.event_model_type == 'large_venue':
                    self._save_day1_state(results, run_id=run_id)

                self._memory_optimization(results)
                all_runs_results.append(results)
        return all_runs_results

    def _print_progress(self, run_id: int):
        """Print per-run progress on a single in-place line (updated every ~10%)."""
        total = self.cfg.n_iterations
        if total == 0:
            return
        update_interval = max(1, total // 10)
        if (run_id + 1) % update_interval == 0 or run_id + 1 == total:
            pct = 100 * (run_id + 1) / total
            sys.stdout.write(f"\r  {self.event_model_type}: {run_id+1}/{total} ({pct:.0f}%)")
            sys.stdout.flush()
            if run_id + 1 == total:
                sys.stdout.write("\n")
                sys.stdout.flush()

    def _memory_optimization(self, results):
        """Aggregates spatial arrays to save memory."""
        n_patches = self.sim_data['n_patches']
        if 'E_ij' in results:
            results['E_j'] = [np.sum(mat.reshape(n_patches, n_patches), axis=0) for mat in results['E_ij']]
            del results['E_ij']
        if 'I_ij' in results:
            results['I_j'] = [np.sum(mat.reshape(n_patches, n_patches), axis=0) for mat in results['I_ij']]
            del results['I_ij']

        keys_to_drop = ['S_ij', 'R_ij', 'N_ij', 'S_all', 'E_all', 'I_all', 'R_all', 'rt', 'new_exposed', 'new_infected']
        for key in keys_to_drop:
            if key in results:
                del results[key]

    def _initialize_realizations(self, n_frames: int) -> Dict[str, np.ndarray]:
        """Prepares arrays to store results from all runs."""
        frame_times = np.arange(0, self.cfg.n_days + 0.5, 0.5) # type: ignore
        realizations = {}
        for comp in ['S', 'E', 'I', 'R']:
            realizations[comp] = np.zeros((n_frames, self.cfg.n_iterations + 1))
            realizations[comp][:, 0] = frame_times
        return realizations

    def _aggregate_run_results(self, realizations: Dict[str, np.ndarray], frame_times: np.ndarray, results: Dict[str, Any], run_id: int):
        """Interpolates and stores results from a single run."""
        for comp in ['S', 'E', 'I', 'R', 'rt']:
            source_key = 'rt' if comp == 'rt' else f'{comp}_all'
            
            if source_key not in results:
                continue

            if comp not in realizations:
                realizations[comp] = np.zeros((len(frame_times), self.cfg.n_iterations + 1))
                realizations[comp][:, 0] = frame_times
            
            realizations[comp][:, run_id + 1] = np.interp(frame_times, results['t'], results[source_key])

        if self.cfg.write_full_results:
            plotting.save_full_results(run_id, results, self.cfg)

    def _analyze_and_save_results(self, realizations: Dict[str, np.ndarray], all_runs_results: List[Dict[str, Any]]):
        """Calculates analysis statistics and saves essential records for the completed scenario."""
        # Save essential records (CSVs/TXTs)
        plotting.analyze_and_save_peak_time(realizations['E'], realizations['I'], self.cfg)
        plotting.save_district_active_cases_per_run(all_runs_results, self.sim_data, self.cfg)
        plotting.save_arrival_time_records(all_runs_results, self.sim_data, self.cfg)
        plotting.save_arrival_time_records_I(all_runs_results, self.sim_data, self.cfg)
        plotting.save_invasion_edge_records(all_runs_results, self.sim_data, self.cfg)
        plotting.save_invasion_tree_from_records(
            all_runs_results,
            self.sim_data,
            self.cfg,
            method=getattr(self.cfg, 'invasion_tree_method', 'arborescence')
        )
        
        # Generate Event-Induced Infection Figures (Depends on beta_event)
        # Figures are now generated by run_sim_visualization.py, not during simulation
        # if not self.event_model_type.startswith('no_event'):
        #     plotting.plot_event_induced_infections_map(all_runs_results, self.sim_data, self.cfg)
        
        # Perform arrival time analysis and get stats (plotting is conditionally skipped based on self.cfg.plot_on_the_fly)
        arrival_stats = plotting.analyze_and_plot_arrival_times(all_runs_results, self.sim_data, self.cfg)
        
        # Plotting functions called here will respect self.cfg.plot_on_the_fly via guards in plotting.py
        # Figures are now generated by run_sim_visualization.py, not during simulation
        # plotting.plot_peak_time_distribution_by_district(all_runs_results, self.sim_data, self.cfg)
        
        # Generate specialized active case curve plot
        # Figures are now generated by run_sim_visualization.py, not during simulation
        # total_pop = self.sim_data['population_df']['population'].sum()
        # plotting.plot_active_case_curves_only(realizations, total_pop, self.cfg)

        return arrival_stats

    # --- Diagnostic output saving ---

    def _save_diagnostics(self, run_id=0):
        """Save diagnostic data for verification of seed propagation.
        
        Called once after the first run of the large_venue scenario.
        Saves attendee distribution, event-generated seed, and day-1 state.
        """
        if self.event_model_type != 'large_venue':
            return
        if run_id != 0:
            return  # Only save diagnostics for run 0

        results_dir = getattr(self.cfg, 'results_dir', Path("results"))
        seed_id = getattr(self.cfg, 'initial_infection_patch_id', 0)
        seed_str = "seed{:02d}".format(seed_id)

        # 1. Attendee distribution (from the cached event data for run 0)
        if self.event_cache and run_id in self.event_cache:
            micro_results = self.event_cache[run_id]
            attendee_df_initial = pd.DataFrame(micro_results.get("attendee_df_initial", []))
            if len(attendee_df_initial) > 0:
                # Origin district -> number of attendees recruited
                attendee_dist = attendee_df_initial.groupby('home_patch_id').agg(
                    number_of_attendees=('home_patch_id', 'size'),
                ).reset_index()
                attendee_dist.rename(columns={'home_patch_id': 'origin_district'}, inplace=True)

                # Exposed and infectious by origin district
                exposed = attendee_df_initial[attendee_df_initial['status'] == 'E'].groupby('home_patch_id').size()
                infectious = attendee_df_initial[attendee_df_initial['status'] == 'I'].groupby('home_patch_id').size()
                attendee_dist['number_exposed'] = attendee_dist['origin_district'].map(exposed).fillna(0).astype(int)
                attendee_dist['number_infectious'] = attendee_dist['origin_district'].map(infectious).fillna(0).astype(int)

                # Ensure all districts are present
                n_patches = self.sim_data['n_patches']
                for d in range(n_patches):
                    if d not in attendee_dist['origin_district'].values:
                        attendee_dist = pd.concat([attendee_dist, pd.DataFrame([{
                            'origin_district': d,
                            'number_of_attendees': 0,
                            'number_exposed': 0,
                            'number_infectious': 0,
                        }])], ignore_index=True)
                attendee_dist = attendee_dist.sort_values('origin_district').reset_index(drop=True)
                attendee_dist.to_csv(results_dir / "{}_attendee_distribution.csv".format(seed_str), index=False)

                # 2. Event-generated epidemic seed (E, I per district after event)
                infected_counts_by_patch = micro_results.get("infected_counts_by_patch", np.zeros(n_patches))
                seed_records = []
                for d in range(n_patches):
                    e_count = int((attendee_df_initial[attendee_df_initial['home_patch_id'] == d]['status'] == 'E').sum())
                    i_count = int(infected_counts_by_patch[d]) if d < len(infected_counts_by_patch) else 0
                    seed_records.append({
                        'district': d,
                        'E': e_count,
                        'I': i_count,
                        'total_seed': e_count + i_count,
                    })
                seed_df = pd.DataFrame(seed_records)
                seed_df.to_csv(results_dir / "event_generated_seed_{}.csv".format(seed_str), index=False)

    def _save_day1_state(self, results, run_id=0):
        """Save the district-level SEIR state after the first simulation day.
        
        Called after the large_venue scenario completes for run 0.
        Saves day1_district_state_{seedXX}.csv with S, E, I, R per district.
        """
        if run_id != 0:
            return

        results_dir = getattr(self.cfg, 'results_dir', Path("results"))
        seed_id = getattr(self.cfg, 'initial_infection_patch_id', 0)
        seed_str = "seed{:02d}".format(seed_id)
        n_patches = self.sim_data['n_patches']

        # Get diagnostic state from results dict (added in simulation_orchestrator.py)
        day1_state = results.get('district_state_day1', {})
        if day1_state and 'S' in day1_state:
            S_d1 = day1_state['S']
            E_d1 = day1_state['E']
            I_d1 = day1_state['I']
            R_d1 = day1_state['R']
        else:
            S_d1 = E_d1 = I_d1 = R_d1 = np.zeros(n_patches)

        day1_records = []
        for d in range(n_patches):
            day1_records.append({
                'district': d,
                'S': float(S_d1[d]) if d < len(S_d1) else 0.0,
                'E': float(E_d1[d]) if d < len(E_d1) else 0.0,
                'I': float(I_d1[d]) if d < len(I_d1) else 0.0,
                'R': float(R_d1[d]) if d < len(R_d1) else 0.0,
            })
        day1_df = pd.DataFrame(day1_records)
        day1_df.to_csv(results_dir / "day1_district_state_{}.csv".format(seed_str), index=False)

        # Also save day0 state (initial seeding, before any dynamics)
        day0_state = results.get('district_state_day0', {})
        if day0_state and 'S' in day0_state:
            S_d0 = day0_state['S']
            E_d0 = day0_state['E']
            I_d0 = day0_state['I']
            R_d0 = day0_state['R']

            day0_records = []
            for d in range(n_patches):
                day0_records.append({
                    'district': d,
                    'S': float(S_d0[d]) if d < len(S_d0) else 0.0,
                    'E': float(E_d0[d]) if d < len(E_d0) else 0.0,
                    'I': float(I_d0[d]) if d < len(I_d0) else 0.0,
                    'R': float(R_d0[d]) if d < len(R_d0) else 0.0,
                })
            day0_df = pd.DataFrame(day0_records)
            day0_df.to_csv(results_dir / "day0_district_state_{}.csv".format(seed_str), index=False)


class ScenarioManager:
    """
    Manages and runs a sequence of simulation scenarios.
    """    
    def __init__(self, sim_data, config, current_R_0, current_I_ss, current_beta_event, event_cache=None):
        print(f"\n{'='*15} Starting Simulation for R0 = {current_R_0}, I_ss = {current_I_ss}, Beta_Event = {current_beta_event} {'='*15}")
        self.sim_data = sim_data
        # Create a shallow copy to ensure parameter changes here don't affect other manager instances
        self.cfg = config.model_copy()
        self.all_scenario_results: Dict[str, Any] = {}
        self.event_cache = event_cache

        # Set the current parameters for this entire manager instance
        self.cfg.R_0 = current_R_0
        self.cfg.beta = current_R_0 * (1.0 / self.cfg.period_infectious)
        self.cfg.I_ss = current_I_ss # type: ignore
        # This is the key. Update the event-specific parameter with the current sweep value.
        self.cfg.large_venue_initially_infected = current_I_ss
        self.cfg.event_base_transmission_rate = current_beta_event

    def run_all_scenarios(self, baseline_results=None, plot_combined_summary=True):
        """Runs all four predefined scenarios in sequence."""
        # Store the original R0 to restore it later
        original_R0 = self.cfg.R_0
        modes = self.cfg.event_simulation_modes

        # 1. Large-venue scenario (must run first to generate attendee file)
        if 'large_venue' in modes:
            large_venue_runner = ScenarioRunner('large_venue', self.sim_data, self.cfg, event_cache=self.event_cache)
            lv_results = large_venue_runner.run()
            self.all_scenario_results['large_venue'] = {
                "realizations": lv_results["realizations"],
                "all_runs_results": lv_results["all_runs_results"],
                "risk_metrics": lv_results.get("risk_metrics", {}),
                "arrival_stats": lv_results.get("arrival_stats")
            }
            attendee_log_filename = lv_results.get("attendee_log_filename")
        else:
            # If not running large_venue, we might need to find a pre-existing log file
            # This maintains functionality for running only multi-venue if the file exists.
            results_dir = getattr(self.cfg, 'results_dir', Path("results"))
            filename = results_dir / f"ini_infectious_attendees_R{int(self.cfg.R_0*100)}_Iss{self.cfg.I_ss}.txt"
            if not filename.exists():
                filename = results_dir / f"ini_infectious_attendees_R{int(self.cfg.R_0*100)}_betaEvent{int(self.cfg.event_base_transmission_rate*100)}_Iss{self.cfg.I_ss}.txt"
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

                for group_id, venue_id in self.cfg.multi_venue_group_venues.items():
                    count = infected_counts_by_group.get(group_id, 0)
                    print(f"  -> Venue at District {venue_id} (for Group '{group_id}'): {count} attendees")
            # ----------------------------------------------------------------

            # 2. Multi-venue scenario
            if 'multi_venue' in modes:
                if initial_attendees_df is not None:
                    self.cfg.R_0 = original_R0
                    self.cfg.beta = original_R0 * (1.0 / self.cfg.period_infectious)
                    mv_runner = ScenarioRunner('multi_venue', self.sim_data, self.cfg, initial_attendees_df)
                    mv_results = mv_runner.run()
                    self.all_scenario_results['multi_venue'] = {
                        "realizations": mv_results["realizations"],
                        "all_runs_results": mv_results["all_runs_results"],
                        "arrival_stats": mv_results.get("arrival_stats")
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
                    self.cfg.R_0 = original_R0
                    self.cfg.beta = original_R0 * (1.0 / self.cfg.period_infectious)
                    ne_runner = ScenarioRunner('no_event', self.sim_data, self.cfg)
                    ne_results = ne_runner.run()
                    self.all_scenario_results['no_event'] = {
                        "realizations": ne_results["realizations"],
                        "all_runs_results": ne_results["all_runs_results"],
                        "arrival_stats": ne_results.get("arrival_stats")
                    } # No post-event threshold for baseline

        # After all scenarios are run, generate the combined plot
        # Figures are now generated by run_sim_visualization.py, not during simulation
        # if len(self.all_scenario_results) > 0:
        #     # Only plot combined summary if requested AND plot_on_the_fly is True
        #     if plot_combined_summary and getattr(self.cfg, 'plot_on_the_fly', True):
        #         plotting.create_combined_summary_plot(self.all_scenario_results, self.sim_data, self.cfg)
        #     # plotting.plot_peak_scatter_comparison(self.all_scenario_results, self.cfg)
        #     # plotting.plot_patch_peak_comparison(self.all_scenario_results, self.sim_data, self.cfg)

def _enumerate_sweep_space(base_cfg):
    """Compute the full sweep space from config for upfront logging.

    Returns a dict with: seed_ids, iss_values, betas, scenarios,
    r0_by_iss ({i_ss: [r0, ...]}), phase1_combos (per-seed-independent combo
    count), phase2_combos (top-level combos total across all seeds, each doing
    n_iterations), and the *_total_runs / grand_total_runs derived counts.
    """
    seed_ids = getattr(base_cfg, 'seed_patch_ids', [0])
    iss_values = list(base_cfg.parameter_sweep_config.keys())
    betas = list(getattr(base_cfg, 'event_base_transmission_rate_values', []))
    scenarios = list(getattr(base_cfg, 'current_event_scenario_names', []))
    n_iter = getattr(base_cfg, 'n_iterations', 1)

    r0_by_iss = {}
    for i_ss_value, r0_params in base_cfg.parameter_sweep_config.items():
        if r0_params['R0_step'] > 0:
            num_steps = int(round((r0_params['R0_end'] - r0_params['R0_start']) / r0_params['R0_step'])) + 1
            r0_vals = [float(v) for v in np.round(np.linspace(r0_params['R0_start'], r0_params['R0_end'], num_steps), 6)]
        else:
            r0_vals = [r0_params['R0_start']]
        r0_by_iss[i_ss_value] = r0_vals

    phase1_combos = len(scenarios) * len(iss_values) * len(betas)
    phase2_combos = 0
    for _ in seed_ids:
        for i_ss_value in iss_values:
            for _r0 in r0_by_iss[i_ss_value]:
                phase2_combos += 1 + len(betas) * len(scenarios)  # baseline + events
    return {
        'seed_ids': seed_ids,
        'iss_values': iss_values,
        'betas': betas,
        'scenarios': scenarios,
        'r0_by_iss': r0_by_iss,
        'n_iterations': n_iter,
        'phase1_combos': phase1_combos,
        'phase2_combos': phase2_combos,
        'phase1_total_runs': phase1_combos * n_iter,
        'phase2_total_runs': phase2_combos * n_iter,
        'grand_total_runs': (phase1_combos + phase2_combos) * n_iter,
    }


def _log_sweep_plan(base_cfg):
    """Print the parameter combinations that will be simulated, plus totals."""
    sp = _enumerate_sweep_space(base_cfg)
    print("\n" + "=" * 60)
    print("SIMULATION SWEEP PLAN — parameter combinations to be executed")
    print("=" * 60)
    print(f"  Seed districts : {sp['seed_ids']}  ({len(sp['seed_ids'])} seeds)")
    print(f"  I_ss values    : {sp['iss_values']}  ({len(sp['iss_values'])})")
    print(f"  Beta event vals: {sp['betas']}  ({len(sp['betas'])})")
    print(f"  Scenarios      : {sp['scenarios']}  ({len(sp['scenarios'])})")
    print("  R0 per I_ss    :")
    for i_ss_val, r0s in sp['r0_by_iss'].items():
        print(f"      I_ss={i_ss_val} -> R0={r0s}  ({len(r0s)} values)")
    print(f"  Monte Carlo runs per combo: {sp['n_iterations']}")
    print("-" * 60)
    print(f"  Phase 1 precompute combos: {sp['phase1_combos']}  ({sp['phase1_total_runs']} sim runs)")
    print(f"  Phase 2 sweep    combos: {sp['phase2_combos']}  ({sp['phase2_total_runs']} sim runs)")
    print(f"  Total estimated runs     : {sp['grand_total_runs']}")
    print("=" * 60 + "\n")
    return sp


def _run_phase2(base_cfg, sim_data, total_population, cache_dir, seed_patch_id, sweep_plan=None):
    """
    Runs the Phase 2 parameter sweep for a given seed district.
    All result files are written to base_cfg.results_dir.
    """
    baseline_cache = {}
    max_infectious_ratio_data = []
    final_size_data = []
    affected_districts_data = []
    extinction_vs_size_data = []

    results_dir = getattr(base_cfg, 'results_dir', Path("results"))
    max_ratio_summary_file = results_dir / "max_infectious_ratio_summary.csv"
    final_size_summary_file = results_dir / "final_size_summary.csv"
    affected_districts_summary_file = results_dir / "affected_districts_summary.csv"

    print("\n" + "="*60)
    print(f"PHASE 2: Main Simulation Loop (Seed District={seed_patch_id})")
    print("="*60)
    r0_groups = []
    for i_ss_value, r0_params in base_cfg.parameter_sweep_config.items():
        if r0_params['R0_step'] > 0:
            num_steps = int(round((r0_params['R0_end'] - r0_params['R0_start']) / r0_params['R0_step'])) + 1
            r0_values = np.linspace(r0_params['R0_start'], r0_params['R0_end'], num_steps)
            r0_values = np.round(r0_values, 6)
        else:
            r0_values = np.array([r0_params['R0_start']])
        for r_0_value in r0_values:
            r0_groups.append((i_ss_value, r_0_value))

    n_r0_groups = len(r0_groups)
    for combo_idx, (i_ss_value, r_0_value) in enumerate(r0_groups):
            print(f"  Phase 2: scenario-batch {combo_idx+1}/{n_r0_groups} (R0={r_0_value}, I_ss={i_ss_value})")

            # --- Run Baseline (No Event) Once per (R0, I_ss) ---
            baseline_key = (r_0_value, i_ss_value)
            
            if baseline_key not in baseline_cache:
                print(f"\n>>> Running Baseline (No Event) for R0={r_0_value}, Iss={i_ss_value} <<<")
                base_cfg.R_0 = r_0_value
                base_cfg.I_ss = i_ss_value
                base_cfg.beta = r_0_value * (1.0 / base_cfg.period_infectious)
                # Use the first beta value for filename generation purposes
                base_cfg.event_base_transmission_rate = base_cfg.event_base_transmission_rate_values[0]

                # --- Metapopulation Percolation Check ---
                mp_frac, mp_count = simulation.analyze_metapopulation_percolation_risk(sim_data, base_cfg)
                print(f"  -> [Metapopulation Risk] R0={r_0_value:.2f}: Reachable Patches from Seed {base_cfg.initial_infection_patch_id}: {mp_count}/{sim_data['n_patches']} ({mp_frac:.1%})")

                baseline_runner = ScenarioRunner('no_event', sim_data, base_cfg)
                res = baseline_runner.run()
                baseline_cache[baseline_key] = {
                    "realizations": res["realizations"],
                    "all_runs_results": res["all_runs_results"],
                    "arrival_stats": res["arrival_stats"]
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

            # Calculate geographic spread for baseline
            arrival_stats_base = res_base['arrival_stats']
            base_affected_item = {
                "scenario_name": "no_event", "R0": r_0_value, "beta_event": 0.0, "I_ss": i_ss_value,
                "mean_affected": arrival_stats_base['mean_affected'],
                "std_affected": arrival_stats_base['std_affected'],
                "values": arrival_stats_base['affected_values']
            }
            affected_districts_data.append(base_affected_item)
            plotting.update_summary_csv(affected_districts_summary_file, [base_affected_item])

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
            betas = base_cfg.event_base_transmission_rate_values
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

                for scenario_name in base_cfg.current_event_scenario_names:
                    # Update the global configuration to match the current scenario
                    base_cfg.set_active_scenario(scenario_name)

                    # Re-apply seed_patch_id to large_venue_ini_infected_attendee_patch
                    # (set_active_scenario resets it to the scenario's default of 0)
                    seed_patch_id = getattr(base_cfg, 'initial_infection_patch_id', 0)
                    base_cfg.large_venue_ini_infected_attendee_patch = seed_patch_id

                    # Load pre-computed cache from disk for this specific combination
                    current_cache = {}
                    for run_id in range(base_cfg.n_iterations):
                        cache_file = cache_dir / f"{scenario_name}_Iss{i_ss_value}_Beta{beta_event_value}_Run{run_id}.pkl"
                        if cache_file.exists():
                            with open(cache_file, 'rb') as f:
                                current_cache[run_id] = pickle.load(f)
                    
                    manager = ScenarioManager(sim_data, base_cfg, current_R_0=r_0_value, current_I_ss=i_ss_value, current_beta_event=beta_event_value, event_cache=current_cache)
                    # Run individual plots, but suppress the old combined plot
                    manager.run_all_scenarios(baseline_results=baseline_results, plot_combined_summary=False)

                    # Collect results, re-keying from generic 'large_venue' to specific scenario name
                    if 'large_venue' in manager.all_scenario_results:
                        all_results_this_beta[scenario_name] = manager.all_scenario_results['large_venue']
                        
                        # Also collect exposure data for the other comparison plot
                        all_runs = manager.all_scenario_results['large_venue']['all_runs_results']
                        event_exposures_all_runs = [r['event_exposures_by_day'] for r in all_runs]
                        first_event_day = base_cfg.event_days[0] if base_cfg.event_days else 0
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
                        
                        # Collect geographic spread for event
                        arrival_stats = manager.all_scenario_results['large_venue']['arrival_stats']
                        event_affected_item = {
                            "scenario_name": scenario_name, "R0": r_0_value, "beta_event": beta_event_value, "I_ss": i_ss_value,
                            "mean_affected": arrival_stats['mean_affected'],
                            "std_affected": arrival_stats['std_affected'],
                            "values": arrival_stats['affected_values']
                        }
                        affected_districts_data.append(event_affected_item)
                        plotting.update_summary_csv(affected_districts_summary_file, [event_affected_item])

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
                    scenario_groups = {"AMS": base_cfg.empirical_scenario_set_1, "Leipzig": base_cfg.empirical_scenario_set_2}
                    for group_name, scenarios_in_group in scenario_groups.items():
                        # Create a dictionary with results for only the scenarios in this group + baseline
                        group_results = {s: all_results_this_beta[s] for s in scenarios_in_group if s in all_results_this_beta}
                        group_results['no_event'] = baseline_results
                        
                        # if len(group_results) > 1: # Only plot if there are event scenarios to compare
                        #     pass

                    # Only plot exposure comparison and risk metrics for the first R0 if plot_on_the_fly is True
                    # Figures are now generated by run_sim_visualization.py, not during simulation
                    # if i_r0 == 0 and getattr(base_cfg, 'plot_on_the_fly', False):
                    #     if all_exposure_data_this_beta:
                    #         plotting.plot_scenario_exposure_comparison_boxplot(all_exposure_data_this_beta, base_cfg)
                    #     
                    #     if risk_comparison_data:
                    #         plotting.plot_event_risk_metrics(risk_comparison_data, base_cfg)
                    
                    # plotting.plot_phase_plane_comparison(all_results_this_beta, sim_data, base_cfg)
                    base_cfg.event_base_transmission_rate = beta_event_value
                    plotting.save_arrival_time_statistics(all_results_this_beta, sim_data, base_cfg)
                    plotting.analyze_arrival_time_sequence(all_results_this_beta, sim_data, base_cfg)


def main():
    """Main entry point for the simulation."""
    parser = argparse.ArgumentParser(
        description="Hybrid Epidemic Simulation Model"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file (overrides defaults)")
    args = parser.parse_args()

    # --- Config loading ---
    if args.config:
        base_cfg = load_config_from_json(args.config)
    else:
        base_cfg = create_default_config()

    loader = DataLoader(base_cfg.dataset_name)
    sim_data = loader.prepare_all()
    total_population = sim_data['population_df']['population'].sum()

    # --- Setup results directory ---
    results_dir = base_cfg.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- PHASE 2: Main Simulation Loop over Seed Districts ---
    # Pre-computation is now INSIDE the seed loop because the event micro-simulation
    # depends on the seed district (large_venue_ini_infected_attendee_patch).
    seed_patch_ids = getattr(base_cfg, 'seed_patch_ids', [0])
    print("Seed Patch IDs:", seed_patch_ids)

    sweep_plan = _log_sweep_plan(base_cfg)

    for seed_patch_id in seed_patch_ids:
        print(f"\n{'#'*20} Seed District: {seed_patch_id} {'#'*20}")
        base_cfg.initial_infection_patch_id = seed_patch_id
        # FIX: Also set the event seed attendee patch to the current seed district.
        # This ensures forced infected attendees are recruited from the seed district,
        # making outbreak dynamics differ across seed locations.
        base_cfg.large_venue_ini_infected_attendee_patch = seed_patch_id

        seed_results_dir = results_dir / f"seed_{seed_patch_id}"
        seed_results_dir.mkdir(parents=True, exist_ok=True)
        base_cfg.results_dir = seed_results_dir

        # --- PRE-COMPUTATION PHASE (per seed) ---
        # Pre-calculate event outcomes for unique (Scenario, I_ss, Beta_Event) combinations.
        # This avoids re-running the expensive agent-based model for every R0 value.
        # MUST run per seed because attendee recruitment depends on the seed district.
        print("\n" + "="*60)
        print(f"PHASE 1: Pre-computing Event Micro-Simulations (Seed={seed_patch_id})")
        print("="*60)

        # Setup temporary cache directory (per seed to avoid collisions)
        cache_dir = Path(f"temp_event_cache_seed_{seed_patch_id}")
        cache_dir.mkdir(parents=True, exist_ok=True)

        unique_iss_values = list(base_cfg.parameter_sweep_config.keys())
        unique_betas = base_cfg.event_base_transmission_rate_values

        for scenario_name in base_cfg.current_event_scenario_names:
            print(f"Pre-computing for scenario: {scenario_name}")
            base_cfg.set_active_scenario(scenario_name)
            # Re-apply seed_patch_id (set_active_scenario resets large_venue_ini_infected_attendee_patch to 0)
            base_cfg.large_venue_ini_infected_attendee_patch = seed_patch_id

            # Set event_model_type explicitly for the pre-computation phase
            base_cfg.event_model_type = 'large_venue'
            base_cfg.current_day = 0
            base_cfg.current_time = 0.0

            if getattr(base_cfg, 'large_venue_ids_are_strings', False):
                 processed_contact_file, n_attendees, header_map = simulation.events.preprocess_contact_file(base_cfg)
                 base_cfg.large_venue_contact_file = processed_contact_file
                 base_cfg.large_venue_total_attendees = n_attendees
                 base_cfg.event_contact_header_map = header_map

            for iss in unique_iss_values:
                base_cfg.large_venue_initially_infected = iss
                for beta in unique_betas:
                    base_cfg.event_base_transmission_rate = beta
                    for run_id in range(base_cfg.n_iterations):
                        # FIX: Include seed_patch_id in RNG seed so each seed gets a unique trajectory
                        run_seed = base_cfg.rnd_seed_0 * (seed_patch_id + 1) * (run_id + 1)
                        rng = np.random.default_rng(run_seed)
                        res = simulation.precompute_event_outcomes(sim_data, base_cfg, rng, run_id)
                        cache_file = cache_dir / f"{scenario_name}_Iss{iss}_Beta{beta}_Run{run_id}.pkl"
                        with open(cache_file, 'wb') as f:
                            pickle.dump(res, f)
                        update_interval = max(1, base_cfg.n_iterations // 10)
                        if (run_id + 1) % update_interval == 0 or run_id + 1 == base_cfg.n_iterations:
                            pct = 100 * (run_id + 1) / base_cfg.n_iterations
                            sys.stdout.write(f"\r  Pre-computing {scenario_name} (I_ss={iss}, Beta={beta}): {run_id+1}/{base_cfg.n_iterations} ({pct:.0f}%)")
                            sys.stdout.flush()
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    print(f"  -> I_ss={iss}, Beta={beta} ... Saved to disk.")

        _run_phase2(base_cfg, sim_data, total_population, cache_dir, seed_patch_id, sweep_plan=sweep_plan)

        # Cleanup temporary cache for this seed
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    print(f"\n{'='*20} All simulation experiments complete. {'='*20}")

    # Cleanup any leftover temp cache directories
    for seed_patch_id in seed_patch_ids:
        temp_cache = Path(f"temp_event_cache_seed_{seed_patch_id}")
        if temp_cache.exists():
            shutil.rmtree(temp_cache)
    temp_cache_old = Path("temp_event_cache")
    if temp_cache_old.exists():
        shutil.rmtree(temp_cache_old)

if __name__ == "__main__":
    main()
