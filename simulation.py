import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, TextIO
import event_modeling as events
import os
from pathlib import Path

class InfectionLogger:
    """Manages logging of initial and new infections for event-based scenarios."""
    def __init__(self, event_model_type: str, n_patches: int, cfg: Any):
        self.event_model_type = event_model_type
        self.n_patches = n_patches
        self.cfg = cfg
        self.ini_writer: Optional[TextIO] = None
        self.new_writer: Optional[TextIO] = None
        self.attendee_log_writer: Optional[TextIO] = None
        self.ini_filename: Optional[str] = None
        self.new_filename: Optional[str] = None
        self.attendee_log_filename: Optional[str] = None
        self._prepare_filenames()

    def _generate_base_filename(self) -> str:
        """Creates the common part of the log filename based on current config."""
        return (f"R{int(self.cfg.R_0*100)}_"
                f"betaEvent{int(self.cfg.event_base_transmission_rate*100)}_Iss{self.cfg.I_ss}.txt")

    def _prepare_filenames(self):
        """Sets up the full filenames for the log files."""
        if self.event_model_type not in ['large_venue', 'multi_venue']:
            return

        results_dir = getattr(self.cfg, 'results_dir', Path("results"))
        base_filename = self._generate_base_filename()

        # Unified filenames: no longer include scenario-specific prefixes like 'large_N...'
        self.attendee_log_filename = str(results_dir / f"ini_infectious_attendees_{base_filename}")
        self.ini_filename = str(results_dir / f"n_ini_infections_{base_filename}")
        self.new_filename = str(results_dir / f"n_new_infections_{base_filename}")

    def _setup_writer(self, filename: str) -> TextIO:
        """Opens a file for writing and adds the header."""
        # Check if file exists and has content to decide whether to write header
        file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
        header = f"scenario_name,run_id,event_day,n_all," + ",".join([f"n_{i}" for i in range(self.n_patches)]) + "\n"
        f = open(filename, 'a') # Append mode
        if not file_exists:
            f.write(header)
        return f

    def _setup_attendee_writer(self, filename: str) -> TextIO:
        """Opens a file for writing attendee details and adds the header."""
        file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
        header = "scenario_name,run_id,event_day,attendee_id,district_id,commuting_district_id,event_model,recruitment_model,initially_infected,event_patch_id,beta_event,R0,initially_infected_seed,initially_infected_patch\n"
        f = open(filename, 'a') # Append mode
        if not file_exists:
            f.write(header)
        return f

    def record_infections(self, writer: Optional[TextIO], run_id: int, day: int, data_array: np.ndarray):
        """Writes a row of infection data to the specified file."""
        if writer is None:
            return
        n_all = np.sum(data_array)
        data_str = ",".join(map(str, data_array.astype(int)))
        writer.write(f"{self.cfg.current_event_scenario_name},{run_id},{day},{int(n_all)},{data_str}\n")
        writer.flush()  # Force write to disk immediately

    def record_attendee_details(self, run_id: int, day: int, attendee_df: pd.DataFrame, cfg: Any):
        """Writes detailed records for a dataframe of attendees."""
        if self.attendee_log_writer is None:
            return
        
        # Build data row programmatically for robustness
        common_data = [
            self.cfg.event_model_type, self.cfg.attendee_recruitment_model,
            self.cfg.large_venue_initially_infected, self.cfg.large_venue_patch_id,
            self.cfg.event_base_transmission_rate, self.cfg.R_0,
            self.cfg.large_venue_ini_infected_attendee_seed,
            self.cfg.large_venue_ini_infected_attendee_patch
        ]
        common_data_str = ",".join(map(str, common_data))

        for _, row in attendee_df.iterrows():
            row_data = [
                run_id, day, row['agent_id'], row['home_patch_id'], row['current_patch_id']
            ]
            self.attendee_log_writer.write(f"{self.cfg.current_event_scenario_name},{','.join(map(str, row_data))},{common_data_str}\n")
        self.attendee_log_writer.flush()

    def __enter__(self):
        """Opens files and returns self, for use as a context manager."""
        if self.ini_filename: self.ini_writer = self._setup_writer(self.ini_filename)
        if self.new_filename: self.new_writer = self._setup_writer(self.new_filename)
        if self.attendee_log_filename: self.attendee_log_writer = self._setup_attendee_writer(self.attendee_log_filename)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes all open file writers."""
        if self.ini_writer:
            self.ini_writer.close()
        if self.new_writer:
            self.new_writer.close()
        if self.attendee_log_writer:
            self.attendee_log_writer.close()

    @staticmethod
    def load_initial_infections(filename: str, n_iterations: int, event_days: List[int]) -> Optional[Dict[int, Dict[int, np.ndarray]]]:
        """Loads the initial infection data from a file."""
        if not Path(filename).exists():
            print(f"Warning: Initial infection file not found: {filename}")
            return None
        
        df = pd.read_csv(filename)
        if df.empty:
            print("\n" + "*"*60)
            print(f"FATAL ERROR: The infection data file '{filename}' is empty.")
            print("This means the 'large_venue' scenario failed to write its output, preventing the 'multi_venue' scenario from starting.")
            print("Please check the console output for the 'large_venue' run for other errors.")
            print("*"*60)
            return None # Return None to allow graceful handling
        n_cols = [col for col in df.columns if col.startswith('n_') and col != 'n_all']
        
        # Group by run and day for efficient lookup
        grouped = df.groupby(['run_id', 'event_day'])
        initial_infections = {run_id: {} for run_id in range(n_iterations)}
        for (run_id, day), group in grouped:
            if run_id < n_iterations and day in event_days:
                initial_infections[run_id][day] = group[n_cols].to_numpy().flatten()
        return initial_infections

    @staticmethod
    def load_attendee_details(filename: str) -> Optional[pd.DataFrame]:
        """Loads the detailed infectious attendee records from a file."""
        if not Path(filename).exists():
            print(f"Warning: Detailed attendee file not found: {filename}")
            return None
        df = pd.read_csv(filename)
        if df.empty:
            print(f"Warning: Detailed attendee file is empty: {filename}")
            return None
        return df

def compute_lambda(beta, dt, n_patches, I_ij_t, N_ij_t, is_day):
    """Computes the force of infection lambda for each patch."""
    I_ij_reshaped = I_ij_t.reshape((n_patches, n_patches))
    N_ij_reshaped = N_ij_t.reshape((n_patches, n_patches))

    if is_day:
        sum_Iji = I_ij_reshaped.sum(axis=0)
        sum_Nji = N_ij_reshaped.sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            lambda_val = (dt * beta) * sum_Iji / sum_Nji
        # Broadcast lambda_val[j] to all i (rows)
        prob_infection = np.nan_to_num(lambda_val[np.newaxis, :])
        prob_infection = np.broadcast_to(prob_infection, (n_patches, n_patches)).ravel()
    else: # Night
        sum_Iij = I_ij_reshaped.sum(axis=1)
        sum_Nij = N_ij_reshaped.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            lambda_val = (dt * beta) * sum_Iij / sum_Nij
        # Broadcast lambda_val[i] to all j (cols)
        prob_infection = np.nan_to_num(lambda_val[:, np.newaxis])
        prob_infection = np.broadcast_to(prob_infection, (n_patches, n_patches)).ravel()
        
    return prob_infection

def compute_SEIR(beta, alpha, gamma, n_patches, S_t, E_t, I_t, R_t, N_t, TOD, rng):
    """Performs one step of the metapopulation SEIR model."""
    is_day = (TOD == "day-time")
    dt = 0.5

    prob_infection = compute_lambda(beta, dt, n_patches, I_t, N_t, is_day)

    new_exposed = rng.binomial(S_t, prob_infection)
    new_infected = rng.binomial(E_t, dt * alpha)
    new_recovered = rng.binomial(I_t, dt * gamma)

    S_new = np.maximum(S_t - new_exposed, 0)
    E_new = np.maximum(E_t + new_exposed - new_infected, 0)
    I_new = np.maximum(I_t + new_infected - new_recovered, 0)
    R_new = np.maximum(R_t + new_recovered, 0)

    N_new = S_new + E_new + I_new + R_new
    new_exposed_all = np.sum(new_exposed)
    new_infected_all = np.sum(new_infected)

    return dt, S_new, E_new, I_new, R_new, N_new, new_exposed_all, new_infected_all

def _recruit_and_transmit_large_venue(data: Dict[str, Any], cfg: Any, rng: np.random.Generator, S_t: np.ndarray, E_t: np.ndarray, I_t: np.ndarray, R_t: np.ndarray, run_id: int) -> Dict[str, Any]:
    """
    Helper: Performs the micro-scale event logic (Recruitment + Transmission).
    Returns the list of attendees (post-transmission) and stats.
    """
    n_patches = data['n_patches']
    N_ij_initial = data['N_ij_initial']

    # Refactored recruitment using Strategy Pattern
    recruitment_strategies = {
        'commuter': events.CommuterStrategy(),
        'gravity': events.GravityStrategy(),
        'population': events.PopulationStrategy(),
        'single_patch': events.SinglePatchStrategy()
    }
    strategy = recruitment_strategies.get(cfg.attendee_recruitment_model)
    if not strategy:
        raise ValueError(f"Unknown recruitment model: '{cfg.attendee_recruitment_model}'")

    # Determine if we need to force infected attendees (Day 0 only)
    force_infected_count = 0
    imported_infected_count = 0
    force_infected_patch = None

    if cfg.current_day == 0 and cfg.large_venue_initially_infected > 0:
        if getattr(cfg, 'large_venue_ini_infected_attendee_seed', None) == 'single_patch':
            if getattr(cfg, 'seeding_method', 'recruit') == 'recruit':
                force_infected_count = cfg.large_venue_initially_infected
                force_infected_patch = getattr(cfg, 'large_venue_ini_infected_attendee_patch', 0)
                # print(f"  -> Prioritizing recruitment of {force_infected_count} infected attendees from patch {force_infected_patch}.")
            else:
                # Import method: Add to imported count instead of forcing recruitment from population
                imported_infected_count += cfg.large_venue_initially_infected

    strategy_kwargs = {
        'event_patch_id': cfg.large_venue_patch_id,
        'flow_ij_df': data['flow_ij_df'],
        'distance_matrix': data['distance_matrix_km'],
        'single_patch_source_id': getattr(cfg, 'single_patch_recruitment_id', cfg.large_venue_patch_id)
    }
    attendees, S_non_attendees, E_non_attendees, I_non_attendees, R_non_attendees, rec_data = events.recruit_attendees(
        strategy, cfg.large_venue_total_attendees, S_t, E_t, I_t, R_t, n_patches, N_ij_initial, rng, data['population_df'],
        force_infected_count=force_infected_count,
        force_infected_patch=force_infected_patch,
        imported_infected_count=imported_infected_count,
        **strategy_kwargs
    )
    attendee_df = pd.DataFrame(attendees)
    e_before = attendee_df[attendee_df['status'] == 'E'].shape[0]
    i_before = attendee_df[attendee_df['status'] == 'I'].shape[0]

    # Recalculate infected counts *after* seeding to get the total initial infected for the event
    infected_counts_by_patch = attendee_df[attendee_df['status'] == 'I'].groupby('home_patch_id').size().reindex(range(n_patches), fill_value=0).to_numpy()

    # Use cached dataframe if available, otherwise fall back to file path
    contact_data = data.get('contact_cache', {}).get('large_venue', cfg.large_venue_contact_file)

    attendees_after_tx = events.simulate_event_transmission(
        attendee_df.to_dict('records'), contact_data, cfg.event_base_transmission_rate,
        cfg.event_contact_fps, rng
        , cfg.event_contact_header_map
    )
    
    # Calculate stats
    df_after = pd.DataFrame(attendees_after_tx)
    e_after = df_after[df_after['status'] == 'E'].shape[0]
    
    return {
        "attendees_after_tx": attendees_after_tx,
        "rec_data": rec_data,
        "infected_counts_by_patch": infected_counts_by_patch,
        "i_before": i_before,
        "e_before": e_before,
        "e_after": e_after,
        "attendee_df_initial": attendee_df # Needed for logging initial state
    }

def _run_large_venue_event(data: Dict[str, Any], cfg: Any, rng: np.random.Generator, S_t: np.ndarray, E_t: np.ndarray, I_t: np.ndarray, R_t: np.ndarray, logger: InfectionLogger, run_id: int, N_t: np.ndarray, precomputed_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Handles the logic for a 'large_venue' event day."""
    n_patches = data['n_patches']
    N_ij_initial = data['N_ij_initial']

    # 1. Get Event Micro-Simulation Results (Recruitment + Transmission)
    if precomputed_data:
        # Use cached results
        micro_results = precomputed_data
        
        # We need to reconstruct S_non_attendees etc. by removing attendees from current S_t
        # Since precomputed data only has the attendee list, we perform removal now.
        attendees_after_tx = micro_results["attendees_after_tx"]
        attendee_df = pd.DataFrame(attendees_after_tx) # This has status AFTER transmission
        
        # IMPORTANT: To get non-attendees correctly, we must remove the attendees based on their 
        # status *at the moment of recruitment*. However, the cached list has status *after* transmission.
        # But since only S->E happens, and I/R don't change, we can infer.
        # Actually, simpler: _remove_attendees_from_population just subtracts counts.
        # We should subtract the attendees based on their CURRENT status in the cache? 
        # No, we must subtract them from the pools they came from.
        # If an agent was S and became E, they were removed from S pool.
        # So we should treat them as their *pre-event* status for removal? 
        # events._remove_attendees_from_population uses the 'status' column.
        # If we pass the post-transmission DF, we remove 'E's from the 'E' pool. 
        # But they were 'S' when they left the 'S' pool.
        # FIX: We need to use the status they had *before* transmission for removal from S_t.
        # However, `recruit_attendees` returns `S_after` directly.
        # If we use cache, we don't have `S_after`. We have to compute it.
        # To avoid complexity, let's assume we remove them based on their *post-event* status 
        # implies they are removed from that pool. But that's wrong for S->E.
        # OPTIMIZATION SHORTCUT: The `micro_results` should ideally return `S_non_attendees` 
        # but `S_non_attendees` depends on `S_t` which depends on R0 if t > 0.
        # Since we assume t=0, `S_t` is constant.
        # So we can just re-run the removal logic using the cached attendee list, 
        # BUT we need to revert S->E changes in the list to remove them from S pool?
        # Actually, `recruit_attendees` does removal.
        # Let's just re-run `_remove_attendees_from_population` using the cached list, 
        # but we need to know who was S.
        # The cached `attendees_after_tx` has the final status.
        # If we simply re-run `recruit_attendees` it defeats the purpose.
        # SOLUTION: The `micro_results` contains `attendee_df_initial` which has status BEFORE transmission.
        attendee_df_initial = micro_results["attendee_df_initial"]
        S_non_attendees, E_non_attendees, I_non_attendees, R_non_attendees = events._remove_attendees_from_population(
            attendee_df_initial, 
            S_t.reshape(n_patches, n_patches), 
            E_t.reshape(n_patches, n_patches), 
            I_t.reshape(n_patches, n_patches), 
            R_t.reshape(n_patches, n_patches), 
            N_ij_initial.reshape(n_patches, n_patches), 
            n_patches, rng
        )
    else:
        # Run full logic
        micro_results = _recruit_and_transmit_large_venue(data, cfg, rng, S_t, E_t, I_t, R_t, run_id)
        S_non_attendees = events._remove_attendees_from_population(
            micro_results["attendee_df_initial"], 
            S_t.reshape(n_patches, n_patches), 
            E_t.reshape(n_patches, n_patches), 
            I_t.reshape(n_patches, n_patches), 
            R_t.reshape(n_patches, n_patches), 
            N_ij_initial.reshape(n_patches, n_patches), 
            n_patches, rng
        )

    # Extract results
    attendees_after_tx = micro_results["attendees_after_tx"]
    rec_data = micro_results["rec_data"]
    infected_counts_by_patch = micro_results["infected_counts_by_patch"]
    i_before = micro_results["i_before"]
    e_before = micro_results["e_before"]
    e_after = micro_results["e_after"]
    
    new_exposures_today = e_after - e_before

    # Logging (Idempotent: only log if not already logged or if we need to log for this specific R0 run)
    # Since output filenames include R0, we must log every time.
    if cfg.current_day == 0 and cfg.large_venue_initially_infected > 0:
        attendee_df_initial = micro_results["attendee_df_initial"]
        infected_indices = attendee_df_initial[attendee_df_initial['status'] == 'I'].index
        newly_infected_df = attendee_df_initial.loc[infected_indices]
        logger.record_attendee_details(run_id, int(cfg.current_time), newly_infected_df, cfg)

    # --- Run Metapopulation Model for Non-Attendees ---
    # This depends on R0, so it must run every time.
    N_non_attendees = S_non_attendees + E_non_attendees + I_non_attendees + R_non_attendees
    meta_step_results = _run_metapopulation_step(data, cfg, rng, S_non_attendees, E_non_attendees, I_non_attendees, R_non_attendees, N_non_attendees)

    # Reintegrate attendees into the *updated* non-attendee population (post-metapopulation step)
    S_updated_non, E_updated_non, I_updated_non, R_updated_non = meta_step_results["S_new"], meta_step_results["E_new"], meta_step_results["I_new"], meta_step_results["R_new"]

    S_final, E_final, I_final, R_final, new_exposures_by_patch = events.reintegrate_attendees(
        attendees_after_event=attendees_after_tx, 
        S_after_recruitment=S_updated_non, 
        E_after_recruitment=E_updated_non, 
        I_after_recruitment=I_updated_non, 
        R_after_recruitment=R_updated_non, 
        n_patches=n_patches, 
        N_ij_initial=N_ij_initial, 
        rng=rng
    )
    
    # Add community infections to the event impact for total daily change tracking (optional, but good for consistency)
    # However, the return dict separates them usually. Let's keep new_exposures_today as EVENT specific.
    # The main loop calculates total delta based on S_final vs S_t.

    # --- Calculate and Log Event Impact Metrics ---
    districts_affected = np.count_nonzero(new_exposures_by_patch)
    print(f"  -> [Event Impact] New Exposures: {new_exposures_today} | Spread to {districts_affected} districts")

    event_results = {
        "S_new": S_final, "E_new": E_final, "I_new": I_final, "R_new": R_final,
        "new_exposures_today": new_exposures_today,
        "infected_recruited_today": i_before,
        "attendees_today": np.sum(rec_data),
        "recruitment_data": rec_data,
        "infected_counts_by_patch": infected_counts_by_patch,
        "new_exposures_by_patch": new_exposures_by_patch
    }
    return event_results

def _run_multi_venue_event(data: Dict[str, Any], cfg: Any, rng: np.random.Generator, S_t: np.ndarray, E_t: np.ndarray, I_t: np.ndarray, R_t: np.ndarray, initial_attendees_df: Optional[pd.DataFrame], run_id: int, N_t: np.ndarray) -> Dict[str, Any]:
    """Handles the logic for a 'multi_venue' event day."""
    # print(f"\n*** Day {int(cfg.current_time)}: Running MULTI-VENUE events for all groups ***")
    n_patches = data['n_patches']
    N_ij_initial = data['N_ij_initial']
    population_df = data['population_df']

    # Start with copies of the current population state. These will be depleted as attendees are recruited.
    S_non_attendees, E_non_attendees, I_non_attendees, R_non_attendees = S_t.copy(), E_t.copy(), I_t.copy(), R_t.copy()

    all_attendees_after_tx = []
    total_exposures_today = 0
    total_infected_recruited_today = 0
    total_attendees_today = 0
    recruitment_data_for_plotting = np.zeros(n_patches)
    total_infected_attendees_series = pd.Series(0, index=range(n_patches))

    day_initial_attendees_df = initial_attendees_df[(initial_attendees_df['event_day'] == int(cfg.current_time)) & (initial_attendees_df['run_id'] == run_id)] if initial_attendees_df is not None else None

    recruitment_strategies = {
        'commuter': events.CommuterStrategy(), 
        'gravity': events.GravityStrategy(), 
        'population': events.PopulationStrategy(),
        'single_patch': events.SinglePatchStrategy()
    }
    strategy = recruitment_strategies.get(cfg.attendee_recruitment_model)
    if not strategy: raise ValueError(f"Unknown recruitment model: '{cfg.attendee_recruitment_model}'")

    for group_id, venue_id in cfg.multi_venue_group_venues.items():
        # Reset start_id to 0 for each group so that agent_ids (0..N) match the local contact file IDs
        strategy_kwargs = {
            'flow_ij_df': data['flow_ij_df'], 
            'distance_matrix': data['distance_matrix_km'],
            'single_patch_source_id': getattr(cfg, 'single_patch_recruitment_id', None)
        }
        group_attendees, S_non_attendees, E_non_attendees, I_non_attendees, R_non_attendees, rec_data, _ = events.recruit_attendees_for_multi_venue_event(
            group_id, venue_id, cfg.multi_venue_total_attendees, strategy,
            S_non_attendees, E_non_attendees, I_non_attendees, R_non_attendees,
            n_patches, population_df, N_ij_initial, rng,
            day_initial_attendees_df, start_id=0, **strategy_kwargs
        )

        recruitment_data_for_plotting += rec_data
        total_attendees_today += rec_data.sum()

        if not group_attendees:
            print(f"  -> No attendees recruited for group {group_id}. Skipping event.")
            continue

        group_attendee_df = pd.DataFrame(group_attendees)

        e_before = group_attendee_df[group_attendee_df['status'] == 'E'].shape[0]
        infected_in_group = group_attendee_df[group_attendee_df['status'] == 'I']

        # print(f"  -> Venue {venue_id} (Group {group_id}): Assigned {len(infected_in_group)} initially infected attendees.")
        total_infected_recruited_today += len(infected_in_group)

        if cfg.current_day == cfg.event_days[0]:
            infected_counts = infected_in_group.groupby('home_patch_id').size()
            total_infected_attendees_series = total_infected_attendees_series.add(infected_counts, fill_value=0)

        # Determine effective transmission rate
        contact_file = Path("data_crowd-contacts") / f"contacts_aggregated_N{cfg.multi_venue_total_attendees}_T36000_{group_id}.csv"
        contact_data = data.get('contact_cache', {}).get(group_id, contact_file)

        attendees_after_tx = events.simulate_event_transmission(
            group_attendee_df.to_dict('records'), contact_data, cfg.event_base_transmission_rate,
            cfg.event_contact_fps, rng
            , cfg.event_contact_header_map
        )
        e_after = pd.DataFrame(attendees_after_tx)[pd.DataFrame(attendees_after_tx)['status'] == 'E'].shape[0]
        total_exposures_today += (e_after - e_before)
        all_attendees_after_tx.extend(attendees_after_tx)

    all_attendees_df = pd.DataFrame(all_attendees_after_tx) if all_attendees_after_tx else pd.DataFrame()
    infected_counts_by_patch = all_attendees_df[all_attendees_df['status'] == 'I'].groupby('home_patch_id').size().reindex(range(n_patches), fill_value=0).to_numpy() if not all_attendees_df.empty else np.zeros(n_patches)

    """
    # print("--- Initially Infected Individuals (Multi-Venue) ---")
    # print("district_id, n_initially_infected_individuals")
    for district_id, count in enumerate(infected_counts_by_patch):
        if count > 0:
            print(f"{district_id},{int(count)}")
    """

    # --- Run Metapopulation Model for Non-Attendees ---
    N_non_attendees = S_non_attendees + E_non_attendees + I_non_attendees + R_non_attendees
    meta_step_results = _run_metapopulation_step(data, cfg, rng, S_non_attendees, E_non_attendees, I_non_attendees, R_non_attendees, N_non_attendees)
    S_updated_non, E_updated_non, I_updated_non, R_updated_non = meta_step_results["S_new"], meta_step_results["E_new"], meta_step_results["I_new"], meta_step_results["R_new"]

    # print(f"{total_infected_recruited_today} infected attendees recruited from population across all venues.")
    # print(f"{total_exposures_today} newly exposed attendees from multi-venue events on day {int(cfg.current_time)}.")
    S_final, E_final, I_final, R_final, new_exposures_by_patch = events.reintegrate_attendees(
        attendees_after_event=all_attendees_after_tx, 
        S_after_recruitment=S_updated_non, 
        E_after_recruitment=E_updated_non, 
        I_after_recruitment=I_updated_non, 
        R_after_recruitment=R_updated_non, 
        n_patches=n_patches, 
        N_ij_initial=N_ij_initial, 
        rng=rng
    )

    event_results = {
        "S_new": S_final, "E_new": E_final, "I_new": I_final, "R_new": R_final,
        "new_exposures_today": total_exposures_today,
        "infected_recruited_today": total_infected_recruited_today,
        "attendees_today": total_attendees_today,
        "recruitment_data": recruitment_data_for_plotting,
        "infected_counts_by_patch": infected_counts_by_patch,
        "new_exposures_by_patch": new_exposures_by_patch,
        "infected_attendee_series": total_infected_attendees_series
    }
    return event_results

def _handle_event_day(data: Dict[str, Any], cfg: Any, rng: np.random.Generator, S_t: np.ndarray, E_t: np.ndarray, I_t: np.ndarray, R_t: np.ndarray, logger: InfectionLogger, run_id: int, initial_infections_data: Optional[pd.DataFrame], N_t: np.ndarray, precomputed_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Orchestrates event simulation for a given day."""
    current_day = int(cfg.current_time)
    event_results = {}

    if cfg.event_model_type == 'large_venue':
        event_results = _run_large_venue_event(data, cfg, rng, S_t, E_t, I_t, R_t, logger, run_id, N_t, precomputed_data)
    elif cfg.event_model_type == 'multi_venue':
        event_results = _run_multi_venue_event(data, cfg, rng, S_t, E_t, I_t, R_t, initial_infections_data, run_id, N_t)
    else:
        raise ValueError(f"Unknown event_model_type: '{cfg.event_model_type}'")

    # Log infection data
    logger.record_infections(logger.ini_writer, run_id, current_day, event_results['infected_counts_by_patch'])
    logger.record_infections(logger.new_writer, run_id, current_day, event_results['new_exposures_by_patch'])

    # Prepare results to be returned
    S_new, E_new, I_new, R_new = event_results["S_new"], event_results["E_new"], event_results["I_new"], event_results["R_new"]
    N_new = S_new + E_new + I_new + R_new
    new_exp, new_inf = np.sum(E_new) - np.sum(E_t), np.sum(I_new) - np.sum(I_t) # Simplified change

    return {
        "S_new": S_new, "E_new": E_new, "I_new": I_new, "R_new": R_new, "N_new": N_new,
        "new_exp": new_exp, "new_inf": new_inf,
        "event_specific_results": event_results # Pass through for main simulation loop
    }

def _run_gillespie_step(data: Dict[str, Any], cfg: Any, rng: np.random.Generator, S_t: np.ndarray, E_t: np.ndarray, I_t: np.ndarray, R_t: np.ndarray) -> Dict[str, Any]:
    """Performs one step of the stochastic simulation using the Gillespie algorithm."""
    n_patches = data['n_patches']
    alpha = 1.0 / cfg.period_incubation
    gamma = 1.0 / cfg.period_infectious
    beta = cfg.beta
    # Reshape for easier processing
    S_ij = S_t.reshape((n_patches, n_patches))
    E_ij = E_t.reshape((n_patches, n_patches))
    I_ij = I_t.reshape((n_patches, n_patches))
    
    # --- 1. Calculate all event rates ---
    
    # Progression (E -> I) rates for each subpopulation
    progression_rates = alpha * E_ij
    
    # Recovery (I -> R) rates for each subpopulation
    recovery_rates = gamma * I_ij
    
    # Infection (S -> E) rates
    is_day = (cfg.current_time % 1.0) == 0.0
    N_ij = (S_ij + E_ij + I_ij + R_t.reshape((n_patches, n_patches)))
    
    if is_day:
        # Day: infection at current location (j)
        I_in_j = I_ij.sum(axis=0) # Total infected in patch j
        N_in_j = N_ij.sum(axis=0) # Total population in patch j
        with np.errstate(divide='ignore', invalid='ignore'):
            lambda_j = beta * I_in_j / N_in_j
        lambda_j = np.nan_to_num(lambda_j)
        infection_rates = S_ij * lambda_j[np.newaxis, :]
    else:
        # Night: infection at home location (i)
        I_in_i = I_ij.sum(axis=1) # Total infected from home patch i
        N_in_i = N_ij.sum(axis=1) # Total population from home patch i
        with np.errstate(divide='ignore', invalid='ignore'):
            lambda_i = beta * I_in_i / N_in_i
        lambda_i = np.nan_to_num(lambda_i)
        infection_rates = S_ij * lambda_i[:, np.newaxis]

    # --- 2. Sum all rates ---
    R_infection = infection_rates.sum()
    R_progression = progression_rates.sum()
    R_recovery = recovery_rates.sum()
    R_total = R_infection + R_progression + R_recovery

    if R_total <= 0:
        # No more events can happen, advance time to end of simulation to stop the run
        return {"tau": cfg.n_days, "S_new": S_t, "E_new": E_t, "I_new": I_t, "R_new": R_t, "N_new": S_t+E_t+I_t+R_t, "new_exp": 0, "new_inf": 0}

    # --- 3. Calculate time to next event ---
    tau = -np.log(rng.random()) / R_total
    
    # --- 4. Choose which event happens ---
    S_new, E_new, I_new, R_new = S_t.copy(), E_t.copy(), I_t.copy(), R_t.copy()
    new_exp, new_inf = 0, 0
    
    event_choice = rng.random() * R_total
    
    if event_choice < R_infection:
        # It's an infection event
        target = event_choice
        cum_rates = np.cumsum(infection_rates.ravel())
        event_idx = np.searchsorted(cum_rates, target)
        
        S_new[event_idx] -= 1
        E_new[event_idx] += 1
        new_exp = 1
    elif event_choice < R_infection + R_progression:
        # It's a progression event
        target = event_choice - R_infection
        cum_rates = np.cumsum(progression_rates.ravel())
        event_idx = np.searchsorted(cum_rates, target)
        
        E_new[event_idx] -= 1
        I_new[event_idx] += 1
        new_inf = 1
    else:
        # It's a recovery event
        target = event_choice - R_infection - R_progression
        cum_rates = np.cumsum(recovery_rates.ravel())
        event_idx = np.searchsorted(cum_rates, target)
        
        I_new[event_idx] -= 1
        R_new[event_idx] += 1

    # Ensure no negative populations
    np.clip(S_new, 0, None, out=S_new)
    np.clip(E_new, 0, None, out=E_new)
    np.clip(I_new, 0, None, out=I_new)

    N_new = S_new + E_new + I_new + R_new
    
    return {"tau": tau, "S_new": S_new, "E_new": E_new, "I_new": I_new, "R_new": R_new, "N_new": N_new, "new_exp": new_exp, "new_inf": new_inf}

def _run_metapopulation_step(data: Dict[str, Any], cfg: Any, rng: np.random.Generator, S_t: np.ndarray, E_t: np.ndarray, I_t: np.ndarray, R_t: np.ndarray, N_t: np.ndarray) -> Dict[str, Any]:
    """Performs one step of the SEIR model, dispatching to either Gillespie or Tau-leaping."""
    E_total = np.sum(E_t)
    I_total = np.sum(I_t)
    
    # The condition `(E_total + I_total) > 0` prevents using Gillespie when there are no active cases, which would lead to R_total=0 and an infinite tau.
    use_gillespie = (E_total + I_total) > 0 and (E_total + I_total) < getattr(cfg, 'TauLeaping_threshold', 0)

    if use_gillespie:
        # Use exact stochastic simulation for low numbers of infected
        return _run_gillespie_step(data, cfg, rng, S_t, E_t, I_t, R_t)
    else:
        # Use faster tau-leaping approximation for high numbers
        time_of_day = cfg.current_time % 1.0
        tod = "day-time" if time_of_day == 0.0 else "night"
        alpha = 1.0 / cfg.period_incubation
        gamma = 1.0 / cfg.period_infectious
        beta = cfg.beta
        
        tau, S_new, E_new, I_new, R_new, N_new, new_exp, new_inf = compute_SEIR(
            beta, alpha, gamma, data['n_patches'], S_t, E_t, I_t, R_t, N_t, tod, rng
        )
        return {"tau": tau, "S_new": S_new, "E_new": E_new, "I_new": I_new, "R_new": R_new, "N_new": N_new, "new_exp": new_exp, "new_inf": new_inf}


def _seed_initial_infections(cfg, S, I, n_patches, run_id, initial_infections_data):
    """Seeds the initial infections based on configuration and data."""
    if cfg.no_event_seeding_mode == 'from_file' and initial_infections_data is not None and isinstance(initial_infections_data, pd.DataFrame):
        day0_infections_df = initial_infections_data[
            (initial_infections_data['run_id'] == run_id) &
            (initial_infections_data['event_day'] == 0)
        ]
        # print(f"Seeding {len(day0_infections_df)} infected individuals based on large-venue distribution file.")
        S_reshaped = S.reshape((n_patches, n_patches))
        I_reshaped = I.reshape((n_patches, n_patches))
        for _, row in day0_infections_df.iterrows():
            home_patch = int(row['home_patch_id'])
            comm_patch = int(row['commuting_district_id'])
            if S_reshaped[home_patch, comm_patch] > 0:
                S_reshaped[home_patch, comm_patch] -= 1
                I_reshaped[home_patch, comm_patch] += 1
        return S_reshaped.ravel(), I_reshaped.ravel()

    elif cfg.no_event_seeding_mode == 'single_patch' and cfg.I_ss > 0:
        # Determine seed patch: use event-specific patch if defined, else baseline patch
        if cfg.event_model_type == 'large_venue' and getattr(cfg, 'large_venue_ini_infected_attendee_seed', None) == 'single_patch':
            seed_patch_idx = getattr(cfg, 'large_venue_ini_infected_attendee_patch', cfg.initial_infection_patch_id)
        else:
            seed_patch_idx = cfg.initial_infection_patch_id

        seed_subpopulation_idx = seed_patch_idx * n_patches + seed_patch_idx
        
        if getattr(cfg, 'seeding_method', 'recruit') == 'recruit':
            num_to_infect = min(cfg.I_ss, S[seed_subpopulation_idx])
            S[seed_subpopulation_idx] -= num_to_infect
            I[seed_subpopulation_idx] += num_to_infect
        else: # import
            I[seed_subpopulation_idx] += cfg.I_ss
        
    return S, I


def _check_and_record_die_out(cfg, current_time, E_total, I_total, R_total, run_id):
    """Checks if the disease has died out and records the event if so."""
    if (E_total + I_total) < 1 and current_time > 1:
        return True
    return False

def run_simulation(data: Dict[str, Any], cfg: Any, rng: np.random.Generator, run_id: int, logger: InfectionLogger, initial_infections_data: Optional[pd.DataFrame] = None, precomputed_event_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Runs a full SEIR simulation, including optional hybrid event logic.
    """
    n_patches = data['n_patches']
    N_ij_initial = data['N_ij_initial'].ravel().astype(np.int32)

    # beta = cfg.event_base_transmission_rate * cfg.k_assumed
    beta = cfg.beta

    # Initialize populations
    S, E, I, R = N_ij_initial.copy(), np.zeros_like(N_ij_initial), np.zeros_like(N_ij_initial), np.zeros_like(N_ij_initial)
    N = S + E + I + R

    S, I = _seed_initial_infections(cfg, S, I, n_patches, run_id, initial_infections_data)
    
    # Recalculate N because seeding might have changed it (if 'import' method used)
    N = S + E + I + R

    # History tracking
    # Optimization: Decouple current state from history storage.
    # We maintain S_curr, E_curr... for the simulation logic (full N^2 arrays).
    # We only append to S_hist, E_hist... if write_full_results is True, or we append aggregated versions.
    S_curr, E_curr, I_curr, R_curr, N_curr = S.copy(), E.copy(), I.copy(), R.copy(), N.copy()
    
    t_hist = [0.0]
    gamma = 1.0 / cfg.period_infectious
    rt_initial = compute_rt_from_state(S_curr.reshape(n_patches, n_patches), N_curr.reshape(n_patches, n_patches), beta, gamma, n_patches)
    rt_hist = [rt_initial]
    last_rt_time = 0.0

    S_all_hist, E_all_hist, I_all_hist, R_all_hist = [np.sum(S)], [np.sum(E)], [np.sum(I)], [np.sum(R)]
    new_exposed_hist, new_infected_hist = [0], [0]
    
    # Determine what to store based on config
    store_detailed = cfg.write_full_results
    
    if store_detailed:
        S_hist = [S_curr.copy()]
        E_hist = [E_curr.copy()]
        I_hist = [I_curr.copy()]
        R_hist = [R_curr.copy()]
        N_hist = [N_curr.copy()]
    else:
        # Store aggregated E and I (sum over axis 0 = total in patch j) for patch-level plots
        # S and R are typically only plotted as global sums or not at patch level in standard plots
        E_agg = E_curr.reshape(n_patches, n_patches).sum(axis=0)
        I_agg = I_curr.reshape(n_patches, n_patches).sum(axis=0)
        E_hist = [E_agg]
        I_hist = [I_agg]
        S_hist = [] # Not stored
        R_hist = [] # Not stored
        N_hist = [] # Not stored

    event_exposures_by_day = {}
    event_attendees_by_day = {}
    infected_attendees_recruited_by_day = {}
    event_dispersal_by_day = {}
    recruitment_data_for_plotting = np.zeros(n_patches)
    infected_attendee_counts = np.zeros(n_patches)
    
    while t_hist[-1] < cfg.n_days:
        # Check for die-out
        if _check_and_record_die_out(cfg, t_hist[-1], E_all_hist[-1], I_all_hist[-1], R_all_hist[-1], run_id):
            break

        cfg.current_time = t_hist[-1]
        cfg.current_day = int(cfg.current_time)
        time_of_day = cfg.current_time % 1.0
        
        # --- Handle Transient Imported Infections in Baseline ---
        # If we are in a baseline scenario with 'imported' seeding, we remove the seeds 
        # after Day 0 (at t=1.0) to simulate them leaving the region, matching the 
        # behavior of the event scenarios where attendees leave.
        if (cfg.current_time == 1.0 and 
            cfg.event_model_type.startswith('no_event') and 
            getattr(cfg, 'seeding_method', 'recruit') == 'import' and 
            getattr(cfg, 'no_event_seeding_mode', 'single_patch') == 'single_patch' and
            cfg.I_ss > 0):
            
            seed_patch_idx = cfg.initial_infection_patch_id
            # Imported individuals are added to the resident population (i, i)
            seed_subpop_idx = seed_patch_idx * n_patches + seed_patch_idx
            
            amount_to_remove = cfg.I_ss
            N_curr[seed_subpop_idx] = max(0, N_curr[seed_subpop_idx] - amount_to_remove)
            
            # Remove primarily from I, remainder from R (if they recovered during Day 0)
            if I_curr[seed_subpop_idx] >= amount_to_remove:
                I_curr[seed_subpop_idx] -= amount_to_remove
            else:
                remainder = amount_to_remove - I_curr[seed_subpop_idx]
                I_curr[seed_subpop_idx] = 0
                R_curr[seed_subpop_idx] = max(0, R_curr[seed_subpop_idx] - remainder)

        # Use current state variables instead of history lookups
        S_t, E_t, I_t, R_t, N_t = S_curr, E_curr, I_curr, R_curr, N_curr

        # FOR TEST: Print state for the baseline no_event scenario on day 1
        """
        if cfg.event_model_type == 'no_event' and cfg.current_day == 1 and time_of_day == 0.0:
            print(f"  State for Day 1 (start):")
            print(f"    S: {int(np.sum(S_t)):,}")
            print(f"    E: {int(np.sum(E_t)):,}")
            print(f"    I: {int(np.sum(I_t)):,}")
            print(f"    R: {int(np.sum(R_t)):,}")
        """
        
        if cfg.current_day in cfg.event_days and time_of_day == 0.0 and not cfg.event_model_type.startswith('no_event'):
            # --- HYBRID SIMULATION FOR EVENT DAY ---
            day_results = _handle_event_day(data, cfg, rng, S_t, E_t, I_t, R_t, logger, run_id, initial_infections_data, N_t, precomputed_event_data)
            S_new, E_new, I_new, R_new = day_results["S_new"], day_results["E_new"], day_results["I_new"], day_results["R_new"]
            N_new, new_exp, new_inf = day_results["N_new"], day_results["new_exp"], day_results["new_inf"]
            event_specific_results = day_results["event_specific_results"]

            # Store event-specific results
            event_exposures_by_day[cfg.current_day] = event_specific_results['new_exposures_today']
            infected_attendees_recruited_by_day[cfg.current_day] = event_specific_results['infected_recruited_today']
            event_attendees_by_day[cfg.current_day] = event_specific_results['attendees_today']
            
            if 'new_exposures_by_patch' in event_specific_results:
                event_dispersal_by_day[cfg.current_day] = np.count_nonzero(event_specific_results['new_exposures_by_patch'])

            if cfg.current_day == cfg.event_days[0]:
                recruitment_data_for_plotting = event_specific_results['recruitment_data']
                if cfg.event_model_type == 'multi_venue':
                    infected_attendee_counts = event_specific_results['infected_attendee_series'].reindex(range(n_patches), fill_value=0).values
                else:
                    infected_attendee_counts = event_specific_results['infected_counts_by_patch']

            tau = 0.5

        else:
            # --- REGULAR METAPOPULATION STEP ---
            step_results = _run_metapopulation_step(data, cfg, rng, S_t, E_t, I_t, R_t, N_t)
            tau = step_results["tau"]
            S_new, E_new, I_new, R_new = step_results["S_new"], step_results["E_new"], step_results["I_new"], step_results["R_new"]
            N_new, new_exp, new_inf = step_results["N_new"], step_results["new_exp"], step_results["new_inf"]

        # Final safety clip to ensure no negative values propagate to history
        S_new = np.maximum(S_new, 0)
        E_new = np.maximum(E_new, 0)
        I_new = np.maximum(I_new, 0)
        R_new = np.maximum(R_new, 0)

        # Update current state
        S_curr, E_curr, I_curr, R_curr, N_curr = S_new, E_new, I_new, R_new, N_new

        # Calculate Rt for the new state
        # Optimization: Calculate Rt only if time advanced by >= 0.5 days or it's the start
        if (t_hist[-1] - last_rt_time >= 0.5):
            rt = compute_rt_from_state(S_curr.reshape(n_patches, n_patches), N_curr.reshape(n_patches, n_patches), beta, gamma, n_patches)
            last_rt_time = t_hist[-1]
        else:
            rt = rt_hist[-1]

        # Append to histories (Aggregated or Detailed)
        S_all_hist.append(np.sum(S_new)); E_all_hist.append(np.sum(E_new)); I_all_hist.append(np.sum(I_new)); R_all_hist.append(np.sum(R_new))
        new_exposed_hist.append(new_exp); new_infected_hist.append(new_inf)
        t_hist.append(t_hist[-1] + tau)
        rt_hist.append(rt)

        if store_detailed:
            S_hist.append(S_new.copy()); E_hist.append(E_new.copy()); I_hist.append(I_new.copy()); R_hist.append(R_new.copy()); N_hist.append(N_new.copy())
        else:
            # Aggregate E and I for patch-level plots (sum over axis 0 = total in patch j)
            E_agg = E_new.reshape(n_patches, n_patches).sum(axis=0)
            I_agg = I_new.reshape(n_patches, n_patches).sum(axis=0)
            E_hist.append(E_agg)
            I_hist.append(I_agg)

    # Construct results dictionary
    results = {
        "t": np.array(t_hist, dtype=np.float32),
        "rt": np.array(rt_hist, dtype=np.float32),
        "S_all": np.array(S_all_hist, dtype=np.float32), "E_all": np.array(E_all_hist, dtype=np.float32), 
        "I_all": np.array(I_all_hist, dtype=np.float32), "R_all": np.array(R_all_hist, dtype=np.float32),
        "new_exposed": np.array(new_exposed_hist, dtype=np.int32), "new_infected": np.array(new_infected_hist, dtype=np.int32),
        "event_exposures_by_day": event_exposures_by_day,
        "event_attendees_by_day": event_attendees_by_day,
        "infected_attendees_recruited_by_day": infected_attendees_recruited_by_day,
        "event_dispersal_by_day": event_dispersal_by_day,
        "recruitment_data": recruitment_data_for_plotting,
        "infected_attendee_counts": infected_attendee_counts
    }
    
    if store_detailed:
        results.update({"S_ij": [x.astype(np.int32) for x in S_hist], "E_ij": [x.astype(np.int32) for x in E_hist], "I_ij": [x.astype(np.int32) for x in I_hist], "R_ij": [x.astype(np.int32) for x in R_hist], "N_ij": [x.astype(np.int32) for x in N_hist]})
    else:
        results.update({"E_j": [x.astype(np.int32) for x in E_hist], "I_j": [x.astype(np.int32) for x in I_hist]})
        
    return results

def compute_rt_from_state(S_ij_t: np.ndarray, N_ij: np.ndarray, beta: float, gamma: float, n_patches: int) -> float:
    """
    Computes the effective reproduction number Rt at a specific time t using the Next Generation Matrix method.

    Args:
        S_ij_t: Current susceptible subpopulations array (n_patches, n_patches).
        N_ij: Current total subpopulations array (n_patches, n_patches).
        beta: Community transmission rate.
        gamma: Recovery rate.
        n_patches: Number of patches.

    Returns:
        The spectral radius of the time-dependent NGM, representing Rt.
    """
    # 1. Calculate susceptible fractions for day and night locations
    # Day: Susceptible fraction in each location j
    S_in_j = S_ij_t.sum(axis=0)
    N_in_j = N_ij.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        sus_frac_day = np.nan_to_num(S_in_j / N_in_j)
    D_day = np.diag(sus_frac_day)

    # Night: Susceptible fraction in each home patch i
    S_at_home_i = np.diag(S_ij_t)
    N_at_home_i = np.diag(N_ij)
    with np.errstate(divide='ignore', invalid='ignore'):
        sus_frac_night = np.nan_to_num(S_at_home_i / N_at_home_i)
    D_night = np.diag(sus_frac_night)

    # 2. Get structural mobility matrices P (origin i to dest j) and M (dest j from origin u)
    N_i = N_ij.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        P_ij = np.nan_to_num(N_ij / N_i[:, np.newaxis])
        M_uj = np.nan_to_num(N_ij / N_in_j[np.newaxis, :])
    
    # 3. Construct the time-dependent Next Generation Matrix K(t)
    T_day, T_night = 0.5, 0.5
    K_day_t = (T_day * beta) * (P_ij @ D_day @ M_uj.T)
    K_night_t = (T_night * beta) * D_night
    K_t = (1.0 / gamma) * (K_day_t + K_night_t)

    # 4. Rt is the spectral radius of K(t)
    eigenvalues = np.linalg.eigvals(K_t)
    return np.max(np.abs(eigenvalues))

def analyze_metapopulation_percolation_risk(data: Dict[str, Any], cfg: Any) -> tuple[float, int]:
    """
    Analyzes the percolation risk of the metapopulation network.
    Constructs a Next Generation Matrix (NGM) proxy and calculates the
    fraction of patches reachable from the seed patch.
    """
    n_patches = data['n_patches']
    N_ij = data['N_ij_initial'] # Expecting (N, N) from data_loader
    
    # Ensure N_ij is 2D
    if N_ij.ndim == 1:
        N_ij = N_ij.reshape(n_patches, n_patches)
    
    # Population per patch (residents)
    N_i = N_ij.sum(axis=1)
    
    # Day population in each patch j (commuters + residents staying)
    N_j_day = N_ij.sum(axis=0)
    
    # Probability of resident i being in j during day: P(i->j)
    with np.errstate(divide='ignore', invalid='ignore'):
        P_ij = N_ij / N_i[:, np.newaxis]
    P_ij = np.nan_to_num(P_ij)
    
    beta = cfg.beta
    gamma = 1.0 / cfg.period_infectious
    # Simulation uses dt=0.5 for day and night steps.
    T_day = 0.5
    T_night = 0.5
    
    # --- Construct K_uv (Next Generation Matrix Proxy) ---
    # K_uv = Expected secondary infections in u caused by one infected in v
    
    # 1. Day Contribution: v goes to j, infects u present in j
    # M_uj = Fraction of people in j who are from u
    with np.errstate(divide='ignore', invalid='ignore'):
        M_uj = N_ij / N_j_day[np.newaxis, :]
    M_uj = np.nan_to_num(M_uj)
    
    # K_day[v, u] = T_day * beta * (P_ij @ M_uj.T)[v, u]
    K_day = (T_day * beta) * (P_ij @ M_uj.T)
    
    # 2. Night Contribution: v stays at home v, infects u=v
    K_night = np.zeros((n_patches, n_patches))
    np.fill_diagonal(K_night, T_night * beta)
    
    # Total K matrix (infections over full infectious period)
    K_total = (K_day + K_night) * (1.0 / gamma)
    
    # --- Percolation Analysis ---
    # Bond probability p_vu = 1 - exp(-K_total[v, u]) (Edge v -> u)
    p_vu = 1 - np.exp(-K_total)
    
    # Sample one realization
    rng = np.random.default_rng(cfg.rnd_seed_0)
    rand_vals = rng.random((n_patches, n_patches))
    
    # Adjacency matrix: adj[v, u] = 1 if v infects u
    adj = (p_vu > rand_vals).astype(int)
    
    # Calculate Reachability from Seed Patch
    seed_patch = cfg.initial_infection_patch_id
    visited = set()
    stack = [seed_patch]
    visited.add(seed_patch)
    while stack:
        curr = stack.pop()
        neighbors = np.where(adj[curr, :] == 1)[0]
        for n in neighbors:
            if n not in visited:
                visited.add(n)
                stack.append(n)
                
    reachable_count = len(visited)
    reachable_fraction = reachable_count / n_patches
    
    return reachable_fraction, reachable_count

def compute_structural_ngm(data: Dict[str, Any], current_N_ij: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Computes the structural component of the Next Generation Matrix (S_mat).
    K_total = R0 * S_mat.
    
    Args:
        data: Dictionary containing simulation data (must have 'n_patches' and 'N_ij_initial').
        current_N_ij: Optional (n_patches, n_patches) array representing the current
                      mobility/population distribution. If None, uses the initial configuration.

    S_mat[v, u] represents the relative expected number of secondary infections 
    in patch v caused by an infected individual from patch u, normalized by R0.
    """
    n_patches = data['n_patches']
    N_ij = current_N_ij if current_N_ij is not None else data['N_ij_initial']

    if N_ij.ndim == 1:
        N_ij = N_ij.reshape(n_patches, n_patches)
    
    N_i = N_ij.sum(axis=1)
    N_j_day = N_ij.sum(axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        P_ij = N_ij / N_i[:, np.newaxis]
        M_uj = N_ij / N_j_day[np.newaxis, :]
    P_ij = np.nan_to_num(P_ij)
    M_uj = np.nan_to_num(M_uj)
    
    # Simulation uses dt=0.5 for day and night steps
    T_day = 0.5
    T_night = 0.5
    
    S_day = T_day * (P_ij @ M_uj.T)
    
    S_night = np.zeros((n_patches, n_patches))
    np.fill_diagonal(S_night, T_night)
    
    return S_day + S_night

def precompute_event_outcomes(data: Dict[str, Any], cfg: Any, rng: np.random.Generator, run_id: int) -> Dict[str, Any]:
    """
    Pre-calculates the micro-scale event outcomes (recruitment + transmission) for t=0.
    Returns a dictionary containing the attendees list and related stats.
    """
    # 1. Setup Initial State (Identical to run_simulation start)
    n_patches = data['n_patches']
    N_ij_initial = data['N_ij_initial'].ravel().astype(np.int32)
    S, E, I, R = N_ij_initial.copy(), np.zeros_like(N_ij_initial), np.zeros_like(N_ij_initial), np.zeros_like(N_ij_initial)
    
    # Seed infections (using None for initial_infections_data as this is for large_venue precompute)
    S, I = _seed_initial_infections(cfg, S, I, n_patches, run_id, None)
    
    # 2. Run Micro-Scale Logic
    micro_results = _recruit_and_transmit_large_venue(data, cfg, rng, S, E, I, R, run_id)
    
    return micro_results