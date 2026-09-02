"""
Event simulation logic for the Hybrid Epidemic Simulation Model.
"""
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import os
from pathlib import Path

from .core import compute_SEIR
from .models.gillespie import _infer_metapop_infection_source
from .network import analyze_event_percolation_risk  # noqa: F401
from .data.loader import preprocess_contact_file  # noqa: F401
from .recruitment.strategies import (
    CommuterStrategy,
    GravityStrategy,
    PopulationStrategy,
    SinglePatchStrategy,
)
from .recruitment.recruitment import (
    recruit_attendees,
    recruit_attendees_for_multi_venue_event,
    reintegrate_attendees,
    _remove_attendees_from_population,
    _resolve_column,
)


def simulate_event_transmission(
    attendees,
    contact_data,
    base_transmission_rate,
    fps,
    rng,
    column_map=None,
    return_provenance: bool = False,
    provenance_mode: str = "max_duration",
):
    """
    Simulates transmission within a large-gathering event based on a contact network.
    Args:
        contact_data: File path (str/Path) OR pre-loaded pandas DataFrame.
    """
    # print(f"  -> running individual-based simulation using '{contact_data_file}'...")
    if isinstance(contact_data, (str, Path)):
        try:
            contacts_df = pd.read_csv(contact_data)
        except FileNotFoundError:
            print(f"Error: Event contact data file not found at '{contact_data}'. Skipping event transmission.")
            return attendees
    else:
        contacts_df = contact_data

    cfg_col_i = column_map.get("agent_i") if column_map else None
    col_i = _resolve_column(contacts_df, cfg_col_i, ["agent_i", "ped1_id", "id_1", "u", "p1"])

    cfg_col_j = column_map.get("agent_j") if column_map else None
    col_j = _resolve_column(contacts_df, cfg_col_j, ["agent_j", "ped2_id", "id_2", "v", "p2"])

    cfg_col_dur = column_map.get("duration_frames") or column_map.get("tc_s") or column_map.get("frames") if column_map else None
    col_dur = _resolve_column(contacts_df, cfg_col_dur, ["total_duration_frames", "tc_s", "duration", "frames"])

    if not col_i or not col_j or not col_dur:
        print(f"Error: Could not identify required columns in contact data.")
        print(f"  Looking for agent_i (found {col_i}), agent_j (found {col_j}), duration (found {col_dur})")
        print(f"  Available columns: {list(contacts_df.columns)}")
        if column_map:
            print(f"  Used column map: {column_map}")
        return attendees

    # Optimization: Only cast if not already integer (avoids copy overhead on cached DF)
    if not pd.api.types.is_integer_dtype(contacts_df[col_i]):
        contacts_df[col_i] = contacts_df[col_i].astype(int)
    if not pd.api.types.is_integer_dtype(contacts_df[col_j]):
        contacts_df[col_j] = contacts_df[col_j].astype(int)

    attendee_df = pd.DataFrame(attendees)
    initial_infected_ids = set(attendee_df[attendee_df['status'] == 'I']['agent_id'])

    infected_contacts = contacts_df[
        contacts_df[col_i].isin(initial_infected_ids) |
        contacts_df[col_j].isin(initial_infected_ids)
    ]

    exposure_pairs = []
    for _, row in infected_contacts.iterrows():
        p1, p2 = row[col_i], row[col_j]
        duration = row[col_dur] / fps

        if p1 in initial_infected_ids and p2 not in initial_infected_ids:
            exposure_pairs.append({'susceptible_agent': p2, 'infected_agent': p1, 'duration': duration})
        if p2 in initial_infected_ids and p1 not in initial_infected_ids:
            exposure_pairs.append({'susceptible_agent': p1, 'infected_agent': p2, 'duration': duration})

    if not exposure_pairs:
        # print(" No contacts between infected and susceptible individuals. No new exposures.")
        # print("")
        return (attendees, []) if return_provenance else attendees

    exposure_df = pd.DataFrame(exposure_pairs)

    # Identify one likely infectious source per susceptible attendee.
    # We keep only the strongest exposure pair per target to preserve a clean provenance edge.
    source_choice_df = exposure_df.sort_values("duration", ascending=False).drop_duplicates("susceptible_agent")

    # Calculate total duration of contact with infected individuals for each susceptible agent
    # q_i = product(q_ij) = product(exp(-beta * t_ij)) = exp(-beta * sum(t_ij))
    # p_i = 1 - q_i = 1 - exp(-beta * sum(t_ij))
    total_duration_by_agent = exposure_df.groupby('susceptible_agent')['duration'].sum()
    total_contact_duration_hr = total_duration_by_agent/3600 # total contact duration with infectious individuals in hours

    # based on aggregated contact duration
    prob_infection = 1 - np.exp(-base_transmission_rate * total_contact_duration_hr)

    infection_roll = rng.random(size=len(prob_infection))
    newly_infected_agents = prob_infection[infection_roll < prob_infection].index

    # Only change status of Susceptible individuals
    susceptible_mask = attendee_df['agent_id'].isin(newly_infected_agents) & (attendee_df['status'] == 'S')
    attendee_df.loc[susceptible_mask, 'status'] = 'E'

    num_newly_exposed = susceptible_mask.sum()
    # print(f"  -> large gathering event resulted in {num_newly_exposed} new exposures.\n")
    # print(f"large gathering event resulted in {num_newly_exposed} new exposures.\n")
    if not return_provenance:
        return attendee_df.to_dict('records')

    provenance_records: List[Dict[str, Any]] = []
    newly_exposed_ids = attendee_df.loc[susceptible_mask, 'agent_id'].tolist()
    newly_exposed_set = set(newly_exposed_ids)

    for _, source_row in source_choice_df.iterrows():
        target_agent_id = int(source_row['susceptible_agent'])
        if target_agent_id not in newly_exposed_set:
            continue

        target_row = attendee_df.loc[attendee_df['agent_id'] == target_agent_id].iloc[0]
        source_agent_id = int(source_row['infected_agent'])
        source_row_attendee = attendee_df.loc[attendee_df['agent_id'] == source_agent_id].iloc[0]

        provenance_records.append({
            'source_agent_id': source_agent_id,
            'target_agent_id': target_agent_id,
            'source_home_patch_id': int(source_row_attendee['home_patch_id']),
            'target_home_patch_id': int(target_row['home_patch_id']),
            'source_current_patch_id': int(source_row_attendee['current_patch_id']),
            'target_current_patch_id': int(target_row['current_patch_id']),
            'infection_context': 'event',
            'event_day': None,
            'source_patch_id': int(source_row_attendee['home_patch_id']),
            'target_patch_id': int(target_row['home_patch_id']),
        })

    return attendee_df.to_dict('records'), provenance_records


class InfectionLogger:
    """Manages logging of initial and new infections for event-based scenarios."""
    def __init__(self, event_model_type: str, n_patches: int, cfg: Any):
        self.event_model_type = event_model_type
        self.n_patches = n_patches
        self.cfg = cfg
        self.ini_writer = None
        self.new_writer = None
        self.attendee_log_writer = None
        self.ini_filename = None
        self.new_filename = None
        self.attendee_log_filename = None
        self._prepare_filenames()

    def _generate_base_filename(self) -> str:
        """Creates the common part of the log filename based on current config."""
        r0_part = f"R{int(self.cfg.R_0*100)}_"
        if self.event_model_type.startswith('no_event'):
            return f"{r0_part}Iss{self.cfg.I_ss}.txt"
        return f"{r0_part}betaEvent{int(self.cfg.event_base_transmission_rate*100)}_Iss{self.cfg.I_ss}.txt"

    def _prepare_filenames(self):
        """Sets up the full filenames for the log files."""
        if self.event_model_type not in ['large_venue', 'multi_venue']:
            return

        results_dir = getattr(self.cfg, 'results_dir', Path("results"))
        base_filename = self._generate_base_filename()

        self.attendee_log_filename = str(results_dir / f"ini_infectious_attendees_{base_filename}")
        self.ini_filename = str(results_dir / f"n_ini_infections_{base_filename}")
        self.new_filename = str(results_dir / f"n_new_infections_{base_filename}")

    def _setup_writer(self, filename: str) -> Any:
        """Opens a file for writing and adds the header."""
        file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
        header = f"scenario_name,run_id,event_day,n_all," + ",".join([f"n_{i}" for i in range(self.n_patches)]) + "\n"
        f = open(filename, 'a')
        if not file_exists:
            f.write(header)
        return f

    def _setup_attendee_writer(self, filename: str) -> Any:
        """Opens a file for writing attendee details and adds the header."""
        file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
        header = "scenario_name,run_id,event_day,attendee_id,district_id,commuting_district_id,event_model,recruitment_model,initially_infected,event_patch_id,beta_event,R0,initially_infected_seed,initially_infected_patch\n"
        f = open(filename, 'a')
        if not file_exists:
            f.write(header)
        return f

    def record_infections(self, writer: Any, run_id: int, day: int, data_array: np.ndarray):
        """Writes a row of infection data to the specified file."""
        if writer is None:
            return
        n_all = np.sum(data_array)
        data_str = ",".join(map(str, data_array.astype(int)))
        writer.write(f"{self.cfg.current_event_scenario_name},{run_id},{day},{int(n_all)},{data_str}\n")
        writer.flush()

    def record_attendee_details(self, run_id: int, day: int, attendee_df: pd.DataFrame):
        """Writes detailed records for a dataframe of attendees."""
        if self.attendee_log_writer is None:
            return

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
            return None
        n_cols = [col for col in df.columns if col.startswith('n_') and col != 'n_all']

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


def _recruit_and_transmit_large_venue(data: Dict[str, Any], cfg: Any, rng: np.random.Generator,
                                       S_t: np.ndarray, E_t: np.ndarray, I_t: np.ndarray, R_t: np.ndarray,
                                       run_id: int) -> Dict[str, Any]:
    """
    Helper: Performs the micro-scale event logic (Recruitment + Transmission).
    Returns the list of attendees (post-transmission) and stats.
    """
    n_patches = data['n_patches']
    N_ij_initial = data['N_ij_initial']

    recruitment_strategies = {
        'commuter': CommuterStrategy(),
        'gravity': GravityStrategy(),
        'population': PopulationStrategy(),
        'single_patch': SinglePatchStrategy()
    }
    strategy = recruitment_strategies.get(getattr(cfg, 'attendee_recruitment_model', 'gravity'))
    if not strategy:
        raise ValueError(f"Unknown recruitment model: '{cfg.attendee_recruitment_model}'")

    force_infected_count = 0
    imported_infected_count = 0
    force_infected_patch = None

    if getattr(cfg, 'current_day', 0) == 0 and getattr(cfg, 'large_venue_initially_infected', 0) > 0:
        if getattr(cfg, 'large_venue_ini_infected_attendee_seed', None) == 'single_patch':
            if getattr(cfg, 'seeding_method', 'recruit') == 'recruit':
                force_infected_count = getattr(cfg, 'large_venue_initially_infected', 0)
                force_infected_patch = getattr(cfg, 'large_venue_ini_infected_attendee_patch', 0)
            else:
                # For 'import' seeding method: recruit forced infected from the seed district
                # (initial_infection_patch_id) instead of importing from outside.
                # This ensures epidemic dynamics differ across seed locations.
                force_infected_count = getattr(cfg, 'large_venue_initially_infected', 0)
                force_infected_patch = getattr(cfg, 'initial_infection_patch_id',
                                                getattr(cfg, 'large_venue_ini_infected_attendee_patch', 0))

    strategy_kwargs = {
        'event_patch_id': getattr(cfg, 'large_venue_patch_id', 0),
        'flow_ij_df': data['flow_ij_df'],
        'distance_matrix': data['distance_matrix_km'],
        'single_patch_source_id': getattr(cfg, 'single_patch_recruitment_id', getattr(cfg, 'large_venue_patch_id', 0)),
        'gravity_model_distance_decay_exponent': getattr(cfg, 'gravity_model_distance_decay_exponent', 1.5),
        'gravity_model_min_dist_km': getattr(cfg, 'gravity_model_min_dist_km', 0.5),
        'gravity_model_self_dist_factor': getattr(cfg, 'gravity_model_self_dist_factor', 0.5)
    }
    attendees, S_non_attendees, E_non_attendees, I_non_attendees, R_non_attendees, rec_data = recruit_attendees(
        strategy, getattr(cfg, 'large_venue_total_attendees', 12000), S_t, E_t, I_t, R_t, n_patches, N_ij_initial, rng, data['population_df'],
        force_infected_count=force_infected_count,
        force_infected_patch=force_infected_patch,
        imported_infected_count=imported_infected_count,
        **strategy_kwargs
    )
    attendee_df = pd.DataFrame(attendees)
    e_before = attendee_df[attendee_df['status'] == 'E'].shape[0]
    i_before = attendee_df[attendee_df['status'] == 'I'].shape[0]

    infected_counts_by_patch = attendee_df[attendee_df['status'] == 'I'].groupby('home_patch_id').size().reindex(range(n_patches), fill_value=0).to_numpy()

    contact_data = data.get('contact_cache', {}).get('large_venue', getattr(cfg, 'large_venue_contact_file', ''))

    transmit_result = simulate_event_transmission(
        attendee_df.to_dict('records'), contact_data, getattr(cfg, 'event_base_transmission_rate', 0.1),
        getattr(cfg, 'event_contact_fps', 5), rng,
        getattr(cfg, 'event_contact_header_map', None),
        return_provenance=getattr(cfg, 'track_invasion_provenance', False)
    )
    if getattr(cfg, 'track_invasion_provenance', False):
        attendees_after_tx, event_provenance_records = transmit_result
    else:
        attendees_after_tx = transmit_result
        event_provenance_records = []

    df_after = pd.DataFrame(attendees_after_tx)
    e_after = df_after[df_after['status'] == 'E'].shape[0]

    return {
        "attendees_after_tx": attendees_after_tx,
        "rec_data": rec_data,
        "infected_counts_by_patch": infected_counts_by_patch,
        "i_before": i_before,
        "e_before": e_before,
        "e_after": e_after,
        "attendee_df_initial": attendee_df,
        "provenance_records": event_provenance_records
    }


def _run_large_venue_event(data: Dict[str, Any], cfg: Any, rng: np.random.Generator,
                            S_t: np.ndarray, E_t: np.ndarray, I_t: np.ndarray, R_t: np.ndarray,
                            logger: InfectionLogger, run_id: int, N_t: np.ndarray,
                            precomputed_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Handles the logic for a 'large_venue' event day."""
    n_patches = data['n_patches']
    N_ij_initial = data['N_ij_initial']

    if precomputed_data:
        micro_results = precomputed_data
        attendees_after_tx = micro_results["attendees_after_tx"]
        attendee_df = pd.DataFrame(attendees_after_tx)
        attendee_df_initial = micro_results["attendee_df_initial"]
        S_non_attendees, E_non_attendees, I_non_attendees, R_non_attendees = _remove_attendees_from_population(
            attendee_df_initial,
            S_t.reshape(n_patches, n_patches),
            E_t.reshape(n_patches, n_patches),
            I_t.reshape(n_patches, n_patches),
            R_t.reshape(n_patches, n_patches),
            N_ij_initial.reshape(n_patches, n_patches),
            n_patches, rng
        )
    else:
        micro_results = _recruit_and_transmit_large_venue(data, cfg, rng, S_t, E_t, I_t, R_t, run_id)
        S_non_attendees, E_non_attendees, I_non_attendees, R_non_attendees = _remove_attendees_from_population(
            micro_results["attendee_df_initial"],
            S_t.reshape(n_patches, n_patches),
            E_t.reshape(n_patches, n_patches),
            I_t.reshape(n_patches, n_patches),
            R_t.reshape(n_patches, n_patches),
            N_ij_initial.reshape(n_patches, n_patches),
            n_patches, rng
        )

    attendees_after_tx = micro_results["attendees_after_tx"]
    rec_data = micro_results["rec_data"]
    infected_counts_by_patch = micro_results["infected_counts_by_patch"]
    i_before = micro_results["i_before"]
    e_before = micro_results["e_before"]
    e_after = micro_results["e_after"]

    new_exposures_today = e_after - e_before

    if getattr(cfg, 'current_day', 0) == 0 and getattr(cfg, 'large_venue_initially_infected', 0) > 0:
        attendee_df_initial = micro_results["attendee_df_initial"]
        infected_indices = attendee_df_initial[attendee_df_initial['status'] == 'I'].index
        newly_infected_df = attendee_df_initial.loc[infected_indices]
        logger.record_attendee_details(run_id, int(getattr(cfg, 'current_time', 0)), newly_infected_df)

    N_non_attendees = S_non_attendees + E_non_attendees + I_non_attendees + R_non_attendees
    meta_step_results = _run_metapopulation_step(
        data, cfg, rng, S_non_attendees, E_non_attendees, I_non_attendees, R_non_attendees, N_non_attendees,
        track_provenance=getattr(cfg, 'track_invasion_provenance', False)
    )

    S_updated_non, E_updated_non, I_updated_non, R_updated_non = (
        meta_step_results["S_new"], meta_step_results["E_new"],
        meta_step_results["I_new"], meta_step_results["R_new"]
    )

    S_final, E_final, I_final, R_final, new_exposures_by_patch = reintegrate_attendees(
        attendees_after_event=attendees_after_tx,
        S_after_recruitment=S_updated_non,
        E_after_recruitment=E_updated_non,
        I_after_recruitment=I_updated_non,
        R_after_recruitment=R_updated_non,
        n_patches=n_patches,
        N_ij_initial=N_ij_initial,
        rng=rng
    )

    districts_affected = np.count_nonzero(new_exposures_by_patch)
    print(f"  -> [Event Impact] New Exposures: {new_exposures_today} | Spread to {districts_affected} districts")

    event_results = {
        "S_new": S_final, "E_new": E_final, "I_new": I_final, "R_new": R_final,
        "new_exposures_today": new_exposures_today,
        "infected_recruited_today": i_before,
        "attendees_today": np.sum(rec_data),
        "recruitment_data": rec_data,
        "infected_counts_by_patch": infected_counts_by_patch,
        "new_exposures_by_patch": new_exposures_by_patch,
        "metapop_provenance_records": meta_step_results.get('provenance_records', []),
        "provenance_records": micro_results.get('provenance_records', []) + meta_step_results.get('provenance_records', [])
    }
    return event_results


def _run_multi_venue_event(data: Dict[str, Any], cfg: Any, rng: np.random.Generator,
                            S_t: np.ndarray, E_t: np.ndarray, I_t: np.ndarray, R_t: np.ndarray,
                            initial_attendees_df: Optional[pd.DataFrame], run_id: int,
                            N_t: np.ndarray) -> Dict[str, Any]:
    """Handles the logic for a 'multi_venue' event day."""
    n_patches = data['n_patches']
    N_ij_initial = data['N_ij_initial']
    population_df = data['population_df']

    S_non_attendees, E_non_attendees, I_non_attendees, R_non_attendees = S_t.copy(), E_t.copy(), I_t.copy(), R_t.copy()

    all_attendees_after_tx = []
    group_provenance_records: List[Dict[str, Any]] = []
    total_exposures_today = 0
    total_infected_recruited_today = 0
    total_attendees_today = 0
    recruitment_data_for_plotting = np.zeros(n_patches)
    total_infected_attendees_series = pd.Series(0, index=range(n_patches))

    day_initial_attendees_df = initial_attendees_df[(initial_attendees_df['event_day'] == int(getattr(cfg, 'current_time', 0))) & (initial_attendees_df['run_id'] == run_id)] if initial_attendees_df is not None else None

    recruitment_strategies = {
        'commuter': CommuterStrategy(),
        'gravity': GravityStrategy(),
        'population': PopulationStrategy(),
        'single_patch': SinglePatchStrategy()
    }
    strategy = recruitment_strategies.get(getattr(cfg, 'attendee_recruitment_model', 'gravity'))
    if not strategy:
        raise ValueError(f"Unknown recruitment model: '{cfg.attendee_recruitment_model}'")

    for group_id, venue_id in getattr(cfg, 'multi_venue_group_venues', {}).items():
        strategy_kwargs = {
            'flow_ij_df': data['flow_ij_df'],
            'distance_matrix': data['distance_matrix_km'],
            'single_patch_source_id': getattr(cfg, 'single_patch_recruitment_id', None),
            'gravity_model_distance_decay_exponent': getattr(cfg, 'gravity_model_distance_decay_exponent', 1.5),
            'gravity_model_min_dist_km': getattr(cfg, 'gravity_model_min_dist_km', 0.5),
            'gravity_model_self_dist_factor': getattr(cfg, 'gravity_model_self_dist_factor', 0.5)
        }
        group_attendees, S_non_attendees, E_non_attendees, I_non_attendees, R_non_attendees, rec_data, _ = recruit_attendees_for_multi_venue_event(
            group_id, venue_id, getattr(cfg, 'multi_venue_total_attendees', 2000), strategy,
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

        total_infected_recruited_today += len(infected_in_group)

        if getattr(cfg, 'current_day', 0) == getattr(cfg, 'event_days', [0])[0]:
            infected_counts = infected_in_group.groupby('home_patch_id').size()
            total_infected_attendees_series = total_infected_attendees_series.add(infected_counts, fill_value=0)

        contact_file = Path("data_crowd-contacts") / f"contacts_aggregated_N{getattr(cfg, 'multi_venue_total_attendees', 2000)}_T36000_{group_id}.csv"
        contact_data = data.get('contact_cache', {}).get(group_id, contact_file)

        transmit_result = simulate_event_transmission(
            group_attendee_df.to_dict('records'), contact_data, getattr(cfg, 'event_base_transmission_rate', 0.1),
            getattr(cfg, 'event_contact_fps', 5), rng,
            getattr(cfg, 'event_contact_header_map', None),
            return_provenance=getattr(cfg, 'track_invasion_provenance', False)
        )
        if getattr(cfg, 'track_invasion_provenance', False):
            attendees_after_tx, provenance_records = transmit_result
        else:
            attendees_after_tx = transmit_result
            provenance_records = []
        e_after = pd.DataFrame(attendees_after_tx)[pd.DataFrame(attendees_after_tx)['status'] == 'E'].shape[0]
        total_exposures_today += (e_after - e_before)
        all_attendees_after_tx.extend(attendees_after_tx)
        if provenance_records:
            for rec in provenance_records:
                rec['group_id'] = group_id
                rec['event_patch_id'] = venue_id
            group_provenance_records.extend(provenance_records)

    all_attendees_df = pd.DataFrame(all_attendees_after_tx) if all_attendees_after_tx else pd.DataFrame()
    infected_counts_by_patch = all_attendees_df[all_attendees_df['status'] == 'I'].groupby('home_patch_id').size().reindex(range(n_patches), fill_value=0).to_numpy() if not all_attendees_df.empty else np.zeros(n_patches)

    N_non_attendees = S_non_attendees + E_non_attendees + I_non_attendees + R_non_attendees
    meta_step_results = _run_metapopulation_step(
        data, cfg, rng, S_non_attendees, E_non_attendees, I_non_attendees, R_non_attendees, N_non_attendees,
        track_provenance=getattr(cfg, 'track_invasion_provenance', False)
    )
    S_updated_non, E_updated_non, I_updated_non, R_updated_non = (
        meta_step_results["S_new"], meta_step_results["E_new"],
        meta_step_results["I_new"], meta_step_results["R_new"]
    )

    S_final, E_final, I_final, R_final, new_exposures_by_patch = reintegrate_attendees(
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
        "infected_attendee_series": total_infected_attendees_series,
        "metapop_provenance_records": meta_step_results.get('provenance_records', []),
        "provenance_records": group_provenance_records + meta_step_results.get('provenance_records', [])
    }
    return event_results


def _handle_event_day(data: Dict[str, Any], cfg: Any, rng: np.random.Generator,
                       S_t: np.ndarray, E_t: np.ndarray, I_t: np.ndarray, R_t: np.ndarray,
                       logger: InfectionLogger, run_id: int, initial_infections_data: Optional[pd.DataFrame],
                       N_t: np.ndarray, precomputed_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Orchestrates event simulation for a given day."""
    event_results = {}

    if getattr(cfg, 'event_model_type', '') == 'large_venue':
        event_results = _run_large_venue_event(data, cfg, rng, S_t, E_t, I_t, R_t, logger, run_id, N_t, precomputed_data)
    elif getattr(cfg, 'event_model_type', '') == 'multi_venue':
        event_results = _run_multi_venue_event(data, cfg, rng, S_t, E_t, I_t, R_t, initial_infections_data, run_id, N_t)
    else:
        raise ValueError(f"Unknown event_model_type: '{cfg.event_model_type}'")

    logger.record_infections(logger.ini_writer, run_id, int(getattr(cfg, 'current_time', 0)), event_results['infected_counts_by_patch'])
    logger.record_infections(logger.new_writer, run_id, int(getattr(cfg, 'current_time', 0)), event_results['new_exposures_by_patch'])

    S_new, E_new, I_new, R_new = event_results["S_new"], event_results["E_new"], event_results["I_new"], event_results["R_new"]
    N_new = S_new + E_new + I_new + R_new
    new_exp, new_inf = np.sum(E_new) - np.sum(E_t), np.sum(I_new) - np.sum(I_t)

    return {
        "S_new": S_new, "E_new": E_new, "I_new": I_new, "R_new": R_new, "N_new": N_new,
        "new_exp": new_exp, "new_inf": new_inf,
        "event_specific_results": event_results
    }


def _run_gillespie_step(data: Dict[str, Any], cfg: Any, rng: np.random.Generator,
                        S_t: np.ndarray, E_t: np.ndarray, I_t: np.ndarray, R_t: np.ndarray,
                        track_provenance: bool = False) -> Dict[str, Any]:
    """Performs one step of the stochastic simulation using the Gillespie algorithm."""
    n_patches = data['n_patches']
    alpha = 1.0 / getattr(cfg, 'period_incubation', 4.0)
    gamma = 1.0 / getattr(cfg, 'period_infectious', 5.0)
    beta = getattr(cfg, 'beta', 0.2)

    S_ij = S_t.reshape((n_patches, n_patches))
    E_ij = E_t.reshape((n_patches, n_patches))
    I_ij = I_t.reshape((n_patches, n_patches))

    progression_rates = alpha * E_ij
    recovery_rates = gamma * I_ij

    is_day = (getattr(cfg, 'current_time', 0) % 1.0) == 0.0
    N_ij = (S_ij + E_ij + I_ij + R_t.reshape((n_patches, n_patches)))

    if is_day:
        I_in_j = I_ij.sum(axis=0)
        N_in_j = N_ij.sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            lambda_j = beta * I_in_j / N_in_j
        lambda_j = np.nan_to_num(lambda_j)
        infection_rates = S_ij * lambda_j[np.newaxis, :]
    else:
        I_in_i = I_ij.sum(axis=1)
        N_in_i = N_ij.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            lambda_i = beta * I_in_i / N_in_i
        lambda_i = np.nan_to_num(lambda_i)
        infection_rates = S_ij * lambda_i[:, np.newaxis]

    R_infection = infection_rates.sum()
    R_progression = progression_rates.sum()
    R_recovery = recovery_rates.sum()
    R_total = R_infection + R_progression + R_recovery

    if R_total <= 0:
        return {"tau": getattr(cfg, 'n_days', 250), "S_new": S_t, "E_new": E_t, "I_new": I_t, "R_new": R_t, "N_new": S_t+E_t+I_t+R_t, "new_exp": 0, "new_inf": 0, "provenance_records": []}

    tau = -np.log(rng.random()) / R_total

    S_new, E_new, I_new, R_new = S_t.copy(), E_t.copy(), I_t.copy(), R_t.copy()
    new_exp, new_inf = 0, 0
    provenance_records: List[Dict[str, Any]] = []

    event_choice = rng.random() * R_total

    if event_choice < R_infection:
        target = event_choice
        cum_rates = np.cumsum(infection_rates.ravel())
        event_idx = np.searchsorted(cum_rates, target)

        S_new[event_idx] -= 1
        E_new[event_idx] += 1
        new_exp = 1
        if track_provenance:
            I_ij = I_t.reshape((n_patches, n_patches))
            provenance_records.append(_infer_metapop_infection_source(n_patches, I_ij, int(event_idx), is_day, rng))
    elif event_choice < R_infection + R_progression:
        target = event_choice - R_infection
        cum_rates = np.cumsum(progression_rates.ravel())
        event_idx = int(np.searchsorted(cum_rates, target))

        E_new[event_idx] -= 1
        I_new[event_idx] += 1
        new_inf = 1
    else:
        target = event_choice - R_infection - R_progression
        cum_rates = np.cumsum(recovery_rates.ravel())
        event_idx = int(np.searchsorted(cum_rates, target))

        I_new[event_idx] -= 1
        R_new[event_idx] += 1

    np.clip(S_new, 0, None, out=S_new)
    np.clip(E_new, 0, None, out=E_new)
    np.clip(I_new, 0, None, out=I_new)

    N_new = S_new + E_new + I_new + R_new

    return {"tau": tau, "S_new": S_new, "E_new": E_new, "I_new": I_new, "R_new": R_new, "N_new": N_new, "new_exp": new_exp, "new_inf": new_inf, "provenance_records": provenance_records}


def _run_metapopulation_step(data: Dict[str, Any], cfg: Any, rng: np.random.Generator,
                              S_t: np.ndarray, E_t: np.ndarray, I_t: np.ndarray, R_t: np.ndarray,
                              N_t: np.ndarray, track_provenance: bool = False) -> Dict[str, Any]:
    """Performs one step of the SEIR model, dispatching to either Gillespie or Tau-leaping."""
    E_total = np.sum(E_t)
    I_total = np.sum(I_t)

    tau_leap_threshold = getattr(cfg, 'TauLeaping_threshold', None)
    if tau_leap_threshold is None:
        tau_leap_threshold = getattr(cfg, 'tau_leaping_threshold', 100)
    use_gillespie = (E_total + I_total) > 0 and (E_total + I_total) < tau_leap_threshold

    if use_gillespie:
        return _run_gillespie_step(data, cfg, rng, S_t, E_t, I_t, R_t, track_provenance=track_provenance)
    else:
        time_of_day = getattr(cfg, 'current_time', 0) % 1.0
        tod = "day-time" if time_of_day == 0.0 else "night"
        alpha = 1.0 / getattr(cfg, 'period_incubation', 4.0)
        gamma = 1.0 / getattr(cfg, 'period_infectious', 5.0)
        beta = getattr(cfg, 'beta', 0.2)

        return compute_SEIR(
            beta, alpha, gamma, data['n_patches'], S_t, E_t, I_t, R_t, N_t, tod, rng, track_provenance=track_provenance
        )
