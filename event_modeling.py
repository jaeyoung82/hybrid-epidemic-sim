import numpy as np
import pandas as pd
from pathlib import Path

def _remove_attendees_from_population(attendee_df, S_reshaped, E_reshaped, I_reshaped, R_reshaped, N_ij_initial_reshaped, n_patches, rng):
    """
    Helper function to remove recruited attendees from the main SEIR arrays.
    This logic is shared by all recruitment models.
    """
    S_after, E_after, I_after, R_after = S_reshaped.copy(), E_reshaped.copy(), I_reshaped.copy(), R_reshaped.copy()
    attendee_counts_by_patch = attendee_df.groupby(['home_patch_id', 'status']).size().unstack(fill_value=0)

    for home_patch_i, counts in attendee_counts_by_patch.iterrows():
        sub_pops = N_ij_initial_reshaped[home_patch_i, :]
        if sub_pops.sum() == 0:
            continue
        sub_pop_dist = sub_pops / sub_pops.sum()

        s_to_remove = rng.multinomial(counts.get('S', 0), sub_pop_dist)
        e_to_remove = rng.multinomial(counts.get('E', 0), sub_pop_dist)
        i_to_remove = rng.multinomial(counts.get('I', 0), sub_pop_dist)
        r_to_remove = rng.multinomial(counts.get('R', 0), sub_pop_dist)

        S_after[home_patch_i, :] -= s_to_remove
        E_after[home_patch_i, :] -= e_to_remove
        I_after[home_patch_i, :] -= i_to_remove
        R_after[home_patch_i, :] -= r_to_remove

    np.clip(S_after, 0, None, out=S_after)
    np.clip(E_after, 0, None, out=E_after)
    np.clip(I_after, 0, None, out=I_after)
    np.clip(R_after, 0, None, out=R_after)

    return S_after.ravel(), E_after.ravel(), I_after.ravel(), R_after.ravel()


def _select_agents_and_locations(attendees_to_recruit, S_reshaped, E_reshaped, I_reshaped, R_reshaped, N_ij_initial_reshaped, n_patches, rng):
    """
    Common logic to select specific agents (S/E/I/R) and assign their current location
    based on the number of attendees to recruit from each patch. Does NOT assign agent_id.
    """
    attendees = []
    
    # Normalize input to an iterable of (patch_id, count)
    if isinstance(attendees_to_recruit, (pd.Series, dict)):
        iterator = attendees_to_recruit.items()
    else:
        # Assume array-like
        iterator = enumerate(attendees_to_recruit)

    for home_patch_i, num_to_recruit in iterator:
        home_patch_i = int(home_patch_i)
        if num_to_recruit <= 0:
            continue

        total_S = S_reshaped[home_patch_i, :].sum()
        total_E = E_reshaped[home_patch_i, :].sum()
        total_I = I_reshaped[home_patch_i, :].sum()
        total_R = R_reshaped[home_patch_i, :].sum()
        total_pop = total_S + total_E + total_I + total_R

        if total_pop <= 0:
            continue

        probs = np.array([total_S, total_E, total_I, total_R]) / total_pop
        probs /= probs.sum() # Normalize to ensure sum is 1.0
        
        recruited_counts = rng.multinomial(num_to_recruit, probs)
        status_map = {0: 'S', 1: 'E', 2: 'I', 3: 'R'}
        
        # Pre-calculate flow distribution for location assignment
        flow_dist = N_ij_initial_reshaped[home_patch_i, :]
        flow_sum = flow_dist.sum()
        flow_probs = flow_dist / flow_sum if flow_sum > 0 else None

        for status_idx, count in enumerate(recruited_counts):
            if count == 0: continue
            status = status_map[status_idx]
            locations = rng.choice(n_patches, size=count, p=flow_probs) if flow_probs is not None else np.full(count, home_patch_i)
            
            for loc in locations:
                attendees.append({
                    'home_patch_id': home_patch_i,
                    'status': status,
                    'current_patch_id': loc
                })
                
    return attendees


class RecruitmentStrategy:
    """Abstract base class for attendee recruitment strategies."""
    def calculate_proportions(self, n_patches, population_df, **kwargs):
        raise NotImplementedError

class PopulationStrategy(RecruitmentStrategy):
    """Recruits attendees based on the population proportion of each district."""
    def calculate_proportions(self, n_patches, population_df, **kwargs):
        populations = population_df['population'].to_numpy()
        return populations / populations.sum()

class GravityStrategy(RecruitmentStrategy):
    """Recruits attendees using a gravity model based on population and distance."""
    def calculate_proportions(self, n_patches, population_df, **kwargs):
        event_patch_id = kwargs['event_patch_id']
        distance_matrix = kwargs['distance_matrix']
        populations = population_df['population'].to_numpy()
        distances_to_venue = distance_matrix[:, event_patch_id].copy()
        
        # Improved self-distance handling to prevent venue patch domination
        non_zero_dists = distances_to_venue[distances_to_venue > 1e-6]
        if len(non_zero_dists) > 0:
            self_dist = max(np.min(non_zero_dists) / 2.0, 0.5)
        else:
            self_dist = 1.0
        distances_to_venue[distances_to_venue <= 1e-6] = self_dist
        
        # Use alpha=1.5 for less aggressive decay than squared distance
        gravity_scores = populations / (distances_to_venue**1.5)
        return gravity_scores / gravity_scores.sum()

class SinglePatchStrategy(RecruitmentStrategy):
    """Recruits attendees exclusively from a single specified patch."""
    def calculate_proportions(self, n_patches, population_df, **kwargs):
        target_patch = kwargs.get('single_patch_source_id')
        if target_patch is None:
             target_patch = kwargs.get('event_patch_id', 0)
        
        proportions = np.zeros(n_patches)
        if 0 <= target_patch < n_patches:
            proportions[int(target_patch)] = 1.0
        return proportions

class CommuterStrategy(RecruitmentStrategy):
    """Recruits attendees based on commuter flows to the event venue."""
    def calculate_proportions(self, n_patches, population_df, **kwargs):
        event_patch_id = kwargs['event_patch_id']
        flow_ij_df = kwargs['flow_ij_df']
        inflows = flow_ij_df[flow_ij_df['patch_j'] == event_patch_id]
        total_inflow = inflows['N_ij'].sum()
        
        proportions = pd.Series(0.0, index=range(n_patches))
        if total_inflow > 0:
            inflow_proportions = inflows.set_index('patch_i')['N_ij'] / total_inflow
            proportions.update(inflow_proportions)
        return proportions.to_numpy()

def recruit_attendees_for_multi_venue_event(
    target_group, event_patch_id, total_attendees, strategy: RecruitmentStrategy,
    S_t, E_t, I_t, R_t, n_patches, population_df, N_ij_initial, rng,
    initial_attendees_df=None, start_id=0, **strategy_kwargs
):
    """
    Recruits attendees for a multi-venue event from a specific group using a given strategy.
    This version uses the Strategy pattern, removes recruits from population pools, and is consistent
    with the large-venue recruitment function.
    """
    group_districts = population_df[population_df['group'] == target_group]
    if group_districts.empty:
        print(f"Warning: No districts found for group '{target_group}'. No attendees recruited.")
        return [], S_t, E_t, I_t, R_t, np.zeros(n_patches), start_id

    group_district_ids = group_districts['id'].tolist()

    # Calculate proportions for all patches using the strategy, then filter and re-normalize for the group
    all_proportions = strategy.calculate_proportions(n_patches, population_df, event_patch_id=event_patch_id, **strategy_kwargs)
    group_mask = np.isin(np.arange(n_patches), group_district_ids)
    group_proportions = all_proportions * group_mask

    total_group_prop = group_proportions.sum()
    if total_group_prop > 0:
        final_proportions = group_proportions / total_group_prop
    else:
        # Fallback to population if strategy yields no one (e.g., commuter model with no commuters from group)
        print(f"Warning: Recruitment model resulted in zero proportions for group {target_group}. Falling back to 'population'.")
        group_pops = population_df[population_df['id'].isin(group_district_ids)].set_index('id')['population']
        final_proportions = np.zeros(n_patches)
        if not group_pops.empty and group_pops.sum() > 0:
            normalized_pops = group_pops / group_pops.sum()
            final_proportions[normalized_pops.index] = normalized_pops.values

    attendees_to_recruit = (final_proportions * total_attendees).round().astype(int)
    diff = total_attendees - attendees_to_recruit.sum()
    if diff != 0 and attendees_to_recruit.size > 0:
        attendees_to_recruit[final_proportions.argmax()] += diff

    # Reshape pools for processing
    S_reshaped, E_reshaped, I_reshaped, R_reshaped = [comp.reshape((n_patches, n_patches)) for comp in (S_t, E_t, I_t, R_t)]
    N_ij_reshaped = N_ij_initial.reshape((n_patches, n_patches))

    attendees = []
    remaining_to_recruit = pd.Series(attendees_to_recruit, index=range(n_patches))

    if initial_attendees_df is not None and not initial_attendees_df.empty:
        # 1. Identify which of the master infected attendees belong to this venue's group
        infected_for_this_group_df = initial_attendees_df[initial_attendees_df['district_id'].isin(group_district_ids)]

        # 2. "Recruit" these specific infected individuals
        for _, infected_row in infected_for_this_group_df.iterrows():
            home_patch_i = infected_row['district_id']
            attendees.append({
                'home_patch_id': home_patch_i,
                'status': 'I',
                'current_patch_id': infected_row['commuting_district_id']
            })
            remaining_to_recruit[home_patch_i] -= 1

        # 3. Recruit the *remaining* attendees from the S, E, R pools
        remaining_to_recruit.clip(lower=0, inplace=True)
        for home_patch_i, num_to_recruit in remaining_to_recruit.items():
            if num_to_recruit <= 0: continue
            total_S, total_E, total_R = S_reshaped[home_patch_i, :].sum(), E_reshaped[home_patch_i, :].sum(), R_reshaped[home_patch_i, :].sum()
            total_non_I_pop = total_S + total_E + total_R
            if total_non_I_pop <= 0: continue

            probs = np.array([total_S, total_E, total_R]) / total_non_I_pop
            recruited_counts = rng.multinomial(num_to_recruit, probs)
            status_map = {0: 'S', 1: 'E', 2: 'R'}

            flow_dist = N_ij_reshaped[home_patch_i, :]
            flow_sum = flow_dist.sum()
            flow_probs = flow_dist / flow_sum if flow_sum > 0 else None

            for status_idx, count in enumerate(recruited_counts):
                if count == 0: continue
                status = status_map[status_idx]
                locations = rng.choice(n_patches, size=count, p=flow_probs) if flow_probs is not None else np.full(count, home_patch_i)
                for loc in locations:
                    attendees.append({'home_patch_id': home_patch_i, 'status': status, 'current_patch_id': loc})
    else:
        # Standard recruitment from S, E, I, R pools if no pre-seeding
        new_attendees = _select_agents_and_locations(
            attendees_to_recruit, S_reshaped, E_reshaped, I_reshaped, R_reshaped, N_ij_reshaped, n_patches, rng
        )
        attendees.extend(new_attendees)

    # --- SHUFFLE AND ASSIGN IDs ---
    # Shuffle to ensure random assignment of agent_ids (which map to contact network nodes)
    rng.shuffle(attendees)
    for i, attendee in enumerate(attendees):
        attendee['agent_id'] = start_id + i

    # Remove recruited attendees from the main population pools
    S_after, E_after, I_after, R_after = _remove_attendees_from_population(pd.DataFrame(attendees), S_reshaped, E_reshaped, I_reshaped, R_reshaped, N_ij_reshaped, n_patches, rng) if attendees else (S_t, E_t, I_t, R_t)

    return attendees, S_after, E_after, I_after, R_after, attendees_to_recruit, start_id + len(attendees)

def recruit_attendees(strategy: RecruitmentStrategy, total_attendees, S_t, E_t, I_t, R_t, n_patches, N_ij_initial, rng, population_df, force_infected_count=0, force_infected_patch=None, imported_infected_count=0, **strategy_kwargs):
    """
    Recruits attendees using a specified strategy, selects their status, removes them
    from the general population, and returns the remaining population pools.
    """
    proportions = strategy.calculate_proportions(n_patches, population_df, **strategy_kwargs)
    
    # Ensure proportions is a numpy array for consistent processing
    if isinstance(proportions, pd.Series):
        proportions = proportions.to_numpy()

    attendees_to_recruit = (proportions * total_attendees).round().astype(int)
    
    # Adjust for rounding errors to match total_attendees
    diff = total_attendees - attendees_to_recruit.sum()
    if diff != 0 and attendees_to_recruit.size > 0:
        attendees_to_recruit[proportions.argmax()] += diff

    S_reshaped = S_t.reshape((n_patches, n_patches))
    E_reshaped = E_t.reshape((n_patches, n_patches))
    I_reshaped = I_t.reshape((n_patches, n_patches))
    R_reshaped = R_t.reshape((n_patches, n_patches))
    N_ij_reshaped = N_ij_initial.reshape((n_patches, n_patches))

    attendees = []
    
    # Create working copies of the pools to track availability during this recruitment step
    S_pool = S_reshaped.copy()
    E_pool = E_reshaped.copy()
    I_pool = I_reshaped.copy()
    R_pool = R_reshaped.copy()

    # 1. Force recruitment of infected individuals if specified
    if force_infected_count > 0 and force_infected_patch is not None:
        patch_id = int(force_infected_patch)
        # Check available infected in the patch
        available_I = I_pool[patch_id, :].sum()
        num_to_take = min(force_infected_count, available_I)
        
        if num_to_take > 0:
            # Create infected attendees
            # We distribute them across commuting locations based on flow
            flow_dist = N_ij_reshaped[patch_id, :]
            flow_probs = flow_dist / flow_dist.sum() if flow_dist.sum() > 0 else None
            locations = rng.choice(n_patches, size=num_to_take, p=flow_probs) if flow_probs is not None else np.full(num_to_take, patch_id)
            
            for loc in locations:
                attendees.append({'home_patch_id': patch_id, 'status': 'I', 'current_patch_id': loc})
                # Decrement from the pool so they aren't picked again in step 2
                I_pool[patch_id, loc] = max(0, I_pool[patch_id, loc] - 1)
            
            # Reduce the number we need to recruit from the general pool for this patch
            attendees_to_recruit[patch_id] = max(0, attendees_to_recruit[patch_id] - num_to_take)
        
        # STRICT ENFORCEMENT:
        # If we are forcing a specific number of infected attendees, we assume this is the 
        # target total. We prevent random background recruitment of additional infected 
        # individuals by clearing the I_pool.
        I_pool.fill(0)
    
    # 2. Recruit the rest
    # We pass the updated pools (S_pool, etc.) instead of the original reshaped arrays
    new_attendees = _select_agents_and_locations(
        attendees_to_recruit, S_pool, E_pool, I_pool, R_pool, N_ij_reshaped, n_patches, rng
    )
    attendees.extend(new_attendees)

    # 3. Add externally imported infected individuals if specified
    if imported_infected_count > 0:
        # print(f"  -> Importing {imported_infected_count} external infected individuals for the event.")
        event_patch_id = strategy_kwargs.get('event_patch_id', -1)
        for _ in range(imported_infected_count):
            attendees.append({
                'home_patch_id': -1, # Special ID for external agents
                'status': 'I',
                'current_patch_id': event_patch_id
            })

    # --- SHUFFLE AND ASSIGN IDs ---
    rng.shuffle(attendees)
    for i, attendee in enumerate(attendees):
        attendee['agent_id'] = i

    # Only proceed with removal if attendees were actually selected
    if attendees:
        attendee_df = pd.DataFrame(attendees)
        S_after, E_after, I_after, R_after = _remove_attendees_from_population(
            attendee_df, S_reshaped, E_reshaped, I_reshaped, R_reshaped, N_ij_reshaped, n_patches, rng
        )
    else:
        S_after, E_after, I_after, R_after = S_t, E_t, I_t, R_t

    return attendees, S_after, E_after, I_after, R_after, attendees_to_recruit


def reintegrate_attendees(attendees_after_event: list, S_after_recruitment: np.ndarray, E_after_recruitment: np.ndarray, I_after_recruitment: np.ndarray, R_after_recruitment: np.ndarray, n_patches: int, N_ij_initial: np.ndarray, rng: np.random.Generator):
    """Adds attendees back to their home patches after the event."""
    # print("Re-integrating attendees back into the main population...")

    S_final = S_after_recruitment.reshape((n_patches, n_patches))
    E_final = E_after_recruitment.reshape((n_patches, n_patches))
    I_final = I_after_recruitment.reshape((n_patches, n_patches))
    R_final = R_after_recruitment.reshape((n_patches, n_patches))
    
    # Ensure N_ij_initial is 2D for lookups
    if N_ij_initial.ndim == 1:
        N_ij_initial = N_ij_initial.reshape((n_patches, n_patches))

    attendees_df = pd.DataFrame(attendees_after_event)

    # Separate internal attendees (who go home) from external ones (who leave the simulation)
    internal_attendees_df = attendees_df[attendees_df['home_patch_id'] != -1].copy()

    if internal_attendees_df.empty:
        # If all attendees were external or no one attended, no one is reintegrated.
        # We still need to calculate exposures for logging, but they don't return to the population.
        new_exposures_by_patch = np.zeros(n_patches)
        return S_after_recruitment, E_after_recruitment, I_after_recruitment, R_after_recruitment, new_exposures_by_patch

    # Calculate newly exposed by home patch (only for internal attendees who will spread the disease back home)
    newly_exposed_df = internal_attendees_df[internal_attendees_df['status'] == 'E']
    new_exposures_by_patch = newly_exposed_df.groupby('home_patch_id').size().reindex(range(n_patches), fill_value=0).to_numpy()

    final_counts = internal_attendees_df.groupby(['home_patch_id', 'status']).size().unstack(fill_value=0)

    for home_patch_i, counts in final_counts.iterrows():
        idx = int(home_patch_i)
        
        # Get the commuting distribution for this home patch to preserve mobility patterns
        flow_dist = N_ij_initial[idx, :]
        total_flow = flow_dist.sum()
        
        if total_flow > 0:
            probs = flow_dist / total_flow
        else:
            probs = np.zeros(n_patches)
            probs[idx] = 1.0

        # Distribute each compartment back according to the mobility matrix
        for status, arr_final in [('S', S_final), ('E', E_final), ('I', I_final), ('R', R_final)]:
            count = counts.get(status, 0)
            if count > 0:
                arr_final[idx, :] += rng.multinomial(count, probs)

    return S_final.ravel(), E_final.ravel(), I_final.ravel(), R_final.ravel(), new_exposures_by_patch


def _resolve_column(df: pd.DataFrame, config_col_name: str, candidates: list) -> str:
    """Helper to resolve a column name from config or candidates."""
    if config_col_name and config_col_name in df.columns:
        return config_col_name
    for cand in candidates:
        if cand in df.columns:
            return cand
    return None


def simulate_event_transmission(attendees, contact_data, base_transmission_rate, fps, rng, column_map=None):
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
        return attendees
    
    exposure_df = pd.DataFrame(exposure_pairs)

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
    return attendee_df.to_dict('records')

def analyze_event_percolation_risk(contact_data_file, beta, fps, header_map=None):
    """
    Performs a static bond percolation analysis on the contact network.
    Calculates the size of the Giant Connected Component (GCC) assuming
    edges exist with probability p = 1 - exp(-beta * duration).
    
    Also calculates First-Generation Risk metrics (Expected Secondary Infections).
    """
    try:
        contacts_df = pd.read_csv(contact_data_file)
    except FileNotFoundError:
        return 0.0, 0, 0.0, 0.0

    # Resolve columns
    cfg_col_i = header_map.get("agent_i") if header_map else None
    col_i = _resolve_column(contacts_df, cfg_col_i, ["agent_i", "ped1_id", "id_1", "u", "p1"])
    cfg_col_j = header_map.get("agent_j") if header_map else None
    col_j = _resolve_column(contacts_df, cfg_col_j, ["agent_j", "ped2_id", "id_2", "v", "p2"])
    cfg_col_dur = header_map.get("duration_frames") or header_map.get("tc_s") if header_map else None
    col_dur = _resolve_column(contacts_df, cfg_col_dur, ["total_duration_frames", "tc_s", "duration", "frames"])

    if not col_i or not col_j or not col_dur:
        return 0.0, 0, 0.0, 0.0

    # Aggregate durations between pairs
    contacts_df['duration_sec'] = pd.to_numeric(contacts_df[col_dur], errors='coerce') / fps
    pair_durations = contacts_df.groupby([col_i, col_j])['duration_sec'].sum().reset_index()

    # Calculate Bond Probabilities (Edge weights)
    # Convert duration to hours for consistency with simulation units if beta is per hour
    # Assuming beta is per hour based on simulation.py usage
    pair_durations['prob'] = 1 - np.exp(-beta * (pair_durations['duration_sec'] / 3600.0))

    # --- First-Generation Risk Analysis (Expected Secondary Infections) ---
    # Sum of probabilities connected to each node = Expected Degree
    node_probs = pd.concat([
        pair_durations[[col_i, 'prob']].rename(columns={col_i: 'node'}),
        pair_durations[[col_j, 'prob']].rename(columns={col_j: 'node'})
    ])
    # Sum probabilities per node
    node_strengths = node_probs.groupby('node')['prob'].sum()
    avg_expected_infections = node_strengths.mean() if not node_strengths.empty else 0.0
    max_expected_infections = node_strengths.max() if not node_strengths.empty else 0.0

    # Build Adjacency List for edges that "percolate" (exist)
    # Since this is a probabilistic check, we run one realization of the static network structure
    # For a robust metric, we treat edges with p > 0.5 as 'structural' connections, 
    # or we can sample. Here we sample to match the stochastic nature.
    rng = np.random.default_rng()
    random_vals = rng.random(len(pair_durations))
    active_edges = pair_durations[pair_durations['prob'] > random_vals]

    adj = {}
    all_nodes = set(contacts_df[col_i].unique()) | set(contacts_df[col_j].unique())
    
    for _, row in active_edges.iterrows():
        u, v = int(row[col_i]), int(row[col_j])
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append(v)
        adj[v].append(u)

    # BFS to find Connected Components
    visited = set()
    max_component_size = 0
    
    for node in all_nodes:
        if node not in visited:
            component_size = 0
            stack = [node]
            visited.add(node)
            while stack:
                curr = stack.pop()
                component_size += 1
                for neighbor in adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            if component_size > max_component_size:
                max_component_size = component_size

    total_nodes = len(all_nodes)
    gcc_fraction = max_component_size / total_nodes if total_nodes > 0 else 0.0
    
    return gcc_fraction, total_nodes, avg_expected_infections, max_expected_infections

def analyze_event_reachability_from_all_seeds(contact_data_file, beta, fps, header_map=None, rng=None):
    """
    Performs a static bond percolation analysis on the contact network.
    For a single realization of the network, it calculates the size of the
    connected component for every node if it were the seed.
    """
    try:
        contacts_df = pd.read_csv(contact_data_file)
    except FileNotFoundError:
        return np.array([]), []

    # Resolve columns
    cfg_col_i = header_map.get("agent_i") if header_map else None
    col_i = _resolve_column(contacts_df, cfg_col_i, ["agent_i", "ped1_id", "id_1", "u", "p1"])
    cfg_col_j = header_map.get("agent_j") if header_map else None
    col_j = _resolve_column(contacts_df, cfg_col_j, ["agent_j", "ped2_id", "id_2", "v", "p2"])
    cfg_col_dur = header_map.get("duration_frames") or header_map.get("tc_s") if header_map else None
    col_dur = _resolve_column(contacts_df, cfg_col_dur, ["total_duration_frames", "tc_s", "duration", "frames"])

    if not col_i or not col_j or not col_dur:
        return np.array([]), []

    # Aggregate durations between pairs
    contacts_df['duration_sec'] = pd.to_numeric(contacts_df[col_dur], errors='coerce') / fps
    pair_durations = contacts_df.groupby([col_i, col_j])['duration_sec'].sum().reset_index()

    # Calculate Bond Probabilities
    pair_durations['prob'] = 1 - np.exp(-beta * (pair_durations['duration_sec'] / 3600.0))

    # Use provided RNG or create a new one
    if rng is None:
        rng = np.random.default_rng()

    # Sample one realization of the network
    random_vals = rng.random(len(pair_durations))
    active_edges = pair_durations[pair_durations['prob'] > random_vals]

    adj = {}
    all_nodes_set = set(contacts_df[col_i].unique()) | set(contacts_df[col_j].unique())
    all_nodes = sorted(list(all_nodes_set))

    for _, row in active_edges.iterrows():
        u, v = int(row[col_i]), int(row[col_j])
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append(v)
        adj[v].append(u)

    # --- Find all connected components and map nodes to their component size ---
    visited = set()
    component_sizes = {} # map node_id -> size of its component
    
    for node in all_nodes:
        if node not in visited:
            component_nodes = []
            stack = [node]
            visited.add(node)
            while stack:
                curr = stack.pop()
                component_nodes.append(curr)
                for neighbor in adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            
            size = len(component_nodes)
            for comp_node in component_nodes:
                component_sizes[comp_node] = size

    # Create an array of reachability values, one for each node, in a consistent order
    reachability_values = np.array([component_sizes.get(node, 1) for node in all_nodes])

    return reachability_values, all_nodes

def extract_gcc_structure(contact_data_file, beta, fps, header_map=None, rng=None):
    """
    Constructs the probabilistic contact network and extracts the Giant Connected Component.
    Returns:
        gcc_nodes (set): Set of node IDs in the GCC.
        gcc_edges (list): List of (u, v) tuples representing edges in the GCC.
    """
    try:
        contacts_df = pd.read_csv(contact_data_file)
    except FileNotFoundError:
        return set(), []

    # Resolve columns
    cfg_col_i = header_map.get("agent_i") if header_map else None
    col_i = _resolve_column(contacts_df, cfg_col_i, ["agent_i", "ped1_id", "id_1", "u", "p1"])
    cfg_col_j = header_map.get("agent_j") if header_map else None
    col_j = _resolve_column(contacts_df, cfg_col_j, ["agent_j", "ped2_id", "id_2", "v", "p2"])
    cfg_col_dur = header_map.get("duration_frames") or header_map.get("tc_s") if header_map else None
    col_dur = _resolve_column(contacts_df, cfg_col_dur, ["total_duration_frames", "tc_s", "duration", "frames"])

    if not col_i or not col_j or not col_dur:
        return set(), []

    # Aggregate durations between pairs
    contacts_df['duration_sec'] = pd.to_numeric(contacts_df[col_dur], errors='coerce') / fps
    pair_durations = contacts_df.groupby([col_i, col_j])['duration_sec'].sum().reset_index()

    # Calculate Bond Probabilities
    pair_durations['prob'] = 1 - np.exp(-beta * (pair_durations['duration_sec'] / 3600.0))

    if rng is None:
        rng = np.random.default_rng()
    
    random_vals = rng.random(len(pair_durations))
    active_edges_df = pair_durations[pair_durations['prob'] > random_vals]

    adj = {}
    all_nodes = set(contacts_df[col_i].unique()) | set(contacts_df[col_j].unique())
    
    # Build adjacency for traversal
    for _, row in active_edges_df.iterrows():
        u, v = int(row[col_i]), int(row[col_j])
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append(v)
        adj[v].append(u)

    # Find GCC
    visited = set()
    max_component_nodes = set()
    
    for node in all_nodes:
        if node not in visited:
            component_nodes = set()
            stack = [node]
            visited.add(node)
            component_nodes.add(node)
            while stack:
                curr = stack.pop()
                for neighbor in adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        component_nodes.add(neighbor)
                        stack.append(neighbor)
            
            if len(component_nodes) > len(max_component_nodes):
                max_component_nodes = component_nodes

    # Extract edges for GCC
    gcc_edges = []
    if max_component_nodes:
        for _, row in active_edges_df.iterrows():
            u, v = int(row[col_i]), int(row[col_j])
            if u in max_component_nodes and v in max_component_nodes:
                gcc_edges.append((u, v))

    return max_component_nodes, gcc_edges

def preprocess_contact_file(cfg):
    """
    Preprocesses the contact file to convert string IDs to integers.
    Returns the path to the new file, the actual number of attendees, and the header map.
    """
    original_file = cfg.large_venue_contact_file
    print(f"Preprocessing contact file '{original_file}' to convert string IDs to integers...")
    
    df = pd.read_csv(original_file)
    
    # Determine column names
    header_map = cfg.event_contact_header_map or {}
    
    def get_col(df, keys, default_candidates):
        if isinstance(keys, str): keys = [keys]
        for k in keys:
            cfg_name = header_map.get(k)
            if cfg_name and cfg_name in df.columns:
                return cfg_name
        for c in default_candidates:
            if c in df.columns:
                print(f"  -> Warning: Configured column for '{keys}' not found. Using '{c}' instead.")
                return c
        return None

    col_i = get_col(df, "agent_i", ["agent_i", "ped1_id", "id_1", "u", "p1"])
    col_j = get_col(df, "agent_j", ["agent_j", "ped2_id", "id_2", "v", "p2"])
    col_dur = get_col(df, ["duration_frames", "tc_s"], ["total_duration_frames", "tc_s", "duration", "frames"])
    
    if not col_i: raise KeyError(f"Column for agent_i not found in {original_file}. Available: {list(df.columns)}")
    if not col_j: raise KeyError(f"Column for agent_j not found in {original_file}. Available: {list(df.columns)}")
    if not col_dur: raise KeyError(f"Column for duration not found in {original_file}. Available: {list(df.columns)}")
    
    # Update config to ensure downstream simulation uses the correct columns
    if cfg.event_contact_header_map is None: cfg.event_contact_header_map = {}
    cfg.event_contact_header_map["agent_i"] = col_i
    cfg.event_contact_header_map["agent_j"] = col_j
    cfg.event_contact_header_map["duration_frames"] = col_dur

    # Create mapping for string IDs to integers
    s_i = df[col_i].astype(str).str.strip()
    s_j = df[col_j].astype(str).str.strip()
    
    all_agents = pd.concat([s_i, s_j]).unique()
    all_agents = all_agents[all_agents != 'nan']
    all_agents.sort()
    
    n_unique_agents = len(all_agents)
    id_map = {agent_id: idx for idx, agent_id in enumerate(all_agents)}
    
    mapping_filename = f"{Path(original_file).stem}_id_mapping.csv"
    pd.DataFrame(list(id_map.items()), columns=['original_id', 'new_id']).to_csv(mapping_filename, index=False)
    print(f"  -> Saved ID mapping to '{mapping_filename}'")

    df[col_i] = s_i.map(id_map)
    df[col_j] = s_j.map(id_map)
    df.dropna(subset=[col_i, col_j], inplace=True)
    df[col_i] = df[col_i].astype(int)
    df[col_j] = df[col_j].astype(int)
    
    temp_filename = f"{Path(original_file).stem}_int_mapped.csv"
    df.to_csv(temp_filename, index=False)
    print(f"  -> Saved preprocessed file to '{temp_filename}'")

    header_map = {"agent_i": col_i, "agent_j": col_j}
    return temp_filename, n_unique_agents, header_map