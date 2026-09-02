"""Core recruitment logic for event attendees."""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from .strategies import RecruitmentStrategy


def _resolve_column(df: pd.DataFrame, config_col_name: str, candidates: list) -> Optional[str]:
    """Helper to resolve a column name from config or candidates."""
    if config_col_name and config_col_name in df.columns:
        return config_col_name
    for cand in candidates:
        if cand in df.columns:
            return cand
    return None


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
