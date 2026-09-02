"""Main simulation runner for the Hybrid Epidemic Simulation Model.

This module contains the ``run_simulation`` function that orchestrates a single
Monte Carlo iteration of the SEIR metapopulation model, including optional
hybrid event logic.  It ties together the ``events``, ``core``, and ``arrival``
submodules.
"""
from typing import Dict, Any, Optional, List
import numpy as np

from .core import (
    compute_rt_from_state,
    _seed_initial_infections,
)
from .events import (
    InfectionLogger,
    _run_metapopulation_step,
    _handle_event_day,
)
from .arrival import (
    apply_arrival_records,
    apply_arrival_records_I,
    check_and_record_die_out,
)


def run_simulation(data: Dict[str, Any], cfg: Any, rng: np.random.Generator, run_id: int,
                   logger: InfectionLogger, initial_infections_data: Optional[Dict] = None,
                   precomputed_event_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Runs a full SEIR simulation, including optional hybrid event logic.
    """
    n_patches = data['n_patches']
    track_provenance = getattr(cfg, 'track_invasion_provenance', False)
    N_ij_initial = data['N_ij_initial'].ravel().astype(np.int32)

    beta = getattr(cfg, 'beta', 0.2)

    S, E, I, R = N_ij_initial.copy(), np.zeros_like(N_ij_initial), np.zeros_like(N_ij_initial), np.zeros_like(N_ij_initial)
    N = S + E + I + R

    S, I = _seed_initial_infections(cfg, S, I, n_patches, run_id, initial_infections_data)
    N = S + E + I + R

    S_curr, E_curr, I_curr, R_curr, N_curr = S.copy(), E.copy(), I.copy(), R.copy(), N.copy()

    # Diagnostic: capture initial district-level state (after seeding, before loop)
    S_initial = S_curr.reshape(n_patches, n_patches).sum(axis=0).astype(np.float32)
    E_initial = E_curr.reshape(n_patches, n_patches).sum(axis=0).astype(np.float32)
    I_initial = I_curr.reshape(n_patches, n_patches).sum(axis=0).astype(np.float32)
    R_initial = R_curr.reshape(n_patches, n_patches).sum(axis=0).astype(np.float32)
    S_day1 = None; E_day1 = None; I_day1 = None; R_day1 = None

    t_hist = [0.0]
    gamma = 1.0 / getattr(cfg, 'period_infectious', 5.0)
    rt_initial = compute_rt_from_state(S_curr.reshape(n_patches, n_patches), N_curr.reshape(n_patches, n_patches), beta, gamma, n_patches)
    rt_hist = [rt_initial]
    last_rt_time = 0.0

    S_all_hist, E_all_hist, I_all_hist, R_all_hist = [np.sum(S)], [np.sum(E)], [np.sum(I)], [np.sum(R)]
    new_exposed_hist, new_infected_hist = [0], [0]
    arrival_times = np.full(n_patches, np.nan, dtype=np.float32)
    arrival_sources = np.full(n_patches, -1, dtype=np.int32)
    arrival_contexts = np.full(n_patches, "", dtype=object)
    arrival_step_index = np.full(n_patches, -1, dtype=np.int32)
    arrival_event_day = np.full(n_patches, -1, dtype=np.int32)
    invasion_edge_records: List[Dict[str, Any]] = []

    arrival_times_I = np.full(n_patches, np.nan, dtype=np.float32)
    arrival_sources_I = np.full(n_patches, -1, dtype=np.int32)
    arrival_contexts_I = np.full(n_patches, "", dtype=object)
    arrival_step_index_I = np.full(n_patches, -1, dtype=np.int32)
    arrival_event_day_I = np.full(n_patches, -1, dtype=np.int32)
    invasion_edge_records_I: List[Dict[str, Any]] = []

    store_detailed = getattr(cfg, 'write_full_results', False)

    if store_detailed:
        S_hist = [S_curr.copy()]
        E_hist = [E_curr.copy()]
        I_hist = [I_curr.copy()]
        R_hist = [R_curr.copy()]
        N_hist = [N_curr.copy()]
    else:
        E_agg = E_curr.reshape(n_patches, n_patches).sum(axis=0)
        I_agg = E_curr.reshape(n_patches, n_patches).sum(axis=0)
        E_hist = [E_agg]
        I_hist = [I_agg]
        S_hist = []
        R_hist = []
        N_hist = []

    event_exposures_by_day = {}
    event_attendees_by_day = {}
    infected_attendees_recruited_by_day = {}
    event_dispersal_by_day = {}
    event_exposures_by_patch_day0 = np.zeros(n_patches)
    recruitment_data_for_plotting = np.zeros(n_patches)
    infected_attendee_counts = np.zeros(n_patches)

    while t_hist[-1] < getattr(cfg, 'n_days', 250):
        if check_and_record_die_out(t_hist[-1], E_all_hist[-1], I_all_hist[-1]):
            break

        cfg.current_time = t_hist[-1]
        cfg.current_day = int(cfg.current_time)
        time_of_day = cfg.current_time % 1.0

        if (getattr(cfg, 'current_time', 0) == 1.0 and
            getattr(cfg, 'event_model_type', '') .startswith('no_event') and
            getattr(cfg, 'seeding_method', 'recruit') == 'import' and
            getattr(cfg, 'no_event_seeding_mode', 'single_patch') == 'single_patch' and
            getattr(cfg, 'I_ss', 0) > 0):

            seed_patch_idx = getattr(cfg, 'initial_infection_patch_id', 0)
            seed_subpop_idx = seed_patch_idx * n_patches + seed_patch_idx

            amount_to_remove = getattr(cfg, 'I_ss', 0)
            N_curr[seed_subpop_idx] = max(0, N_curr[seed_subpop_idx] - amount_to_remove)

            if I_curr[seed_subpop_idx] >= amount_to_remove:
                I_curr[seed_subpop_idx] -= amount_to_remove
            else:
                remainder = amount_to_remove - I_curr[seed_subpop_idx]
                I_curr[seed_subpop_idx] = 0
                R_curr[seed_subpop_idx] = max(0, R_curr[seed_subpop_idx] - remainder)

        S_t, E_t, I_t, R_t, N_t = S_curr, E_curr, I_curr, R_curr, N_curr

        if getattr(cfg, 'current_day', 0) in getattr(cfg, 'event_days', []) and time_of_day == 0.0 and not getattr(cfg, 'event_model_type', '') .startswith('no_event'):
            day_results = _handle_event_day(data, cfg, rng, S_t, E_t, I_t, R_t, logger, run_id, initial_infections_data, N_t, precomputed_event_data)
            S_new, E_new, I_new, R_new = day_results["S_new"], day_results["E_new"], day_results["I_new"], day_results["R_new"]
            N_new, new_exp, new_inf = day_results["N_new"], day_results["new_exp"], day_results["new_inf"]
            event_specific_results = day_results["event_specific_results"]

            event_exposures_by_day[getattr(cfg, 'current_day', 0)] = event_specific_results['new_exposures_today']
            infected_attendees_recruited_by_day[getattr(cfg, 'current_day', 0)] = event_specific_results['infected_recruited_today']
            event_attendees_by_day[getattr(cfg, 'current_day', 0)] = event_specific_results['attendees_today']

            if 'new_exposures_by_patch' in event_specific_results:
                event_dispersal_by_day[getattr(cfg, 'current_day', 0)] = np.count_nonzero(event_specific_results['new_exposures_by_patch'])

                if getattr(cfg, 'current_day', 0) == getattr(cfg, 'event_days', [0])[0]:
                    event_exposures_by_patch_day0 = event_specific_results['new_exposures_by_patch']

            if getattr(cfg, 'current_day', 0) == getattr(cfg, 'event_days', [0])[0]:
                recruitment_data_for_plotting = event_specific_results['recruitment_data']
                if getattr(cfg, 'event_model_type', '') == 'multi_venue':
                    infected_attendee_counts = event_specific_results['infected_attendee_series'].reindex(range(n_patches), fill_value=0).values
                else:
                    infected_attendee_counts = event_specific_results['infected_counts_by_patch']

            tau = 0.5
            prev_e_by_patch = E_t.reshape(n_patches, n_patches).sum(axis=0)
            curr_e_by_patch = E_new.reshape(n_patches, n_patches).sum(axis=0)
            apply_arrival_records(
                n_patches, prev_e_by_patch, curr_e_by_patch,
                event_specific_results.get('provenance_records', []),
                arrival_times, arrival_sources, arrival_contexts,
                arrival_step_index, arrival_event_day, invasion_edge_records,
                event_time=t_hist[-1] + tau, step_index=len(t_hist),
                infection_context="event_day", event_day=int(getattr(cfg, 'current_day', 0)),
                cfg=cfg, run_id=run_id
            )
            prev_i_by_patch = I_t.reshape(n_patches, n_patches).sum(axis=0)
            curr_i_by_patch = I_new.reshape(n_patches, n_patches).sum(axis=0)
            apply_arrival_records_I(
                n_patches, prev_i_by_patch, curr_i_by_patch,
                event_specific_results.get('provenance_records', []),
                arrival_times_I, arrival_sources_I, arrival_contexts_I,
                arrival_step_index_I, arrival_event_day_I, invasion_edge_records_I,
                event_time=t_hist[-1] + tau, step_index=len(t_hist),
                infection_context="event_day", event_day=int(getattr(cfg, 'current_day', 0)),
                cfg=cfg, run_id=run_id
            )

        else:
            step_results = _run_metapopulation_step(data, cfg, rng, S_t, E_t, I_t, R_t, N_t)
            tau = step_results["tau"]
            S_new, E_new, I_new, R_new = step_results["S_new"], step_results["E_new"], step_results["I_new"], step_results["R_new"]
            N_new, new_exp, new_inf = step_results["N_new"], step_results["new_exp"], step_results["new_inf"]

            S_new = np.maximum(S_new, 0)
            E_new = np.maximum(E_new, 0)
            I_new = np.maximum(I_new, 0)
            R_new = np.maximum(R_new, 0)

            prev_e_by_patch = E_t.reshape(n_patches, n_patches).sum(axis=0)
            curr_e_by_patch = E_new.reshape(n_patches, n_patches).sum(axis=0)
            apply_arrival_records(
                n_patches, prev_e_by_patch, curr_e_by_patch,
                step_results.get('provenance_records', []),
                arrival_times, arrival_sources, arrival_contexts,
                arrival_step_index, arrival_event_day, invasion_edge_records,
                event_time=t_hist[-1] + tau, step_index=len(t_hist),
                infection_context="metapop", event_day=int(getattr(cfg, 'current_day', 0)),
                cfg=cfg, run_id=run_id
            )
            prev_i_by_patch = I_t.reshape(n_patches, n_patches).sum(axis=0)
            curr_i_by_patch = I_new.reshape(n_patches, n_patches).sum(axis=0)
            apply_arrival_records_I(
                n_patches, prev_i_by_patch, curr_i_by_patch,
                step_results.get('provenance_records', []),
                arrival_times_I, arrival_sources_I, arrival_contexts_I,
                arrival_step_index_I, arrival_event_day_I, invasion_edge_records_I,
                event_time=t_hist[-1] + tau, step_index=len(t_hist),
                infection_context="metapop", event_day=int(getattr(cfg, 'current_day', 0)),
                cfg=cfg, run_id=run_id
            )

        S_new = np.maximum(S_new, 0)
        E_new = np.maximum(E_new, 0)
        I_new = np.maximum(I_new, 0)
        R_new = np.maximum(R_new, 0)

        S_curr, E_curr, I_curr, R_curr, N_curr = S_new, E_new, I_new, R_new, N_new

        # Diagnostic: capture day-1 district-level state (after first step)
        if S_day1 is None:
            S_day1 = S_new.reshape(n_patches, n_patches).sum(axis=0).astype(np.float32)
            E_day1 = E_new.reshape(n_patches, n_patches).sum(axis=0).astype(np.float32)
            I_day1 = I_new.reshape(n_patches, n_patches).sum(axis=0).astype(np.float32)
            R_day1 = R_new.reshape(n_patches, n_patches).sum(axis=0).astype(np.float32)

        if (t_hist[-1] - last_rt_time >= 0.5):
            rt = compute_rt_from_state(S_curr.reshape(n_patches, n_patches), N_curr.reshape(n_patches, n_patches), beta, gamma, n_patches)
            last_rt_time = t_hist[-1]
        else:
            rt = rt_hist[-1]

        S_all_hist.append(np.sum(S_new)); E_all_hist.append(np.sum(E_new)); I_all_hist.append(np.sum(I_new)); R_all_hist.append(np.sum(R_new))
        new_exposed_hist.append(new_exp); new_infected_hist.append(new_inf)
        t_hist.append(t_hist[-1] + tau)
        rt_hist.append(rt)

        if store_detailed:
            S_hist.append(S_new.copy()); E_hist.append(E_new.copy()); I_hist.append(I_new.copy()); R_hist.append(R_new.copy()); N_hist.append(N_new.copy())
        else:
            E_agg = E_new.reshape(n_patches, n_patches).sum(axis=0)
            I_agg = E_new.reshape(n_patches, n_patches).sum(axis=0)
            E_hist.append(E_agg)
            I_hist.append(I_agg)

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
        "event_exposures_by_patch": event_exposures_by_patch_day0,
        "recruitment_data": recruitment_data_for_plotting,
        "infected_attendee_counts": infected_attendee_counts,
        # Diagnostic: per-district SEIR state at day 0 (initial) and day 1 (after first step)
        # Always saved to enable seed propagation verification
        "district_state_day0": {"S": S_initial.copy(), "E": E_initial.copy(), "I": I_initial.copy(), "R": R_initial.copy()},
        "district_state_day1": {"S": S_day1.copy(), "E": E_day1.copy(), "I": I_day1.copy(), "R": R_day1.copy()},
        "arrival_times": arrival_times,
        "arrival_sources": arrival_sources,
        "arrival_contexts": arrival_contexts,
        "arrival_step_index": arrival_step_index,
        "arrival_event_day": arrival_event_day,
        "invasion_edge_records": invasion_edge_records,
        "arrival_times_I": arrival_times_I,
        "arrival_sources_I": arrival_sources_I,
        "arrival_contexts_I": arrival_contexts_I,
        "arrival_step_index_I": arrival_step_index_I,
        "arrival_event_day_I": arrival_event_day_I,
        "invasion_edge_records_I": invasion_edge_records_I,
    }

    if store_detailed:
        results.update({"S_ij": [x.astype(np.int32) for x in S_hist], "E_ij": [x.astype(np.int32) for x in E_hist], "I_ij": [x.astype(np.int32) for x in I_hist], "R_ij": [x.astype(np.int32) for x in R_hist], "N_ij": [x.astype(np.int32) for x in N_hist]})
    else:
        results.update({"E_j": [x.astype(np.int32) for x in E_hist], "I_j": [x.astype(np.int32) for x in I_hist]})

    return results
