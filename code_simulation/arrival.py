"""
Arrival time tracking logic for the Hybrid Epidemic Simulation Model.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def apply_arrival_records(
    n_patches: int,
    prev_e_by_patch: np.ndarray,
    curr_e_by_patch: np.ndarray,
    provenance_records: List[Dict[str, Any]],
    arrival_times: np.ndarray,
    arrival_sources: np.ndarray,
    arrival_contexts: np.ndarray,
    arrival_step_index: np.ndarray,
    arrival_event_day: np.ndarray,
    invasion_edge_records: List[Dict[str, Any]],
    event_time: float,
    step_index: int,
    infection_context: str,
    event_day: int,
    cfg: Any,
    run_id: int,
) -> None:
    """Record first-arrival times for districts based on E transition."""
    provenance_by_target: Dict[int, Dict[str, Any]] = {}
    for rec in provenance_records:
        try:
            target_patch = int(rec.get('target_patch_id', -1))
        except Exception:
            continue
        if target_patch < 0 or target_patch >= n_patches:
            continue
        provenance_by_target.setdefault(target_patch, rec)

    newly_arrived = np.where((prev_e_by_patch <= 0) & (curr_e_by_patch > 0))[0]
    for target_patch in newly_arrived:
        if np.isnan(arrival_times[target_patch]):
            arrival_times[target_patch] = np.float32(event_time)
            arrival_step_index[target_patch] = int(step_index)
            arrival_event_day[target_patch] = int(event_day)

            rec = provenance_by_target.get(int(target_patch))
            scenario_name = getattr(cfg, 'current_event_scenario_name', getattr(cfg, 'event_model_type', 'unknown'))
            if rec is not None:
                source_patch = int(rec.get('source_patch_id', -1))
                arrival_sources[target_patch] = source_patch
                arrival_contexts[target_patch] = str(rec.get('infection_context', infection_context))
                invasion_edge_records.append({
                    "scenario_name": scenario_name,
                    "run_id": int(run_id),
                    "source_district_id": source_patch,
                    "target_district_id": int(target_patch),
                    "arrival_time": float(event_time),
                    "source_agent_id": int(rec.get('source_agent_id', -1)) if rec.get('source_agent_id', -1) is not None else -1,
                    "target_agent_id": int(rec.get('target_agent_id', -1)) if rec.get('target_agent_id', -1) is not None else -1,
                    "infection_context": str(rec.get('infection_context', infection_context)),
                    "event_day": int(rec.get('event_day', event_day)) if rec.get('event_day', event_day) is not None else int(event_day),
                    "weight_hint": 1,
                })
            else:
                arrival_contexts[target_patch] = infection_context
                invasion_edge_records.append({
                    "scenario_name": scenario_name,
                    "run_id": int(run_id),
                    "source_district_id": -1,
                    "target_district_id": int(target_patch),
                    "arrival_time": float(event_time),
                    "source_agent_id": -1,
                    "target_agent_id": -1,
                    "infection_context": infection_context,
                    "event_day": int(event_day),
                    "weight_hint": 1,
                })


def apply_arrival_records_I(
    n_patches: int,
    prev_i_by_patch: np.ndarray,
    curr_i_by_patch: np.ndarray,
    provenance_records: List[Dict[str, Any]],
    arrival_times_I: np.ndarray,
    arrival_sources_I: np.ndarray,
    arrival_contexts_I: np.ndarray,
    arrival_step_index_I: np.ndarray,
    arrival_event_day_I: np.ndarray,
    invasion_edge_records_I: List[Dict[str, Any]],
    event_time: float,
    step_index: int,
    infection_context: str,
    event_day: int,
    cfg: Any,
    run_id: int,
) -> None:
    """I-based arrival detection: tracks when districts first have infectious individuals (I > 0)."""
    provenance_by_target: Dict[int, Dict[str, Any]] = {}
    for rec in provenance_records:
        try:
            target_patch = int(rec.get('target_patch_id', -1))
        except Exception:
            continue
        if target_patch < 0 or target_patch >= n_patches:
            continue
        provenance_by_target.setdefault(target_patch, rec)

    newly_arrived = np.where((prev_i_by_patch <= 0) & (curr_i_by_patch > 0))[0]
    for target_patch in newly_arrived:
        if np.isnan(arrival_times_I[target_patch]):
            arrival_times_I[target_patch] = np.float32(event_time)
            arrival_step_index_I[target_patch] = int(step_index)
            arrival_event_day_I[target_patch] = int(event_day)

            rec = provenance_by_target.get(int(target_patch))
            scenario_name = getattr(cfg, 'current_event_scenario_name', getattr(cfg, 'event_model_type', 'unknown'))
            if rec is not None:
                source_patch = int(rec.get('source_patch_id', -1))
                arrival_sources_I[target_patch] = source_patch
                arrival_contexts_I[target_patch] = str(rec.get('infection_context', infection_context))
                invasion_edge_records_I.append({
                    "scenario_name": scenario_name,
                    "run_id": int(run_id),
                    "source_district_id": source_patch,
                    "target_district_id": int(target_patch),
                    "arrival_time": float(event_time),
                    "source_agent_id": int(rec.get('source_agent_id', -1)) if rec.get('source_agent_id', -1) is not None else -1,
                    "target_agent_id": int(rec.get('target_agent_id', -1)) if rec.get('target_agent_id', -1) is not None else -1,
                    "infection_context": str(rec.get('infection_context', infection_context)),
                    "event_day": int(rec.get('event_day', event_day)) if rec.get('event_day', event_day) is not None else int(event_day),
                    "weight_hint": 1,
                })
            else:
                arrival_contexts_I[target_patch] = infection_context
                invasion_edge_records_I.append({
                    "scenario_name": scenario_name,
                    "run_id": int(run_id),
                    "source_district_id": -1,
                    "target_district_id": int(target_patch),
                    "arrival_time": float(event_time),
                    "source_agent_id": -1,
                    "target_agent_id": -1,
                    "infection_context": infection_context,
                    "event_day": int(event_day),
                    "weight_hint": 1,
                })


def check_and_record_die_out(current_time: float, E_total: float, I_total: float) -> bool:
    """Checks if the disease has died out and records the event if so."""
    return (E_total + I_total) < 1 and current_time > 1