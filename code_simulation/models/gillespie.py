"""Gillespie stochastic simulation algorithm for SEIR."""
from typing import Dict, Any, List
import numpy as np


def _infer_metapop_infection_source(
    n_patches: int,
    I_ij: np.ndarray,
    target_idx: int,
    is_day: bool,
    rng: np.random.Generator,
) -> Dict[str, int]:
    """Infer a likely source/target patch pair for one infection event."""
    target_home_patch = int(target_idx // n_patches)
    target_current_patch = int(target_idx % n_patches)
    source_patch_id = target_home_patch
    infection_context = "day" if is_day else "night"
    
    if is_day:
        infected_in_current_patch = I_ij[:, target_current_patch].astype(float)
        total = infected_in_current_patch.sum()
        if total > 0:
            probs = infected_in_current_patch / total
            source_patch_id = int(rng.choice(np.arange(n_patches), p=probs))
        else:
            source_patch_id = target_current_patch
    else:
        source_patch_id = target_home_patch
    
    return {
        "source_patch_id": source_patch_id,
        "target_patch_id": target_current_patch,
        "source_current_patch_id": target_current_patch,
        "target_current_patch_id": target_current_patch,
        "infection_context": infection_context,
    }


class GillespieEngine:
    """Gillespie SSA implementation for stochastic epidemic simulation."""
    
    def __init__(self, n_patches: int, cfg: Any):
        self.n_patches = n_patches
        self.cfg = cfg
    
    def step(self, S_t: np.ndarray, E_t: np.ndarray, I_t: np.ndarray, 
             R_t: np.ndarray, rng: np.random.Generator,
             track_provenance: bool = False) -> Dict[str, Any]:
        """Perform one Gillespie step."""
        alpha = 1.0 / getattr(self.cfg, 'period_incubation', 4.0)
        gamma = 1.0 / getattr(self.cfg, 'period_infectious', 5.0)
        beta = getattr(self.cfg, 'beta', 0.2)
        
        S_ij = S_t.reshape((self.n_patches, self.n_patches))
        E_ij = E_t.reshape((self.n_patches, self.n_patches))
        I_ij = I_t.reshape((self.n_patches, self.n_patches))
        
        progression_rates = alpha * E_ij
        recovery_rates = gamma * I_ij
        
        is_day = (getattr(self.cfg, 'current_time', 0) % 1.0) == 0.0
        N_ij = (S_ij + E_ij + I_ij + R_t.reshape((self.n_patches, self.n_patches)))
        
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
            return {
                "tau": getattr(self.cfg, 'n_days', 250),
                "S_new": S_t, "E_new": E_t, "I_new": I_t, "R_new": R_t,
                "N_new": S_t + E_t + I_t + R_t,
                "new_exp": 0, "new_inf": 0, "provenance_records": []
            }
        
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
                provenance_records.append(_infer_metapop_infection_source(self.n_patches, I_ij, int(event_idx), is_day, rng))
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
        
        return {
            "tau": tau,
            "S_new": S_new, "E_new": E_new, "I_new": I_new, "R_new": R_new,
            "N_new": N_new,
            "new_exp": new_exp, "new_inf": new_inf,
            "provenance_records": provenance_records
        }