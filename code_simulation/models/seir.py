"""SEIR model implementation."""
from typing import Dict, Any, List
import numpy as np

from .gillespie import _infer_metapop_infection_source


def compute_lambda(beta: float, dt: float, n_patches: int, 
                   I_t: np.ndarray, N_t: np.ndarray, is_day: bool) -> np.ndarray:
    """Compute the force of infection lambda for each patch."""
    I_ij = I_t.reshape((n_patches, n_patches))
    N_ij = N_t.reshape((n_patches, n_patches))
    
    if is_day:
        sum_Iji = I_ij.sum(axis=0)
        sum_Nji = N_ij.sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            lambda_val = (dt * beta) * sum_Iji / sum_Nji
        prob_infection = np.nan_to_num(lambda_val[np.newaxis, :])
        prob_infection = np.broadcast_to(prob_infection, (n_patches, n_patches)).ravel()
    else:
        sum_Iij = I_ij.sum(axis=1)
        sum_Nij = N_ij.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            lambda_val = (dt * beta) * sum_Iij / sum_Nij
        prob_infection = np.nan_to_num(lambda_val[:, np.newaxis])
        prob_infection = np.broadcast_to(prob_infection, (n_patches, n_patches)).ravel()
    
    return prob_infection


def compute_SEIR(beta: float, alpha: float, gamma: float, n_patches: int,
                 S_t: np.ndarray, E_t: np.ndarray, I_t: np.ndarray, R_t: np.ndarray,
                 N_t: np.ndarray, TOD: str, rng: np.random.Generator,
                 track_provenance: bool = False) -> Dict[str, Any]:
    """Perform one step of the metapopulation SEIR model."""
    is_day = (TOD == "day-time")
    dt = 0.5
    
    prob_infection = compute_lambda(beta, dt, n_patches, I_t, N_t, is_day)
    
    new_exposed = rng.binomial(S_t, prob_infection)
    new_infected = rng.binomial(E_t, dt * alpha)
    new_recovered = rng.binomial(I_t, dt * gamma)
    
    provenance_records: List[Dict[str, Any]] = []
    if track_provenance and np.any(new_exposed > 0):
        I_ij = I_t.reshape((n_patches, n_patches))
        exposed_indices = np.where(new_exposed > 0)[0]
        for idx in exposed_indices:
            for _ in range(int(new_exposed[idx])):
                provenance_records.append(
                    _infer_metapop_infection_source(n_patches, I_ij, int(idx), is_day, rng)
                )
    
    S_new = np.maximum(S_t - new_exposed, 0)
    E_new = np.maximum(E_t + new_exposed - new_infected, 0)
    I_new = np.maximum(I_t + new_infected - new_recovered, 0)
    R_new = np.maximum(R_t + new_recovered, 0)
    
    N_new = S_new + E_new + I_new + R_new
    new_exposed_all = np.sum(new_exposed)
    new_infected_all = np.sum(new_infected)
    
    return {
        "tau": dt,
        "S_new": S_new,
        "E_new": E_new,
        "I_new": I_new,
        "R_new": R_new,
        "N_new": N_new,
        "new_exp": new_exposed_all,
        "new_inf": new_infected_all,
        "provenance_records": provenance_records,
    }