"""Next Generation Matrix and reproduction number calculations."""
import numpy as np
from typing import Dict, Any


def compute_rt_from_state(S_ij_t: np.ndarray, N_ij: np.ndarray, 
                          beta: float, gamma: float, n_patches: int) -> float:
    """Compute the effective reproduction number Rt using NGM method."""
    S_in_j = S_ij_t.sum(axis=0)
    N_in_j = N_ij.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        sus_frac_day = np.nan_to_num(S_in_j / N_in_j)
    D_day = np.diag(sus_frac_day)
    
    S_at_home_i = np.diag(S_ij_t)
    N_at_home_i = np.diag(N_ij)
    with np.errstate(divide='ignore', invalid='ignore'):
        sus_frac_night = np.nan_to_num(S_at_home_i / N_at_home_i)
    D_night = np.diag(sus_frac_night)
    
    N_i = N_ij.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        P_ij = np.nan_to_num(N_ij / N_i[:, np.newaxis])
        M_uj = np.nan_to_num(N_ij / N_in_j[np.newaxis, :])
    
    T_day, T_night = 0.5, 0.5
    K_day_t = (T_day * beta) * (P_ij @ D_day @ M_uj.T)
    K_night_t = (T_night * beta) * D_night
    K_t = (1.0 / gamma) * (K_day_t + K_night_t)
    
    eigenvalues = np.linalg.eigvals(K_t)
    return np.max(np.abs(eigenvalues))


def compute_structural_ngm(N_ij: np.ndarray) -> np.ndarray:
    """Compute structural component of Next Generation Matrix (S_mat)."""
    n_patches = N_ij.shape[0]
    
    N_i = N_ij.sum(axis=1)
    N_j_day = N_ij.sum(axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        P_ij = np.nan_to_num(N_ij / N_i[:, np.newaxis])
        M_uj = np.nan_to_num(N_ij / N_j_day[np.newaxis, :])
    
    T_day, T_night = 0.5, 0.5
    
    S_day = T_day * (P_ij @ M_uj.T)
    S_night = np.zeros((n_patches, n_patches))
    np.fill_diagonal(S_night, T_night)
    
    return S_day + S_night