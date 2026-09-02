"""Tau-leaping algorithm for stochastic epidemic simulation."""
from typing import Dict, Any
import numpy as np


class TauLeapEngine:
    """Tau-leaping implementation for faster stochastic simulation with larger populations."""
    
    def __init__(self, n_patches: int, cfg: Any):
        self.n_patches = n_patches
        self.cfg = cfg
    
    def step(self, S_t: np.ndarray, E_t: np.ndarray, I_t: np.ndarray,
             R_t: np.ndarray, N_t: np.ndarray, TOD: str,
             rng: np.random.Generator, track_provenance: bool = False) -> Dict[str, Any]:
        """Perform one tau-leaping step."""
        from .seir import compute_SEIR
        
        alpha = 1.0 / getattr(self.cfg, 'period_incubation', 4.0)
        gamma = 1.0 / getattr(self.cfg, 'period_infectious', 5.0)
        beta = getattr(self.cfg, 'beta', 0.2)
        
        return compute_SEIR(
            beta, alpha, gamma, self.n_patches,
            S_t, E_t, I_t, R_t, N_t, TOD, rng,
            track_provenance=track_provenance
        )