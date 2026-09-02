"""Typed accessors for survival analysis datasets."""
from typing import List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class PressureVariables:
    """Named pressure covariates for time-dependent survival analysis."""
    Z_seed: float
    log_Z_seed: float
    phi: float
    Phi: float
    phi_EI: float = 0.0
    lagged_phi: float = 0.0
    lagged_phi_EI: float = 0.0
    P_w1: float = 0.0
    P_w3: float = 0.0
    P_w5: float = 0.0
    P_w7: float = 0.0
    P_w14: float = 0.0
    Mobility_in: int = 0
    Mobility_out: int = 0
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class SurvivalDataset:
    """
    Typed wrapper for survival analysis data.
    
    Provides type-safe accessors for pressure variables and covariates.
    """
    
    PRESSURE_COLS = [
        'Z_seed', 'log_Z_seed', 'phi', 'Phi', 'phi_EI', 
        'lagged_phi', 'lagged_phi_EI', 'P_w1', 'P_w3', 'P_w5', 
        'P_w7', 'P_w14', 'Mobility_in', 'Mobility_out'
    ]
    
    REQUIRED_COLS = ['start', 'stop', 'event', 'id', 'scenario', 'district']
    
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._validate()
    
    def _validate(self):
        """Validate that required columns exist."""
        missing = [c for c in self.REQUIRED_COLS if c not in self._df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    @property
    def df(self) -> pd.DataFrame:
        """Access underlying DataFrame."""
        return self._df
    
    def get_pressure(self, variable: str) -> np.ndarray:
        """Get a specific pressure variable as array."""
        if variable not in self.PRESSURE_COLS:
            raise ValueError(f"Unknown pressure variable: {variable}")
        return self._df[variable].to_numpy()
    
    def get_standardized(self, variable: str) -> np.ndarray:
        """Get standardized version of a variable (z_ prefixed)."""
        z_var = f'z_{variable}'
        if z_var in self._df.columns:
            return self._df[z_var].to_numpy()
        
        mean_val = self._df[variable].mean()
        std_val = self._df[variable].std()
        if std_val > 0:
            return ((self._df[variable] - mean_val) / std_val).to_numpy()
        return np.zeros(len(self._df))
    
    def filter_by_scenario(self, scenario: str) -> 'SurvivalDataset':
        """Filter to a specific scenario."""
        return SurvivalDataset(self._df[self._df['scenario'] == scenario].copy())
    
    def filter_by_time_window(self, t_min: float, t_max: float) -> 'SurvivalDataset':
        """Filter to a specific time window."""
        return SurvivalDataset(
            self._df[(self._df['start'] >= t_min) & (self._df['stop'] <= t_max)].copy()
        )
    
    def get_event_times(self) -> np.ndarray:
        """Get event times (uncensored)."""
        return self._df[self._df['event'] == 1]['stop'].to_numpy()
    
    def get_censoring_times(self) -> np.ndarray:
        """Get censoring times (censored)."""
        return self._df[self._df['event'] == 0]['stop'].to_numpy()