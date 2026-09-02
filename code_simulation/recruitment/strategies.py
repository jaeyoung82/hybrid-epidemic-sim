"""Recruitment strategies for event attendees."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class RecruitmentStrategy(ABC):
    """Abstract base class for attendee recruitment strategies."""
    
    @abstractmethod
    def calculate_proportions(self, n_patches: int, population_df: pd.DataFrame, **kwargs) -> np.ndarray:
        """Calculate recruitment proportions for each patch."""
        pass


class PopulationStrategy(RecruitmentStrategy):
    """Recruit attendees based on population proportion."""
    
    def calculate_proportions(self, n_patches: int, population_df: pd.DataFrame, **kwargs) -> np.ndarray:
        populations = population_df['population'].to_numpy()
        return populations / populations.sum()


class GravityStrategy(RecruitmentStrategy):
    """Recruit attendees using gravity model based on population and distance."""
    
    def calculate_proportions(self, n_patches: int, population_df: pd.DataFrame, **kwargs) -> np.ndarray:
        event_patch_id = kwargs['event_patch_id']
        distance_matrix = kwargs['distance_matrix']
        alpha = kwargs.get('gravity_model_distance_decay_exponent', 1.5)
        min_dist_floor = kwargs.get('gravity_model_min_dist_km', 0.5)
        self_dist_factor = kwargs.get('gravity_model_self_dist_factor', 0.5)
        
        populations = population_df['population'].to_numpy()
        distances_to_venue = distance_matrix[:, event_patch_id].copy()
        
        non_zero_dists = distances_to_venue[distances_to_venue > 1e-6]
        if len(non_zero_dists) > 0:
            self_dist = max(np.min(non_zero_dists) * self_dist_factor, min_dist_floor)
        else:
            self_dist = 1.0
        distances_to_venue[distances_to_venue <= 1e-6] = self_dist
        
        gravity_scores = populations / (distances_to_venue ** alpha)
        return gravity_scores / gravity_scores.sum()


class SinglePatchStrategy(RecruitmentStrategy):
    """Recruit attendees from a single specified patch."""
    
    def calculate_proportions(self, n_patches: int, population_df: pd.DataFrame, **kwargs) -> np.ndarray:
        target_patch = kwargs.get('single_patch_source_id')
        if target_patch is None:
            target_patch = kwargs.get('event_patch_id', 0)
        
        proportions = np.zeros(n_patches)
        if 0 <= target_patch < n_patches:
            proportions[int(target_patch)] = 1.0
        return proportions


class CommuterStrategy(RecruitmentStrategy):
    """Recruit attendees based on commuter flows to the event venue."""
    
    def calculate_proportions(self, n_patches: int, population_df: pd.DataFrame, **kwargs) -> np.ndarray:
        event_patch_id = kwargs['event_patch_id']
        flow_ij_df = kwargs['flow_ij_df']
        
        inflows = flow_ij_df[flow_ij_df['patch_j'] == event_patch_id]
        total_inflow = inflows['N_ij'].sum()
        
        proportions = pd.Series(0.0, index=range(n_patches))
        if total_inflow > 0:
            inflow_proportions = inflows.set_index('patch_i')['N_ij'] / total_inflow
            proportions.update(inflow_proportions)
        
        return proportions.to_numpy()


def get_strategy(name: str) -> RecruitmentStrategy:
    """Get recruitment strategy by name."""
    strategies = {
        'population': PopulationStrategy,
        'gravity': GravityStrategy,
        'single_patch': SinglePatchStrategy,
        'commuter': CommuterStrategy,
    }
    strategy_class = strategies.get(name)
    if not strategy_class:
        raise ValueError(f"Unknown recruitment strategy: '{name}'")
    return strategy_class()