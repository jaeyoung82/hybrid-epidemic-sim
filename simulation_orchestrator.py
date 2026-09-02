"""
Hybrid Epidemic Simulation Model - facade module.

This module provides backward compatibility for imports like
``import simulation_orchestrator as simulation``.  All real functionality
now lives in the :mod:`code_simulation` package; this module simply
re-exports it.
"""
from code_simulation import events
from code_simulation.events import InfectionLogger

from code_simulation.core import (
    compute_lambda,
    compute_SEIR,
    compute_rt_from_state,
    analyze_metapopulation_percolation_risk,
    compute_structural_ngm,
    compute_percolation_metrics,
    compute_shortest_effective_paths,
    precompute_event_outcomes,
    _seed_initial_infections,
)
from code_simulation.runner import run_simulation
from code_simulation.arrival import (
    apply_arrival_records,
    apply_arrival_records_I,
    check_and_record_die_out,
)

__all__ = [
    'run_simulation',
    'events',
    'InfectionLogger',
    'compute_lambda',
    'compute_SEIR',
    'compute_rt_from_state',
    'analyze_metapopulation_percolation_risk',
    'compute_structural_ngm',
    'compute_percolation_metrics',
    'compute_shortest_effective_paths',
    'precompute_event_outcomes',
    '_seed_initial_infections',
    'apply_arrival_records',
    'apply_arrival_records_I',
    'check_and_record_die_out',
]
