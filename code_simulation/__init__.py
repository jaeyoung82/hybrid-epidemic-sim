"""
Simulation package for the Hybrid Epidemic Simulation Model.
"""

# Import order is structured to avoid circular imports:
#   models (leaf) -> arrival (leaf) -> recruitment (leaf) -> network -> data -> core -> events -> runner

from .models import (
    compute_lambda,
    compute_SEIR,
    compute_rt_from_state,
    compute_structural_ngm,
)
from .models.gillespie import GillespieEngine
from .models.tau_leap import TauLeapEngine

from .arrival import (
    apply_arrival_records,
    apply_arrival_records_I,
    check_and_record_die_out,
)
from .recruitment.strategies import (
    RecruitmentStrategy,
    PopulationStrategy,
    GravityStrategy,
    SinglePatchStrategy,
    CommuterStrategy,
    get_strategy,
)
from .recruitment.recruitment import (
    recruit_attendees,
    recruit_attendees_for_multi_venue_event,
    reintegrate_attendees,
    _remove_attendees_from_population,
)
from .network import (
    analyze_event_percolation_risk,
    analyze_event_reachability_from_all_seeds,
    extract_gcc_structure,
)
from .core import (
    analyze_metapopulation_percolation_risk,
    compute_percolation_metrics,
    compute_shortest_effective_paths,
    precompute_event_outcomes,
    _seed_initial_infections,
)
from .events import (
    InfectionLogger,
    simulate_event_transmission,
    _run_metapopulation_step,
    _handle_event_day,
    _run_large_venue_event,
    _run_multi_venue_event,
    _run_gillespie_step,
    _recruit_and_transmit_large_venue,
)
from .runner import run_simulation

__all__ = [
    'run_simulation',
    'InfectionLogger',
    'compute_lambda',
    'compute_SEIR',
    'compute_rt_from_state',
    'analyze_metapopulation_percolation_risk',
    'compute_structural_ngm',
    'compute_percolation_metrics',
    'compute_shortest_effective_paths',
    'precompute_event_outcomes',
    'apply_arrival_records',
    'apply_arrival_records_I',
    'check_and_record_die_out',
    'RecruitmentStrategy',
    'PopulationStrategy',
    'GravityStrategy',
    'SinglePatchStrategy',
    'CommuterStrategy',
    'get_strategy',
    'GillespieEngine',
    'TauLeapEngine',
    'recruit_attendees',
    'recruit_attendees_for_multi_venue_event',
    'reintegrate_attendees',
    'simulate_event_transmission',
    '_remove_attendees_from_population',
    '_run_metapopulation_step',
    '_handle_event_day',
    '_run_large_venue_event',
    '_run_multi_venue_event',
    '_run_gillespie_step',
    '_recruit_and_transmit_large_venue',
    'analyze_event_percolation_risk',
    'analyze_event_reachability_from_all_seeds',
    'extract_gcc_structure',
]
