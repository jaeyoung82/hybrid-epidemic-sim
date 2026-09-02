"""Validated configuration models using pydantic for epidemic simulation.

This module provides a single source of truth for simulation configuration,
consolidating all parameters from the legacy ``config.py`` module-level
attributes into a validated, mutable pydantic model.

Backward-compatible property aliases (``r0``, ``i_ss``, ``tau_leaping_threshold``)
are provided so that both legacy naming (``cfg.R_0``) and new naming
(``cfg.r0``) work interchangeably.
"""
from dataclasses import dataclass, field as dc_field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

try:
    from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

    def Field(default=None, default_factory=None, **kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    def ConfigDict(**kwargs):
        return kwargs

    def _passthrough_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    field_validator = model_validator = _passthrough_validator

    class BaseModel:
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)
        model_config = None
        def model_copy(self, **kwargs):
            import copy
            return copy.copy(self)


class RecruitmentStrategy(str, Enum):
    """Available recruitment strategies for event attendees."""
    POPULATION = "population"
    GRAVITY = "gravity"
    SINGLE_PATCH = "single_patch"
    COMMUTER = "commuter"


class SimulationMode(str, Enum):
    """Available simulation event modes."""
    LARGE_VENUE = "large_venue"
    MULTI_VENUE = "multi_venue"
    NO_EVENT = "no_event"
    NO_EVENT_DISTRIBUTED = "no_event_distributed"


# Mapping from legacy event model type strings to SimulationMode
SIMULATION_MODES = {
    "large_venue": SimulationMode.LARGE_VENUE,
    "multi_venue": SimulationMode.MULTI_VENUE,
    "no_event": SimulationMode.NO_EVENT,
    "no_event_distributed": SimulationMode.NO_EVENT_DISTRIBUTED,
}


# Default event scenario configurations (mirrors legacy config.event_configurations)
DEFAULT_EVENT_CONFIGURATIONS: Dict[str, Dict[str, Any]] = {
    "N12000_T36000": {
        "patch_id": 0,
        "total_attendees": 12000,
        "contact_file": "data_crowd-contacts/contacts_aggregated_N12000_T36000_0.csv",
        "contact_fps": 5,
        "initially_infected": 1,
        "ini_infected_attendee_seed": 'single_patch',
        "ini_infected_attendee_patch": 0,
        "contact_header_map": None,
        "simulation_modes": ["large_venue", "no_event"],
        "ids_are_strings": False,
    },
    "N2000_T36000": {
        "patch_id": 0,
        "total_attendees": 2000,
        "contact_file": "data_crowd-contacts/contacts_aggregated_N2000_T36000_0.csv",
        "contact_fps": 5,
        "initially_infected": 1,
        "ini_infected_attendee_seed": 'single_patch',
        "ini_infected_attendee_patch": 0,
        "contact_header_map": None,
        "simulation_modes": ["large_venue", "no_event"],
        "ids_are_strings": False,
    },
    "AMS_dance": {
        "patch_id": 0,
        "total_attendees": 1048,
        "contact_file": "data_crowd-contacts/contacts_aggregated_AMS_dance.csv",
        "contact_fps": 1,
        "initially_infected": 1,
        "ini_infected_attendee_seed": 'single_patch',
        "ini_infected_attendee_patch": 0,
        "contact_header_map": {"agent_i": "ped1_id", "agent_j": "ped2_id", "tc_s": "total_duration_frames"},
        "simulation_modes": ["large_venue", "no_event"],
        "ids_are_strings": False,
    },
    "AMS_football": {
        "patch_id": 0,
        "total_attendees": 362,
        "contact_file": "data_crowd-contacts/contacts_aggregated_AMS_football.csv",
        "contact_fps": 1,
        "initially_infected": 1,
        "ini_infected_attendee_seed": 'single_patch',
        "ini_infected_attendee_patch": 0,
        "contact_header_map": {"agent_i": "ped1_id", "agent_j": "ped2_id", "tc_s": "total_duration_frames"},
        "simulation_modes": ["large_venue", "no_event"],
        "ids_are_strings": False,
    },
    "Leipzig_1": {
        "patch_id": 0,
        "total_attendees": 1194,
        "contact_file": "data_crowd-contacts/contacts_aggregated_Leipzig_1.csv",
        "contact_fps": 1,
        "initially_infected": 1,
        "ini_infected_attendee_seed": 'single_patch',
        "ini_infected_attendee_patch": 0,
        "contact_header_map": {"agent_i": "ped1_id", "agent_j": "ped2_id", "tc_s": "total_duration_frames"},
        "simulation_modes": ["large_venue", "no_event"],
        "ids_are_strings": False,
    },
    "Leipzig_2": {
        "patch_id": 0,
        "total_attendees": 1158,
        "contact_file": "data_crowd-contacts/contacts_aggregated_Leipzig_2.csv",
        "contact_fps": 1,
        "initially_infected": 1,
        "ini_infected_attendee_seed": 'single_patch',
        "ini_infected_attendee_patch": 0,
        "contact_header_map": {"agent_i": "ped1_id", "agent_j": "ped2_id", "tc_s": "total_duration_frames"},
        "simulation_modes": ["large_venue", "no_event"],
        "ids_are_strings": False,
    },
    "Leipzig_3": {
        "patch_id": 0,
        "total_attendees": 1054,
        "contact_file": "data_crowd-contacts/contacts_aggregated_Leipzig_3.csv",
        "contact_fps": 1,
        "initially_infected": 1,
        "ini_infected_attendee_seed": 'single_patch',
        "ini_infected_attendee_patch": 0,
        "contact_header_map": {"agent_i": "ped1_id", "agent_j": "ped2_id", "tc_s": "total_duration_frames"},
        "simulation_modes": ["large_venue", "no_event"],
        "ids_are_strings": False,
    },
}


@dataclass
class EventScenarioConfig:
    """Configuration for a single event scenario."""
    patch_id: int = 0
    total_attendees: int = 2000
    contact_file: str = ""
    contact_fps: int = 5
    initially_infected: int = 1
    ini_infected_attendee_seed: str = "single_patch"
    ini_infected_attendee_patch: int = 0
    contact_header_map: Optional[Dict[str, str]] = None
    simulation_modes: List[str] = dc_field(default_factory=lambda: ["large_venue", "no_event"])
    ids_are_strings: bool = False


class SimulationConfig(BaseModel):
    """
    Centralized, validated configuration for the epidemic simulation.

    Supports both legacy uppercase attributes (``R_0``, ``I_ss``,
    ``TauLeaping_threshold``) and lowercase aliases (``r0``, ``i_ss``,
    ``tau_leaping_threshold``) via read-only properties.

    The model is mutable (pydantic v2 default) and allows extra attributes
    (``model_config = ConfigDict(extra='allow')``) so that run scripts can
    set dynamic values like ``cfg.current_day`` and ``cfg.current_time``.
    """

    if HAS_PYDANTIC:
        model_config = ConfigDict(extra='allow', populate_by_name=True)

    # ------------------------------------------------------------------
    # Simulation setup
    # ------------------------------------------------------------------
    n_iterations: int = Field(default=1000, ge=1)
    n_days: int = Field(default=250, ge=1)
    population_factor: float = Field(default=1.0, gt=0)
    write_full_results: bool = False
    plot_on_the_fly: bool = False
    track_invasion_provenance: bool = True
    save_arrival_time_records: bool = True
    save_invasion_edge_records: bool = True
    save_invasion_tree_records: bool = True
    invasion_tree_method: str = "arborescence"
    TauLeaping_threshold: int = Field(default=100, ge=1)
    # snake_case alias used by code_simulation.core
    tau_leaping_threshold: int = Field(default=100, ge=1)

    # ------------------------------------------------------------------
    # Epidemiological parameters
    # ------------------------------------------------------------------
    period_incubation: float = Field(default=4.0, gt=0)
    period_infectious: float = Field(default=5.0, gt=0)

    # R_0 (basic reproduction number) — primary field name
    R_0: float = Field(default=1.5, ge=0.1, le=10.0)

    @property
    def r0(self) -> float:
        """Lowercase alias for :attr:`R_0`."""
        return self.R_0

    @property
    def get_beta(self) -> float:
        """Compute beta from R0 and infectious period."""
        return self.R_0 * (1.0 / self.period_infectious)

    # beta is a mutable field — run scripts set it explicitly during sweeps
    beta: float = Field(default=None)  # type: ignore

    @model_validator(mode='after')
    def _compute_beta(self):
        if self.beta is None:
            object.__setattr__(self, 'beta', self.R_0 * (1.0 / self.period_infectious))
        return self

    # ------------------------------------------------------------------
    # Large-Gathering Event Parameters
    # ------------------------------------------------------------------
    event_base_transmission_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    event_base_transmission_rate_values: List[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])


    attendee_recruitment_model: str = "gravity"
    single_patch_recruitment_id: int = 0
    gravity_model_distance_decay_exponent: float = Field(default=1.5)
    gravity_model_min_dist_km: float = Field(default=0.5)
    gravity_model_self_dist_factor: float = Field(default=0.5)
    event_contact_fps: int = 5
    event_contact_header_map: Optional[Dict[str, str]] = None

    # ------------------------------------------------------------------
    # Event configurations
    # ------------------------------------------------------------------
    event_configurations: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_EVENT_CONFIGURATIONS.items()})
    event_simulation_modes: List[str] = Field(default_factory=lambda: ["large_venue", "multi_venue", "no_event"])
    large_venue_ids_are_strings: bool = False

    # Large venue specific
    large_venue_patch_id: int = 0
    large_venue_total_attendees: int = 12000
    large_venue_contact_file: str = ""
    large_venue_initially_infected: int = 1
    large_venue_ini_infected_attendee_seed: str = "single_patch"
    large_venue_ini_infected_attendee_patch: int = 0

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------
    initial_infection_patch_id: int = 0
    event_venue_patch_id: int = 0
    ini_infected_attendee_patch: int = 0
    no_event_seeding_mode: str = "single_patch"
    seeding_method: str = "import"

    # I_ss (initial seed size) — primary field name
    I_ss: int = Field(default=1, ge=0)

    @property
    def i_ss(self) -> int:
        """Lowercase alias for :attr:`I_ss`."""
        return self.I_ss

    # Seed districts
    seed_patch_ids: List[int] = Field(default_factory=lambda: list(range(21)))

    # ------------------------------------------------------------------
    # Multi-venue parameters
    # ------------------------------------------------------------------
    multi_venue_group_venues: Dict[str, int] = Field(default_factory=lambda: {'5': 0})
    multi_venue_total_attendees: int = 2000
    districts_to_visualize_attendees: List[int] = Field(default_factory=lambda: [5])

    # ------------------------------------------------------------------
    # Parameter sweep
    # ------------------------------------------------------------------
    parameter_sweep_config: Dict = Field(default_factory=lambda: {
        1: {'R0_start': 1.0, 'R0_end': 3.0, 'R0_step': 0.5},
    })
    rnd_seed_0: int = 1234567
    k_assumed: int = 10

    # ------------------------------------------------------------------
    # Scenario selections
    # ------------------------------------------------------------------
    current_event_scenario_names: List[str] = Field(default_factory=lambda: ["AMS_dance"])
    empirical_scenario_set_1: List[str] = Field(default_factory=lambda: ["AMS_dance", "AMS_football"])
    empirical_scenario_set_2: List[str] = Field(default_factory=lambda: ["Leipzig_1", "Leipzig_2", "Leipzig_3"])

    # ------------------------------------------------------------------
    # Plot configuration
    # ------------------------------------------------------------------
    custom_spatial_comparison_scenarios: List[str] = Field(default_factory=lambda: ["no_event", "AMS_dance", "AMS_football", "Leipzig_1", "Leipzig_2", "Leipzig_3"])
    custom_spatial_comparison_days: List[int] = Field(default_factory=lambda: [100, 150, 175, 200])
    custom_spatial_comparison_titles: Dict[str, str] = Field(default_factory=lambda: {
        "no_event": "Baseline",
        "AMS_dance": "AMS_dance",
        "AMS_football": "AMS_football",
        "Leipzig_1": "Leipzig_1",
        "Leipzig_2": "Leipzig_2",
        "Leipzig_3": "Leipzig_3",
    })
    arrival_time_comparison_scenarios: List[str] = Field(default_factory=lambda: ["no_event", "AMS_dance", "AMS_football", "Leipzig_1", "Leipzig_2", "Leipzig_3"])
    arrival_time_comparison_titles: Dict[str, str] = Field(default_factory=lambda: {
        "no_event": "Baseline",
        "AMS_dance": "AMS_dance",
        "AMS_football": "AMS_football",
        "Leipzig_1": "Leipzig_1",
        "Leipzig_2": "Leipzig_2",
        "Leipzig_3": "Leipzig_3",
    })
    target_days_for_active_case_distribution: List[int] = Field(default_factory=lambda: [250])

    # ------------------------------------------------------------------
    # Dataset configuration
    # ------------------------------------------------------------------
    dataset_name: str = "Madrid"
    dataset_folder_name: str = "data_mobility_Madrid/"
    filename_population: Path = Field(default_factory=lambda: Path("data_mobility_Madrid/population_Madrid.csv"))
    filename_flow_matrix: Path = Field(default_factory=lambda: Path("data_mobility_Madrid/mobility_matrix_Madrid.csv"))
    event_days: List[int] = Field(default_factory=lambda: [0])
    results_dir: Path = Field(default_factory=lambda: Path("results_hybrid_sim"))

    # ------------------------------------------------------------------
    # Event scenarios (typed)
    # ------------------------------------------------------------------
    event_scenarios: Dict[str, EventScenarioConfig] = Field(default_factory=dict)
    current_event_scenario_name: str = "no_event"

    # ------------------------------------------------------------------
    # Runtime state (mutated during simulation)
    # ------------------------------------------------------------------
    event_model_type: str = "no_event"
    current_day: int = 0
    current_time: float = 0.0

    # ------------------------------------------------------------------
    # Pydantic validators
    # ------------------------------------------------------------------
    @field_validator('results_dir', mode='before')
    @classmethod
    def ensure_path(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------
    def get_beta(self) -> float:
        """Get the current beta value."""
        return self.R_0 * (1.0 / self.period_infectious)

    def set_scenario(self, scenario_name: str) -> None:
        """Update configuration for a specific event scenario."""
        if scenario_name not in self.event_scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found in event_scenarios.")

        config = self.event_scenarios[scenario_name]
        self.current_event_scenario_name = scenario_name
        self.large_venue_patch_id = config.patch_id
        self.large_venue_total_attendees = config.total_attendees
        self.large_venue_contact_file = config.contact_file
        self.event_contact_fps = config.contact_fps
        self.large_venue_initially_infected = config.initially_infected
        self.large_venue_ini_infected_attendee_seed = config.ini_infected_attendee_seed
        self.large_venue_ini_infected_attendee_patch = config.ini_infected_attendee_patch
        self.event_contact_header_map = config.contact_header_map
        self.large_venue_ids_are_strings = config.ids_are_strings

    def set_active_scenario(self, scenario_name: str) -> None:
        """Update configuration from event_configurations dict for the selected scenario.

        This is the legacy-compatible method used by run_hybrid_simulation.py.
        It does NOT set initial_infection_patch_id, ini_infected_attendee_patch,
        beta, R_0, or I_ss — those are managed by the sweep loop.
        """
        if scenario_name not in self.event_configurations:
            raise ValueError(f"Scenario '{scenario_name}' not found in event_configurations.")

        selected_config = self.event_configurations[scenario_name]

        self.current_event_scenario_name = scenario_name
        self.large_venue_patch_id = selected_config.get("patch_id", 0)
        self.large_venue_total_attendees = selected_config.get("total_attendees", 12000)
        self.large_venue_initially_infected = selected_config.get("initially_infected", 1)
        self.large_venue_contact_file = selected_config.get("contact_file", "")
        self.event_contact_fps = selected_config.get("contact_fps", 5)
        self.large_venue_ini_infected_attendee_seed = selected_config.get("ini_infected_attendee_seed", "single_patch")
        self.large_venue_ini_infected_attendee_patch = selected_config.get("ini_infected_attendee_patch", 0)
        self.event_contact_header_map = selected_config.get("contact_header_map", None)
        self.event_simulation_modes = selected_config.get("simulation_modes", ["large_venue", "multi_venue", "no_event"])
        self.large_venue_ids_are_strings = selected_config.get("ids_are_strings", False)

    def model_copy(self, *, update: Optional[Dict[str, Any]] = None) -> 'SimulationConfig':
        """Create a mutable copy of this configuration."""
        import copy
        new = copy.copy(self)
        if update:
            for k, v in update.items():
                setattr(new, k, v)
        return new


def create_default_config() -> SimulationConfig:
    """Create a simulation configuration with default values.

    Populates ``event_scenarios`` from ``event_configurations`` and applies
    the first entry in ``current_event_scenario_names`` via
    :meth:`SimulationConfig.set_active_scenario`.
    """
    config = SimulationConfig()

    # Build typed EventScenarioConfig objects from the raw dicts
    for name, sc_dict in DEFAULT_EVENT_CONFIGURATIONS.items():
        config.event_scenarios[name] = EventScenarioConfig(
            patch_id=sc_dict.get("patch_id", 0),
            total_attendees=sc_dict.get("total_attendees", 2000),
            contact_file=sc_dict.get("contact_file", ""),
            contact_fps=sc_dict.get("contact_fps", 5),
            initially_infected=sc_dict.get("initially_infected", 1),
            ini_infected_attendee_seed=sc_dict.get("ini_infected_attendee_seed", "single_patch"),
            ini_infected_attendee_patch=sc_dict.get("ini_infected_attendee_patch", 0),
            contact_header_map=sc_dict.get("contact_header_map", None),
            simulation_modes=sc_dict.get("simulation_modes", ["large_venue", "no_event"]),
            ids_are_strings=sc_dict.get("ids_are_strings", False),
        )

    # Apply the first scenario as the default active scenario
    if config.current_event_scenario_names:
        config.set_active_scenario(config.current_event_scenario_names[0])

    return config


def load_config_from_json(path: str | Path) -> 'SimulationConfig':
    """Load SimulationConfig from JSON, overriding default values.

    Precedence: create_default_config() -> JSON overrides (setattr).
    Skips: _-prefixed keys, event_configurations, event_scenarios.
    """
    import json

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cfg = create_default_config()

    # Edge case A/G: detect Path-typed fields for manual str->Path conversion
    if HAS_PYDANTIC:
        field_annotations = {k: fi.annotation for k, fi in cfg.model_fields.items()}
    else:
        # Edge case G: hardcoded for non-pydantic fallback
        field_annotations = {'results_dir': Path, 'filename_population': Path,
                             'filename_flow_matrix': Path}

    for key, value in data.items():
        if key.startswith('_'):
            continue  # edge case: comment keys
        if key in ('event_configurations', 'event_scenarios'):
            continue  # edge case E
        if isinstance(value, str) and field_annotations.get(key) is Path:
            value = Path(value)  # edge case A
        if key == 'parameter_sweep_config' and isinstance(value, dict):
            value = {int(k) if k.lstrip('-').isdigit() else k: v
                     for k, v in value.items()}  # edge case D
        setattr(cfg, key, value)

    # Edge case B/C: only re-apply scenario if name is valid
    if 'current_event_scenario_name' in data:
        name = data['current_event_scenario_name']
        if name in cfg.event_configurations:
            cfg.set_active_scenario(name)

    return cfg


def save_config_to_json(cfg: 'SimulationConfig', path: str | Path) -> None:
    """Serialize SimulationConfig to JSON for inspection/export."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if HAS_PYDANTIC:
        raw = cfg.model_dump(mode='json')
    else:
        raw = {k: v for k, v in cfg.__dict__.items()
               if not k.startswith('_')}

    # Path -> str for JSON readability (pydantic mode='json' does this;
    # non-pydantic fallback needs manual handling)
    if not HAS_PYDANTIC:
        raw = {k: str(v) if isinstance(v, Path) else v for k, v in raw.items()}

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(raw, f, indent=2, default=str)


# Re-export EventScenarioConfig for backward compatibility
__all__ = [
    'RecruitmentStrategy',
    'SimulationMode',
    'SIMULATION_MODES',
    'EventScenarioConfig',
    'SimulationConfig',
    'create_default_config',
    'load_config_from_json',
    'save_config_to_json',
    'DEFAULT_EVENT_CONFIGURATIONS',
]
