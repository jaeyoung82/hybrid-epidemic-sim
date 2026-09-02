"""Configuration module with validated models for epidemic simulation."""
from .models import (
    EventScenarioConfig,
    SimulationConfig,
    RecruitmentStrategy,
    SimulationMode,
    SIMULATION_MODES,
    DEFAULT_EVENT_CONFIGURATIONS,
    create_default_config,
    load_config_from_json,
    save_config_to_json,
)
from .models import SimulationConfig as Config

__all__ = [
    'EventScenarioConfig',
    'SimulationConfig',
    'RecruitmentStrategy',
    'SimulationMode',
    'SIMULATION_MODES',
    'DEFAULT_EVENT_CONFIGURATIONS',
    'create_default_config',
    'load_config_from_json',
    'save_config_to_json',
    'Config',
]
