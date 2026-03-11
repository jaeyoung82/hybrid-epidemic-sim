"""
Configuration file for the Hybrid Epidemic Simulation Model.
"""
from pathlib import Path
import numpy as np

### Simulation Setups
n_iterations = 100 # Number of iterations (default 100)
n_days = 250  # Simulation duration in days
population_factor = 1.0  # Factor to scale population and commuter flow, 1.0 for simulation, smaller than 1 for test
write_full_results = False  # Whether to write detailed subpopulation results for each run

# Threshold for switching between Gillespie (exact stochastic) and Tau-leaping (approximate stochastic) methods.
# If E+I < threshold, use Gillespie. Set to 0 to always use Tau-leaping.
# Set to a value larger than total population to force Gillespie
TauLeaping_threshold = 1000 

### Large-Gathering Event (LGE) Parameters
# --- Attendee Recruitment Model ---
# Options: 'population', 'commuter', 'gravity', 'single_patch'
# population: based on the population size of each district
# commuter: based on commuter traffic volume to the event venue district
# gravity: based on gravity model
# single_patch: recruits all attendees from a specific district defined by single_patch_recruitment_id
attendee_recruitment_model = 'gravity'  
single_patch_recruitment_id = 0 # Patch ID to recruit from if 'single_patch' is selected

# A list of days on which events occur.
event_days = [0]
"""
event_days = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 
              105, 112, 119, 126, 133, 140, 147, 154, 161, 168, 175, 182, 
              189, 196, 203, 210, 217, 224, 231, 238, 245]  # Example for a single event at day 0
"""

### Dataset configuration for metapopulation model 
dataset_name = "Madrid" # "Madrid", "NE-England"
dataset_folder_name = "mobility_"+dataset_name+"/"
filename_population = Path(dataset_folder_name+f"population_{dataset_name}.csv")
filename_flow_matrix = Path(dataset_folder_name+f"mobility_matrix_{dataset_name}.csv")

### Randomness
rnd_seed_0 = 1234567  # Base seed for the random number generator

### Epidemiological Parameters
period_incubation = 4.0  # Exposed period (days)
period_infectious = 5.0  # Infectious period (days)

# --- Parameter Sweep Configuration ---
# Defines the R0 ranges to test for each I_ss value (number of imported individuals).
# The simulation will iterate through each I_ss key and then through the R0 range defined for it.
parameter_sweep_config = {
    # I_ss value: {'R0_start': float, 'R0_end': float, 'R0_step': float}
    1: {'R0_start': 1.0, 'R0_end': 3.0, 'R0_step': 0.2},
    # 2:  {'R0_start': 1.0, 'R0_end': 3.0, 'R0_step': 0.5},
    # 4:  {'R0_start': 1.0, 'R0_end': 3.0, 'R0_step': 0.5},
    # 6:  {'R0_start': 1.0, 'R0_end': 3.0, 'R0_step': 0.5},
    # 8:  {'R0_start': 1.0, 'R0_end': 3.0, 'R0_step': 0.5},
    # 10: {'R0_start': 1.0, 'R0_end': 3.0, 'R0_step': 0.5},
}

# For components that need a default value before the main loop starts
# These will be overwritten in the main simulation loop.
I_ss = list(parameter_sweep_config.keys())[0]
_default_R0_params = parameter_sweep_config[I_ss]
if _default_R0_params['R0_step'] > 0:
    _default_R0_values = np.round(np.arange(_default_R0_params['R0_start'], _default_R0_params['R0_end'] + _default_R0_params['R0_step'], _default_R0_params['R0_step']), 6)
else:
    _default_R0_values = np.array([_default_R0_params['R0_start']])
R_0 = _default_R0_values[0]

k_assumed = 10  # Assumed average daily contacts for metapopulation model
gamma = 1.0 / period_infectious
beta = R_0 * gamma

# --- Fine-Grained Event Simulation Parameters ---
event_base_transmission_rate_values = [0.1, 0.2, 0.3, 0.4, 0.5] # 0.1, 0.3, 0.5, 1
# event_base_transmission_rate_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # List of event transmission probabilities to test
# event_base_transmission_rate_values = [0.1] # List of event transmission probabilities to test
event_base_transmission_rate = event_base_transmission_rate_values[0] # Default value, will be overwritten in the main loop

### Initial Conditions
initial_infection_patch_id = 0 # patch ID to seed initial infections for 'no_event' or background seeding.
event_venue_patch_id = 0  # patch ID for the event 
ini_infected_attendee_patch = 0

# --- 'no_event' Seeding Configuration ---
# 'single_patch': Seeds I_ss individuals in one patch (current baseline).
# 'from_file': Seeds infected individuals based on the large-venue output file (for direct comparison).
no_event_seeding_mode = 'single_patch'

# --- Seeding Method ---
# 'recruit': Converts existing susceptible individuals to infected (keeps N constant).
# 'import': Adds new infected individuals to the population (increases N).
seeding_method = 'import'

# --- Event Scenarios Configuration ---
# Define different large gathering event scenarios with their specific parameters.
# The 'current_event_scenario_name' variable will select which one is active.
#
# contact header map
# N12000_T36000 and N2000_T36000: specify as None
# AMS_dance and AMS_football: specify as {"agent_i": "ped1_id", "agent_j": "ped2_id", "tc_s": "total_duration_frames"}
# Leipzig_1, Leipzig_2, and Leipzig_3: specify as {"agent_i": "ped1_id", "agent_j": "ped2_id", "tc_s": "total_duration_frames"}
#
# NOTE: 'initially_infected' captures the value of I_ss at definition time. 
# If sweeping I_ss, ensure large_venue_initially_infected is updated dynamically in the main loop.
event_configurations = {
    "N12000_T36000": {
        "patch_id": event_venue_patch_id, 
        "total_attendees": 12000,
        "contact_file": "data_crowd-contacts/contacts_aggregated_N12000_T36000_0.csv",
        "contact_fps": 5,  # Frames per second for contact data recording
        "initially_infected": I_ss,  # Number of initially infected individuals seeded at the event
        "ini_infected_attendee_seed": 'single_patch',
        "ini_infected_attendee_patch": ini_infected_attendee_patch,  # Patch ID for seeding infected attendees
        "contact_header_map": None, # Example: {"agent_a": "ped1_id", "agent_b": "ped2_id", "frames": "total_duration_frames"}
        "simulation_modes": ["large_venue", "no_event"],
        "ids_are_strings": False,
    },
    "N2000_T36000": {
        "patch_id": event_venue_patch_id,
        "total_attendees": 2000,
        "contact_file": "data_crowd-contacts/contacts_aggregated_N2000_T36000_0.csv",
        "contact_fps": 5,
        "initially_infected": I_ss,
        "ini_infected_attendee_seed": 'single_patch',
        "ini_infected_attendee_patch": ini_infected_attendee_patch,
        "contact_header_map": None, # Example: {"agent_a": "ped1_id", "agent_b": "ped2_id", "frames": "total_duration_frames"}
        "simulation_modes": ["large_venue", "no_event"],
        "ids_are_strings": False,
    },
    "AMS_dance": {
        "patch_id": event_venue_patch_id,
        "total_attendees": 1048,
        "contact_file": "data_crowd-contacts/contacts_aggregated_AMS_dance.csv",
        "contact_fps": 1,
        "initially_infected": I_ss,
        "ini_infected_attendee_seed": 'single_patch',
        "ini_infected_attendee_patch": ini_infected_attendee_patch,
        "contact_header_map": {"agent_i": "ped1_id", "agent_j": "ped2_id", "tc_s": "total_duration_frames"},
        "simulation_modes": ["large_venue", "no_event"],
        "ids_are_strings": False,
    },
    "AMS_football": {
        "patch_id": event_venue_patch_id,
        "total_attendees": 362,
        "contact_file": "data_crowd-contacts/contacts_aggregated_AMS_football.csv",
        "contact_fps": 1,
        "initially_infected": I_ss,
        "ini_infected_attendee_seed": 'single_patch',
        "ini_infected_attendee_patch": ini_infected_attendee_patch,
        "contact_header_map": {"agent_i": "ped1_id", "agent_j": "ped2_id", "tc_s": "total_duration_frames"},
        "simulation_modes": ["large_venue", "no_event"],
        "ids_are_strings": False,
    },
    "Leipzig_1": {
        "patch_id": event_venue_patch_id,
        "total_attendees": 1194,
        "contact_file": "data_crowd-contacts/contacts_aggregated_Leipzig_1.csv",
        "contact_fps": 1,
        "initially_infected": I_ss,
        "ini_infected_attendee_seed": 'single_patch',
        "ini_infected_attendee_patch": ini_infected_attendee_patch,
        "contact_header_map": {"agent_i": "ped1_id", "agent_j": "ped2_id", "tc_s": "total_duration_frames"},
        "simulation_modes": ["large_venue", "no_event"],
        "ids_are_strings": False,
    },
    "Leipzig_2": {
        "patch_id": event_venue_patch_id,
        "total_attendees": 1158,
        "contact_file": "data_crowd-contacts/contacts_aggregated_Leipzig_2.csv",
        "contact_fps": 1,
        "initially_infected": I_ss,
        "ini_infected_attendee_seed": 'single_patch',
        "ini_infected_attendee_patch": ini_infected_attendee_patch,
        "contact_header_map": {"agent_i": "ped1_id", "agent_j": "ped2_id", "tc_s": "total_duration_frames"},
        "simulation_modes": ["large_venue", "no_event"],
        "ids_are_strings": False,
    },
    "Leipzig_3": {
        "patch_id": event_venue_patch_id,
        "total_attendees": 1054,
        "contact_file": "data_crowd-contacts/contacts_aggregated_Leipzig_3.csv",
        "contact_fps": 1,
        "initially_infected": I_ss,
        "ini_infected_attendee_seed": 'single_patch',
        "ini_infected_attendee_patch": ini_infected_attendee_patch,
        "contact_header_map": {"agent_i": "ped1_id", "agent_j": "ped2_id", "tc_s": "total_duration_frames"},
        "simulation_modes": ["large_venue", "no_event"],
        "ids_are_strings": False,
    },
}

# Select the active event scenario
empirical_scenario_set_1 = ["AMS_dance", "AMS_football"]
empirical_scenario_set_2 = ["Leipzig_1", "Leipzig_2", "Leipzig_3"]
# for instance, ["N12000_T36000", "AMS_dance", "Leipzig_1"]
"""
current_event_scenario_names = ["AMS_dance", "AMS_football",
                                "Leipzig_1", "Leipzig_2", "Leipzig_3", 
                                "N2000_T36000", "N12000_T36000"]  
"""
current_event_scenario_names = empirical_scenario_set_1+empirical_scenario_set_2


def set_active_scenario(cfg_module, scenario_name):
    """
    Updates the attributes of the cfg module with the parameters of the selected scenario.
    """
    if scenario_name not in cfg_module.event_configurations:
        raise ValueError(f"Scenario '{scenario_name}' not found in event_configurations.")

    cfg_module.current_event_scenario_name = scenario_name
    selected_config = cfg_module.event_configurations[scenario_name]

    # Update all relevant attributes in the cfg_module
    cfg_module.large_venue_patch_id = selected_config["patch_id"]
    cfg_module.large_venue_total_attendees = selected_config["total_attendees"]
    cfg_module.large_venue_initially_infected = selected_config["initially_infected"]
    cfg_module.large_venue_contact_file = selected_config["contact_file"]
    cfg_module.event_contact_fps = selected_config["contact_fps"]
    cfg_module.large_venue_ini_infected_attendee_seed = selected_config["ini_infected_attendee_seed"]
    cfg_module.large_venue_ini_infected_attendee_patch = selected_config["ini_infected_attendee_patch"]
    cfg_module.event_contact_header_map = selected_config.get("contact_header_map", None)
    cfg_module.event_simulation_modes = selected_config.get("simulation_modes", ["large_venue", "multi_venue", "no_event"])
    cfg_module.large_venue_ids_are_strings = selected_config.get("ids_are_strings", False)

# Initialize with the first scenario in the list as default
if current_event_scenario_names:
    # This is needed so that importing the module provides some default values.
    set_active_scenario(__import__(__name__), current_event_scenario_names[0])

# --- 'multi_venue' Model Parameters ---
# This dictionary maps a group_id to the patch_id of its designated venue.
# The group_ids should match those in your population.csv file.
multi_venue_group_venues = {
    '5': event_venue_patch_id,   # Group '1' -> Venue in patch 5 (Downtown Durham)
}
multi_venue_total_attendees = 2000
# Example format: "contacts_aggregated_N2000_T36000_{group_id}.csv"

# Specify district IDs to visualize attendee numbers (e.g., [0, 1, 2] for the first three districts)
districts_to_visualize_attendees = [5]