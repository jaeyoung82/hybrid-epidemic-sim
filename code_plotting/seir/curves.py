"""SEIR evolution plotting functions."""
import numpy as np
from pathlib import Path
from typing import Dict, Any

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_PLOTTING = True
except Exception:
    HAS_PLOTTING = False


def plot_seir_evolution(ax, time, total_population, n_iterations, 
                        S_realizations, E_realizations, I_realizations, R_realizations):
    """Plot full SEIR evolution on a given axis."""
    colors = {
        'S': {'mean': 'blue', 'runs': 'lightblue'}, 
        'E': {'mean': 'orange', 'runs': 'moccasin'},
        'I': {'mean': 'red', 'runs': 'lightcoral'}, 
        'R': {'mean': 'green', 'runs': 'lightgreen'}
    }
    
    def plot_curves(data, color_map, label):
        for i in range(1, n_iterations + 1):
            ax.plot(time, data[:, i] / total_population, color=color_map['runs'], alpha=0.1)
        mean_prop = np.mean(data[:, 1:], axis=1) / total_population
        ax.plot(time, mean_prop, color=color_map['mean'], lw=2.5, label=f'Mean {label}')
    
    plot_curves(S_realizations, colors['S'], 'Susceptible')
    plot_curves(E_realizations, colors['E'], 'Exposed')
    plot_curves(I_realizations, colors['I'], 'Infected')
    plot_curves(R_realizations, colors['R'], 'Recovered')
    
    ax.set_title('Full SEIR Evolution', fontsize=14)
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Proportion of Population', fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)


def plot_exposed_infected_zoom(ax, time, total_population, n_iterations, 
                               E_realizations, I_realizations):
    """Plot zoomed-in E, I, and E+I curves."""
    colors = {
        'E': {'mean': 'orange', 'runs': 'moccasin'}, 
        'I': {'mean': 'red', 'runs': 'lightcoral'},
        'E+I': {'mean': 'purple', 'runs': 'thistle'}
    }
    
    def plot_curves(data, color_map, label):
        for i in range(1, n_iterations + 1):
            ax.plot(time, (data[:, i] / total_population) * 1000, color=color_map['runs'], alpha=0.1)
        mean_prop = (np.mean(data[:, 1:], axis=1) / total_population) * 1000
        ax.plot(time, mean_prop, color=color_map['mean'], lw=2.5, label=f'Mean {label}')
    
    plot_curves(E_realizations, colors['E'], 'Exposed')
    plot_curves(I_realizations, colors['I'], 'Infected')
    
    EI_realizations = E_realizations[:, 1:] + I_realizations[:, 1:]
    EI_with_time = np.hstack((time[:, np.newaxis], EI_realizations))
    plot_curves(EI_with_time, colors['E+I'], 'Exposed + Infected')
    
    ax.set_title('Exposed & Infected Dynamics', fontsize=14)
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Incidence per 1000', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)


def plot_peak_time_distribution(ax, time, n_iterations, E_realizations, I_realizations):
    """Plot distribution of epidemic peak times (E+I)."""
    peak_times = [time[np.argmax(E_realizations[:, i] + I_realizations[:, i])] for i in range(1, n_iterations + 1)]
    ax.hist(peak_times, bins=25, color='mediumpurple', edgecolor='black', rwidth=0.85)
    
    mean_peak_time = np.mean(peak_times)
    ax.axvline(mean_peak_time, color='red', linestyle='--', lw=2, label=f'Mean: {mean_peak_time:.1f} days')
    
    ax.set_title('Distribution of Epidemic Peak Time (E+I)', fontsize=14)
    ax.set_xlabel('Time to Peak (days)')
    ax.set_ylabel('Frequency (Number of Runs)')
    ax.legend()
    ax.grid(axis='y', alpha=0.75)