"""CLI entry point for hybrid epidemic simulation."""
import typer
from pathlib import Path
import numpy as np
from typing import Optional, List

app = typer.Typer(help="Hybrid Epidemic Simulation CLI")

# Import refactored modules
try:
    from code_config_models import SimulationConfig, create_default_config, load_config_from_json, save_config_to_json
    from code_simulation.data.loader import DataLoader, load_and_prepare_data
    from code_simulation.recruitment.strategies import RecruitmentStrategy
except ImportError as e:
    print(f"Import error: {e}")
    print("Some modules may require pydantic. Install with: pip install pydantic")


@app.command()
def init_config(
    output: str = typer.Option("simulation_config.json", "--output", "-o",
        help="Output path for generated config template (default: simulation_config.json)"),
):
    """Generate a full JSON config template from simulation defaults."""
    save_config_to_json(create_default_config(), output)
    typer.echo(f"Config template saved to {output}")


@app.command()
def run(
    config: Optional[str] = typer.Option(None, "--config", "-c",
        help="Path to JSON config file. CLI flags override JSON values."),
    r0: Optional[float] = typer.Option(None, "--r0", "-r", help="Reproduction number (0.1-10.0)"),
    beta: Optional[float] = typer.Option(None, "--beta", "-b", help="Event transmission rate (0.0-1.0)"),
    iss: Optional[int] = typer.Option(None, "--iss", "-i", help="Initial seed size"),
    scenario: str = typer.Option("all", "--scenario", "-s", help="Scenario to run (or 'all')"),
    iterations: Optional[int] = typer.Option(None, "--iterations", "-n", help="Number of Monte Carlo runs"),
    days: Optional[int] = typer.Option(None, "--days", "-d", help="Simulation duration in days"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Run epidemic simulation with specified parameters."""
    cfg = load_config_from_json(config) if config else create_default_config()

    if r0 is not None:
        cfg.R_0 = r0
    if beta is not None:
        cfg.event_base_transmission_rate = beta
    if iss is not None:
        cfg.I_ss = iss
    if iterations is not None:
        cfg.n_iterations = iterations
    if days is not None:
        cfg.n_days = days
    if output is not None:
        cfg.results_dir = Path(output)

    if scenario != "all":
        cfg.current_event_scenario_names = [scenario]
        if scenario in cfg.event_configurations:
            cfg.set_active_scenario(scenario)

    print(f"Running simulation: R0={cfg.R_0}, beta={cfg.event_base_transmission_rate}, ISS={cfg.I_ss}")
    print(f"Iterations: {cfg.n_iterations}, Days: {cfg.n_days}")

    loader = DataLoader(cfg.dataset_name)
    data = loader.prepare_all()

    # Placeholder for actual simulation execution
    # In practice, this would call the simulation pipeline
    typer.echo("Simulation complete (stub - implement full pipeline)")


@app.command()
def analyze(
    mode: str = typer.Option("scenario", "--mode", "-m", help="Analysis mode: 'scenario' or 'parameter'"),
    r0: float = typer.Option(1.5, "--r0", "-r", help="Fixed R0 for Mode A"),
    beta: float = typer.Option(0.5, "--beta", "-b", help="Fixed beta for Mode A"),
    iss: int = typer.Option(1, "--iss", "-i", help="Fixed ISS for Mode A"),
    fixed_scenario: Optional[str] = typer.Option(None, "--fixed-scenario", "-f", help="Fixed scenario for Mode B"),
    results_dir: str = typer.Option("results_hybrid_sim", "--results-dir", help="Results directory"),
    seed_locations: Optional[List[int]] = typer.Option(None, "--seed-locations", "-s", help="Seed district IDs to include (auto-discovered if omitted)"),
    per_seed: bool = typer.Option(False, "--per-seed", help="Run analysis independently for each seed (separate output folders per seed)"),
):
    """Run survival analysis on simulation results."""
    from code_survival import run_analysis
    
    result = run_analysis(
        analysis_mode=mode,
        results_dir=results_dir,
        r0=r0,
        beta=beta,
        iss=iss,
        fixed_scenario=fixed_scenario,
        seed_locations=seed_locations,
        per_seed=per_seed,
    )
    
    if result:
        typer.echo("Analysis complete")


@app.command()
def list_scenarios(
    config: Optional[str] = typer.Option(None, "--config", "-c",
        help="Path to JSON config file"),
):
    """List available event scenarios."""
    cfg = load_config_from_json(config) if config else create_default_config()
    for name in cfg.current_event_scenario_names:
        if name in cfg.event_scenarios:
            sc = cfg.event_scenarios[name]
            print(f"  {name}: {sc.total_attendees} attendees, file={sc.contact_file}")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()