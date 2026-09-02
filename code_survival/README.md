# SurvivalAnalysis Framework

A modular survival analysis framework for epidemic invasion timing, designed
for mass-gathering effect estimation on epidemic invasion timing.

## Architecture

The framework separates **primary** and **supplementary** analyses:

### Primary analysis (always run by default)

The cluster-robust Cox proportional hazards model is the primary statistical
analysis. It properly accounts for within-simulation dependence using
Huber-White sandwich standard errors clustered on `simulation_run_id`.

1. **Kaplan-Meier analysis** — survival curves, at-risk tables, global and
   pairwise log-rank tests (Holm corrected)
2. **Cluster-robust Cox PH models** — sequential covariate models (M0, M1, M2
   for Mode A; P0, P1, P2 for Mode B)
3. **Sequential model comparison** — AIC, delta-AIC, likelihood ratio tests
4. **Model performance evaluation** — concordance index, AIC
5. **Figures** — KM curves, forest plots, heatmaps, parsimony plots
6. **Interpretation** — mediation metrics and scientific interpretation

### Supplementary analyses (opt-in via `--supplementary`)

Supplementary analyses are **not** executed unless explicitly requested.
They serve as sensitivity checks or robustness assessments.

| Analysis         | `--supplementary` value | Description                                           |
|------------------|-------------------------|-------------------------------------------------------|
| Shared frailty   | `frailty`               | Mixed-effects Cox with Gaussian random intercepts     |
| Bootstrap        | `bootstrap`             | Non-parametric bootstrap for HR uncertainty           |
| Diagnostics      | `diagnostics`           | Model diagnostics (reserved for future expansion)     |
| All of the above | `all`                   | Runs every registered supplementary analysis          |

#### Why the cluster-robust Cox model is the primary analysis

The cluster-robust Cox model is reported as the primary analysis because it
accounts for within-simulation correlation through cluster-robust standard
errors (Huber-White sandwich estimator clustered on `simulation_run_id`).
This approach:

- Yields unbiased coefficient estimates under the Cox proportional hazards
  assumption
- Provides valid standard errors that account for within-run correlation
- Does not require distributional assumptions about random effects
- Is widely used and well-understood in the epidemiological literature

#### Why the shared frailty model is supplementary

The shared frailty model is provided as an **optional sensitivity analysis**
to assess whether unobserved simulation-level heterogeneity materially changes
the substantive conclusions. It:

- Models between-run heterogeneity explicitly via Gaussian random intercepts
- Provides frailty variance and ICC to quantify heterogeneity
- Compares adjusted hazard ratios against cluster-robust estimates
- **Is not the default workflow** — it requires the `--supplementary frailty`
  flag

The shared frailty model is intended as supplementary analysis rather than
the default workflow because it introduces additional modeling assumptions
(Gaussian random effects) and is computationally intensive.

## CLI Usage

```bash
# Mode A (scenario comparison) — primary analysis only
python run_survival_analysis.py mode_a

# Mode B (parameter sensitivity) for a specific scenario — primary analysis only
python run_survival_analysis.py mode_b AMS_dance

# Mode A + supplementary shared frailty analysis
python run_survival_analysis.py mode_a --supplementary frailty

# Mode B + supplementary shared frailty analysis
python run_survival_analysis.py mode_b AMS_dance --supplementary frailty

# All supplementary analyses (Mode A)
python run_survival_analysis.py mode_a --supplementary all

# Custom results directory + per-seed output.
# Flag order does not affect behavior (argparse is order-independent),
# but document examples consistently with --results-dir before --per-seed.
python run_survival_analysis.py mode_a --results-dir results_hybrid_sim --per-seed
python run_survival_analysis.py mode_b AMS_dance --results-dir results_hybrid_sim --per-seed
```

## Python API

```python
from code_survival import run_analysis

# Mode A (scenario comparison) — primary analysis only
result = run_analysis(analysis_mode="scenario", r0=1.5, beta=0.5, iss=1)

# Mode B (parameter sensitivity) — primary analysis only
result = run_analysis(analysis_mode="parameter", fixed_scenario="AMS_dance")

# Mode A + supplementary frailty analysis
result = run_analysis(
    analysis_mode="scenario", r0=1.5, beta=0.5, iss=1,
    supplementary=["frailty"],
)

# All supplementary analyses
result = run_analysis(
    analysis_mode="scenario", r0=1.5, beta=0.5, iss=1,
    supplementary=["all"],
)
```

## Output Structure

```
results_survival_analysis_a/                         # or _b for Mode B
├── km_curve_data.csv                 # from km/
├── km_pairwise_logrank_results.csv
├── km_survival_R150_beta50_Iss1.png   # (Mode A, with seed info)
├── M0_gathering_summary.csv          # robust (cluster-robust Cox) results
├── M1_gathering_seed_summary.csv
├── M2_pressure_3d_summary.csv
├── M2_pressure_7d_summary.csv
├── M2_pressure_14d_summary.csv
├── model_comparison.csv
├── model_comparison_naive_vs_clustered.csv
├── discrimination_performance_R150_beta50_Iss1.png
├── forest_plot_R150_beta50_Iss1.png
├── model_parsimony_R150_beta50_Iss1.png
├── ...                               # (Mode B: parameter heatmap, forest plots)
├── scenario_analysis_R150_beta50_Iss1.csv  # survival dataset
├── scenario_analysis_R150_beta50_Iss1_interpretation.txt
├── scenario_analysis_R150_beta50_Iss1_timing_log.txt
├── naive/                             # Naive (model-based SE) results
│   ├── M0_gathering_summary.csv
│   ├── ...
│   ├── forest_plot.png
│   └── model_comparison.csv
└── supplementary/
    └── frailty/                          # Only created with --supplementary frailty
        ├── summaries/
        │   └── frailty_model_summary.csv
        ├── tables/
        │   ├── frailty_coefficients.csv
        │   ├── frailty_variance.csv
        │   └── frailty_model_comparison.csv
        ├── figures/                       # Reserved for future plots
        └── interpretation/
            └── frailty_interpretation.txt
```

## Code Organization

### Primary analysis modules (`code_survival/reporting.py`)

| Function                      | Description                                           |
|-------------------------------|-------------------------------------------------------|
| `run_analysis()`              | Main entry point — coordinates primary + supplementary  |
| `run_primary_survival_analysis()` | Orchestrates the complete primary pipeline       |
| `run_kaplan_meier()`          | KM curves, log-rank tests, at-risk tables, figures    |
| `run_clustered_cox_models()`  | Fits naive + cluster-robust Cox models, saves summaries |
| `compare_models()`            | Model comparison and naive-vs-clustered tables        |
| `save_primary_results()`      | Saves survival dataset CSV                             |
| `generate_primary_figures()`  | Forest plots, heatmaps, performance plots             |
| `generate_primary_interpretation()` | Generates interpretation text                   |

### Primary analysis helpers (`code_survival/`)

- `cox_models.py` — Cox PH regression with cluster-robust SEs
- `km_analysis.py` — Kaplan-Meier curves and log-rank tests
- `forest_plot.py` — Publication-quality figure generation
- `mediation.py` — Mediation analysis and interpretation
- `calibration.py` — Model calibration assessment
- `bootstrap.py` — Bootstrap estimation for HRs

### Supplementary analysis framework (`code_survival/supplementary/`)

```
code_survival/supplementary/
├── __init__.py          # Registry: SUPPLEMENTARY_ANALYSES, resolve_supplementary_requests()
├── frailty.py           # Shared frailty model wrapper
├── bootstrap.py         # Bootstrap analysis wrapper
├── diagnostics.py       # Model diagnostics (reserved/future)
```

Each supplementary module provides:
- `<name>()` — main entry point that runs the analysis and saves outputs
- `save_<name>_results()` — saves outputs to structured subdirectories
- `generate_<name>_figures()` — generates plots (future)
- `interpret_<name>_models()` — generates interpretation text

## Backward Compatibility

The numerical results produced by the primary analysis are identical to the
previous implementation. The cluster-robust Cox model fitting, Kaplan-Meier
computation, model comparison tables, and all hazard ratios, AIC values,
concordance indices, and interpretation text remain unchanged.

The output directory structure has been flattened to place primary results
directly under the output directory (no intermediate subfolders), with
only ``naive/`` kept as a subdirectory for comparison results, and all
file contents are numerically identical.

The only behavioral change is that supplementary analyses (e.g., the shared
frailty model) are now only executed when explicitly requested via the
`--supplementary` flag.
