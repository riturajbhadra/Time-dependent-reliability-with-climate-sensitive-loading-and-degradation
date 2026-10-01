# Time-dependent reliability with climate-sensitive loading and degradation

Code and data accompanying **Reliability Analysis of Infrastructure Systems
under Climate-Induced Loading and Deterioration**.

The analysis combines nonstationary environmental loading with time-dependent
structural resistance. Two stochastic deterioration models are implemented:
a shifted power-law random-rate model and a shifted gamma process. Reliability
is evaluated using simulated resistance trajectories and an analytical
non-homogeneous Poisson load-event formulation.

The supplied example considers wind loading and corrosion-related deterioration.
Baseline and prescribed warming temperature histories are used in the
deterioration analysis. The repository includes calibration inputs, fitted
parameters, reference results, and scripts for model estimation and reliability
assessment.

## Installation

Run all commands from the repository root. Python 3.13 was used for validation.

```sh
python -m venv .venv
```

Activate the environment on **Windows PowerShell**:

```powershell
.venv\Scripts\Activate.ps1
```

On **Linux or macOS**:

```sh
source .venv/bin/activate
```

Install the package and dependencies:

```sh
python -m pip install -e . -c requirements-reproducible.txt
```

The constraints file specifies the dependency versions used for validation.

## Run the reliability analysis

```sh
python main.py --validate-reference
```

This command uses the supplied calibration tables to evaluate both degradation
models under baseline and warming conditions. The default simulation uses
100,000 trajectories per model and case, with random seed `20260526`.
The validation option compares the computed results with the reference curves.

Results are written to `outputs/reproduction/`:

| Output | Contents |
|---|---|
| `annual_reliability_results.csv` | Annual and cumulative failure probabilities, reliability indices, and mean resistance |
| `service_life_summary.csv` | First crossing of the adopted annual reliability-index target |
| `final_horizon_summary.csv` | Reliability results at the end of the analysis period |
| `simulation_summary.csv` | Simulation input summary |
| `run_metadata.json` | Configuration, sample size, seed, software versions, input hashes, and validation results |
| PNG figures | Annual and cumulative probability-of-failure and reliability-index curves |

For a short execution check:

```sh
python main.py --n-sim 1000 --output-dir outputs/smoke
```

The default service-life target is an annual reliability index of `3.0`.
Use `--target-beta` to specify another target. Additional options include
`--n-sim`, `--seed`, `--config`, `--input-dir`, and `--output-dir`.
Reference validation uses the default simulation settings and supplied inputs.

## Model estimation and input generation

Scripts are provided for wind-event extraction, occurrence and magnitude model
fitting, mechanistic resistance-history generation, and stochastic degradation
model estimation. The reliability entry point uses the supplied fitted tables;
these estimation steps can be run separately.

See the [reproducibility guide](docs/REPRODUCIBILITY.md) for commands, input
requirements, and output locations. The [model documentation](docs/MODELS.md)
describes the equations, assumptions, and implementation details, and the
[data documentation](data/README.md) describes the input datasets and scenario
definitions.

## Verification and convergence

Run the numerical and input-integrity checks:

```sh
python -m unittest discover -s tests -v
```

Run the representative-case Monte Carlo convergence study:

```sh
python scripts/check_reliability_convergence.py
```

The study compares sample sizes from 1,000 to 100,000 across ten random seeds
against an independent 500,000-trajectory reference. It evaluates variability
in reliability estimates and reliability-based service life. The procedure
and results are described in [validation](docs/VALIDATION.md).

## Repository structure

| Path | Contents |
|---|---|
| `main.py` | Reliability-analysis entry point |
| `scripts/` | Data processing, model fitting, reliability analysis, convergence, and figure generation |
| `src/wind_reliability/` | Configuration, random sampling, and mechanistic deterioration functions |
| `config/` | Model and analysis settings |
| `data/raw/` | Wind input data |
| `data/reference/` | Calibration inputs, fitted parameters, and reference results |
| `docs/` | Model, reproduction, and validation documentation |
| `tests/` | Numerical consistency and data-integrity checks |
| `outputs/` | Generated results, created when scripts run |

