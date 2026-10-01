# Climate-sensitive infrastructure reliability

Research code for **Reliability Analysis of Infrastructure Systems under
Climate-Induced Loading and Deterioration**. The retained analysis combines
nonstationary wind-event loading with shifted power-law random-rate and
gamma-process deterioration models. Resistance paths are simulated; the NHPP
load process enters the reliability calculation analytically.

The primary entry point is `main.py`. It uses the archived calibration inputs
in `data/reference/` and reproduces the current 100,000-path reliability run.
It does not refit parameters or overwrite those reference inputs.

## Installation

Python 3.13 is the tested interpreter. Create an environment in a fresh checkout:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` on Windows PowerShell, or
`source .venv/bin/activate` on Linux/macOS. Then install:

```sh
python -m pip install -e . -c requirements-reproducible.txt
```

The constraints file records the versions used for validation. `pyproject.toml`
lists the required libraries; no external application or hosted service is
needed to run the analysis.

## Reproduce the reliability results

```sh
python main.py --validate-reference
```

This runs both degradation models in both temperature cases, using 100,000
paths per model/case and seed 20260526, and checks all resulting curves against
the stored results. Outputs go to `outputs/reproduction/`:

- `annual_reliability_results.csv`: annual and cumulative failure probabilities,
  reliability indices, and mean resistance.
- `service_life_summary.csv`: first annual-index crossing of the selected target.
- `final_horizon_summary.csv`: cumulative results at the end of the analysis.
- `run_metadata.json`: sample size, seed, configuration, package versions, input
  hashes, and reference-comparison results.
- Four reliability plots and a simulation summary.

For a quick execution check rather than a publication calculation:

```sh
python main.py --n-sim 1000 --output-dir outputs/smoke
```

Use `--target-beta 3.75` to change the service-life criterion. The default is
3.0. Use `--config`, `--input-dir`, and `--output-dir` for alternate inputs.
Changing the sample size, seed, or calibrated inputs is incompatible with an
exact reproduction of the archived run. Simulation size, reliability seed and
target reliability index are explicit command-line options.

## Convergence and verification

```sh
python -m unittest discover -s tests -v
python scripts/check_reliability_convergence.py
```

The convergence study uses ten seeds, sample sizes from 1,000 to 100,000, and
an independent 500,000-path reference. It checks the vectorized sampler against
the primary implementation before computing annual/cumulative statistics and
service-life variability. See [validation](docs/VALIDATION.md).

## Refit or regenerate inputs

These are separate research steps, not prerequisites for reproducing the stored
reliability curves. Re-estimation can change fitted parameters and results.
Commands and input/output relationships are in
[reproducibility instructions](docs/REPRODUCIBILITY.md).

| File | Purpose |
|---|---|
| `main.py`, `scripts/run_analysis.py` | Reproduce reliability from frozen fits |
| `scripts/fit_power_law_degradation_reliability.py` | Fit both degradation models and run reliability |
| `scripts/generate_degradation_paths.py` | Mechanistic initiation, corrosion, and normalized capacity histories |
| `scripts/select_high_nonstationarity_realization.py` | Extract/rank wind realizations and threshold events |
| `scripts/fit_selected_realization_nhpp_leyp.py` | Fit occurrence models and history-parameter profiles |
| `scripts/fit_selected_event_magnitude_distributions.py` | Compare event-magnitude models |
| `scripts/check_reliability_convergence.py` | Repeated-seed convergence study |
| `scripts/generate_methodology_flowchart.py` | Editable methodology figure, PDF and PNG |
| `src/wind_reliability/` | Configuration, sampling, and mechanistic pitting model |
| `data/reference/` | Immutable calibration and regression-reference tables |
| `tests/` | Input integrity and mathematical consistency checks |

## Interpretation and data provenance

Read [model assumptions and limitations](docs/MODELS.md) and
[data provenance](data/README.md) before interpreting the results. In particular,
the internal label `SSP5 delta_T=4 C` denotes a prescribed warming proxy; it is
not evidence that the wind data are from an SSP experiment. Earlier manuscript
provenance identifies CanRCM4/CanESM2 under RCP8.5. The raw CSV does not contain
original climate-model identifiers.

The current analysis retains the calibrated mean initial resistance of 1621 Pa
and COV 0.09. It does not silently replace this with an alternative nominal-load
calibration. Several manuscript/code differences are documented in `docs/MODELS.md`.

## Repository contents

Generated results, environments, editor settings, caches and local backups are
excluded by `.gitignore`. The reference tables are deliberately included so the
reproduction workflow starts from a clean checkout. Build a source archive with:

```sh
python scripts/package_release.py
```

The archive contains the source, documentation, tests, configuration and data;
it excludes local working directories and historical results. No repository is
published automatically. A redistribution license and final manuscript citation
have not been assigned in these files.
