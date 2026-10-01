# Reproducibility guide

All commands below run from the repository root, in the installed environment.
The default workflow starts from archived fits and therefore does not depend
on existing contents of `outputs/`.

## 1. Frozen-fit reproduction

```sh
python main.py --validate-reference
```

The order of random draws matches the stored analysis: baseline random-rate,
baseline gamma, warming random-rate, warming gamma. Each uses 100,000 paths.
Changing batching or order changes individual draws even with the same seed.
The comparison allows small floating-point/library differences; it checks every
annual and cumulative curve, rather than only an end-of-life statistic.

`data/reference/manifest.json` records SHA-256 hashes of the archived CSV files.
The integrity test verifies that they remain unchanged. Write experimental
results to a separate directory instead of editing these inputs.

## 2. Refit degradation models from the archived mechanistic histories

```sh
python scripts/fit_power_law_degradation_reliability.py --n-sim 100000
```

Inputs: `data/reference/degradation_model_fits/normalized_resistance_paths_used.csv`
and the two loading-fit tables in `data/reference/`. Outputs: `outputs/refit/`.
The run fits trajectory parameters, marginal distributions, common propagation
parameters, and gamma increments, then simulates reliability. Optimizer versions
and numerical convergence can affect results; this is not the frozen-fit mode.

To run reliability from the refitted parameters, create a separate input folder
with the same layout as `data/reference/`, replace its degradation-fit tables
with `outputs/refit/tables/degradation_power_law_fits/`, and pass `--input-dir`.
Keep the original inputs unchanged.

## 3. Generate new mechanistic histories

```sh
python scripts/generate_degradation_paths.py --n-paths 40
```

Output: `outputs/generated/normalized_resistance_paths_used.csv`. Options include
`--surface-chloride`, `--delta-temperature`, `--config` and `--output`. The
mechanistic equations and random sampling were extracted unchanged from the
retained path-generation implementation. The exact original invocation that
created the archived calibration paths was not recorded. Regeneration from
the default configuration is therefore not claimed to reproduce them exactly.
The archived paths remain the authoritative calibration inputs for reproduction.

## 4. Loading-data extraction and model fitting

```sh
python scripts/select_high_nonstationarity_realization.py
python scripts/fit_selected_realization_nhpp_leyp.py
python scripts/fit_selected_event_magnitude_distributions.py
```

Extraction reads `data/raw/combined_output.csv` and writes provisional realization
IDs, ranking and selected events to `outputs/tables/`. By default, the fitting
scripts use the archived selected realization/events in `data/reference/`.
To fit a newly selected series, pass `--realization` and `--events` pointing to
the extraction outputs. Fitting outputs are in `outputs/tables/` and figures in
`outputs/figures/`; they do not overwrite the archived inputs.

The occurrence script also produces a Gumbel magnitude diagnostic and approximate
LEYP percentile curves. These diagnostics are not the Weibull mark model used by
the primary reliability calculation. The magnitude-comparison script fits the
pressure excess over the prescribed threshold.

## 5. Convergence study

```sh
python scripts/check_reliability_convergence.py
```

For a shorter execution check:

```sh
python scripts/check_reliability_convergence.py --sizes 1000 5000 --repetitions 2 --reference-n 10000 --output-dir outputs/convergence_smoke
```

Sizes must be multiples of 1,000. Runs are independent across seeds; checkpoints
within each seed share a nested sample. Monte Carlo standard errors use the
variance of conditional failure probabilities, not a Bernoulli-event formula.
The independent reference has its own sampling error. All fitted parameters and
the annual integration grid remain fixed during this study.

## 6. Figures and package

```sh
python scripts/generate_methodology_flowchart.py
python scripts/package_release.py
```

The flowchart is a general methodology diagram; it intentionally omits numerical
case parameters and source equation numbers. SVG text remains editable, the PDF
uses vector content and embedded fonts, and the PNG is rendered at 600 dpi.
The source archive is written to `dist/wind_reliability.zip`, using an explicit list of public
directories rather than bundling the entire working folder.
