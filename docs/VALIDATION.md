# Validation record

Validation performed on 2026-10-01 using Python 3.13 and the versions in
`requirements-reproducible.txt`.

## Reference reproduction

`python main.py --validate-reference` evaluated all four combinations of
degradation model and temperature case, with 100,000 trajectories per case
and seed 20260526. Comparison against the archived curves passed.

| Quantity | Maximum absolute difference across all cases/years |
|---|---:|
| Annual failure probability | 3.87e-14 |
| Cumulative failure probability | 3.11e-14 |
| Annual reliability index | 1.10e-13 |
| Cumulative reliability index | 9.07e-14 |
| Mean resistance | 2.18e-11 |

The reference files were copied byte-for-byte before cleanup and are covered
by the hash manifest. The new runner changes orchestration and output naming,
not the calibrated probability calculations. It also reports first-crossing
service life separately from the final-horizon cumulative statistics.

## Mathematical and input checks

`python -m unittest discover -s tests -v` checks:

- Integrity of every archived input/reference CSV.
- Additivity of NHPP cumulative intensity, including the stationary limit.
- Gamma-increment log likelihood against a shape/scale density evaluation.
- No degradation before initiation and the upper loss bound.
- First-crossing interpolation and a target not reached within the horizon.

The convergence script additionally compares its vectorized gamma sampler
and conditional failure contributions with the retained primary functions
using identical random seeds, before running the larger experiment.

## Convergence experiment

Representative case: power-law gamma degradation, prescribed warming exposure,
nonstationary loading. Fixed calibrated parameters and an annual time grid.
Ten independent seeds were checked at N=1,000, 5,000, 10,000, 20,000, 50,000
and 100,000, against a separate 500,000-trajectory reference.

| N | Mean annual beta at year 37 | Between-run SD | Service life at annual beta=3.0, mean +/- SD (years) |
|---:|---:|---:|---:|
| 20,000 | 2.9758 | 0.0141 | 36.649 +/- 0.205 |
| 50,000 | 2.9757 | 0.0057 | 36.648 +/- 0.083 |
| 100,000 | 2.9788 | 0.0040 | 36.694 +/- 0.058 |

The reference beta is 2.9774 and service life 36.674 years. At 100,000, the
relative Monte Carlo standard error in year-37 failure probability is about
1.8%; the independent reference itself also has sampling error. At the
alternative annual target beta=3.75, service-life SD is about 0.106 years
at N=100,000. These are checks of simulation precision for one model/case,
not of model-form uncertainty or annual time-step accuracy.

See `docs/MODELS.md` for remaining interpretation and manuscript-reconciliation
issues. Unit tests and numerical agreement do not resolve those issues.
