# Data and reference inputs

`raw/combined_output.csv` contains Year, Month, Day and MaxSfcWind for the wind
analysis. Speed is in m/s and pressure is computed in Pa. Earlier manuscript
versions describe a CanRCM4 regional ensemble driven by CanESM2 under RCP8.5
for the Toronto area. The CSV does not retain the original dataset identifier,
download URL, ensemble-member labels, or acquisition command. Do not infer
SSP provenance from later scenario labels. Add the original dataset citation
and applicable redistribution terms when they become available.

The selection script infers provisional realization boundaries from dates
resetting in file order. It selects among sufficiently long future blocks
using positive trends in event counts and pressure magnitudes. Threshold
exceedances are treated as individual events, without a declustering step.

`reference/` is an immutable copy of the calibration inputs and results retained
for reproducibility. `manifest.json` records their SHA-256 hashes and their
original relative output paths. These checksums document this snapshot, not an
original climate-data acquisition history.

| Content | Meaning |
|---|---|
| Selected realization and events | Wind observations used for occurrence/mark fitting |
| NHPP/LEYP fit summary | Fitted occurrence-rate and history parameters |
| Magnitude comparison | Candidate excess-magnitude fits; Weibull used in reliability |
| Normalized resistance paths | Mechanistic calibration histories, two temperature cases |
| `degradation_power_law_fits/` | Fitted random-rate/gamma parameters and diagnostics |
| `nonstationary_reliability_power_law/` | Stored curves and summaries for regression checking |

In the path table, `time` is service age in years; `calendar_year` uses a 2026
origin; `temperature_c` is the prescribed temperature proxy; `G` is normalized
resistance. Each scenario contains 40 histories at 75 grid points (ages 0--74).
The exact original generator invocation was not recorded; current defaults do
not exactly regenerate these histories. They are supplied directly so the
calibration and reliability stages remain accessible.

The key `Base temperature` denotes the baseline. `SSP5 delta_T=4 C` is retained
as a compatibility identifier for the warmer proxy. It does not change the
documented wind source to an SSP experiment. No individual-level personal data
are used by the analysis.
