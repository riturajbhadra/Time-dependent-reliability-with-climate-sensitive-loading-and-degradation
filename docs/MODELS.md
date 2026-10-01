# Models, conventions, and limitations

## Loading and reliability

The retained illustrative analysis uses `lambda(t)=exp(theta0+theta1*t)` and
Weibull-distributed wind-pressure excesses with scale `exp(alpha0+alpha1*t)`.
The LEYP occurrence model is fitted as an alternative; its saved history
parameter is zero. The Weibull family is explicitly selected for reliability,
although GPD has a slightly smaller AIC in the candidate comparison.

For a sampled resistance history and dead load, conditional survival is
`exp(-integral lambda(t)*P[W(t)>R(t)-B] dt)`. Interval intensities are integrated
analytically and the resistance/mark parameters evaluated at interval midpoints.
Annual failure probabilities and cumulative failure probabilities are computed
separately, then averaged over structural samples. The annual result is not
explicitly conditioned on survival through all previous years.

The baseline freezes the load parameters at their initial fitted future-process
values. The changed case retains the time trends and uses the warming proxy.
Only baseline/baseline and changed/changed are evaluated by the retained main
workflow. Load-only and deterioration-only cases require separate calculations.

Initial resistance is independently lognormal with mean 1621 Pa and COV .09;
dead load is independently normal with mean 359 Pa and COV .10. Resistance and
loads use pressure-equivalent units. Loading, degradation parameters/increments,
initial resistance and dead load are independently sampled or conditioned on the
prescribed scenario. A shared scenario does not introduce joint dependence.

## Degradation

The effective exposure time integrates the Arrhenius temperature acceleration
after initiation. Random-rate loss is `min(eta*tau^m,1)`. Its default simulation
resamples initiation from the relevant scenario, draws eta from pooled fitted
rates, and uses the common fitted exponent. This is not independent sampling of
all the fitted marginal families printed in the fit tables.

The gamma process uses independent increments of shape `c*Delta(tau^m)` and
scale `nu`; accumulated fractional loss is capped at one. Parameters c, nu and m
are pooled across scenarios. Initiation distributions are fitted separately.
Both models use `R(t)=R0*(1-loss(t))` and have no loss before initiation.

The mechanistic path generator retains the implemented chloride concentration
approximation, temperature-dependent apparent diffusion, first threshold
crossing, blended corrosion-current model, pitting penetration and four-bar
rectangular-section capacity calculation. See `generate_degradation_paths.py`
and `src/wind_reliability/pitting.py` for the complete expressions.

## Limits preserved rather than silently changed

- Propagation in `pitting_simplified` uses an internal 10-to-14 degree C ramp
  for both temperature cases. Scenario temperature influences initiation and
  the subsequent fitted effective-time transformation; it is not directly
  passed into the mechanistic propagation temperature series.
- The calibrated path grid runs from service age 0 through 74 years. The final
  reliability interval extends to year 75 and interpolation holds the last
  available path value constant at its midpoint. The reproduction preserves
  this convention; time-grid convergence has not been established.
- Initiation simulation retries samples that do not initiate within its horizon.
  It is not a censored initiation-time likelihood. Gamma fitting filters
  nonpositive increments and is not a measurement-error model.
- The apparent-diffusivity error-function expression is the implemented
  approximation, not an exact general PDE solution for arbitrary time-varying
  diffusion. Pitting integration retains its original discrete correction.
- Some archived distribution tables count a fixed location parameter in AIC/BIC.
  These tables are preserved for reproduction; information-criterion rankings
  should be reviewed before selecting different models. Cross-model comparisons
  also require likelihoods for the same observed data.
- Selection favors a wind realization with strong positive trends. Ensemble
  uncertainty and selection effects are not represented by this single case.
- No maintenance, bond-loss, spalling, or detailed pier finite-element model is
  included. Normalized section capacity does not independently establish an
  absolute pressure-equivalent initial resistance.

## Manuscript reconciliation

The saved current results were identified as a 100,000-path run by replaying the
recorded seed and matching the saved parameter averages. A 20,000-path description
does not describe these archived curves. The convergence study supports 100,000
for the checked reliability-index and service-life estimates.

The earlier wind provenance is CanRCM4/CanESM2 under RCP8.5. The preserved CSV key
`SSP5 delta_T=4 C` is a legacy identifier for the prescribed warming proxy, not
an assertion of an SSP wind experiment.

Initial-resistance conventions differ across historical analyses and manuscript
text. The retained mean 1621 Pa is fixed by the current code configuration.
The nominal-load relation `Rn=(1.2*Dn+Wn)/0.9`, with `Dn=Wn/2`, instead gives
`Rn=1.778*Wn` and `E[R0]=1.08*Rn=1.92*Wn`. These are not interchangeable, and
the manuscript's `2.37*Rn` statement does not follow from those factors.

Service life depends on the chosen annual reliability target. The default
entry point now records the target explicitly and interpolates its first
crossing; it reports no crossing as NaN with `target_reached=False`. Final-horizon
cumulative statistics are stored separately. A target of 3.0 and a target of
3.75 must not be compared as if they defined the same service life.
