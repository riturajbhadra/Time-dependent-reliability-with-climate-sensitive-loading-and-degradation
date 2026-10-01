"""Reproducible Monte Carlo convergence audit of one existing reliability case.

Uses saved power-law gamma-process fits, warming exposure and nonstationary
loading. No model refitting or modification of existing analysis outputs.
Vectorized sampling is checked against the original functions with equal seeds.
"""
from pathlib import Path
import sys
import json
import time
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fit_power_law_degradation_reliability as model

OUT = ROOT / 'outputs' / 'convergence_check'
TABLES = ROOT / 'data' / 'reference'
FITS = TABLES / 'degradation_power_law_fits'
CONFIG = model.WorkflowConfig.from_json(ROOT / 'config/defaults.json')
SCENARIOS = model.read_paths(TABLES / 'degradation_model_fits/normalized_resistance_paths_used.csv')
SCENARIO = next(s for s in SCENARIOS if s != 'Base temperature')
DATA = SCENARIOS[SCENARIO]
TIME = DATA['time']
DATA['cumulative_accel'] = model.cumulative_trapezoid(TIME,
    model.acceleration_factor(DATA['temperature_c'], 40000., 8.314, 20.))
GP = pd.read_csv(FITS / 'power_gamma_process_fitted_parameters.csv')
INIT = pd.read_csv(FITS / 'power_gamma_process_initiation_distribution_fits.csv')
ROW = GP[GP.scenario == SCENARIO].iloc[0]
C, NU, M = (float(ROW[k]) for k in ('c', 'nu', 'm_gamma'))
LOAD = model.read_load_model(TABLES)
theta0, theta1, threshold, alpha0, alpha1, shape = LOAD
START = np.arange(0., CONFIG.analysis.time_horizon_years, CONFIG.analysis.time_step_years)
YEARS = START + CONFIG.analysis.time_step_years
MID = (START + YEARS) / 2
LAMBDA = model.nhpp_cumulative(theta0, theta1, START, YEARS)
SCALE = np.exp(alpha0 + alpha1 * MID)
SIZES = [1000, 5000, 10000, 20000, 50000, 100000]
REPS, REFERENCE_N, BATCH = 10, 500000, 1000

def fast_resistance(rng, n):
    ti = model.sample_from_best_distribution(rng, INIT, SCENARIO, 't_ini_gamma', n)
    cum = DATA['cumulative_accel']
    tau = np.maximum(cum[None, :] - np.interp(ti, TIME, cum)[:, None], 0.)
    da = np.diff(tau ** M, axis=1)
    inc = np.zeros_like(da)
    active = da > 0
    inc[active] = rng.gamma(C * da[active], scale=NU)
    loss = np.column_stack((np.zeros(n), np.cumsum(inc, axis=1)))
    loss = np.clip(loss, 0, 1)
    mu, sigma = model.lognormal_mu_sigma(CONFIG.resistance.initial_mean, CONFIG.resistance.initial_cov)
    r0 = rng.lognormal(mu, sigma, n)
    return r0[:, None] * (1 - loss)

def contributions(rng, resistance):
    # Match np.interp, including the original final-year endpoint clamping.
    ix = np.clip(np.searchsorted(TIME, MID, side='right') - 1, 0, len(TIME)-2)
    weight = np.clip((MID-TIME[ix])/(TIME[ix+1]-TIME[ix]), 0, 1)
    rmid = resistance[:, ix]*(1-weight) + resistance[:, ix+1]*weight
    dead = rng.normal(CONFIG.dead_load.mean, CONFIG.dead_load.mean*CONFIG.dead_load.cov, len(resistance))
    excess = rmid - dead[:, None] - threshold
    tail = np.ones_like(excess)
    positive = excess > 0
    tail[positive] = np.exp(-((excess/ SCALE[None, :])[positive]) ** shape)
    h = LAMBDA[None, :] * tail
    # expm1 avoids cancellation for small probabilities, equivalent to source.
    return -np.expm1(-h), -np.expm1(-np.cumsum(h, axis=1))

def crossing(beta, target):
    below = np.flatnonzero(beta <= target)
    if not len(below):
        return np.nan
    j = below[0]
    if j == 0:
        return float(YEARS[0])
    return float(YEARS[j-1] + (target-beta[j-1])*(YEARS[j]-YEARS[j-1])/(beta[j]-beta[j-1]))

def run(seed, checkpoints):
    rng = np.random.default_rng(seed)
    sums = np.zeros((2, len(YEARS)))
    squares = sums.copy()
    result = {}
    for lo in range(0, max(checkpoints), BATCH):
        n = min(BATCH, max(checkpoints)-lo)
        vals = contributions(rng, fast_resistance(rng, n))
        for k, v in enumerate(vals):
            sums[k] += v.sum(axis=0)
            squares[k] += (v*v).sum(axis=0)
        total = lo+n
        if total in checkpoints:
            mean = sums/total
            variance = np.maximum((squares-total*mean**2)/(total-1), 0)
            se = np.sqrt(variance/total)
            beta = -norm.ppf(np.clip(mean, 1e-15, 1-1e-15))
            result[total] = (mean.copy(), se, beta)
    return result

def main():
    global OUT, SIZES, REPS, REFERENCE_N
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=SIZES)
    parser.add_argument('--repetitions', type=int, default=REPS)
    parser.add_argument('--reference-n', type=int, default=REFERENCE_N)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    SIZES, REPS, REFERENCE_N, OUT = sorted(set(args.sizes)), args.repetitions, args.reference_n, args.output_dir
    if REPS < 2 or any(n < BATCH or n % BATCH for n in SIZES + [REFERENCE_N]):
        parser.error('Use at least two repetitions and sample sizes that are positive multiples of 1000.')
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # Replay the original initial random-rate draws to identify saved run size.
    rr = pd.read_csv(FITS / 'power_random_rate_fitted_parameters.csv')
    pool = rr[rr.scenario == 'Base temperature'].reset_index(drop=True)
    saved = pd.read_csv(TABLES / 'nonstationary_reliability_power_law/simulation_input_summary.csv')
    saved = saved[(saved.scenario == 'Base temperature') & (saved.degradation_model == 'power_random_rate')].iloc[0]
    provenance = []
    for n in (20000, 100000):
        rng = np.random.default_rng(20260526)
        ti = pool.iloc[rng.integers(0, len(pool), n)].t_ini.to_numpy()
        eta = rr.eta.to_numpy()[rng.integers(0, len(rr), n)]
        provenance.append(dict(N=n, mean_t_ini=float(ti.mean()), mean_eta=float(eta.mean()),
            matches_saved=bool(np.isclose(ti.mean(), saved.mean_t_ini, rtol=0, atol=1e-10)
            and np.isclose(eta.mean(), saved.mean_eta, rtol=0, atol=1e-12))))
    print('Sample-size replay:', provenance, flush=True)

    seed, n = 98137, 128
    original_rng = np.random.default_rng(seed)
    original, _ = model.simulate_gp_resistance(original_rng, CONFIG, SCENARIO, GP, INIT, DATA, n)
    expected = model.reliability_from_resistance(original_rng, CONFIG, original, TIME, *LOAD, stationary_load=False)
    fast_rng = np.random.default_rng(seed)
    fast = fast_resistance(fast_rng, n)
    np.testing.assert_allclose(fast, original, rtol=1e-12, atol=1e-10)
    annual, cumulative = contributions(fast_rng, fast)
    np.testing.assert_allclose(annual.mean(axis=0), expected.Annual_Pf, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(cumulative.mean(axis=0), expected.Cumulative_Pf, rtol=1e-10, atol=1e-14)
    print('Original implementation equivalence checks passed.', flush=True)

    reference = run(890123, [REFERENCE_N])[REFERENCE_N]
    print(f'Independent {REFERENCE_N:,}-path reference complete.', flush=True)
    ref_mean, ref_se, ref_beta = reference
    curves = []
    for k, metric in enumerate(('Annual', 'Cumulative')):
        for j, year in enumerate(YEARS):
            curves.append(dict(metric=metric, year=year, Pf=ref_mean[k,j], Pf_MCSE=ref_se[k,j],
                Beta=ref_beta[k,j], Beta_MCSE=ref_se[k,j]/norm.pdf(ref_beta[k,j])))
    pd.DataFrame(curves).to_csv(OUT/'reference_curves.csv', index=False)
    target_year = int(round(crossing(ref_beta[0], 3.0)))
    target375_year = int(round(crossing(ref_beta[0], 3.75)))
    selected_years = sorted(set((1, target_year, target375_year, 50, 75)))
    rows, lives = [], []
    for rep in range(REPS):
        results = run(620000+rep, SIZES)
        for size, (mean, se, beta) in results.items():
            for target in (3., 3.75):
                lives.append(dict(N=size, replicate=rep+1, target=target,
                    life=crossing(beta[0], target), reference_life=crossing(ref_beta[0], target)))
            for k, metric in enumerate(('Annual', 'Cumulative')):
                for year in selected_years:
                    j = year-1
                    rows.append(dict(N=size, replicate=rep+1, metric=metric, year=year,
                        Pf=mean[k,j], Pf_MCSE=se[k,j], relative_MCSE=se[k,j]/mean[k,j],
                        Beta=beta[k,j], reference_Pf=ref_mean[k,j], reference_Beta=ref_beta[k,j]))
        print(f'Replicate {rep+1}/{REPS} complete.', flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT/'replicate_results.csv', index=False)
    summary = raw.groupby(['N','metric','year']).agg(
        Pf_mean=('Pf','mean'), Pf_between_seed_SD=('Pf','std'),
        Pf_mean_MCSE=('Pf_MCSE','mean'), relative_MCSE_mean=('relative_MCSE','mean'),
        Beta_mean=('Beta','mean'), Beta_SD=('Beta','std'), Beta_min=('Beta','min'), Beta_max=('Beta','max'),
        reference_Pf=('reference_Pf','first'), reference_Beta=('reference_Beta','first')).reset_index()
    summary.to_csv(OUT/'convergence_summary.csv', index=False)
    life = pd.DataFrame(lives)
    life.to_csv(OUT/'service_life_replicates.csv', index=False)
    life_summary = life.groupby(['N','target']).agg(mean=('life','mean'), SD=('life','std'),
        minimum=('life','min'), maximum=('life','max'), reference=('reference_life','first')).reset_index()
    life_summary.to_csv(OUT/'service_life_summary.csv', index=False)
    meta = dict(case='Power-law gamma process, warming exposure, nonstationary wind loading',
        internal_scenario_label=SCENARIO, sample_sizes=SIZES, independent_replicates=REPS,
        reference_N=REFERENCE_N, reference_seed=890123, replicate_seeds=list(range(620000,620000+REPS)),
        target_year=target_year, target375_year=target375_year,
        sample_size_replay=provenance, original_equivalence_check='passed',
        initial_resistance_mean=CONFIG.resistance.initial_mean, batch_size=BATCH,
        elapsed_seconds=time.perf_counter()-started,
        interpretation='Monte Carlo error only; fitted parameters and annual grid held fixed. Nested sample sizes within each independent replicate. SE uses conditional failure contributions, not Bernoulli events.')
    (OUT/'run_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.size':10, 'pdf.fonttype':42,'svg.fonttype':'none'})
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for size in SIZES:
        vals = raw[(raw.N==size)&(raw.metric=='Annual')&(raw.year==target_year)].Beta
        axs[0].scatter(np.full(len(vals),size), vals, s=15, color='#31536a', alpha=.55)
        vals2 = life[(life.N==size)&(life.target==3.)].life
        axs[1].scatter(np.full(len(vals2),size),vals2,s=15,color='#31536a',alpha=.55)
    axs[0].axhline(ref_beta[0,target_year-1],ls='--',color='black',label=f'Independent {REFERENCE_N:,}-path reference')
    axs[1].axhline(crossing(ref_beta[0],3.),ls='--',color='black')
    axs[0].set(ylabel=f'Annual reliability index, year {target_year}',xlabel='Number of resistance paths')
    axs[1].set(ylabel='Service life at annual target beta = 3.0 (years)',xlabel='Number of resistance paths')
    for ax in axs:
        ax.set_xscale('log'); ax.grid(alpha=.2)
    axs[0].legend(fontsize=8)
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'convergence_check.{ext}',dpi=300)
    plt.close(fig)
    print('\nAnnual results near target:\n',summary[(summary.metric=='Annual')&(summary.year==target_year)].to_string(index=False),flush=True)
    print('\nService life:\n',life_summary.to_string(index=False),flush=True)

if __name__ == '__main__':
    main()
