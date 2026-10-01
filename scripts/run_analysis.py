"""Reproduce reliability curves from the archived calibration inputs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import fit_power_law_degradation_reliability as model


def first_crossing(years, beta, target):
    """Interpolate the first target crossing; NaN means not reached in the grid."""
    indices = np.flatnonzero(np.asarray(beta) <= target)
    if not len(indices):
        return float('nan')
    i = indices[0]
    if i == 0:
        return float(years[0])
    return float(years[i-1] + (target-beta[i-1])*(years[i]-years[i-1])/(beta[i]-beta[i-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/defaults.json')
    parser.add_argument('--input-dir', type=Path, default=ROOT/'data/reference')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/reproduction')
    parser.add_argument('--n-sim', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=20260526)
    parser.add_argument('--target-beta', type=float, default=3.0)
    parser.add_argument('--validate-reference', action='store_true',
                        help='Compare all computed curves with the stored 100,000-path run.')
    args = parser.parse_args()
    if args.n_sim < 1:
        parser.error('--n-sim must be positive')
    if args.validate_reference and (args.n_sim != 100000 or args.seed != 20260526):
        parser.error('Reference comparison requires --n-sim 100000 --seed 20260526.')
    cfg = model.WorkflowConfig.from_json(args.config)
    inputs = args.input_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    fits_dir = inputs/'degradation_power_law_fits'
    rr = pd.read_csv(fits_dir/'power_random_rate_fitted_parameters.csv')
    distributions = pd.read_csv(fits_dir/'power_random_rate_distribution_fits.csv')
    gp = pd.read_csv(fits_dir/'power_gamma_process_fitted_parameters.csv')
    gp_inits = pd.read_csv(fits_dir/'power_gamma_process_initiation_distribution_fits.csv')
    common_m = float(pd.read_csv(fits_dir/'power_random_rate_common_propagation_parameters.csv').m_common.iloc[0])
    scenarios = model.read_paths(inputs/'degradation_model_fits/normalized_resistance_paths_used.csv')
    load = model.read_load_model(inputs)
    rng = np.random.default_rng(args.seed)
    rows, simulation = [], []
    for scenario, data in scenarios.items():
        data['cumulative_accel'] = model.cumulative_trapezoid(data['time'],
            model.acceleration_factor(data['temperature_c'], 40000., 8.314, 20.))
        for name in ('power_random_rate', 'power_gamma_process'):
            if name == 'power_random_rate':
                resistance, sampled = model.simulate_rr_resistance(rng, cfg, scenario,
                    distributions, data, args.n_sim, fitted_triplets=rr, common_m=common_m)
            else:
                resistance, sampled = model.simulate_gp_resistance(rng, cfg, scenario,
                    gp, gp_inits, data, args.n_sim)
            result = model.reliability_from_resistance(rng, cfg, resistance, data['time'],
                *load, stationary_load=(scenario == 'Base temperature'))
            result.insert(0, 'scenario', scenario)
            result.insert(0, 'degradation_model', name)
            rows.append(result)
            simulation.append(dict(scenario=scenario, degradation_model=name, N=args.n_sim,
                mean_R0=float(sampled['R0'].mean()), mean_t_ini=float(sampled['t_ini'].mean())))
            print(f'Completed {name}: {scenario} ({args.n_sim:,} paths)', flush=True)
            del resistance, sampled
    results = pd.concat(rows, ignore_index=True)
    results.to_csv(out/'annual_reliability_results.csv',index=False)
    pd.DataFrame(simulation).to_csv(out/'simulation_summary.csv',index=False)
    results[results.Service_Year == cfg.analysis.time_horizon_years].to_csv(out/'final_horizon_summary.csv',index=False)
    lives = []
    for (name,scenario), group in results.groupby(['degradation_model','scenario'], sort=False):
        life = first_crossing(group.Service_Year.to_numpy(),group.Annual_Beta.to_numpy(),args.target_beta)
        lives.append(dict(degradation_model=name,scenario=scenario,target_annual_beta=args.target_beta,
            service_life_years=life,target_reached=bool(np.isfinite(life))))
    pd.DataFrame(lives).to_csv(out/'service_life_summary.csv',index=False)
    model.configure_matplotlib()
    for col, title, log_y in [('Annual_Pf','Annual probability of failure',True),
            ('Cumulative_Pf','Cumulative probability of failure',True),
            ('Annual_Beta','Annual reliability index',False),
            ('Cumulative_Beta','Cumulative reliability index',False)]:
        model.plot_reliability(results,col,title,out/f'{col.lower()}.png',log_y)
    validation = None
    if args.validate_reference:
        ref = pd.read_csv(inputs/'nonstationary_reliability_power_law/annual_reliability_results.csv')
        keys = ['degradation_model','scenario','Service_Year']
        actual = results.sort_values(keys).reset_index(drop=True)
        expected = ref.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[keys],expected[keys])
        differences = {}
        for col in ['Annual_Pf','Cumulative_Pf','Annual_Beta','Cumulative_Beta','Mean_Resistance']:
            np.testing.assert_allclose(actual[col],expected[col],rtol=2e-6,atol=1e-9,
                                       err_msg=f'Reference mismatch: {col}')
            differences[col] = float(np.max(np.abs(actual[col]-expected[col])))
        validation = {'passed':True,'max_absolute_differences':differences}
        print('Stored-reference regression passed.',flush=True)
    metadata = dict(N=args.n_sim,seed=args.seed,target_annual_beta=args.target_beta,
        configuration=asdict(cfg),python=platform.python_version(),
        packages={name:importlib.metadata.version(name) for name in ('numpy','pandas','scipy','matplotlib')},
        input_hashes={str(p.relative_to(inputs)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(inputs.rglob('*.csv'))},reference_validation=validation)
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
