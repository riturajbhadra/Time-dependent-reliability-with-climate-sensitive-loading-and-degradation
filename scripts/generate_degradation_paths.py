"""Generate normalized mechanistic resistance histories used for calibration."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from wind_reliability import pitting
from wind_reliability import random_sample_generator as rsg
from wind_reliability.config import WorkflowConfig

def chloride_ingress_no_temp_ccrit(Dc, T, d_cover, C_surf, c_crit, t):
    t = np.asarray(t, dtype=float)
    T = np.asarray(T, dtype=float)
    C_surf = np.asarray(C_surf, dtype=float)

    e_diff = 47.0
    r_gas = 8.314e-3
    t0 = 0.0767
    n = 0.15
    dc_eff = Dc * np.exp(e_diff / r_gas * (1 / 296.0 - 1 / (273.0 + T))) * np.power(t0 / t, n)
    chloride = np.array(
        [C_surf[i] * (1.0 - math.erf(d_cover / (2.0 * math.sqrt(t[i] * dc_eff[i])))) for i in range(len(t))]
    )
    exceed = np.where(chloride > c_crit)[0]
    return np.nan if exceed.size == 0 else float(t[exceed[0]])


def sample_initiation_times(config: WorkflowConfig, corrosivity_idx: int, delta_t: float, n_paths: int) -> np.ndarray:
    t = np.linspace(1, config.corrosion.t_end, round(config.corrosion.t_end / config.corrosion.dt))
    temperature = 10 + delta_t / 80 * t

    samples = []
    attempts = 0
    while len(samples) < n_paths and attempts < n_paths * 100:
        attempts += 1
        c_surf = rsg.g_rand(
            "lognormal",
            float(config.corrosion.c_surf_mean[corrosivity_idx]),
            float(config.corrosion.c_surf_cov[corrosivity_idx]),
            1,
        ).item()
        dc = rsg.g_rand("lognormal", 240, 0.2, 1).item()
        c_crit = rsg.g_rand("normal", 0.5, 0.2, 1).item()
        d_cover = rsg.g_rand("normal", float(config.analysis.cover_mm), 0.01, 1).item()
        t_init = chloride_ingress_no_temp_ccrit(
            Dc=dc,
            T=temperature,
            d_cover=d_cover,
            C_surf=np.full(t.shape, c_surf),
            c_crit=c_crit,
            t=t,
        )
        if np.isfinite(t_init):
            samples.append(t_init)

    if len(samples) < n_paths:
        raise RuntimeError("Could not sample enough finite initiation times.")
    return np.asarray(samples, dtype=float)


def build_paths(config: WorkflowConfig, delta_t: float, n_paths: int, service_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corrosivity_idx = 1
    propagation_time = np.linspace(1, config.degradation.t_life, round(config.degradation.t_life / config.degradation.dt))
    i_th = float(np.asarray(config.degradation.i_th_base, dtype=float)[corrosivity_idx])

    t_init = sample_initiation_times(config, corrosivity_idx, delta_t, n_paths)
    fck = rsg.g_rand("normal", 40, 0.15, n_paths)
    fy = rsg.g_rand("normal", 600, 0.10, n_paths)
    alpha = rsg.g_rand("gumbel", config.degradation.alpha_mean, config.degradation.alpha_cov, n_paths)

    paths = np.ones((n_paths, service_time.size))
    for i in range(n_paths):
        propagation_path = pitting.pitting_simplified(
            i_th=i_th,
            tp=propagation_time,
            fck=float(fck[i]),
            fy=float(fy[i]),
            alpha=float(alpha[i]),
            d_cover=float(config.analysis.cover_mm),
        )
        time_after_init = service_time - t_init[i]
        active = time_after_init > 0.0
        if np.any(active):
            paths[i, active] = np.interp(
                time_after_init[active],
                propagation_time,
                propagation_path,
                left=1.0,
                right=propagation_path[-1],
            )
    return paths, t_init


def temperature_series(time: np.ndarray, delta_t: float) -> np.ndarray:
    return 10.0 + delta_t / 80.0 * np.asarray(time, dtype=float)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT/"config/defaults.json")
    parser.add_argument("--n-paths", type=int, default=40)
    parser.add_argument("--delta-temperature", type=float, default=4.0)
    parser.add_argument("--surface-chloride", type=float)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT/"outputs/generated/normalized_resistance_paths_used.csv")
    args = parser.parse_args()
    if args.n_paths < 1:
        parser.error("--n-paths must be positive")
    config = WorkflowConfig.from_json(args.config)
    if args.surface_chloride is not None:
        config.corrosion.c_surf_mean = list(config.corrosion.c_surf_mean)
        config.corrosion.c_surf_mean[1] = args.surface_chloride
    # Retain the calibrated service grid (0--74) for historical reproducibility.
    time = np.arange(0, 75, dtype=float)
    rows = []
    for name, delta in [("Base temperature",0.0),(f"SSP5 delta_T={args.delta_temperature:g} C",args.delta_temperature)]:
        # The second key is retained for compatibility; it denotes a warming proxy.
        np.random.seed(config.runtime.random_seed)
        paths, _ = build_paths(config, delta, args.n_paths, time)
        for i,path in enumerate(paths):
            for t,temp,g in zip(time,temperature_series(time,delta),path):
                rows.append(dict(scenario=name,path_id=i,time=t,calendar_year=2026+t,temperature_c=temp,G=g))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output,index=False)
    print(f"Wrote {len(rows)} observations to {args.output}")

if __name__ == "__main__":
    main()
