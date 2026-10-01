"""Fit shifted power-law degradation models and run reliability analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from scipy.special import gammaln
from scipy.stats import gamma as gamma_dist
from scipy.stats import lognorm, norm, weibull_min


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wind_reliability.config import WorkflowConfig  # noqa: E402


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 14,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "legend.frameon": False,
            "figure.dpi": 120,
        }
    )


def read_paths(path_file: Path) -> dict[str, dict[str, np.ndarray]]:
    table = pd.read_csv(path_file)
    scenarios = {}
    for scenario, frame in table.groupby("scenario", sort=False):
        time = np.sort(frame["time"].unique().astype(float))
        temp = frame.drop_duplicates("time").sort_values("time")["temperature_c"].to_numpy(float)
        path_ids = sorted(frame["path_id"].unique())
        paths = []
        for path_id in path_ids:
            p = frame[frame["path_id"] == path_id].sort_values("time")["G"].to_numpy(float)
            paths.append(p)
        scenarios[scenario] = {"time": time, "temperature_c": temp, "paths": np.asarray(paths), "path_ids": np.asarray(path_ids)}
    return scenarios


def acceleration_factor(temperature_c: np.ndarray, ea_j_per_mol: float, r_gas: float, t_ref_c: float) -> np.ndarray:
    t_k = np.asarray(temperature_c, dtype=float) + 273.15
    t_ref_k = t_ref_c + 273.15
    return np.exp((ea_j_per_mol / r_gas) * (1.0 / t_ref_k - 1.0 / t_k))


def cumulative_trapezoid(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(time, dtype=float)
    out[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(time))
    return out


def tau_after_initiation(time: np.ndarray, cumulative_accel: np.ndarray, t_ini: float) -> np.ndarray:
    h_ini = float(np.interp(t_ini, time, cumulative_accel))
    return np.maximum(cumulative_accel - h_ini, 0.0)


def onset_time(time: np.ndarray, loss: np.ndarray, threshold: float = 0.005) -> float:
    idxs = np.where(loss > threshold)[0]
    if idxs.size == 0:
        return float(time[-1])
    idx = int(idxs[0])
    if idx == 0:
        return float(time[0])
    l0, l1 = float(loss[idx - 1]), float(loss[idx])
    t0, t1 = float(time[idx - 1]), float(time[idx])
    if abs(l1 - l0) < 1e-12:
        return t1
    frac = np.clip((threshold - l0) / (l1 - l0), 0.0, 1.0)
    return float(t0 + frac * (t1 - t0))


def power_law_loss(time: np.ndarray, cumulative_accel: np.ndarray, t_ini: float, eta: float, m: float) -> np.ndarray:
    tau = tau_after_initiation(time, cumulative_accel, t_ini)
    return np.clip(eta * np.power(tau, m), 0.0, 1.0)


def fit_random_rate_power_law(scenario: str, time: np.ndarray, cumulative_accel: np.ndarray, paths: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    fitted = np.empty_like(paths)
    for path_id, g in enumerate(paths):
        loss = 1.0 - g
        t0 = onset_time(time, loss)
        active = loss > 0.005
        tau0 = tau_after_initiation(time, cumulative_accel, t0)
        eta0 = 1e-3
        m0 = 1.0
        if np.any(active):
            x = np.log(np.maximum(tau0[active], 1e-9))
            y = np.log(np.maximum(loss[active], 1e-9))
            if x.size >= 2 and np.std(x) > 1e-9:
                slope, intercept = np.polyfit(x, y, 1)
                m0 = float(np.clip(slope, 0.2, 5.0))
                eta0 = float(np.clip(np.exp(intercept), 1e-10, 10.0))

        def residual(params: np.ndarray) -> np.ndarray:
            return loss - power_law_loss(time, cumulative_accel, float(params[0]), float(params[1]), float(params[2]))

        result = least_squares(
            residual,
            x0=np.array([t0, eta0, m0]),
            bounds=([time[0], 1e-12, 0.2], [time[-1], 10.0, 5.0]),
            max_nfev=10000,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        t_ini, eta, m = result.x
        fit_loss = power_law_loss(time, cumulative_accel, float(t_ini), float(eta), float(m))
        fitted[path_id, :] = 1.0 - fit_loss
        rows.append(
            {
                "scenario": scenario,
                "path_id": path_id,
                "t_ini": float(t_ini),
                "eta": float(eta),
                "m": float(m),
                "RMSE_loss": float(np.sqrt(np.mean((loss - fit_loss) ** 2))),
                "success": bool(result.success),
            }
        )
    return pd.DataFrame(rows), fitted


def fit_common_random_rate_eta_m(scenarios: dict, rr_params: pd.DataFrame) -> tuple[float, float, float]:
    def objective(params: np.ndarray) -> float:
        eta, m = float(params[0]), float(params[1])
        sse = 0.0
        for scenario, data in scenarios.items():
            scenario_params = rr_params[rr_params["scenario"] == scenario].set_index("path_id")
            for path_id, g in enumerate(data["paths"]):
                loss = 1.0 - g
                t_ini = float(scenario_params.loc[path_id, "t_ini"])
                fit = power_law_loss(data["time"], data["cumulative_accel"], t_ini, eta, m)
                sse += float(np.sum((loss - fit) ** 2))
        return sse

    starts = [
        np.array([float(rr_params["eta"].mean()), float(rr_params["m"].mean())]),
        np.array([0.054731, 0.635198]),
        np.array([0.047591, 0.689493]),
    ]
    candidates = []
    for start in starts:
        result = minimize(objective, start, method="L-BFGS-B", bounds=[(1e-8, 1.0), (0.2, 5.0)], options={"maxiter": 5000})
        if np.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("Common random-rate eta/m fit failed.")
    best = min(candidates, key=lambda item: item.fun)
    return float(best.x[0]), float(best.x[1]), float(best.fun)


def gamma_increment_loglik(log_params: np.ndarray, delta_l: np.ndarray, delta_a: np.ndarray) -> float:
    c = np.exp(log_params[0])
    nu = np.exp(log_params[1])
    shape = c * delta_a
    ll = np.sum((shape - 1.0) * np.log(delta_l) - delta_l / nu - shape * np.log(nu) - gammaln(shape))
    return float(ll) if np.isfinite(ll) else -np.inf


def gamma_increments_for_m(scenarios: dict, init_table: pd.DataFrame, m: float) -> tuple[np.ndarray, np.ndarray, int]:
    all_dl = []
    all_da = []
    removed = 0
    for scenario, data in scenarios.items():
        time = data["time"]
        cum = data["cumulative_accel"]
        for path_id, g in enumerate(data["paths"]):
            row = init_table[(init_table["scenario"] == scenario) & (init_table["path_id"] == path_id)].iloc[0]
            tau = tau_after_initiation(time, cum, float(row["t_ini_gamma"]))
            a = np.power(tau, m)
            loss = 1.0 - g
            dl = np.diff(loss)
            da = np.diff(a)
            active = da > 0.0
            removed += int(np.sum(active & (dl <= 0.0)))
            keep = active & (dl > 0.0)
            all_dl.append(dl[keep])
            all_da.append(da[keep])
    return np.concatenate(all_dl), np.concatenate(all_da), removed


def fit_gamma_power_law(scenarios: dict, onset_threshold: float = 0.005) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    init_rows = []
    for scenario, data in scenarios.items():
        for path_id, g in enumerate(data["paths"]):
            loss = 1.0 - g
            init_rows.append({"scenario": scenario, "path_id": path_id, "t_ini_gamma": onset_time(data["time"], loss, onset_threshold)})
    init_table = pd.DataFrame(init_rows)

    def nll(params: np.ndarray) -> float:
        log_c, log_nu, raw_m = params
        m = 0.2 + 4.8 / (1.0 + np.exp(-raw_m))
        dl, da, _ = gamma_increments_for_m(scenarios, init_table, m)
        if dl.size == 0:
            return np.inf
        ll = gamma_increment_loglik(np.array([log_c, log_nu]), dl, da)
        return float(-ll) if np.isfinite(ll) else np.inf

    starts = [np.array([np.log(10.0), np.log(1e-3), 0.0]), np.array([np.log(1.0), np.log(1e-2), -1.0]), np.array([np.log(50.0), np.log(1e-4), 1.0])]
    results = []
    for start in starts:
        result = minimize(nll, start, method="L-BFGS-B", bounds=[(-30, 30), (-30, 30), (-10, 10)], options={"maxiter": 5000})
        if np.isfinite(result.fun):
            results.append(result)
    if not results:
        raise RuntimeError("Gamma-process power-law fit failed.")
    best = min(results, key=lambda item: item.fun)
    m = float(0.2 + 4.8 / (1.0 + np.exp(-best.x[2])))

    dl, da, removed = gamma_increments_for_m(scenarios, init_table, m)
    eta0 = max(float(np.sum(dl) / np.sum(da)), 1e-8)

    def nll_c_nu(log_params: np.ndarray) -> float:
        ll = gamma_increment_loglik(log_params, dl, da)
        return float(-ll) if np.isfinite(ll) else np.inf

    candidates = []
    for start in [np.log([1.0, eta0]), np.log([10.0, eta0 / 10.0]), np.log([100.0, eta0 / 100.0])]:
        result = minimize(nll_c_nu, start, method="L-BFGS-B", bounds=[(-30, 30), (-30, 30)], options={"maxiter": 5000})
        if np.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("Pooled gamma-process c/nu fit failed.")
    best_c_nu = min(candidates, key=lambda item: item.fun)
    c = float(np.exp(best_c_nu.x[0]))
    nu = float(np.exp(best_c_nu.x[1]))
    loglik = float(-best_c_nu.fun)
    params = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "c": c,
                "nu": nu,
                "eta_gamma": c * nu,
                "m_gamma": m,
                "loglik": loglik,
                "AIC": 2 * 3 - 2 * loglik,
                "n_increments": int(dl.size),
                "removed_nonpositive_delta_L": int(removed),
                "kinetics_fit": "pooled_c_nu_m_scenario_t_ini",
            }
            for scenario in scenarios
        ]
    )
    fitted = {}
    for scenario, data in scenarios.items():
        paths = []
        for _, row in init_table[init_table["scenario"] == scenario].iterrows():
            loss = power_law_loss(data["time"], data["cumulative_accel"], float(row["t_ini_gamma"]), c * nu, m)
            paths.append(1.0 - loss)
        fitted[scenario] = np.asarray(paths)
    return params, init_table, fitted


def fit_distribution_candidates(samples: np.ndarray, scenario: str, variable: str) -> pd.DataFrame:
    samples = np.asarray(samples, dtype=float)
    rows = []
    candidates = [
        ("lognormal", lambda x: lognorm.fit(x, floc=0.0), lambda x, p: lognorm.logpdf(x, *p)),
        ("gamma", lambda x: gamma_dist.fit(x, floc=0.0), lambda x, p: gamma_dist.logpdf(x, *p)),
        ("weibull", lambda x: weibull_min.fit(x, floc=0.0), lambda x, p: weibull_min.logpdf(x, *p)),
        ("normal", lambda x: norm.fit(x), lambda x, p: norm.logpdf(x, *p)),
    ]
    for name, fitter, logpdf in candidates:
        params = fitter(samples)
        ll = float(np.sum(logpdf(samples, params)))
        k = len(params)
        row = {"scenario": scenario, "variable": variable, "distribution": name, "loglik": ll, "AIC": 2 * k - 2 * ll, "BIC": k * np.log(samples.size) - 2 * ll}
        for idx, value in enumerate(params):
            row[f"param_{idx}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)


def sample_from_best_distribution(rng: np.random.Generator, fits: pd.DataFrame, scenario: str, variable: str, n: int) -> np.ndarray:
    row = fits[(fits["scenario"] == scenario) & (fits["variable"] == variable)].sort_values("AIC").iloc[0]
    params = [row[col] for col in row.index if col.startswith("param_") and pd.notna(row[col])]
    dist = row["distribution"]
    if dist == "lognormal":
        out = lognorm.rvs(*params, size=n, random_state=rng)
    elif dist == "gamma":
        out = gamma_dist.rvs(*params, size=n, random_state=rng)
    elif dist == "weibull":
        out = weibull_min.rvs(*params, size=n, random_state=rng)
    else:
        out = norm.rvs(*params, size=n, random_state=rng)
    return np.maximum(np.asarray(out, dtype=float), 1e-12)


def lognormal_mu_sigma(mean: float, cov: float) -> tuple[float, float]:
    sigma = np.sqrt(np.log1p(cov**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return float(mu), float(sigma)


def nhpp_cumulative(theta0: float, theta1: float, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    if abs(theta1) < 1e-12:
        return np.exp(theta0) * (end - start)
    return np.exp(theta0) / theta1 * (np.exp(theta1 * end) - np.exp(theta1 * start))


def read_load_model(table_dir: Path) -> tuple[float, float, float, float, float, float]:
    rate = pd.read_csv(table_dir / "selected_realization_nhpp_leyp_fit_summary.csv")
    nhpp = rate[rate["Model"] == "NHPP"].iloc[0]
    mag = pd.read_csv(table_dir / "selected_realization_event_magnitude_distribution_comparison.csv")
    weib = mag[mag["Model"] == "Weibull scale trend"].iloc[0]
    return (
        float(nhpp["theta_0"]),
        float(nhpp["theta_1"]),
        float(weib["Pressure_Threshold_Pa"]),
        float(weib["param_0"]),
        float(weib["param_1"]),
        float(np.exp(float(weib["param_2"]))),
    )


def load_survival(margin: np.ndarray, threshold: float, shape: float, scale: float) -> np.ndarray:
    excess = margin - threshold
    out = np.ones_like(excess)
    positive = excess > 0.0
    out[positive] = np.exp(-np.power(excess[positive] / scale, shape))
    return np.clip(out, 0.0, 1.0)


def simulate_rr_resistance(
    rng,
    config,
    scenario,
    fits,
    data,
    n,
    fitted_triplets: pd.DataFrame | None = None,
    constant_m: bool = False,
    common_eta: float | None = None,
    common_m: float | None = None,
):
    if fitted_triplets is None:
        t_ini = sample_from_best_distribution(rng, fits, scenario, "t_ini", n)
        eta = sample_from_best_distribution(rng, fits, scenario, "eta", n)
        m = np.clip(sample_from_best_distribution(rng, fits, scenario, "m", n), 0.2, 5.0)
        sampler = "independent_marginals"
    else:
        pool = fitted_triplets[fitted_triplets["scenario"] == scenario].reset_index(drop=True)
        idx = rng.integers(0, len(pool), size=n)
        sampled = pool.iloc[idx]
        t_ini = sampled["t_ini"].to_numpy(float)
        if common_eta is not None and common_m is not None:
            eta = np.full(n, float(common_eta))
            m = np.full(n, float(common_m))
            sampler = "bootstrap_tini_common_eta_m"
        elif common_m is not None:
            pooled_eta = fitted_triplets["eta"].to_numpy(float)
            eta = pooled_eta[rng.integers(0, pooled_eta.size, size=n)]
            m = np.full(n, float(common_m))
            sampler = "bootstrap_tini_pooled_eta_common_m"
        elif constant_m:
            eta = sampled["eta"].to_numpy(float)
            m = np.full(n, float(fitted_triplets["m"].mean()))
            sampler = "bootstrap_tini_eta_constant_m"
        else:
            eta = sampled["eta"].to_numpy(float)
            m = sampled["m"].to_numpy(float)
            sampler = "bootstrap_triplets"
    losses = np.array([power_law_loss(data["time"], data["cumulative_accel"], ti, e, mi) for ti, e, mi in zip(t_ini, eta, m)])
    caps = int(np.sum(losses >= 1.0))
    r_mu, r_sigma = lognormal_mu_sigma(config.resistance.initial_mean, config.resistance.initial_cov)
    r0 = rng.lognormal(r_mu, r_sigma, size=n)
    return r0[:, None] * (1.0 - losses), {"cap_count": caps, "t_ini": t_ini, "eta": eta, "m": m, "R0": r0, "sampler": sampler}


def simulate_gp_resistance(rng, config, scenario, gamma_params, gamma_init_fits, data, n):
    row = gamma_params[gamma_params["scenario"] == scenario].iloc[0]
    t_ini = sample_from_best_distribution(rng, gamma_init_fits, scenario, "t_ini_gamma", n)
    c, nu, m = float(row["c"]), float(row["nu"]), float(row["m_gamma"])
    losses = []
    cap_count = 0
    for ti in t_ini:
        tau = tau_after_initiation(data["time"], data["cumulative_accel"], ti)
        a = np.power(tau, m)
        da = np.diff(a)
        inc = np.zeros_like(da)
        active = da > 0
        inc[active] = rng.gamma(shape=c * da[active], scale=nu)
        l = np.zeros(data["time"].size)
        l[1:] = np.cumsum(inc)
        cap_count += int(np.sum(l > 1.0))
        losses.append(np.clip(l, 0.0, 1.0))
    r_mu, r_sigma = lognormal_mu_sigma(config.resistance.initial_mean, config.resistance.initial_cov)
    r0 = rng.lognormal(r_mu, r_sigma, size=n)
    return r0[:, None] * (1.0 - np.asarray(losses)), {"cap_count": cap_count, "t_ini": t_ini, "R0": r0}


def reliability_from_resistance(rng, config, resistance, time, theta0, theta1, threshold, alpha0, alpha1, weib_shape, stationary_load):
    n = resistance.shape[0]
    starts = np.arange(0.0, config.analysis.time_horizon_years, config.analysis.time_step_years)
    ends = starts + config.analysis.time_step_years
    mid = 0.5 * (starts + ends)
    lam = nhpp_cumulative(theta0, 0.0 if stationary_load else theta1, starts, ends)
    scale = np.full(mid.shape, np.exp(alpha0)) if stationary_load else np.exp(alpha0 + alpha1 * mid)
    r_mid = np.array([np.interp(mid, time, r) for r in resistance])
    dead = rng.normal(config.dead_load.mean, config.dead_load.mean * config.dead_load.cov, size=n)
    poe = np.empty_like(r_mid)
    for j in range(mid.size):
        poe[:, j] = load_survival(r_mid[:, j] - dead, threshold, weib_shape, scale[j])
    annual_mvf = lam[None, :] * poe
    ann_pf = np.mean(1.0 - np.exp(-annual_mvf), axis=0)
    cum_pf = np.mean(1.0 - np.exp(-np.cumsum(annual_mvf, axis=1)), axis=0)
    return pd.DataFrame(
        {
            "Service_Year": ends.astype(int),
            "Calendar_Year": (2026 + ends).astype(int),
            "Annual_Pf": ann_pf,
            "Cumulative_Pf": cum_pf,
            "Annual_Beta": -norm.ppf(np.clip(ann_pf, 1e-15, 1 - 1e-15)),
            "Cumulative_Beta": -norm.ppf(np.clip(cum_pf, 1e-15, 1 - 1e-15)),
            "Mean_Resistance": np.mean(r_mid, axis=0),
        }
    )


def plot_paths(outputs: dict[str, tuple[np.ndarray, np.ndarray]], out_file: Path, stepped: bool) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    styles = {
        "Base temperature": ("0.02", "--", "Base climate realizations", "Base climate mean"),
        "SSP5 delta_T=4 C": ("#004c99", "-", "Warming realizations", "Warming mean"),
    }
    for scenario, (years, paths) in outputs.items():
        color, ls, lab, mean_lab = styles[scenario]
        for idx, path in enumerate(paths[:14]):
            if stepped:
                ax.step(years, path, where="post", color=color, linestyle=ls, alpha=0.45, linewidth=0.8, label=lab if idx == 0 else None)
            else:
                ax.plot(years, path, color=color, linestyle=ls, alpha=0.45, linewidth=0.8, label=lab if idx == 0 else None)
        ax.plot(years, np.mean(paths, axis=0), color=color, linestyle=ls, linewidth=2.2, label=mean_lab)
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"Normalized resistance, $R(t)/R(0)$")
    ax.margins(x=0, y=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(results: pd.DataFrame, metric: str, ylabel: str, out_file: Path, log_y: bool) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    styles = {
        ("power_random_rate", "Base temperature"): ("0.02", (0, (5, 2)), "Random-rate, base"),
        ("power_random_rate", "SSP5 delta_T=4 C"): ("0.02", "-", "Random-rate, warming"),
        ("power_gamma_process", "Base temperature"): ("#004c99", (0, (3, 1, 1, 1)), "Gamma process, base"),
        ("power_gamma_process", "SSP5 delta_T=4 C"): ("#004c99", (0, (1, 1)), "Gamma process, warming"),
    }
    for key, sub in results.groupby(["degradation_model", "scenario"], sort=False):
        color, ls, label = styles[key]
        ax.plot(sub["Calendar_Year"], sub[metric], color=color, linestyle=ls, linewidth=1.45, label=label)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.margins(x=0, y=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sim", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/defaults.json")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data/reference")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/refit")
    parser.add_argument("--ea-j-per-mol", type=float, default=40000.0)
    parser.add_argument("--r-gas", type=float, default=8.314)
    parser.add_argument("--t-ref-c", type=float, default=20.0)
    args = parser.parse_args()

    if args.n_sim < 1:
        parser.error("--n-sim must be positive")
    config = WorkflowConfig.from_json(args.config)
    paths_file = args.input_dir / "degradation_model_fits/normalized_resistance_paths_used.csv"
    scenarios = read_paths(paths_file)
    for data in scenarios.values():
        accel = acceleration_factor(data["temperature_c"], args.ea_j_per_mol, args.r_gas, args.t_ref_c)
        if not np.all(np.isfinite(accel) & (accel > 0.0)):
            raise ValueError("Temperature acceleration must be positive and finite.")
        data["cumulative_accel"] = cumulative_trapezoid(data["time"], accel)
        if np.any(np.diff(data["cumulative_accel"]) < -1e-12):
            raise ValueError("Adjusted propagation time must be non-decreasing.")
        if not np.all((data["paths"] > 0.0) & (data["paths"] <= 1.0)):
            raise ValueError("All G paths must satisfy 0 < G <= 1.")
        if not np.all((1.0 - data["paths"] >= -1e-12) & (1.0 - data["paths"] <= 1.0 + 1e-12)):
            raise ValueError("All losses must satisfy 0 <= L <= 1.")

    out_table = args.output_dir / "tables/degradation_power_law_fits"
    out_fig = args.output_dir / "figures/degradation_power_law_fits"
    rel_fig = args.output_dir / "figures/nonstationary_reliability_power_law"
    rel_table = args.output_dir / "tables/nonstationary_reliability_power_law"
    for d in [out_table, out_fig, rel_fig, rel_table]:
        d.mkdir(parents=True, exist_ok=True)

    rr_rows = []
    rr_fit_paths = {}
    dist_rows = []
    for scenario, data in scenarios.items():
        rr, fitted = fit_random_rate_power_law(scenario, data["time"], data["cumulative_accel"], data["paths"])
        rr_rows.append(rr)
        rr_fit_paths[scenario] = fitted
        for var in ["t_ini", "eta", "m"]:
            dist_rows.append(fit_distribution_candidates(rr[var].to_numpy(float), scenario, var))
    rr_params = pd.concat(rr_rows, ignore_index=True)
    rr_dists = pd.concat(dist_rows, ignore_index=True)
    common_rr_eta, common_rr_m, common_rr_sse = fit_common_random_rate_eta_m(scenarios, rr_params)
    gp_params, gp_inits, gp_fitted = fit_gamma_power_law(scenarios)
    gp_init_dists = pd.concat(
        [fit_distribution_candidates(gp_inits[gp_inits["scenario"] == scenario]["t_ini_gamma"].to_numpy(float), scenario, "t_ini_gamma") for scenario in scenarios],
        ignore_index=True,
    )

    rr_params.to_csv(out_table / "power_random_rate_fitted_parameters.csv", index=False)
    rr_dists.to_csv(out_table / "power_random_rate_distribution_fits.csv", index=False)
    gp_params.to_csv(out_table / "power_gamma_process_fitted_parameters.csv", index=False)
    gp_inits.to_csv(out_table / "power_gamma_process_path_initiation_times.csv", index=False)
    gp_init_dists.to_csv(out_table / "power_gamma_process_initiation_distribution_fits.csv", index=False)
    pd.DataFrame(
        [
            {
                "eta_common": common_rr_eta,
                "m_common": common_rr_m,
                "SSE_all_paths": common_rr_sse,
                "fit": "random_rate_common_eta_m_scenario_t_ini",
            }
        ]
    ).to_csv(out_table / "power_random_rate_common_propagation_parameters.csv", index=False)

    years = {s: 2026 + data["time"] for s, data in scenarios.items()}
    plot_paths({s: (years[s], rr_fit_paths[s]) for s in scenarios}, out_fig / "power_random_rate_fitted_realizations.png", stepped=False)
    plot_paths({s: (years[s], gp_fitted[s]) for s in scenarios}, out_fig / "power_gamma_process_fitted_realizations.png", stepped=True)

    theta0, theta1, threshold, alpha0, alpha1, weib_shape = read_load_model(args.input_dir)
    rng = np.random.default_rng(args.seed)
    rel_rows = []
    sim_summaries = []
    rr_sim_paths = {}
    gp_sim_paths = {}
    for scenario, data in scenarios.items():
        stationary_load = scenario == "Base temperature"
        rr_res, rr_sim = simulate_rr_resistance(
            rng,
            config,
            scenario,
            rr_dists,
            data,
            args.n_sim,
            fitted_triplets=rr_params,
            constant_m=False,
            common_eta=None,
            common_m=common_rr_m,
        )
        rr_rel = reliability_from_resistance(rng, config, rr_res, data["time"], theta0, theta1, threshold, alpha0, alpha1, weib_shape, stationary_load)
        rr_rel.insert(0, "scenario", scenario)
        rr_rel.insert(0, "degradation_model", "power_random_rate")
        rel_rows.append(rr_rel)
        rr_sim_paths[scenario] = rr_res[:40] / rr_sim["R0"][:40, None]
        sim_summaries.append({"degradation_model": "power_random_rate", "scenario": scenario, "sampler": rr_sim["sampler"], "cap_count": rr_sim["cap_count"], "mean_t_ini": float(np.mean(rr_sim["t_ini"])), "mean_eta": float(np.mean(rr_sim["eta"])), "mean_m": float(np.mean(rr_sim["m"]))})

        gp_res, gp_sim = simulate_gp_resistance(rng, config, scenario, gp_params, gp_init_dists, data, args.n_sim)
        gp_rel = reliability_from_resistance(rng, config, gp_res, data["time"], theta0, theta1, threshold, alpha0, alpha1, weib_shape, stationary_load)
        gp_rel.insert(0, "scenario", scenario)
        gp_rel.insert(0, "degradation_model", "power_gamma_process")
        rel_rows.append(gp_rel)
        gp_sim_paths[scenario] = gp_res[:40] / gp_sim["R0"][:40, None]
        gp_row = gp_params[gp_params["scenario"] == scenario].iloc[0]
        sim_summaries.append({"degradation_model": "power_gamma_process", "scenario": scenario, "sampler": "gamma_process_increments", "cap_count": gp_sim["cap_count"], "mean_t_ini": float(np.mean(gp_sim["t_ini"])), "c": float(gp_row["c"]), "nu": float(gp_row["nu"]), "eta_gamma": float(gp_row["eta_gamma"]), "m_gamma": float(gp_row["m_gamma"])})

    rel = pd.concat(rel_rows, ignore_index=True)
    rel.to_csv(rel_table / "annual_reliability_results.csv", index=False)
    pd.DataFrame(sim_summaries).to_csv(rel_table / "simulation_input_summary.csv", index=False)
    final = rel[rel["Service_Year"] == config.analysis.time_horizon_years][["degradation_model", "scenario", "Cumulative_Pf", "Cumulative_Beta", "Mean_Resistance"]]
    final.to_csv(rel_table / "final_horizon_summary.csv", index=False)
    plot_paths({s: (years[s], rr_sim_paths[s]) for s in scenarios}, out_fig / "power_random_rate_generated_realizations.png", stepped=False)
    plot_paths({s: (years[s], gp_sim_paths[s]) for s in scenarios}, out_fig / "power_gamma_process_generated_realizations.png", stepped=True)
    plot_reliability(rel, "Annual_Pf", "Annual probability of failure", rel_fig / "annual_probability_of_failure.png", log_y=True)
    plot_reliability(rel, "Cumulative_Pf", "Cumulative probability of failure", rel_fig / "cumulative_probability_of_failure.png", log_y=True)
    plot_reliability(rel, "Annual_Beta", r"Annual reliability index, $\beta$", rel_fig / "annual_reliability_index.png", log_y=False)
    plot_reliability(rel, "Cumulative_Beta", r"Cumulative reliability index, $\beta$", rel_fig / "cumulative_reliability_index.png", log_y=False)

    model_summary = rr_params.groupby("scenario").agg(
        mean_t_ini=("t_ini", "mean"),
        cov_t_ini=("t_ini", lambda x: float(np.std(x, ddof=1) / np.mean(x))),
        mean_eta=("eta", "mean"),
        cov_eta=("eta", lambda x: float(np.std(x, ddof=1) / np.mean(x))),
        mean_m=("m", "mean"),
        cov_m=("m", lambda x: float(np.std(x, ddof=1) / np.mean(x))),
        mean_RMSE=("RMSE_loss", "mean"),
    ).reset_index()
    model_summary.to_csv(out_table / "power_law_model_parameter_summary.csv", index=False)

    print("Power-law degradation and reliability analysis")
    print("----------------------------------------------")
    print("Random-rate parameter summary:")
    print(model_summary.to_string(index=False))
    print("\nGamma-process parameters:")
    print(gp_params.to_string(index=False))
    print("\nFinal-horizon cumulative reliability summary:")
    print(final.to_string(index=False))
    print(f"\nTables: {out_table}")
    print(f"Reliability tables: {rel_table}")
    print(f"Figures: {out_fig}")
    print(f"Reliability figures: {rel_fig}")


if __name__ == "__main__":
    main()
