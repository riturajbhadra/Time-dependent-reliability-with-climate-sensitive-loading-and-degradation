"""Fit NHPP/LEYP to the selected wind realization and plot comparisons."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERCENTILE = 0.95
SERVICE_LIFE_YEARS = 75


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.edgecolor": "0.15",
            "axes.labelcolor": "0.05",
            "xtick.color": "0.05",
            "ytick.color": "0.05",
            "text.color": "0.05",
            "legend.frameon": False,
            "figure.dpi": 120,
        }
    )


@dataclass(frozen=True)
class GumbelTrend:
    loc_intercept: float
    loc_slope: float
    scale: float

    def loc(self, t: np.ndarray) -> np.ndarray:
        return self.loc_intercept + self.loc_slope * t


def nhpp_cumulative(theta_0: float, theta_1: float, start, end) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if abs(theta_1) < 1e-12:
        return np.exp(theta_0) * (end - start)
    return np.exp(theta_0) / theta_1 * (np.exp(theta_1 * end) - np.exp(theta_1 * start))


def nhpp_rate(theta_0: float, theta_1: float, t: np.ndarray) -> np.ndarray:
    return np.exp(theta_0 + theta_1 * np.asarray(t, dtype=float))


def leyp_loglik(event_times: np.ndarray, t_obs: float, theta_0: float, theta_1: float, a: float) -> float:
    if a < 0.0:
        return -np.inf
    event_times = np.sort(np.asarray(event_times, dtype=float))
    if event_times.size and (event_times[0] <= 0.0 or event_times[-1] >= t_obs):
        return -np.inf

    m = event_times.size
    left = np.concatenate(([0.0], event_times))
    right = np.concatenate((event_times, [t_obs]))
    intervals = nhpp_cumulative(theta_0, theta_1, left, right)
    if np.any(~np.isfinite(intervals)) or np.any(intervals < 0.0):
        return -np.inf
    event_term = np.sum(np.log1p(a * np.arange(m, dtype=float)))
    baseline_term = np.sum(theta_0 + theta_1 * event_times)
    compensator = np.sum((1.0 + a * np.arange(m + 1, dtype=float)) * intervals)
    ll = event_term + baseline_term - compensator
    return float(ll) if np.isfinite(ll) else -np.inf


def fit_nhpp(event_times: np.ndarray, t_obs: float) -> tuple[float, float, float]:
    event_times = np.sort(np.asarray(event_times, dtype=float))
    empirical_rate = len(event_times) / t_obs

    def nll(params: np.ndarray) -> float:
        theta_0, theta_1 = params
        ll = leyp_loglik(event_times, t_obs, theta_0, theta_1, 0.0)
        return float(-ll) if np.isfinite(ll) else np.inf

    base = np.log(max(empirical_rate, 1e-12))
    starts = [
        np.array([base, 0.0]),
        np.array([base, 0.02]),
        np.array([base, -0.02]),
        np.array([base - 0.5, 0.05]),
        np.array([base + 0.5, -0.05]),
    ]
    candidates = []
    for start in starts:
        result = minimize(
            nll,
            start,
            method="L-BFGS-B",
            bounds=[(-20.0, 20.0), (-5.0, 5.0)],
            options={"maxiter": 5000, "ftol": 1e-10},
        )
        if np.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("NHPP fit failed for all starting points.")
    best = min(candidates, key=lambda item: item.fun)
    return float(best.x[0]), float(best.x[1]), float(-best.fun)


def fit_leyp_joint(event_times: np.ndarray, t_obs: float, a_upper: float = 5.0) -> tuple[float, float, float, float]:
    event_times = np.sort(np.asarray(event_times, dtype=float))
    empirical_rate = len(event_times) / t_obs

    def nll(params: np.ndarray) -> float:
        theta_0, theta_1, a = params
        ll = leyp_loglik(event_times, t_obs, theta_0, theta_1, a)
        return float(-ll) if np.isfinite(ll) else np.inf

    base = np.log(max(empirical_rate, 1e-12))
    starts = [
        np.array([base, 0.0, 0.0]),
        np.array([base, 0.0, 0.05]),
        np.array([base, 0.0, 0.25]),
        np.array([base, 0.05, 0.25]),
        np.array([base, -0.05, 0.25]),
        np.array([base - 0.5, 0.10, 0.50]),
        np.array([base + 0.5, -0.10, 0.50]),
        np.array([0.0, 0.0, 1.0]),
    ]
    candidates = []
    for start in starts:
        result = minimize(
            nll,
            start,
            method="L-BFGS-B",
            bounds=[(-20.0, 20.0), (-5.0, 5.0), (0.0, a_upper)],
            options={"maxiter": 5000, "ftol": 1e-10},
        )
        if np.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("Joint LEYP fit failed for all starting points.")
    best = min(candidates, key=lambda item: item.fun)
    return float(best.x[0]), float(best.x[1]), float(best.x[2]), float(-best.fun)


def profile_likelihood_a(event_times: np.ndarray, t_obs: float, a_grid: np.ndarray) -> pd.DataFrame:
    event_times = np.sort(np.asarray(event_times, dtype=float))
    empirical_rate = len(event_times) / t_obs
    base = np.log(max(empirical_rate, 1e-12))
    prev = np.array([base, 0.0])
    rows = []

    for a in a_grid:
        def nll_theta(theta: np.ndarray) -> float:
            ll = leyp_loglik(event_times, t_obs, float(theta[0]), float(theta[1]), float(a))
            return float(-ll) if np.isfinite(ll) else np.inf

        starts = [prev, np.array([base, 0.0]), np.array([base, 0.05]), np.array([base, -0.05])]
        candidates = []
        for start in starts:
            result = minimize(
                nll_theta,
                start,
                method="L-BFGS-B",
                bounds=[(-20.0, 20.0), (-5.0, 5.0)],
                options={"maxiter": 3000, "ftol": 1e-10},
            )
            if np.isfinite(result.fun):
                candidates.append(result)
        if not candidates:
            rows.append({"a": float(a), "theta_0": np.nan, "theta_1": np.nan, "loglik": -np.inf})
            continue
        best = min(candidates, key=lambda item: item.fun)
        prev = best.x
        rows.append({"a": float(a), "theta_0": float(best.x[0]), "theta_1": float(best.x[1]), "loglik": float(-best.fun)})

    profile = pd.DataFrame(rows)
    profile["loglik_shifted"] = profile["loglik"] - profile["loglik"].max()
    return profile


def leyp_mean_count(theta_0: float, theta_1: float, a: float, t: np.ndarray) -> np.ndarray:
    baseline = nhpp_cumulative(theta_0, theta_1, 0.0, t)
    if a <= 1e-12:
        return baseline
    exponent = np.clip(a * baseline, -700.0, 700.0)
    return np.expm1(exponent) / a


def fit_gumbel_location_trend(times: np.ndarray, values: np.ndarray) -> GumbelTrend:
    slope, intercept = np.polyfit(times, values, deg=1)
    scale0 = max(float(np.std(values, ddof=1)) * np.sqrt(6.0) / np.pi, 1e-6)

    def nll(params: np.ndarray) -> float:
        mu0, mu1, log_scale = params
        scale = np.exp(log_scale)
        z = (values - (mu0 + mu1 * times)) / scale
        val = np.sum(log_scale + z + np.exp(-z))
        return float(val) if np.isfinite(val) else np.inf

    result = minimize(nll, np.array([intercept, slope, np.log(scale0)]), method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Event-magnitude Gumbel trend fit failed: {result.message}")
    mu0, mu1, log_scale = result.x
    return GumbelTrend(float(mu0), float(mu1), float(np.exp(log_scale)))


def compound_poisson_gumbel_q95(loc: np.ndarray, scale: float, intensity: np.ndarray) -> np.ndarray:
    target_event_cdf = 1.0 + np.log(PERCENTILE) / np.asarray(intensity, dtype=float)
    if np.any((target_event_cdf <= 0.0) | (target_event_cdf >= 1.0)):
        print("Warning: percentile-domain condition failed for at least one year.")
    target_event_cdf = np.clip(target_event_cdf, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    return np.asarray(loc, dtype=float) - scale * np.log(-np.log(target_event_cdf))


def main() -> None:
    configure_matplotlib()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--realization",
        type=Path,
        default=Path("data/reference/selected_high_nonstationarity_realization.csv"),
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("data/reference/selected_high_nonstationarity_events.csv"),
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("selected_realization_nhpp_leyp"),
    )
    args = parser.parse_args()

    realization_file = (PROJECT_ROOT / args.realization).resolve() if not args.realization.is_absolute() else args.realization
    events_file = (PROJECT_ROOT / args.events).resolve() if not args.events.is_absolute() else args.events
    output_prefix = args.out_prefix

    realization = pd.read_csv(realization_file)
    events = pd.read_csv(events_file)
    realization["Date"] = pd.to_datetime(realization["Date"], errors="coerce")
    events["Date"] = pd.to_datetime(events["Date"], errors="coerce")

    obs_start = pd.Timestamp("2026-01-01")
    obs_end = pd.Timestamp("2101-01-01")
    t_obs = float((obs_end - obs_start) / pd.Timedelta(days=365.25))

    events = events[(events["Date"] > obs_start) & (events["Date"] < obs_end)].copy()
    event_times = ((events["Date"] - obs_start) / pd.Timedelta(days=365.25)).to_numpy(dtype=float)
    event_magnitudes = events["W_pressure"].to_numpy(dtype=float)
    order = np.argsort(event_times)
    event_times = event_times[order]
    event_magnitudes = event_magnitudes[order]

    if not np.all((event_times > 0.0) & (event_times < t_obs)):
        raise ValueError("Selected realization event times must satisfy 0 < t_i < T_obs.")

    nhpp_theta_0, nhpp_theta_1, nhpp_ll = fit_nhpp(event_times, t_obs)
    leyp_theta_0, leyp_theta_1, a_hat, leyp_ll = fit_leyp_joint(event_times, t_obs)
    magnitude_trend = fit_gumbel_location_trend(event_times, event_magnitudes)

    years = np.arange(1, SERVICE_LIFE_YEARS + 1, dtype=float)
    starts = years - 1.0
    ends = years
    hpp_rate = float(np.exp(nhpp_theta_0))
    lambda_hpp = np.full(years.shape, hpp_rate)
    lambda_nhpp = nhpp_cumulative(nhpp_theta_0, nhpp_theta_1, starts, ends)
    lambda_leyp = leyp_mean_count(leyp_theta_0, leyp_theta_1, a_hat, ends) - leyp_mean_count(
        leyp_theta_0,
        leyp_theta_1,
        a_hat,
        starts,
    )

    q_hpp = compound_poisson_gumbel_q95(magnitude_trend.loc(np.zeros_like(years)), magnitude_trend.scale, lambda_hpp)
    q_nhpp = compound_poisson_gumbel_q95(magnitude_trend.loc(starts), magnitude_trend.scale, lambda_nhpp)
    q_leyp = compound_poisson_gumbel_q95(magnitude_trend.loc(starts), magnitude_trend.scale, lambda_leyp)

    table_dir = PROJECT_ROOT / "outputs/tables"
    figure_dir = PROJECT_ROOT / "outputs/figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    percentile_table = pd.DataFrame(
        {
            "Service_Year": years.astype(int),
            "HPP_q95": q_hpp,
            "NHPP_q95": q_nhpp,
            "LEYP_q95_approx": q_leyp,
            "HPP_Annual_Intensity": lambda_hpp,
            "NHPP_Annual_Intensity": lambda_nhpp,
            "LEYP_Annual_Mean_Count": lambda_leyp,
        }
    )
    percentile_csv = table_dir / f"{output_prefix}_annual_q95.csv"
    percentile_table.to_csv(percentile_csv, index=False)

    count_time = np.linspace(0.0, t_obs, 500)
    count_year_axis = 2026.0 + count_time
    event_year_axis = 2026.0 + event_times
    empirical_count = np.searchsorted(event_times, count_time, side="right")
    hpp_count = hpp_rate * count_time
    nhpp_count = nhpp_cumulative(nhpp_theta_0, nhpp_theta_1, 0.0, count_time)
    leyp_count = leyp_mean_count(leyp_theta_0, leyp_theta_1, a_hat, count_time)

    count_table = pd.DataFrame(
        {
            "Time": count_time,
            "Calendar_Year": count_year_axis,
            "Empirical_Cumulative_Count": empirical_count,
            "HPP_Expected_Cumulative_Count": hpp_count,
            "NHPP_Expected_Cumulative_Count": nhpp_count,
            "LEYP_Expected_Cumulative_Count": leyp_count,
        }
    )
    count_csv = table_dir / f"{output_prefix}_cumulative_counts.csv"
    count_table.to_csv(count_csv, index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        event_year_axis,
        np.arange(1, event_times.size + 1),
        marker="+",
        s=22,
        linewidths=0.7,
        color="0.05",
        label="CanRCM4 realization",
    )
    leyp_label = "LEYP, $\\hat{a}=0$" if a_hat <= 1e-12 else "LEYP"
    ax.plot(count_year_axis, hpp_count, linewidth=1.8, color="0.30", linestyle=":", label="HPP")
    ax.plot(count_year_axis, nhpp_count, linewidth=2.2, color="0.18", linestyle="-", label="NHPP")
    ax.plot(
        count_year_axis,
        leyp_count,
        linewidth=1.2,
        color="0.50",
        linestyle="none",
        marker="s",
        markevery=32,
        markersize=4,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label=leyp_label,
    )
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative number of events")
    ax.margins(x=0.0, y=0.0)
    ax.legend()
    fig.tight_layout()
    count_figure = figure_dir / f"{output_prefix}_cumulative_counts.png"
    fig.savefig(count_figure, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(percentile_table["Service_Year"], percentile_table["HPP_q95"], linewidth=1.8, color="0.30", linestyle=":", label="HPP")
    ax.plot(percentile_table["Service_Year"], percentile_table["NHPP_q95"], linewidth=2.2, color="0.18", linestyle="-", label="NHPP")
    ax.plot(
        percentile_table["Service_Year"],
        percentile_table["LEYP_q95_approx"],
        linewidth=1.2,
        color="0.50",
        linestyle="none",
        marker="s",
        markevery=5,
        markersize=4,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label=leyp_label,
    )
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.set_xlabel(r"Service year, $t$")
    ax.set_ylabel(r"Annual maximum wind load effect, $q_{0.95}$ (Pa)")
    ax.margins(x=0.0, y=0.0)
    ax.legend()
    fig.tight_layout()
    percentile_figure = figure_dir / f"{output_prefix}_annual_q95.png"
    fig.savefig(percentile_figure, dpi=300, bbox_inches="tight")
    plt.close(fig)

    n_events = event_times.size
    nhpp_aic = 2 * 2 - 2 * nhpp_ll
    leyp_aic = 2 * 3 - 2 * leyp_ll
    nhpp_bic = 2 * np.log(max(n_events, 1)) - 2 * nhpp_ll
    leyp_bic = 3 * np.log(max(n_events, 1)) - 2 * leyp_ll
    fit_summary = pd.DataFrame(
        [
            {
                "Model": "HPP",
                "theta_0": np.log(hpp_rate),
                "theta_1": 0.0,
                "a": 0.0,
                "loglik": np.nan,
                "AIC": np.nan,
                "BIC": np.nan,
            },
            {
                "Model": "NHPP",
                "theta_0": nhpp_theta_0,
                "theta_1": nhpp_theta_1,
                "a": 0.0,
                "loglik": nhpp_ll,
                "AIC": nhpp_aic,
                "BIC": nhpp_bic,
            },
            {
                "Model": "LEYP",
                "theta_0": leyp_theta_0,
                "theta_1": leyp_theta_1,
                "a": a_hat,
                "loglik": leyp_ll,
                "AIC": leyp_aic,
                "BIC": leyp_bic,
            },
        ]
    )
    summary_csv = table_dir / f"{output_prefix}_fit_summary.csv"
    fit_summary.to_csv(summary_csv, index=False)

    profile_grid = np.unique(np.r_[0.0, np.logspace(-10.0, 0.0, 160)])
    profile = profile_likelihood_a(event_times, t_obs, profile_grid)
    profile_csv = table_dir / f"{output_prefix}_leyp_profile_likelihood.csv"
    profile.to_csv(profile_csv, index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(profile["a"], profile["loglik_shifted"], linewidth=1.8, color="0.12")
    ax.axvline(a_hat, color="0.45", linewidth=1.2, linestyle=":")
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.set_xlabel(r"Candidate LEYP parameter, $a$")
    ax.set_ylabel("Profile log-likelihood, shifted")
    ax.margins(x=0.0, y=0.0)
    fig.tight_layout()
    profile_figure = figure_dir / f"{output_prefix}_leyp_profile_likelihood.png"
    fig.savefig(profile_figure, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Selected realization NHPP/LEYP fit")
    print("----------------------------------")
    print(f"Events: {n_events}")
    print(f"Observation period: {t_obs:.6g} years")
    print(f"HPP rate exp(NHPP theta_0): {hpp_rate:.6g}")
    print(f"NHPP theta_0: {nhpp_theta_0:.6g}")
    print(f"NHPP theta_1: {nhpp_theta_1:.6g}")
    print(f"NHPP log-likelihood: {nhpp_ll:.6g}")
    print(f"NHPP expected events: {float(nhpp_cumulative(nhpp_theta_0, nhpp_theta_1, 0.0, t_obs)):.6g}")
    print(f"LEYP theta_0: {leyp_theta_0:.6g}")
    print(f"LEYP theta_1: {leyp_theta_1:.6g}")
    print(f"LEYP a_hat: {a_hat:.6g}")
    print(f"LEYP log-likelihood: {leyp_ll:.6g}")
    print(f"NHPP AIC/BIC: {nhpp_aic:.6g} / {nhpp_bic:.6g}")
    print(f"LEYP AIC/BIC: {leyp_aic:.6g} / {leyp_bic:.6g}")
    print(f"Event magnitude Gumbel mu0: {magnitude_trend.loc_intercept:.6g}")
    print(f"Event magnitude Gumbel mu1: {magnitude_trend.loc_slope:.6g}")
    print(f"Event magnitude Gumbel beta: {magnitude_trend.scale:.6g}")
    print(f"Fit summary CSV: {summary_csv}")
    print(f"LEYP profile CSV: {profile_csv}")
    print(f"LEYP profile figure: {profile_figure}")
    print(f"Cumulative count CSV: {count_csv}")
    print(f"Cumulative count figure: {count_figure}")
    print(f"Annual q95 CSV: {percentile_csv}")
    print(f"Annual q95 figure: {percentile_figure}")


if __name__ == "__main__":
    main()
