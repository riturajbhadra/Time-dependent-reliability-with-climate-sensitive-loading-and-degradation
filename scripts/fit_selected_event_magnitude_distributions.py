"""Compare candidate distributions for selected wind-event magnitudes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from wind_reliability.config import WorkflowConfig

from scipy.stats import gamma, genpareto, gumbel_r, lognorm, norm, weibull_min


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
            "legend.fontsize": 10,
            "axes.edgecolor": "0.15",
            "axes.labelcolor": "0.05",
            "xtick.color": "0.05",
            "ytick.color": "0.05",
            "text.color": "0.05",
            "legend.frameon": False,
            "figure.dpi": 120,
        }
    )


def read_events(events_file: Path) -> tuple[np.ndarray, np.ndarray]:
    events = pd.read_csv(events_file)
    events["Date"] = pd.to_datetime(events["Date"], errors="coerce")
    obs_start = pd.Timestamp("2026-01-01")
    obs_end = pd.Timestamp("2101-01-01")
    events = events[(events["Date"] > obs_start) & (events["Date"] < obs_end)].copy()
    times = ((events["Date"] - obs_start) / pd.Timedelta(days=365.25)).to_numpy(dtype=float)
    values = events["W_pressure"].to_numpy(dtype=float)
    order = np.argsort(times)
    return times[order], values[order]


def fit_location_scale_model(
    name: str,
    times: np.ndarray,
    values: np.ndarray,
    nll_func,
    start: np.ndarray,
    bounds: list[tuple[float | None, float | None]],
    cdf_func,
) -> dict:
    result = minimize(nll_func, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000, "ftol": 1e-10})
    if not result.success:
        raise RuntimeError(f"{name} fit failed: {result.message}")
    loglik = -float(result.fun)
    k = len(result.x)
    n = len(values)
    pit = np.clip(cdf_func(result.x, times, values), 1e-12, 1.0 - 1e-12)
    return {
        "model": name,
        "params": result.x.astype(float),
        "loglik": loglik,
        "k": k,
        "AIC": 2 * k - 2 * loglik,
        "BIC": k * np.log(n) - 2 * loglik,
        "PIT": pit,
    }


def fit_candidates(times: np.ndarray, values: np.ndarray) -> list[dict]:
    slope, intercept = np.polyfit(times, values, deg=1)
    sd0 = max(float(np.std(values, ddof=1)), 1e-6)
    mean_positive = np.maximum(intercept + slope * times, 1e-6)
    log_slope, log_intercept = np.polyfit(times, np.log(values), deg=1)
    cv0 = max(sd0 / max(float(np.mean(values)), 1e-6), 1e-3)
    shape0 = max(1.0 / cv0**2, 1e-3)

    fits = []

    def gumbel_nll(p: np.ndarray) -> float:
        mu = p[0] + p[1] * times
        beta = np.exp(p[2])
        z = (values - mu) / beta
        return float(np.sum(np.log(beta) + z + np.exp(-z)))

    fits.append(
        fit_location_scale_model(
            "Gumbel location trend",
            times,
            values,
            gumbel_nll,
            np.array([intercept, slope, np.log(sd0 * np.sqrt(6.0) / np.pi)]),
            [(None, None), (None, None), (None, None)],
            lambda p, t, x: gumbel_r.cdf(x, loc=p[0] + p[1] * t, scale=np.exp(p[2])),
        )
    )

    def normal_nll(p: np.ndarray) -> float:
        mu = p[0] + p[1] * times
        sigma = np.exp(p[2])
        z = (values - mu) / sigma
        return float(np.sum(np.log(sigma) + 0.5 * np.log(2.0 * np.pi) + 0.5 * z**2))

    fits.append(
        fit_location_scale_model(
            "Normal location trend",
            times,
            values,
            normal_nll,
            np.array([intercept, slope, np.log(sd0)]),
            [(None, None), (None, None), (None, None)],
            lambda p, t, x: norm.cdf(x, loc=p[0] + p[1] * t, scale=np.exp(p[2])),
        )
    )

    def lognormal_nll(p: np.ndarray) -> float:
        mu_log = p[0] + p[1] * times
        sigma = np.exp(p[2])
        z = (np.log(values) - mu_log) / sigma
        return float(np.sum(np.log(values) + np.log(sigma) + 0.5 * np.log(2.0 * np.pi) + 0.5 * z**2))

    fits.append(
        fit_location_scale_model(
            "Lognormal log-location trend",
            times,
            values,
            lognormal_nll,
            np.array([log_intercept, log_slope, np.log(max(float(np.std(np.log(values), ddof=1)), 1e-6))]),
            [(None, None), (None, None), (None, None)],
            lambda p, t, x: lognorm.cdf(x, s=np.exp(p[2]), scale=np.exp(p[0] + p[1] * t)),
        )
    )

    def gamma_nll(p: np.ndarray) -> float:
        mean = np.exp(p[0] + p[1] * times)
        shape = np.exp(p[2])
        scale = mean / shape
        return float(np.sum(gammaln(shape) + shape * np.log(scale) - (shape - 1.0) * np.log(values) + values / scale))

    fits.append(
        fit_location_scale_model(
            "Gamma mean trend",
            times,
            values,
            gamma_nll,
            np.array([np.log(np.mean(mean_positive)), 0.0, np.log(shape0)]),
            [(None, None), (None, None), (None, None)],
            lambda p, t, x: gamma.cdf(x, a=np.exp(p[2]), scale=np.exp(p[0] + p[1] * t) / np.exp(p[2])),
        )
    )

    def weibull_nll(p: np.ndarray) -> float:
        shape = np.exp(p[2])
        scale = np.exp(p[0] + p[1] * times)
        z = values / scale
        return float(np.sum(np.log(scale) - np.log(shape) - (shape - 1.0) * np.log(z) + z**shape))

    fits.append(
        fit_location_scale_model(
            "Weibull scale trend",
            times,
            values,
            weibull_nll,
            np.array([np.log(np.mean(values)), 0.0, np.log(2.0)]),
            [(None, None), (None, None), (None, None)],
            lambda p, t, x: weibull_min.cdf(x, c=np.exp(p[2]), scale=np.exp(p[0] + p[1] * t)),
        )
    )

    def gpd_nll(p: np.ndarray) -> float:
        scale = np.exp(p[0] + p[1] * times)
        xi = p[2]
        z = values / scale
        support = 1.0 + xi * z
        if np.any(support <= 0.0):
            return np.inf
        if abs(xi) < 1e-8:
            return float(np.sum(np.log(scale) + z))
        return float(np.sum(np.log(scale) + (1.0 / xi + 1.0) * np.log(support)))

    fits.append(
        fit_location_scale_model(
            "GPD scale trend",
            times,
            values,
            gpd_nll,
            np.array([np.log(np.mean(values)), 0.0, 0.0]),
            [(None, None), (None, None), (-1.0, 1.0)],
            lambda p, t, x: genpareto.cdf(x, c=p[2], loc=0.0, scale=np.exp(p[0] + p[1] * t)),
        )
    )

    return fits


def plot_pit_qq(fits: list[dict], out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    n = len(fits[0]["PIT"])
    empirical = (np.arange(1, n + 1) - 0.5) / n
    colors = ["0.05", "0.20", "0.35", "0.50", "0.65", "0.80"]
    for idx, fit in enumerate(fits):
        ax.plot(empirical, np.sort(fit["PIT"]), linewidth=1.3, color=colors[idx], label=fit["model"])
    ax.plot([0.0, 1.0], [0.0, 1.0], color="0.05", linestyle=":", linewidth=1.0)
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.set_xlabel("Uniform quantile")
    ax.set_ylabel("Fitted probability integral transform")
    ax.margins(x=0.0, y=0.0)
    ax.legend()
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("data/reference/selected_high_nonstationarity_events.csv"))
    parser.add_argument("--out-prefix", type=str, default="selected_realization_event_magnitude_distribution")
    parser.add_argument("--fit-excess", action="store_true", default=True)
    args = parser.parse_args()

    events_file = (PROJECT_ROOT / args.events).resolve() if not args.events.is_absolute() else args.events
    times, values = read_events(events_file)
    cfg = WorkflowConfig.from_json(PROJECT_ROOT / "config/defaults.json")
    threshold_pressure = (
        0.5
        * cfg.wind.rho_air
        * cfg.wind.c_e
        * cfg.wind.c_g
        * cfg.wind.c_h
        * cfg.analysis.threshold_wind_speed**2
    )
    fit_values = values - threshold_pressure if args.fit_excess else values
    if np.any(fit_values <= 0.0):
        raise ValueError("All shifted event magnitudes must be positive for excess-distribution fitting.")
    fits = fit_candidates(times, fit_values)

    rows = []
    for fit in fits:
        row = {
            "Model": fit["model"],
            "Fitted_Variable": "Pressure excess above threshold" if args.fit_excess else "Raw pressure",
            "Pressure_Threshold_Pa": threshold_pressure if args.fit_excess else 0.0,
            "LogLik": fit["loglik"],
            "Num_Params": fit["k"],
            "AIC": fit["AIC"],
            "BIC": fit["BIC"],
        }
        for idx, value in enumerate(fit["params"]):
            row[f"param_{idx}"] = value
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    table["Delta_AIC"] = table["AIC"] - table["AIC"].min()
    table["Delta_BIC"] = table["BIC"] - table["BIC"].min()

    table_dir = PROJECT_ROOT / "outputs/tables"
    figure_dir = PROJECT_ROOT / "outputs/figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    out_csv = table_dir / f"{args.out_prefix}_comparison.csv"
    out_png = figure_dir / f"{args.out_prefix}_pit_qq.png"
    table.to_csv(out_csv, index=False)
    plot_pit_qq(sorted(fits, key=lambda item: item["AIC"]), out_png)

    print("Selected realization event-magnitude distribution comparison")
    print("------------------------------------------------------------")
    print(f"Events: {len(values)}")
    if args.fit_excess:
        print(f"Fitted variable: pressure excess Y = W_pressure - {threshold_pressure:.6g} Pa")
        print("Use X = threshold + Y for annual maximum pressure percentiles.")
    else:
        print("Fitted variable: raw event pressure W_pressure")
    print(table[["Model", "LogLik", "AIC", "BIC", "Delta_AIC", "Delta_BIC"]].to_string(index=False))
    print(f"Best by AIC: {table.iloc[0]['Model']}")
    print(f"Best by BIC: {table.sort_values('BIC').iloc[0]['Model']}")
    print(f"Comparison CSV: {out_csv}")
    print(f"PIT QQ figure: {out_png}")


if __name__ == "__main__":
    main()
