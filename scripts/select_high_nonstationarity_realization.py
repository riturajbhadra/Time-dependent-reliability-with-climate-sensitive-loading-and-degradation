"""Select the inferred wind realization with strongest future non-stationarity.

The raw combined wind file does not contain a realization/model identifier. This
script infers provisional realizations from row-order date resets, then ranks
complete future blocks by positive trends in threshold-event counts and event
magnitudes.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wind_reliability.config import WorkflowConfig  # noqa: E402


def slope_or_nan(x: pd.Index | np.ndarray, y: pd.Series | np.ndarray) -> tuple[float, float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(y_arr)
    if mask.sum() < 3 or np.all(y_arr[mask] == y_arr[mask][0]):
        return np.nan, np.nan, np.nan
    result = linregress(x_arr[mask], y_arr[mask])
    return float(result.slope), float(result.rvalue), float(result.pvalue)


def main() -> None:
    config = WorkflowConfig.from_json(PROJECT_ROOT / "config/defaults.json")
    raw_file = PROJECT_ROOT / config.paths.input_csv
    output_dir = PROJECT_ROOT / config.paths.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_file)
    df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]], errors="coerce")
    date_reset = df["Date"].diff().dt.days.fillna(1) < 0
    df["Realization_Inferred"] = date_reset.cumsum() + 1

    w = config.wind
    df["W_pressure"] = 0.5 * w.rho_air * w.c_e * w.c_g * w.c_h * np.square(df["MaxSfcWind"])

    future = df[df["Year"] > config.analysis.future_start_year].copy()
    events = future[future["MaxSfcWind"] > config.analysis.threshold_wind_speed].copy()

    rows = []
    for rid, block in df.groupby("Realization_Inferred"):
        block_future = future[future["Realization_Inferred"] == rid]
        block_events = events[events["Realization_Inferred"] == rid]
        if block_future.empty:
            continue

        years = np.arange(int(block_future["Year"].min()), int(block_future["Year"].max()) + 1)
        annual_counts = block_events.groupby("Year").size().reindex(years, fill_value=0)
        annual_event_mean = block_events.groupby("Year")["W_pressure"].mean()
        annual_event_p95 = block_events.groupby("Year")["W_pressure"].quantile(0.95)
        annual_max = block_future.groupby("Year")["W_pressure"].max()

        count_slope, count_r, count_p = slope_or_nan(annual_counts.index, annual_counts.values)
        mean_slope, mean_r, mean_p = slope_or_nan(annual_event_mean.index, annual_event_mean.values)
        p95_slope, p95_r, p95_p = slope_or_nan(annual_event_p95.index, annual_event_p95.values)
        max_slope, max_r, max_p = slope_or_nan(annual_max.index, annual_max.values)

        rows.append(
            {
                "Realization_Inferred": int(rid),
                "Rows": int(len(block)),
                "Start": block["Date"].min().date(),
                "End": block["Date"].max().date(),
                "Future_Years": int(len(years)),
                "Events": int(len(block_events)),
                "Event_Rate_per_year": float(len(block_events) / len(years)),
                "Annual_Count_Slope_events_per_year": count_slope,
                "Annual_Count_R": count_r,
                "Annual_Count_P": count_p,
                "Event_Mean_Pressure_Slope_Pa_per_year": mean_slope,
                "Event_Mean_R": mean_r,
                "Event_Mean_P": mean_p,
                "Event_P95_Pressure_Slope_Pa_per_year": p95_slope,
                "Event_P95_R": p95_r,
                "Event_P95_P": p95_p,
                "Annual_Max_Pressure_Slope_Pa_per_year": max_slope,
                "Annual_Max_R": max_r,
                "Annual_Max_P": max_p,
            }
        )

    summary = pd.DataFrame(rows)
    complete = summary[summary["Future_Years"] >= 50].copy()
    for col in [
        "Annual_Count_Slope_events_per_year",
        "Event_Mean_Pressure_Slope_Pa_per_year",
        "Event_P95_Pressure_Slope_Pa_per_year",
    ]:
        values = complete[col].clip(lower=0.0)
        denom = values.max()
        complete[f"{col}_Norm"] = values / denom if denom > 0 else 0.0

    complete["Nonstationarity_Score"] = (
        complete["Annual_Count_Slope_events_per_year_Norm"]
        + complete["Event_Mean_Pressure_Slope_Pa_per_year_Norm"]
        + complete["Event_P95_Pressure_Slope_Pa_per_year_Norm"]
    )
    summary = summary.merge(
        complete[["Realization_Inferred", "Nonstationarity_Score"]],
        on="Realization_Inferred",
        how="left",
    )

    selected_id = int(complete.loc[complete["Nonstationarity_Score"].idxmax(), "Realization_Inferred"])
    selected = df[df["Realization_Inferred"] == selected_id].copy()
    selected_events = events[events["Realization_Inferred"] == selected_id].copy()

    summary.to_csv(table_dir / "inferred_realization_nonstationarity_summary.csv", index=False)
    selected.to_csv(table_dir / "selected_high_nonstationarity_realization.csv", index=False)
    selected_events.to_csv(table_dir / "selected_high_nonstationarity_events.csv", index=False)

    block_future = future[future["Realization_Inferred"] == selected_id]
    years = np.arange(int(block_future["Year"].min()), int(block_future["Year"].max()) + 1)
    annual_counts = selected_events.groupby("Year").size().reindex(years, fill_value=0)
    annual_event_mean = selected_events.groupby("Year")["W_pressure"].mean()
    annual_event_p95 = selected_events.groupby("Year")["W_pressure"].quantile(0.95)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].bar(annual_counts.index, annual_counts.values, color="#4C78A8", alpha=0.8)
    count_slope, _, _ = slope_or_nan(annual_counts.index, annual_counts.values)
    count_years = annual_counts.index.to_numpy(dtype=float)
    count_intercept = annual_counts.mean() - count_slope * count_years.mean()
    ax[0].plot(count_years, count_slope * count_years + count_intercept, color="black")
    ax[0].set_xlabel("Year")
    ax[0].set_ylabel("Threshold-exceedance events")
    ax[0].set_title(f"Inferred realization {selected_id}: event counts")

    ax[1].scatter(annual_event_mean.index, annual_event_mean.values, label="Annual event mean", color="#4C78A8")
    ax[1].scatter(annual_event_p95.index, annual_event_p95.values, label="Annual event 95th", color="#E45756")
    for series, color in [(annual_event_mean, "#4C78A8"), (annual_event_p95, "#E45756")]:
        slope, _, _ = slope_or_nan(series.index, series.values)
        series_years = series.index.to_numpy(dtype=float)
        intercept = series.mean() - slope * series_years.mean()
        ax[1].plot(series_years, slope * series_years + intercept, color=color)
    ax[1].set_xlabel("Year")
    ax[1].set_ylabel("Wind pressure (Pa)")
    ax[1].set_title(f"Inferred realization {selected_id}: event magnitudes")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "selected_high_nonstationarity_realization.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Selected inferred realization: {selected_id}")
    print(summary.to_string(index=False))
    print(f"Summary: {table_dir / 'inferred_realization_nonstationarity_summary.csv'}")
    print(f"Selected realization data: {table_dir / 'selected_high_nonstationarity_realization.csv'}")
    print(f"Selected event data: {table_dir / 'selected_high_nonstationarity_events.csv'}")
    print(f"Diagnostic plot: {figure_dir / 'selected_high_nonstationarity_realization.png'}")


if __name__ == "__main__":
    main()
