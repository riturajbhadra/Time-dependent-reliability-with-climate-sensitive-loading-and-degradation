"""Pitting corrosion and normalized section capacity for the calibrated model."""

import math
import numpy as np

def _section_moment_capacity_ratio(diameter_mm: np.ndarray, fck: float, fy: float) -> np.ndarray:
    """Return M(t)/M(0) for a rectangular section and 4 bars."""

    d_eff = 1000.0
    width = 400.0
    area_steel = math.pi * np.square(diameter_mm)
    neutral_axis = 0.87 * fy * area_steel / (0.362 * fck * width)
    moment = area_steel * fy * (d_eff - 0.42 * neutral_axis)
    return moment / moment[0]


def pitting_simplified(i_th: float, tp: np.ndarray, fck: float, fy: float, alpha: float, d_cover: float) -> np.ndarray:
    """Bastidas-inspired simplified pitting model for capacity reduction."""

    tp = np.asarray(tp, dtype=float)
    wcr = 0.3
    i_corr_20 = 37.88 * (1.0 - wcr) ** -1.64 / (d_cover / 10.0)

    # Temperature trend and transition between initiation/active corrosion branches.
    temperature = np.linspace(10.0, 14.0, tp.size)
    k_t = np.where(temperature < 20.0, 0.025, 0.073)
    i_corr_ini = 0.85 * i_corr_20 * np.power(tp, -0.29) * (1.0 + k_t * (temperature - 20.0))

    mu_ini = 1.0 / (1.0 + np.exp(-(-0.8) * (tp - 6.5)))
    mu_act = 1.0 / (1.0 + np.exp(-(0.7) * (tp - 12.0)))
    i_corr = (mu_ini * i_corr_ini + mu_act * i_th) / (mu_ini + mu_act)

    dpdt = i_corr * alpha * 0.0116
    pit_depth = np.cumsum(dpdt) * (tp[1] - tp[0]) - dpdt[0]

    bar_diameter = 25.0 - pit_depth
    bar_diameter[bar_diameter < 0.0] = 0.0
    return _section_moment_capacity_ratio(bar_diameter, fck=fck, fy=fy)
