"""Sampling utilities for Monte Carlo reliability analyses."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats


def _beta_params_from_mean_cov(mean: float, cov: float) -> tuple[float, float]:
    std_dev = mean * cov
    variance = std_dev**2
    common_factor = mean * (1.0 - mean) / variance - 1.0
    alpha = mean * common_factor
    beta = (1.0 - mean) * common_factor
    if alpha <= 0 or beta <= 0:
        raise ValueError(f"Invalid mean/cov for beta distribution: mean={mean}, cov={cov}")
    return alpha, beta


def g_rand(
    dist: str,
    mean: float,
    cov: float,
    N: int = 1,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    method: str = "random",
) -> np.ndarray:
    """Generate random samples for a supported distribution.

    Parameters use mean and COV for consistency across reliability modules.
    """

    std_dev = mean * cov
    method = method.lower()

    if method not in {"random", "lhs"}:
        raise ValueError(f"Unsupported sampling method: {method}")

    if cov == 0:
        return np.full(N, float(mean), dtype=float)

    if method == "lhs":
        # One-dimensional LHS: stratify [0,1] and randomly permute strata samples.
        u = (np.arange(N, dtype=float) + np.random.random(N)) / float(N)
        np.random.shuffle(u)
        eps = np.finfo(float).eps
        u = np.clip(u, eps, 1.0 - eps)

    if dist == "normal":
        if method == "lhs":
            return stats.norm.ppf(u, loc=mean, scale=std_dev)
        return np.random.normal(mean, std_dev, N)

    if dist == "lognormal":
        sigma = np.sqrt(np.log(1.0 + cov**2))
        mu = np.log(mean) - 0.5 * sigma**2
        if method == "lhs":
            return stats.lognorm.ppf(u, s=sigma, scale=np.exp(mu))
        return np.random.lognormal(mu, sigma, N)

    if dist == "uniform":
        lo = lower if lower is not None else mean - std_dev * np.sqrt(3.0)
        hi = upper if upper is not None else mean + std_dev * np.sqrt(3.0)
        if method == "lhs":
            return lo + (hi - lo) * u
        return np.random.uniform(lo, hi, N)

    if dist == "beta":
        alpha, beta_param = _beta_params_from_mean_cov(mean, cov)
        if method == "lhs":
            return stats.beta.ppf(u, alpha, beta_param)
        return np.random.beta(alpha, beta_param, N)

    if dist == "gumbel":
        scale = std_dev * np.sqrt(6.0) / np.pi
        location = mean - scale * 0.5772156649
        if method == "lhs":
            return stats.gumbel_r.ppf(u, loc=location, scale=scale)
        return np.random.gumbel(location, scale, N)

    raise ValueError(f"Unsupported distribution: {dist}")


def generate_samples(variables: Mapping[str, Dict], N: int) -> pd.DataFrame:
    """Generate a full Monte Carlo sample table from a variable configuration."""

    samples: Dict[str, np.ndarray] = {}
    for name, properties in variables.items():
        mean = float(properties["mean"])
        cov = float(properties["cov"])
        dist = properties["distribution"]
        params = properties.get("parameters", {})
        samples[name] = g_rand(
            dist=dist,
            mean=mean,
            cov=cov,
            N=N,
            lower=params.get("lower_bound"),
            upper=params.get("upper_bound"),
            method=str(params.get("method", "random")),
        )
    return pd.DataFrame(samples)
