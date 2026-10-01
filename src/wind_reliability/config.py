"""Configuration models and loader for wind reliability workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class PathConfig:
    """Filesystem locations for inputs and outputs, relative to project root by default."""

    input_csv: str = "data/raw/combined_output.csv"
    output_dir: str = "outputs"


@dataclass
class WindConfig:
    """Physical constants and load factors for wind pressure conversion."""

    rho_air: float = 1.2
    c_g: float = 2.0
    c_e: float = 1.2
    c_h: float = 2.0


@dataclass
class AnalysisConfig:
    """High-level analysis controls for periods, thresholds, and reliability settings."""

    future_start_year: int = 2025
    threshold_wind_speed: float = 11.0
    time_horizon_years: int = 75
    time_step_years: int = 1
    cover_mm: float = 70.0


@dataclass
class CorrosionConfig:
    """Controls for corrosion initiation Monte Carlo simulation."""

    c_surf_mean: tuple[float, ...] = (1.8, 3.5, 5.3, 7.4)
    c_surf_cov: tuple[float, ...] = (0.5, 0.5, 0.5, 0.5)
    t_end: int = 1000
    dt: int = 1


@dataclass
class DegradationConfig:
    """Controls for degradation-rate fitting from pitting simulations."""

    t_life: int = 200
    dt: int = 1
    i_th_base: tuple[float, ...] = (0.5, 2.0, 5.0, 10.0)
    alpha_mean: float = 5.65
    alpha_cov: float = 0.22


@dataclass
class ResistanceConfig:
    """Initial resistance model for reliability analysis."""

    initial_mean: float = 1621.0
    initial_cov: float = 0.09


@dataclass
class DeadLoadConfig:
    """Dead load model for reliability analysis."""

    mean: float = 359.0
    cov: float = 0.10


@dataclass
class RuntimeConfig:
    """Random seed for the mechanistic-history generator."""

    random_seed: int = 42


@dataclass
class WorkflowConfig:
    """Top-level configuration object composed from all sections."""

    paths: PathConfig
    wind: WindConfig
    analysis: AnalysisConfig
    corrosion: CorrosionConfig
    degradation: DegradationConfig
    resistance: ResistanceConfig
    dead_load: DeadLoadConfig
    runtime: RuntimeConfig

    @classmethod
    def default(cls) -> "WorkflowConfig":
        return cls(
            paths=PathConfig(),
            wind=WindConfig(),
            analysis=AnalysisConfig(),
            corrosion=CorrosionConfig(),
            degradation=DegradationConfig(),
            resistance=ResistanceConfig(),
            dead_load=DeadLoadConfig(),
            runtime=RuntimeConfig(),
        )

    @classmethod
    def from_json(cls, config_path: Path | str) -> "WorkflowConfig":
        """Load config JSON and merge with defaults so missing keys are auto-filled."""

        config_path = Path(config_path)
        payload: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))

        default_cfg = cls.default()

        def merge(section_name: str, default_obj: Any):
            values = payload.get(section_name, {})
            allowed = {f.name for f in fields(type(default_obj))}
            values = {k: v for k, v in values.items() if k in allowed}
            merged = default_obj.__dict__.copy()
            merged.update(values)
            return type(default_obj)(**merged)

        return cls(
            paths=merge("paths", default_cfg.paths),
            wind=merge("wind", default_cfg.wind),
            analysis=merge("analysis", default_cfg.analysis),
            corrosion=merge("corrosion", default_cfg.corrosion),
            degradation=merge("degradation", default_cfg.degradation),
            resistance=merge("resistance", default_cfg.resistance),
            dead_load=merge("dead_load", default_cfg.dead_load),
            runtime=merge("runtime", default_cfg.runtime),
        )
