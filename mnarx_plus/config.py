"""Configuration for mNARX+ wind turbine surrogate modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class SplitConfig:
    """Simulation-level split ratios."""

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}.")


@dataclass
class SpatialPCAConfig:
    """Spatial PCA settings for wind representation."""

    n_components: Optional[int] = None
    explained_variance_threshold: Optional[float] = 0.95


@dataclass
class TemporalPCAConfig:
    """Temporal PCA settings for lag-window compression."""

    enabled: bool = True
    n_components: Optional[int] = None
    explained_variance_threshold: Optional[float] = 0.95


@dataclass
class SparseRegressionConfig:
    """Sparse regression configuration for FNARX."""

    alpha: float = 1e-4
    max_iter: int = 10_000
    tol: float = 1e-5


@dataclass
class MNARXConfig:
    """Top-level configuration for the mNARX+ pipeline."""

    random_seed: int = 42
    data_folder: Optional[Path] = None
    split: SplitConfig = field(default_factory=SplitConfig)
    original_fs: int = 100
    target_fs: int = 20
    memory_seconds_per_signal: Dict[str, float] = field(
        default_factory=lambda: {
            "RotorSpeed": 3.0,
            "Pitch": 3.0,
            "BladeRootFlapwiseMoment": 3.0,
            "wind": 3.0,
        }
    )
    initial_history_seconds: float = 3.0
    wind_representation_mode: str = "spatial_pca"  # direct | spatial_pca | dct_placeholder
    spatial_pca: SpatialPCAConfig = field(default_factory=SpatialPCAConfig)
    temporal_pca: TemporalPCAConfig = field(default_factory=TemporalPCAConfig)
    polynomial_degree: int = 3
    hyperbolic_q: float = 1.0
    polynomial_max_terms: int = 400
    sparse_regression: SparseRegressionConfig = field(default_factory=SparseRegressionConfig)
    correlation_method: str = "pearson"
    correlation_threshold: float = 0.2
    max_selected_features: int = 30
    final_target_column: str = "BladeRootFlapwiseMoment"
    response_columns: tuple[str, ...] = ("RotorSpeed", "Pitch", "BladeRootFlapwiseMoment")
    time_column: str = "Time"
    include_azimuth_harmonics: bool = True
    azimuth_harmonics_k: int = 4
    rotor_speed_unit: str = "rpm"  # rpm | rad_s

    def validate(self) -> None:
        self.split.validate()
        if self.target_fs <= 0 or self.original_fs <= 0:
            raise ValueError("Sampling frequencies must be positive.")
        if self.target_fs > self.original_fs:
            raise ValueError("target_fs must be <= original_fs.")
        if self.wind_representation_mode not in {"direct", "spatial_pca", "dct_placeholder"}:
            raise ValueError("wind_representation_mode must be direct/spatial_pca/dct_placeholder.")
        if self.correlation_method not in {"pearson", "spearman", "kendall"}:
            raise ValueError("correlation_method must be pearson/spearman/kendall.")
