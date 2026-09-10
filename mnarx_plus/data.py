"""Data loading, simulation-level splitting, and downsampling."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from .config import MNARXConfig


@dataclass
class DataSplit:
    """Holds simulation IDs for train/val/test."""

    train: List[str]
    val: List[str]
    test: List[str]


class SimulationDataset:
    """Container for full-simulation dataframes."""

    def __init__(self, simulations: Mapping[str, pd.DataFrame]):
        self.simulations: Dict[str, pd.DataFrame] = dict(simulations)

    def subset(self, names: List[str]) -> "SimulationDataset":
        return SimulationDataset({k: self.simulations[k] for k in names})


def load_simulations_from_folder(folder: str | Path) -> SimulationDataset:
    """Load all CSV files where each file corresponds to one full simulation."""
    path = Path(folder)
    if not path.exists():
        raise FileNotFoundError(f"Data folder does not exist: {path}")

    csv_files = sorted(path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")

    simulations: Dict[str, pd.DataFrame] = {}
    for csv_file in csv_files:
        simulations[csv_file.stem] = pd.read_csv(csv_file)
    return SimulationDataset(simulations)


def split_simulations(dataset: SimulationDataset, config: MNARXConfig) -> DataSplit:
    """Split full simulations into train/val/test without temporal leakage."""
    names = sorted(dataset.simulations.keys())
    if len(names) < 3:
        raise ValueError("Need at least 3 simulation files to make train/val/test split.")
    rng = np.random.default_rng(config.random_seed)
    perm = list(rng.permutation(names))

    n = len(perm)
    n_train = max(1, int(round(config.split.train_ratio * n)))
    n_val = max(1, int(round(config.split.val_ratio * n)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    n_test = n - n_train - n_val

    train = perm[:n_train]
    val = perm[n_train : n_train + n_val]
    test = perm[n_train + n_val : n_train + n_val + n_test]
    return DataSplit(train=train, val=val, test=test)


def _downsample_df(df: pd.DataFrame, original_fs: int, target_fs: int) -> pd.DataFrame:
    if target_fs == original_fs:
        return df.copy()

    ratio = Fraction(target_fs, original_fs).limit_denominator()
    up, down = ratio.numerator, ratio.denominator

    out = {}
    for col in df.columns:
        values = df[col].to_numpy()
        if np.issubdtype(values.dtype, np.number):
            out[col] = resample_poly(values, up=up, down=down)
        else:
            raise ValueError(f"Non-numeric column cannot be downsampled: {col}")
    min_len = min(len(v) for v in out.values())
    return pd.DataFrame({k: v[:min_len] for k, v in out.items()})


def downsample_dataset(dataset: SimulationDataset, config: MNARXConfig) -> SimulationDataset:
    """Apply anti-aliased resampling per simulation independently."""
    simulations = {
        name: _downsample_df(df, config.original_fs, config.target_fs)
        for name, df in dataset.simulations.items()
    }
    return SimulationDataset(simulations)
