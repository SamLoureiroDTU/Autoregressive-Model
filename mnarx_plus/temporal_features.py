"""Lag windows and temporal PCA with strict simulation-boundary isolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import MNARXConfig


@dataclass
class SignalTemporalModel:
    """Per-signal temporal model over lag windows."""

    signal: str
    lag_samples: int
    include_current: bool
    scaler: StandardScaler | None
    pca: PCA | None
    feature_names: List[str]


class TemporalFeatureBuilder:
    """Builds lag-window features without crossing simulation boundaries."""

    def __init__(self, config: MNARXConfig, include_current_by_signal: Mapping[str, bool]):
        self.config = config
        self.include_current_by_signal = dict(include_current_by_signal)
        self.models_: Dict[str, SignalTemporalModel] = {}

    def _lag_samples(self, signal: str) -> int:
        sec = self.config.memory_seconds_per_signal.get(
            signal, self.config.memory_seconds_per_signal.get("wind", 3.0)
        )
        return max(1, int(round(sec * self.config.target_fs)))

    @staticmethod
    def _windows(series: np.ndarray, lag_samples: int, include_current: bool) -> tuple[np.ndarray, np.ndarray]:
        start = lag_samples if include_current else lag_samples + 1
        rows, times = [], []
        for t in range(start, len(series)):
            if include_current:
                win = series[t - lag_samples : t + 1][::-1]
            else:
                win = series[t - lag_samples : t][::-1]
            rows.append(win)
            times.append(t)
        return np.asarray(rows), np.asarray(times)

    def fit(self, train_sims: Mapping[str, pd.DataFrame], signals: Iterable[str]) -> None:
        for signal in signals:
            lag = self._lag_samples(signal)
            include_current = self.include_current_by_signal.get(signal, True)
            windows = []
            for df in train_sims.values():
                W, _ = self._windows(df[signal].to_numpy(), lag, include_current)
                if len(W):
                    windows.append(W)
            if not windows:
                raise ValueError(f"Cannot fit temporal model for signal '{signal}' due to insufficient length.")
            X = np.concatenate(windows, axis=0)

            scaler: StandardScaler | None = None
            pca: PCA | None = None
            if self.config.temporal_pca.enabled:
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                n_comp = self.config.temporal_pca.n_components
                if n_comp is None:
                    n_comp = self.config.temporal_pca.explained_variance_threshold
                pca = PCA(n_components=n_comp, random_state=self.config.random_seed)
                pca.fit(Xs)
                feat_names = [f"{signal}__tpca_{i}" for i in range(pca.n_components_)]
            else:
                width = lag + 1 if include_current else lag
                feat_names = [f"{signal}__lag_{i}" for i in range(width)]

            self.models_[signal] = SignalTemporalModel(
                signal=signal,
                lag_samples=lag,
                include_current=include_current,
                scaler=scaler,
                pca=pca,
                feature_names=feat_names,
            )

    def transform_simulation(self, df: pd.DataFrame, signals: Iterable[str]) -> pd.DataFrame:
        signals = list(signals)
        starts = []
        per_signal_windows: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for signal in signals:
            model = self.models_[signal]
            W, times = self._windows(df[signal].to_numpy(), model.lag_samples, model.include_current)
            per_signal_windows[signal] = (W, times)
            starts.append(times[0] if len(times) else len(df))
        global_start = max(starts)
        rows = []
        for t in range(global_start, len(df)):
            row = {}
            for signal in signals:
                model = self.models_[signal]
                values = df[signal].to_numpy()
                if model.include_current:
                    win = values[t - model.lag_samples : t + 1][::-1]
                else:
                    win = values[t - model.lag_samples : t][::-1]
                if model.pca is not None and model.scaler is not None:
                    z = model.pca.transform(model.scaler.transform(win.reshape(1, -1)))[0]
                    for i, val in enumerate(z):
                        row[model.feature_names[i]] = val
                else:
                    for i, val in enumerate(win):
                        row[model.feature_names[i]] = val
            rows.append(row)
        return pd.DataFrame(rows, index=np.arange(global_start, len(df)))

    def online_features_from_histories(self, histories: Mapping[str, List[float]]) -> Dict[str, float]:
        """Create one feature row from current signal histories for recursive inference."""
        features: Dict[str, float] = {}
        for signal, model in self.models_.items():
            if signal not in histories:
                continue
            hist = np.asarray(histories[signal], dtype=float)
            need = model.lag_samples + (1 if model.include_current else 0)
            if len(hist) < need:
                raise ValueError(f"Signal {signal} has insufficient history {len(hist)} < {need}.")
            win = hist[-need:][::-1]
            if model.pca is not None and model.scaler is not None:
                z = model.pca.transform(model.scaler.transform(win.reshape(1, -1)))[0]
                for i, val in enumerate(z):
                    features[model.feature_names[i]] = float(val)
            else:
                for i, val in enumerate(win):
                    features[model.feature_names[i]] = float(val)
        return features
