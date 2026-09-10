"""Plot utilities for model diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_timeseries(y_true: np.ndarray, y_pred: np.ndarray, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_true, label="true")
    ax.plot(y_pred, label="pred", alpha=0.8)
    ax.set_title(title)
    ax.legend()
    _save(fig, out)


def plot_zoom(y_true: np.ndarray, y_pred: np.ndarray, out: Path, start: int, end: int, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(start, end), y_true[start:end], label="true")
    ax.plot(np.arange(start, end), y_pred[start:end], label="pred")
    ax.set_title(title)
    ax.legend()
    _save(fig, out)


def plot_error(y_true: np.ndarray, y_pred: np.ndarray, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_true - y_pred)
    ax.set_title(title)
    ax.set_ylabel("error")
    _save(fig, out)


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=6, alpha=0.5)
    lim_min = min(float(np.min(y_true)), float(np.min(y_pred)))
    lim_max = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--")
    ax.set_title(title)
    ax.set_xlabel("true")
    ax.set_ylabel("pred")
    _save(fig, out)


def plot_psd(y_true: np.ndarray, y_pred: np.ndarray, out: Path, fs: float, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    f_t, p_t = np.fft.rfftfreq(len(y_true), d=1 / fs), np.abs(np.fft.rfft(y_true)) ** 2
    f_p, p_p = np.fft.rfftfreq(len(y_pred), d=1 / fs), np.abs(np.fft.rfft(y_pred)) ** 2
    ax.semilogy(f_t, p_t + 1e-12, label="true")
    ax.semilogy(f_p, p_p + 1e-12, label="pred")
    ax.set_title(title)
    ax.set_xlabel("Hz")
    ax.legend()
    _save(fig, out)


def plot_metric_bars(metric_per_sim: Mapping[str, float], out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = list(metric_per_sim.keys())
    vals = [metric_per_sim[k] for k in labels]
    ax.bar(labels, vals)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    _save(fig, out)
