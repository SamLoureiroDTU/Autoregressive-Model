"""Evaluation metrics for dynamic surrogate forecasts."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scale = np.std(y_true)
    return float(rmse(y_true, y_pred) / (scale + 1e-12))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    return float(pearsonr(y_true, y_pred)[0])


def simulation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "nRMSE": nrmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "Pearson": pearson(y_true, y_pred),
    }


def summarize_metrics(per_sim: Mapping[str, Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(per_sim, orient="index")
    df.loc["mean"] = df.mean(axis=0)
    df.loc["median"] = df.median(axis=0)
    return df
