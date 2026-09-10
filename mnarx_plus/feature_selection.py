"""Residual-based iterative feature selection for mNARX+."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr


@dataclass
class SelectionStep:
    iteration: int
    selected_so_far: List[str]
    best_candidate: str | None
    best_correlation: float | None
    action: str


class ResidualFeatureSelector:
    """mNARX+ style residual-correlation feature selector."""

    def __init__(
        self,
        correlation_method: str = "pearson",
        threshold: float = 0.2,
        max_features: int = 30,
    ):
        self.correlation_method = correlation_method
        self.threshold = threshold
        self.max_features = max_features
        self.history_: List[SelectionStep] = []

    def _corr(self, a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        if self.correlation_method == "pearson":
            return float(pearsonr(a, b)[0])
        if self.correlation_method == "spearman":
            return float(spearmanr(a, b)[0])
        if self.correlation_method == "kendall":
            return float(kendalltau(a, b)[0])
        raise ValueError(f"Unsupported correlation method {self.correlation_method}")

    def select(
        self,
        candidate_features: pd.DataFrame,
        target: np.ndarray,
        fit_and_forecast: Callable[[List[str]], Tuple[object, np.ndarray]],
    ) -> List[str]:
        selected: List[str] = []
        all_features = list(candidate_features.columns)
        for it in range(self.max_features + 1):
            _, preds = fit_and_forecast(selected)
            residual = target - preds
            unused = [c for c in all_features if c not in selected]
            if not unused or len(selected) >= self.max_features:
                self.history_.append(
                    SelectionStep(it, selected.copy(), None, None, "stop:max_features_or_empty")
                )
                break

            best_feature = None
            best_corr = 0.0
            for feat in unused:
                c = abs(self._corr(residual, candidate_features[feat].to_numpy()))
                if c > best_corr:
                    best_corr = c
                    best_feature = feat

            if best_feature is None or best_corr < self.threshold:
                self.history_.append(
                    SelectionStep(it, selected.copy(), best_feature, best_corr, "stop:below_threshold")
                )
                break

            selected.append(best_feature)
            self.history_.append(
                SelectionStep(it, selected.copy(), best_feature, best_corr, "add")
            )
        return selected

    def history_as_dict(self) -> List[Dict[str, object]]:
        return [
            {
                "iteration": h.iteration,
                "selected_so_far": h.selected_so_far,
                "best_candidate": h.best_candidate,
                "best_correlation": h.best_correlation,
                "action": h.action,
            }
            for h in self.history_
        ]
