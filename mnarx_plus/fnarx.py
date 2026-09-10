"""FNARX model with sparse polynomial regression and recursive forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from sklearn.linear_model import Lasso

from .config import SparseRegressionConfig
from .polynomial_basis import PolynomialBasisGenerator


@dataclass
class FNARXModel:
    """Functional NARX-style surrogate model."""

    basis: PolynomialBasisGenerator
    sparse_config: SparseRegressionConfig

    def __post_init__(self) -> None:
        self.regressor_ = Lasso(
            alpha=self.sparse_config.alpha,
            max_iter=self.sparse_config.max_iter,
            tol=self.sparse_config.tol,
            random_state=0,
        )
        self.feature_names_: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> "FNARXModel":
        self.feature_names_ = list(feature_names)
        self.basis.fit(feature_names)
        Z = self.basis.transform(X)
        self.regressor_.fit(Z, y)
        return self

    def predict_one_step(self, x: np.ndarray) -> float:
        Z = self.basis.transform(x.reshape(1, -1))
        return float(self.regressor_.predict(Z)[0])

    def predict_teacher_forcing(self, X: np.ndarray) -> np.ndarray:
        """Teacher forcing: each step uses measured regressors."""
        Z = self.basis.transform(X)
        return self.regressor_.predict(Z)

    def forecast_recursive(
        self,
        steps: int,
        feature_function: Callable[[int, List[float]], np.ndarray],
        initial_predictions: List[float] | None = None,
    ) -> np.ndarray:
        """
        Recursive free-running forecast.

        feature_function receives (step, predictions_so_far) and must build current regressors
        from known exogenous variables + past predicted outputs when measured future is unavailable.
        """
        preds: List[float] = list(initial_predictions or [])
        out: List[float] = []
        for step in range(steps):
            x = feature_function(step, preds)
            yhat = self.predict_one_step(np.asarray(x, dtype=float))
            preds.append(yhat)
            out.append(yhat)
        return np.asarray(out)
