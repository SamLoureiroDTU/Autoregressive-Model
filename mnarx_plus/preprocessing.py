"""Wind preprocessing and column resolution utilities."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import MNARXConfig

WIND_COLUMN_RE = re.compile(
    r"(?P<prefix>.*?)(?P<component>V[xyz])\s+pos\s+(?P<x>-?[0-9]*\.?[0-9]+),\s*(?P<y>-?[0-9]*\.?[0-9]+),\s*(?P<z>-?[0-9]*\.?[0-9]+)$"
)


@dataclass(frozen=True)
class WindPoint:
    x: float
    y: float
    z: float


@dataclass
class WindColumnMatch:
    column: str
    component: str
    point: WindPoint


class WindPreprocessor:
    """
    Implements wind representation options.

    paper-specified: manifold-like reduced wind representation is allowed.
    implementation choice: spatial PCA over resolved wind columns after scaling.
    """

    def __init__(self, config: MNARXConfig):
        self.config = config
        self.mode = config.wind_representation_mode
        self.matches_: List[WindColumnMatch] = []
        self.columns_: List[str] = []
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None

    @staticmethod
    def resolve_wind_columns(columns: Iterable[str]) -> List[WindColumnMatch]:
        matches: List[WindColumnMatch] = []
        for col in columns:
            m = WIND_COLUMN_RE.match(col)
            if not m:
                continue
            matches.append(
                WindColumnMatch(
                    column=col,
                    component=m.group("component"),
                    point=WindPoint(
                        float(m.group("x")),
                        float(m.group("y")),
                        float(m.group("z")),
                    ),
                )
            )
        return sorted(matches, key=lambda x: (x.point.x, x.point.y, x.point.z, x.component))

    def fit(self, train_sims: Mapping[str, pd.DataFrame]) -> None:
        first = next(iter(train_sims.values()))
        self.matches_ = self.resolve_wind_columns(first.columns)
        if not self.matches_:
            raise ValueError(
                "No wind columns detected. Check naming, expected e.g. '... Vx pos x,y,z'."
            )

        self.columns_ = [m.column for m in self.matches_]
        for name, df in train_sims.items():
            missing = [c for c in self.columns_ if c not in df.columns]
            if missing:
                raise ValueError(f"Simulation {name} missing wind columns: {missing[:5]}")

        if self.mode == "spatial_pca":
            X = np.concatenate([df[self.columns_].to_numpy() for df in train_sims.values()], axis=0)
            self.scaler_ = StandardScaler()
            Xs = self.scaler_.fit_transform(X)
            n_comp = self.config.spatial_pca.n_components
            if n_comp is None:
                n_comp = self.config.spatial_pca.explained_variance_threshold
            self.pca_ = PCA(n_components=n_comp, random_state=self.config.random_seed)
            self.pca_.fit(Xs)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.columns_].to_numpy()
        if self.mode == "direct":
            return pd.DataFrame(X, columns=[f"wind_direct_{i}" for i in range(X.shape[1])], index=df.index)
        if self.mode == "spatial_pca":
            assert self.scaler_ is not None and self.pca_ is not None
            Z = self.pca_.transform(self.scaler_.transform(X))
            return pd.DataFrame(Z, columns=[f"wind_pca_{i}" for i in range(Z.shape[1])], index=df.index)
        if self.mode == "dct_placeholder":
            # implementation choice: placeholder passthrough while keeping stable interface.
            return pd.DataFrame(X, columns=[f"wind_dct_placeholder_{i}" for i in range(X.shape[1])], index=df.index)
        raise ValueError(f"Unsupported wind mode: {self.mode}")

    def transform_all(self, sims: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return {name: self.transform(df) for name, df in sims.items()}
