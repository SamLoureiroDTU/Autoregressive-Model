"""mNARX+ orchestrator with recursive auxiliary model discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import joblib
import networkx as nx
import numpy as np
import pandas as pd

from .config import MNARXConfig
from .data import DataSplit, SimulationDataset, downsample_dataset, load_simulations_from_folder, split_simulations
from .feature_selection import ResidualFeatureSelector
from .fnarx import FNARXModel
from .metrics import simulation_metrics, summarize_metrics
from .polynomial_basis import PolynomialBasisGenerator
from .preprocessing import WindPreprocessor
from .temporal_features import TemporalFeatureBuilder


@dataclass
class ResponseModelBundle:
    target: str
    selected_features: List[str]
    dependency_signals: List[str]
    selector_history: List[dict]
    model: FNARXModel


class MNARXPlus:
    """Main API for training, evaluation, and simulation."""

    def __init__(self, config: MNARXConfig):
        self.config = config
        self.config.validate()
        self.wind_preprocessor = WindPreprocessor(config)
        self.temporal_builder: TemporalFeatureBuilder | None = None
        self.dataset_: SimulationDataset | None = None
        self.split_: DataSplit | None = None
        self.prepared_sims_: Dict[str, pd.DataFrame] = {}
        self.models_: Dict[str, ResponseModelBundle] = {}
        self.dependency_graph_ = nx.DiGraph()
        self.exogenous_signals_: List[str] = []

    def _compute_azimuth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        rotor = df["RotorSpeed"].to_numpy(dtype=float)
        if self.config.rotor_speed_unit == "rpm":
            omega = rotor * (2.0 * np.pi / 60.0)
        else:
            omega = rotor
        dt = 1.0 / self.config.target_fs
        azimuth = np.cumsum(omega) * dt
        cols = {}
        for k in range(1, self.config.azimuth_harmonics_k + 1):
            cols[f"AzimuthSin{k}"] = np.sin(k * azimuth)
            cols[f"AzimuthCos{k}"] = np.cos(k * azimuth)
        return pd.DataFrame(cols, index=df.index)

    def _prepare_simulations(self, sims: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        wind = self.wind_preprocessor.transform_all(sims)
        prepared: Dict[str, pd.DataFrame] = {}
        for name, df in sims.items():
            miss = [c for c in (self.config.time_column, *self.config.response_columns) if c not in df.columns]
            if miss:
                raise ValueError(f"Simulation '{name}' missing required columns: {miss}")
            parts = [
                df[[self.config.time_column, *self.config.response_columns]].copy(),
                wind[name],
            ]
            if self.config.include_azimuth_harmonics:
                parts.append(self._compute_azimuth_features(df))
            prepared[name] = pd.concat(parts, axis=1)
        return prepared

    def _all_signals_for_temporal(self) -> List[str]:
        return [
            c
            for c in self.prepared_sims_[next(iter(self.prepared_sims_))].columns
            if c != self.config.time_column
        ]

    def fit(self, data_folder: str | Path) -> "MNARXPlus":
        self.dataset_ = load_simulations_from_folder(data_folder)
        self.dataset_ = downsample_dataset(self.dataset_, self.config)
        self.split_ = split_simulations(self.dataset_, self.config)

        train_sims = self.dataset_.subset(self.split_.train).simulations
        self.wind_preprocessor.fit(train_sims)
        self.prepared_sims_ = self._prepare_simulations(self.dataset_.simulations)

        all_signals = self._all_signals_for_temporal()
        include_current = {
            s: (s not in self.config.response_columns) for s in all_signals if s != self.config.time_column
        }
        self.temporal_builder = TemporalFeatureBuilder(self.config, include_current)
        train_prepared = {k: self.prepared_sims_[k] for k in self.split_.train}
        self.temporal_builder.fit(train_prepared, [s for s in all_signals if s != self.config.time_column])

        self.exogenous_signals_ = [s for s in all_signals if s not in self.config.response_columns and s != self.config.time_column]
        for r in self.config.response_columns:
            self.dependency_graph_.add_node(r)

        self._build_model_recursive(self.config.final_target_column, ancestors=set())
        return self

    def _make_feature_target_for_split(
        self, names: Iterable[str], target: str, candidate_signals: List[str]
    ) -> tuple[pd.DataFrame, np.ndarray]:
        assert self.temporal_builder is not None
        X_parts = []
        y_parts = []
        for n in names:
            df = self.prepared_sims_[n]
            Xn = self.temporal_builder.transform_simulation(df, candidate_signals)
            yn = df[target].iloc[Xn.index].to_numpy()
            X_parts.append(Xn)
            y_parts.append(yn)
        return pd.concat(X_parts, axis=0, ignore_index=True), np.concatenate(y_parts, axis=0)

    def _recursive_forecast_single(
        self,
        model: FNARXModel,
        sim_name: str,
        target: str,
        selected_features: List[str],
        candidate_signals: List[str],
        aux_predictions: Mapping[str, np.ndarray] | None = None,
        strict_no_future_measurements: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.temporal_builder is not None
        df = self.prepared_sims_[sim_name]
        n = len(df)
        init_hist = max(1, int(round(self.config.initial_history_seconds * self.config.target_fs)))
        lag_need = max(self.temporal_builder.models_[s].lag_samples + 1 for s in candidate_signals)
        start = max(init_hist, lag_need)

        histories: Dict[str, List[float]] = {s: list(df[s].iloc[:start].to_numpy(dtype=float)) for s in candidate_signals}
        y_true = df[target].iloc[start:].to_numpy(dtype=float)
        preds: List[float] = []

        for t in range(start, n):
            for s in self.exogenous_signals_:
                if s in histories:
                    histories[s].append(float(df[s].iloc[t]))

            if aux_predictions:
                for s, vals in aux_predictions.items():
                    if s in histories and len(vals) > (t - start):
                        histories[s].append(float(vals[t - start]))

            for s in self.config.response_columns:
                if s == target or s not in histories:
                    continue
                if strict_no_future_measurements and t >= start and (not aux_predictions or s not in aux_predictions):
                    raise ValueError(
                        f"Future measured value for auxiliary response '{s}' unavailable at t={t}; train dependency model first."
                    )
                if not strict_no_future_measurements:
                    histories[s].append(float(df[s].iloc[t]))

            row = self.temporal_builder.online_features_from_histories(histories)
            x = np.asarray([row[f] for f in selected_features], dtype=float)
            yhat = model.predict_one_step(x)
            preds.append(yhat)
            histories[target].append(yhat)

        return y_true, np.asarray(preds, dtype=float)

    def _build_model_recursive(self, target: str, ancestors: set[str]) -> None:
        if target in self.models_:
            return
        if target in ancestors:
            raise ValueError(f"Dependency cycle detected when building {target}.")

        candidate_signals = self.exogenous_signals_ + [
            s for s in self.config.response_columns if s not in ancestors
        ]
        X_train, y_train = self._make_feature_target_for_split(self.split_.train, target, candidate_signals)  # type: ignore[arg-type]
        X_val, y_val = self._make_feature_target_for_split(self.split_.val, target, candidate_signals)  # type: ignore[arg-type]

        selector = ResidualFeatureSelector(
            correlation_method=self.config.correlation_method,
            threshold=self.config.correlation_threshold,
            max_features=self.config.max_selected_features,
        )

        def fit_and_forecast(selected: List[str]):
            if not selected:
                baseline = np.full_like(y_val, np.mean(y_train), dtype=float)
                return None, baseline
            basis = PolynomialBasisGenerator(
                degree=self.config.polynomial_degree,
                hyperbolic_q=self.config.hyperbolic_q,
                max_terms=self.config.polynomial_max_terms,
            )
            fnarx = FNARXModel(basis=basis, sparse_config=self.config.sparse_regression)
            fnarx.fit(X_train[selected].to_numpy(), y_train, selected)
            preds = fnarx.predict_teacher_forcing(X_val[selected].to_numpy())
            return fnarx, preds

        selected = selector.select(X_val, y_val, fit_and_forecast)
        if not selected:
            selected = list(X_train.columns[: min(3, X_train.shape[1])])

        dependency_signals: List[str] = []
        filtered_selected: List[str] = []
        for feat in selected:
            signal = feat.split("__", 1)[0]
            if signal in self.config.response_columns and signal != target:
                if nx.has_path(self.dependency_graph_, target, signal):
                    continue
                if not self.dependency_graph_.has_edge(signal, target):
                    self.dependency_graph_.add_edge(signal, target)
                dependency_signals.append(signal)
            filtered_selected.append(feat)

        basis = PolynomialBasisGenerator(
            degree=self.config.polynomial_degree,
            hyperbolic_q=self.config.hyperbolic_q,
            max_terms=self.config.polynomial_max_terms,
        )
        final_model = FNARXModel(basis=basis, sparse_config=self.config.sparse_regression)
        X_fit = pd.concat([X_train, X_val], axis=0, ignore_index=True)
        y_fit = np.concatenate([y_train, y_val])
        final_model.fit(X_fit[filtered_selected].to_numpy(), y_fit, filtered_selected)

        self.models_[target] = ResponseModelBundle(
            target=target,
            selected_features=filtered_selected,
            dependency_signals=sorted(set(dependency_signals)),
            selector_history=selector.history_as_dict(),
            model=final_model,
        )

        for dep in sorted(set(dependency_signals)):
            self._build_model_recursive(dep, ancestors=ancestors | {target})

    def _topological_model_order(self) -> List[str]:
        nodes = list(self.models_.keys())
        sub = self.dependency_graph_.subgraph(nodes)
        return list(nx.topological_sort(sub))

    def evaluate_test_set(self) -> Dict[str, object]:
        if self.split_ is None:
            raise RuntimeError("Model not fit.")
        per_target: Dict[str, Dict[str, Dict[str, float]]] = {}
        preds_store: Dict[str, Dict[str, np.ndarray]] = {}
        order = self._topological_model_order()

        for sim_name in self.split_.test:
            aux_preds: Dict[str, np.ndarray] = {}
            for target in order:
                bundle = self.models_[target]
                y_true, y_pred = self._recursive_forecast_single(
                    bundle.model,
                    sim_name,
                    target,
                    bundle.selected_features,
                    self.exogenous_signals_ + [target] + bundle.dependency_signals,
                    aux_predictions=aux_preds,
                    strict_no_future_measurements=True,
                )
                aux_preds[target] = y_pred
                per_target.setdefault(target, {})[sim_name] = simulation_metrics(y_true, y_pred)
                preds_store.setdefault(target, {})[sim_name] = y_pred

        summary = {t: summarize_metrics(m) for t, m in per_target.items()}
        return {"per_target": per_target, "summary": summary, "predictions": preds_store}

    def simulate(self, simulation_csv: str | Path) -> Dict[str, np.ndarray]:
        if self.temporal_builder is None:
            raise RuntimeError("Model not fit.")
        raw = pd.read_csv(simulation_csv)
        raw_ds = raw if self.config.target_fs == self.config.original_fs else raw.iloc[:: int(self.config.original_fs / self.config.target_fs)].reset_index(drop=True)
        self.prepared_sims_["_simulate_"] = self._prepare_simulations({"_simulate_": raw_ds})["_simulate_"]

        out: Dict[str, np.ndarray] = {}
        for target in self._topological_model_order():
            bundle = self.models_[target]
            y_true, y_pred = self._recursive_forecast_single(
                bundle.model,
                "_simulate_",
                target,
                bundle.selected_features,
                self.exogenous_signals_ + [target] + bundle.dependency_signals,
                aux_predictions=out,
                strict_no_future_measurements=True,
            )
            out[target] = y_pred
            out[f"{target}__true_available_for_reference"] = y_true
        return out

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "MNARXPlus":
        return joblib.load(path)
