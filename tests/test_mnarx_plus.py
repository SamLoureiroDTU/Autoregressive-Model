from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mnarx_plus.config import MNARXConfig
from mnarx_plus.data import SimulationDataset, downsample_dataset
from mnarx_plus.fnarx import FNARXModel
from mnarx_plus.mnarx_plus import MNARXPlus
from mnarx_plus.polynomial_basis import PolynomialBasisGenerator
from mnarx_plus.temporal_features import TemporalFeatureBuilder


WIND_COL = "Free wind speed Vx pos 0.00, 0.00,-150.00"


def _make_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 20
    u = np.sin(0.3 * t) + 0.1 * rng.normal(size=n)
    rotor = np.zeros(n)
    pitch = np.zeros(n)
    blade = np.zeros(n)
    for i in range(1, n):
        rotor[i] = 0.8 * rotor[i - 1] + 0.5 * u[i - 1]
        pitch[i] = 0.7 * pitch[i - 1] + 0.6 * rotor[i - 1]
        blade[i] = 0.6 * blade[i - 1] + 0.9 * pitch[i - 1]
    return pd.DataFrame(
        {
            "Time": np.arange(n) / 100.0,
            "RotorSpeed": rotor,
            "Pitch": pitch,
            "BladeRootFlapwiseMoment": blade,
            WIND_COL: u,
        }
    )


def test_windows_do_not_cross_simulation_boundaries() -> None:
    cfg = MNARXConfig(target_fs=20)
    tb = TemporalFeatureBuilder(cfg, {"a": True})
    w1, _ = tb._windows(np.arange(10), lag_samples=3, include_current=True)
    w2, _ = tb._windows(np.arange(100, 110), lag_samples=3, include_current=True)
    assert np.all(w1[0] < 50)
    assert np.all(w2[0] > 50)


def test_pca_fit_only_on_training_data() -> None:
    cfg = MNARXConfig(target_fs=20)
    tb = TemporalFeatureBuilder(cfg, {"x": True})
    train = {"s1": pd.DataFrame({"x": np.zeros(200)})}
    tb.fit(train, ["x"])
    assert tb.models_["x"].scaler is not None
    assert abs(float(tb.models_["x"].scaler.mean_[0])) < 1e-6


def test_recursive_prediction_uses_prior_predictions() -> None:
    basis = PolynomialBasisGenerator(degree=1, hyperbolic_q=1.0, max_terms=5)
    model = FNARXModel(basis=basis, sparse_config=MNARXConfig().sparse_regression)
    model.predict_one_step = lambda x: float(x[0] + 1.0)  # type: ignore[method-assign]

    def feat(step: int, preds: list[float]) -> np.ndarray:
        return np.array([preds[-1] if preds else 0.0])

    out = model.forecast_recursive(steps=4, feature_function=feat)
    assert np.allclose(out, np.array([1.0, 2.0, 3.0, 4.0]))


def test_aux_predictions_replace_future_measured_values(tmp_path: Path) -> None:
    df = _make_df(200)
    cfg = MNARXConfig(target_fs=20, original_fs=20, include_azimuth_harmonics=False)
    m = MNARXPlus(cfg)
    m.prepared_sims_ = {"s": df}
    m.exogenous_signals_ = [WIND_COL]
    tb = TemporalFeatureBuilder(cfg, {WIND_COL: True, "RotorSpeed": False, "Pitch": False})
    tb.fit({"s": df}, [WIND_COL, "RotorSpeed", "Pitch"])
    m.temporal_builder = tb

    dummy = FNARXModel(PolynomialBasisGenerator(degree=1, max_terms=10), cfg.sparse_regression)
    dummy.predict_one_step = lambda x: 0.0  # type: ignore[method-assign]
    dummy.basis.fit(["Pitch__tpca_0"])
    dummy.regressor_.coef_ = np.array([0.0])
    dummy.regressor_.intercept_ = 0.0

    with pytest.raises(ValueError):
        m._recursive_forecast_single(  # noqa: SLF001
            dummy,
            "s",
            "RotorSpeed",
            ["Pitch__tpca_0"],
            [WIND_COL, "RotorSpeed", "Pitch"],
            aux_predictions=None,
            strict_no_future_measurements=True,
        )

    y_true, y_pred = m._recursive_forecast_single(  # noqa: SLF001
        dummy,
        "s",
        "RotorSpeed",
        ["Pitch__tpca_0"],
        [WIND_COL, "RotorSpeed", "Pitch"],
        aux_predictions={"Pitch": np.zeros(180)},
        strict_no_future_measurements=True,
    )
    assert len(y_true) == len(y_pred)


def test_dependency_cycle_prevention_guard() -> None:
    cfg = MNARXConfig(response_columns=("A", "B"), final_target_column="A", include_azimuth_harmonics=False)
    m = MNARXPlus(cfg)
    with pytest.raises(ValueError):
        m._build_model_recursive("A", ancestors={"A"})  # noqa: SLF001


def test_downsampling_expected_frequency() -> None:
    n = 1000
    df = pd.DataFrame(
        {
            "Time": np.arange(n) / 100,
            "RotorSpeed": np.sin(np.arange(n) / 20),
            "Pitch": np.sin(np.arange(n) / 15),
            "BladeRootFlapwiseMoment": np.sin(np.arange(n) / 10),
            WIND_COL: np.sin(np.arange(n) / 25),
        }
    )
    ds = SimulationDataset({"s": df})
    cfg = MNARXConfig(original_fs=100, target_fs=20)
    out = downsample_dataset(ds, cfg)
    assert abs(len(out.simulations["s"]) - n * 0.2) <= 2


def test_save_load_equivalence(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(6):
        _make_df(220, seed=i).to_csv(data_dir / f"sim_{i}.csv", index=False)

    cfg = MNARXConfig(
        original_fs=20,
        target_fs=20,
        include_azimuth_harmonics=False,
        max_selected_features=5,
        polynomial_degree=2,
        polynomial_max_terms=60,
        correlation_threshold=0.05,
    )
    model = MNARXPlus(cfg)
    model.fit(data_dir)
    test_sim = next(iter((data_dir).glob("*.csv")))
    pred1 = model.simulate(test_sim)["BladeRootFlapwiseMoment"]

    p = tmp_path / "model.joblib"
    model.save(p)
    loaded = MNARXPlus.load(p)
    pred2 = loaded.simulate(test_sim)["BladeRootFlapwiseMoment"]
    assert np.allclose(pred1, pred2)


def test_synthetic_hierarchical_dependency_behavior(tmp_path: Path) -> None:
    data_dir = tmp_path / "hier"
    data_dir.mkdir()
    for i in range(8):
        _make_df(260, seed=i + 100).to_csv(data_dir / f"sim_{i}.csv", index=False)

    cfg = MNARXConfig(
        original_fs=20,
        target_fs=20,
        include_azimuth_harmonics=False,
        max_selected_features=6,
        polynomial_degree=1,
        polynomial_max_terms=80,
        correlation_threshold=0.03,
    )
    model = MNARXPlus(cfg)
    model.fit(data_dir)
    order = model._topological_model_order()  # noqa: SLF001
    assert cfg.final_target_column in order
    assert len(model.models_) >= 1
