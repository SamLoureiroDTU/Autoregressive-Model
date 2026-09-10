# mNARX+ for Wind Turbine Dynamic Surrogate Modeling

This repository implements a modular Python project for **mNARX+** dynamic surrogate modeling, adapted to wind turbine simulations.

## 1) mNARX+ overview

mNARX+ combines:
- manifold-style feature compression (here through spatial/temporal PCA options),
- NARX-style autoregressive-with-exogenous dynamic modeling,
- sparse polynomial FNARX regression,
- residual-driven automatic feature selection,
- recursive free-running forecasting.

## 2) Relation to arXiv methodology

Authoritative reference: **mNARX+: A surrogate model for complex dynamical systems using manifold-NARX and automatic feature selection** (arXiv).

Implemented core ideas:
- simulation-level splitting to avoid leakage,
- lag-window functional representation,
- sparse polynomial regression,
- iterative residual-correlation feature addition,
- recursive free-running selection/evaluation logic,
- recursive auxiliary model construction with dependency graph and cycle prevention.

## 3) Dataset format

- Input folder contains multiple CSVs.
- Each CSV is one full simulation.
- Expected default columns:
  - `Time`
  - `RotorSpeed`
  - `Pitch`
  - `BladeRootFlapwiseMoment`
- Wind columns are auto-resolved from names like:
  - `Free wind speed Vx pos 0.00, 0.00,-150.00`
  - similarly for `Vy`, `Vz` and other points.

## 4) How to edit column names / wind points

- Edit `MNARXConfig` in `mnarx_plus/config.py`:
  - `time_column`
  - `response_columns`
  - `final_target_column`
- Wind columns are matched centrally in `mnarx_plus/preprocessing.py` via regex (`WIND_COLUMN_RE`).

## 5) Wind representation options

Set `wind_representation_mode` in `MNARXConfig`:
- `direct`: use resolved wind channels directly.
- `spatial_pca`: fit scaler+PCA on training simulations only.
- `dct_placeholder`: interface stub (documented placeholder).

## 6) Training and evaluation commands

```bash
python run_training.py --data-folder /absolute/path/to/simulations --model-out artifacts/mnarx_wind_turbine.joblib
python run_evaluation.py --model-path artifacts/mnarx_wind_turbine.joblib --output-dir artifacts/evaluation
```

## 7) Recursive forecasting explanation

- Models are evaluated in **free-running recursive mode**.
- Past predicted values are fed back as history.
- During unseen simulation inference, future measured response/auxiliary values are blocked.
- Only initial measured history is allowed (configurable with `initial_history_seconds`).

## 8) Automatic auxiliary model generation

- Final target model is built first.
- Selected features are inspected to detect required auxiliary responses.
- Missing auxiliary response models are built recursively.
- Dependency cycles are prevented with a directed graph check.

## 9) Dependency graph interpretation

- Edge `A -> B` means model for `B` depends on response `A`.
- Topological order defines safe recursive inference order on unseen simulations.

## 10) Known approximations vs exact paper details

- **paper-specified**: residual-driven iterative feature selection and recursive prediction use.
- **implementation choice**: practical sparse Lasso FNARX and hyperbolic truncation via exponent filtering.
- **implementation choice**: spatial manifold proxy implemented with PCA over resolved wind channels.
- **implementation choice**: DCT mode kept as explicit placeholder interface.

---

## Practical guide: How to run on your own dataset

### 1. After cloning

```bash
cd /home/runner/work/Autoregressive-Model/Autoregressive-Model
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import numpy,pandas,scipy,sklearn,matplotlib,joblib,networkx; print('ok')"
```

### 2. Dataset preparation

Expected folder structure:

```text
data/
  sim_001.csv
  sim_002.csv
  sim_003.csv
  ...
```

Required columns per CSV:
- `Time`
- response placeholders (`RotorSpeed`, `Pitch`, `BladeRootFlapwiseMoment` by default)
- wind columns matching `... Vx pos x,y,z`, `... Vy pos x,y,z`, `... Vz pos x,y,z`

If names differ, adapt config fields (`time_column`, `response_columns`, `final_target_column`) and/or update wind regex in `mnarx_plus/preprocessing.py`.

### 3. Quick start (minimal run)

Training:

```bash
python run_training.py --data-folder /absolute/path/to/data --model-out artifacts/mnarx_wind_turbine.joblib
```

Evaluation:

```bash
python run_evaluation.py --model-path artifacts/mnarx_wind_turbine.joblib --output-dir artifacts/evaluation
```

Outputs:
- Model file: `artifacts/mnarx_wind_turbine.joblib`
- Metrics tables: `artifacts/evaluation/*_metrics_summary.csv`
- Plots: `artifacts/evaluation/<target>/<simulation>/...`

### 4. Configuration walkthrough

Edit these first in `MNARXConfig`:
- `data_folder` (or pass via script argument)
- `original_fs`, `target_fs`
- `memory_seconds_per_signal`
- `wind_representation_mode`
- `correlation_threshold`
- `polynomial_degree`
- `initial_history_seconds`

Concrete wind-turbine example:

```python
from mnarx_plus import MNARXConfig

config = MNARXConfig(
    random_seed=42,
    original_fs=100,
    target_fs=20,
    wind_representation_mode="spatial_pca",
    memory_seconds_per_signal={
        "RotorSpeed": 3.0,
        "Pitch": 3.0,
        "BladeRootFlapwiseMoment": 4.0,
        "wind": 3.0,
    },
    correlation_threshold=0.20,
    polynomial_degree=3,
    initial_history_seconds=3.0,
)
```

### 5. Run on unseen simulation

```python
from mnarx_plus import MNARXPlus

model = MNARXPlus.load("artifacts/mnarx_wind_turbine.joblib")
pred = model.simulate("/absolute/path/to/new_simulation.csv")
mbld_hat = pred["BladeRootFlapwiseMoment"]
```

Leakage rule reminders:
- initial measured history is used,
- future measured auxiliary/target values are not used,
- recursive predictions are used when required by dependencies.

### 6. Interpreting outputs

- Metrics tables (`RMSE`, `nRMSE`, `MAE`, `R2`, `Pearson`) are produced per simulation plus mean/median.
- Plots include:
  - full true vs pred,
  - zoomed segment,
  - error trace,
  - scatter true vs pred,
  - PSD comparison,
  - per-simulation metric bar chart.
- Feature selection history is available in each model bundle: `model.models_[target].selector_history`.
- Dependency graph edges (`model.dependency_graph_.edges()`) show auxiliary model dependencies.

### 7. Troubleshooting

- **Missing columns**: check CSV headers, config column names, and wind naming pattern.
- **Wrong sampling assumptions**: verify `original_fs` and `target_fs`; downsampling assumes these are correct.
- **Insufficient initial history**: increase `initial_history_seconds` and/or memory seconds.
- **Shape mismatch after config change**: retrain model after changing memory/PCA/polynomial settings.

### 8. Reproducibility

- Random seed is controlled by `MNARXConfig.random_seed`.
- Train/val/test splits are simulation-level and reproducible with the same seed.
- Save complete fitted object with `model.save(...)` to preserve config, preprocessing, models, and dependencies.

---

## API usage example

```python
from mnarx_plus import MNARXPlus, MNARXConfig

config = MNARXConfig(
    original_fs=100,
    target_fs=20,
    correlation_threshold=0.20,
    polynomial_degree=3,
)

model = MNARXPlus(config)
model.fit(data_folder="path/to/simulations/")
results = model.evaluate_test_set()
model.save("mnarx_wind_turbine.joblib")

pred = model.simulate("unseen_simulation.csv")
mbld_hat = pred["BladeRootFlapwiseMoment"]
```
