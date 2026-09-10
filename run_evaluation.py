"""Evaluate saved mNARX+ model on its held-out test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mnarx_plus import MNARXPlus
from mnarx_plus.plotting import (
    plot_error,
    plot_metric_bars,
    plot_psd,
    plot_scatter,
    plot_timeseries,
    plot_zoom,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="artifacts/evaluation")
    args = parser.parse_args()

    model = MNARXPlus.load(args.model_path)
    results = model.evaluate_test_set()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for target, summary_df in results["summary"].items():
        summary_df.to_csv(out_dir / f"{target}_metrics_summary.csv")

    test_names = model.split_.test  # type: ignore[union-attr]
    for target, by_sim in results["predictions"].items():
        for sim_name in test_names:
            y_true, y_pred = model._recursive_forecast_single(  # noqa: SLF001
                model.models_[target].model,
                sim_name,
                target,
                model.models_[target].selected_features,
                model.exogenous_signals_ + [target] + model.models_[target].dependency_signals,
                aux_predictions=None,
                strict_no_future_measurements=False,
            )
            sim_dir = out_dir / target / sim_name
            sim_dir.mkdir(parents=True, exist_ok=True)
            plot_timeseries(y_true, y_pred, sim_dir / "timeseries.png", f"{target} {sim_name}")
            end = min(len(y_true), 600)
            plot_zoom(y_true, y_pred, sim_dir / "zoom.png", 0, end, f"{target} zoom {sim_name}")
            plot_error(y_true, y_pred, sim_dir / "error.png", f"{target} error {sim_name}")
            plot_scatter(y_true, y_pred, sim_dir / "scatter.png", f"{target} scatter {sim_name}")
            plot_psd(y_true, y_pred, sim_dir / "psd.png", fs=model.config.target_fs, title=f"{target} PSD {sim_name}")

        metric_chart = {sim: results["per_target"][target][sim]["RMSE"] for sim in test_names}
        plot_metric_bars(metric_chart, out_dir / target / "rmse_per_sim.png", f"{target} RMSE per simulation")

    print("Saved evaluation outputs to", out_dir)


if __name__ == "__main__":
    main()
