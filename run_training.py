"""Train mNARX+ model and persist artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from mnarx_plus import MNARXConfig, MNARXPlus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--model-out", default="artifacts/mnarx_wind_turbine.joblib")
    parser.add_argument("--target-fs", type=int, default=20)
    parser.add_argument("--wind-mode", default="spatial_pca", choices=["direct", "spatial_pca", "dct_placeholder"])
    args = parser.parse_args()

    config = MNARXConfig(target_fs=args.target_fs, wind_representation_mode=args.wind_mode)
    model = MNARXPlus(config)
    model.fit(args.data_folder)

    print("Dependency edges:")
    for u, v in model.dependency_graph_.edges():
        print(f"  {u} -> {v}")

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"Saved model to {out}")


if __name__ == "__main__":
    main()
