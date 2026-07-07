"""Log pretrained Suk-baseline predictions through the HemoMesh metric harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hemomesh.experiments import append_experiment
from hemomesh.metrics.fields import approximation_error, cosine_similarity, nmae, rmse


def load_array(path: str | Path) -> np.ndarray:
    """Load a `.npy` or `.npz` array from disk."""

    array_path = Path(path)
    if array_path.suffix == ".npy":
        return np.load(array_path)
    if array_path.suffix == ".npz":
        data = np.load(array_path)
        if len(data.files) != 1:
            raise ValueError(f"{array_path} must contain exactly one array")
        return data[data.files[0]]
    raise ValueError(f"Unsupported array format: {array_path.suffix}")


def evaluate_wss_baseline(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Compute the WSS metrics used for the first baseline reproduction pass."""

    return {
        "wss_approximation_error": approximation_error(prediction, reference),
        "wss_rmse": rmse(prediction, reference),
        "wss_nmae": nmae(prediction, reference),
        "wss_cosine_similarity": cosine_similarity(prediction, reference),
    }


def log_wss_baseline(
    name: str,
    subset: str,
    prediction_path: str | Path,
    truth_path: str | Path,
    output_path: str | Path = "results/experiments.csv",
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate and append one WSS baseline row to `results/experiments.csv`."""

    prediction = load_array(prediction_path)
    reference = load_array(truth_path)
    metrics = evaluate_wss_baseline(prediction, reference)
    config: dict[str, Any] = {
        "run_type": "baseline_reproduction",
        "model": name,
        "subset": subset,
        "split": "pretrained_eval",
        "prediction_path": str(prediction_path),
        "truth_path": str(truth_path),
    }
    if extra_config:
        config.update(extra_config)
    return append_experiment(config=config, metrics=metrics, path=output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Log WSS baseline reproduction metrics.")
    parser.add_argument("--name", required=True, help="Baseline name, e.g. gem_gcn_pretrained.")
    parser.add_argument("--subset", required=True, choices=["single", "bifurcating"])
    parser.add_argument("--prediction", required=True, help="Path to WSS prediction .npy/.npz.")
    parser.add_argument("--truth", required=True, help="Path to WSS reference .npy/.npz.")
    parser.add_argument("--output", default="results/experiments.csv")
    args = parser.parse_args()

    row = log_wss_baseline(
        name=args.name,
        subset=args.subset,
        prediction_path=args.prediction,
        truth_path=args.truth,
        output_path=args.output,
    )
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
