"""Training entry point for HemoMesh backbone experiments."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hemomesh.data import SukHDF5Dataset, build_graph, fit_tensor_normalizer
from hemomesh.experiments import append_experiment
from hemomesh.metrics import (
    approximation_error,
    bootstrap_ci,
    cosine_similarity,
    nmae,
    pressure_r2,
    rmse,
)
from hemomesh.models import MeshGNN, MLPFieldRegressor


def _require_torch():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install the `torch` extra to train models.") from exc
    return torch, functional


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML training config."""

    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("training config must be a YAML mapping")
    return config


def set_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and Torch."""

    torch, _ = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_case_indices(
    num_cases: int,
    train_cases: int,
    val_cases: int,
    test_cases: int,
    seed: int,
) -> dict[str, list[int]]:
    """Create deterministic train/validation/test case-index splits."""

    requested = train_cases + val_cases + test_cases
    if requested > num_cases:
        raise ValueError(f"requested {requested} cases, but dataset contains {num_cases}")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_cases)[:requested].astype(int).tolist()
    train_end = train_cases
    val_end = train_end + val_cases
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def _build_model(config: dict[str, Any], in_features: int, out_features: int):
    model_name = config.get("model", "mlp")
    model_config = config.get("model_params", {})
    hidden_features = tuple(model_config.get("hidden_features", [128, 128, 128]))
    dropout = float(model_config.get("dropout", 0.0))
    output_smoothing_steps = int(model_config.get("output_smoothing_steps", 0))
    output_smoothing_alpha = float(model_config.get("output_smoothing_alpha", 0.25))

    if model_name == "mlp":
        return MLPFieldRegressor(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            dropout=dropout,
        )
    if model_name == "mesh_gnn":
        return MeshGNN(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            dropout=dropout,
            output_smoothing_steps=output_smoothing_steps,
            output_smoothing_alpha=output_smoothing_alpha,
        )
    raise ValueError("model must be either 'mlp' or 'mesh_gnn'")


def _forward(model, graph, model_name: str):
    if model_name == "mlp":
        return model(graph.x)
    return model(graph.x, graph.edge_index)


def _normalize_graphs(graphs, x_normalizer, y_normalizer):
    normalized = []
    for graph in graphs:
        item = graph.clone()
        item.y_raw = item.y.clone()
        item.x = x_normalizer.transform(item.x)
        item.y = y_normalizer.transform(item.y)
        normalized.append(item)
    return normalized


def weighted_mse_loss(prediction, target, target_weights=None):
    """Return MSE with optional per-target weights."""

    _, functional = _require_torch()
    loss = functional.mse_loss(prediction, target, reduction="none")
    if target_weights is not None:
        loss = loss * target_weights.to(prediction.device)
    return loss.mean()


def train_one_epoch(
    model,
    graphs,
    optimizer,
    model_name: str,
    device: str = "cpu",
    target_weights=None,
) -> float:
    """Train one epoch over a list of full-mesh PyG graphs."""

    torch, functional = _require_torch()
    model.train()
    losses: list[float] = []
    for graph in graphs:
        graph = graph.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = _forward(model, graph, model_name)
        loss = weighted_mse_loss(prediction, graph.y, target_weights=target_weights)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        return float("nan")
    return float(np.mean(losses))


def evaluate_loss(
    model,
    graphs,
    model_name: str,
    device: str = "cpu",
    target_weights=None,
) -> float:
    """Return normalized MSE over a list of validation graphs."""

    torch, functional = _require_torch()
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            prediction = _forward(model, graph, model_name)
            loss = weighted_mse_loss(prediction, graph.y, target_weights=target_weights)
            losses.append(float(loss.detach().cpu()))
    if not losses:
        return float("nan")
    return float(np.mean(losses))


def evaluate_model(model, graphs, model_name: str, y_normalizer, device: str = "cpu"):
    """Evaluate case-level field metrics on normalized graphs."""

    torch, _ = _require_torch()
    model.eval()
    case_metrics: list[dict[str, Any]] = []
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            prediction = _forward(model, graph, model_name)
            prediction_raw = y_normalizer.inverse_transform(prediction).detach().cpu().numpy()
            reference_raw = graph.y_raw.detach().cpu().numpy()
            wss_prediction = prediction_raw[:, :3]
            wss_reference = reference_raw[:, :3]
            pressure_prediction = prediction_raw[:, 3:]
            pressure_reference = reference_raw[:, 3:]
            case_metrics.append(
                {
                    "sample_id": graph.sample_id,
                    "subset": graph.subset,
                    "wss_approximation_error": approximation_error(
                        wss_prediction,
                        wss_reference,
                    ),
                    "wss_rmse": rmse(wss_prediction, wss_reference),
                    "wss_nmae": nmae(wss_prediction, wss_reference),
                    "wss_cosine_similarity": cosine_similarity(
                        wss_prediction,
                        wss_reference,
                    ),
                    "pressure_rmse": rmse(pressure_prediction, pressure_reference),
                    "pressure_nmae": nmae(pressure_prediction, pressure_reference),
                    "pressure_r2": pressure_r2(pressure_prediction, pressure_reference),
                }
            )
    return case_metrics


def _mean_metric(case_metrics: list[dict[str, Any]], key: str) -> float:
    values = [float(item[key]) for item in case_metrics]
    return float(np.mean(values)) if values else float("nan")


def summarize_case_metrics(
    case_metrics: list[dict[str, Any]],
    seed: int,
    prefix: str,
) -> dict[str, Any]:
    """Summarize case-level metrics with a bootstrap CI for WSS approximation error."""

    wss_errors = [float(item["wss_approximation_error"]) for item in case_metrics]
    wss_ci = bootstrap_ci(wss_errors, seed=seed) if wss_errors else {}
    summary = {
        f"{prefix}_num_cases": len(case_metrics),
        f"{prefix}_wss_approximation_error_mean": wss_ci.get("mean", float("nan")),
        f"{prefix}_wss_approximation_error_ci_low": wss_ci.get("ci_low", float("nan")),
        f"{prefix}_wss_approximation_error_ci_high": wss_ci.get("ci_high", float("nan")),
        f"{prefix}_wss_rmse_mean": _mean_metric(case_metrics, "wss_rmse"),
        f"{prefix}_wss_nmae_mean": _mean_metric(case_metrics, "wss_nmae"),
        f"{prefix}_wss_cosine_similarity_mean": _mean_metric(
            case_metrics,
            "wss_cosine_similarity",
        ),
        f"{prefix}_pressure_rmse_mean": _mean_metric(case_metrics, "pressure_rmse"),
        f"{prefix}_pressure_nmae_mean": _mean_metric(case_metrics, "pressure_nmae"),
        f"{prefix}_pressure_r2_mean": _mean_metric(case_metrics, "pressure_r2"),
    }
    return summary


def _write_case_metrics(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    """Run one configured backbone experiment and append it to the ledger."""

    torch, _ = _require_torch()
    seed = int(config.get("seed", 0))
    set_seed(seed)

    data_config = config.get("data", {})
    split_config = config.get("split_params", {})
    train_config = config.get("training", {})
    output_config = config.get("outputs", {})
    subset = data_config.get("subset", config.get("subset", "single"))
    dataset = SukHDF5Dataset(data_config.get("root", "."), subset=subset)

    splits = split_case_indices(
        num_cases=len(dataset),
        train_cases=int(split_config.get("train_cases", 24)),
        val_cases=int(split_config.get("val_cases", 8)),
        test_cases=int(split_config.get("test_cases", 8)),
        seed=seed,
    )
    graphs_by_split = {
        split_name: [build_graph(dataset[index]) for index in indices]
        for split_name, indices in splits.items()
    }

    x_normalizer = fit_tensor_normalizer(graph.x for graph in graphs_by_split["train"])
    y_normalizer = fit_tensor_normalizer(graph.y for graph in graphs_by_split["train"])
    graphs_by_split = {
        split_name: _normalize_graphs(graphs, x_normalizer, y_normalizer)
        for split_name, graphs in graphs_by_split.items()
    }

    model_name = config.get("model", "mlp")
    in_features = int(graphs_by_split["train"][0].x.shape[1])
    out_features = int(graphs_by_split["train"][0].y.shape[1])
    model = _build_model(config, in_features, out_features)
    device = train_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("lr", 1e-3)),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
    )
    target_weights = train_config.get("target_weights")
    if target_weights is not None:
        target_weights = torch.as_tensor(target_weights, dtype=torch.float32, device=device)

    history: list[dict[str, float]] = []
    epochs = int(train_config.get("epochs", 25))
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            graphs_by_split["train"],
            optimizer,
            model_name=model_name,
            device=device,
            target_weights=target_weights,
        )
        val_loss = evaluate_loss(
            model,
            graphs_by_split["val"],
            model_name=model_name,
            device=device,
            target_weights=target_weights,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    model.load_state_dict(best_state)

    val_case_metrics = evaluate_model(
        model,
        graphs_by_split["val"],
        model_name,
        y_normalizer,
        device=device,
    )
    test_case_metrics = evaluate_model(
        model,
        graphs_by_split["test"],
        model_name,
        y_normalizer,
        device=device,
    )
    metrics = {
        "train_loss_final": history[-1]["train_loss"] if history else float("nan"),
        "val_loss_best": best_val_loss,
        "best_epoch": best_epoch,
        **summarize_case_metrics(val_case_metrics, seed=seed, prefix="val"),
        **summarize_case_metrics(test_case_metrics, seed=seed, prefix="test"),
    }
    run_payload = {
        "config": config,
        "metrics": metrics,
        "splits": splits,
        "history": history,
    }

    case_metrics_path = output_config.get("case_metrics")
    if case_metrics_path:
        rows = [
            {"split": "val", **row} for row in val_case_metrics
        ] + [{"split": "test", **row} for row in test_case_metrics]
        _write_case_metrics(case_metrics_path, rows)

    summary_path = output_config.get("summary_json")
    if summary_path:
        _write_json(summary_path, run_payload)

    ledger_path = output_config.get("experiment_log", "results/experiments.csv")
    row = append_experiment(
        {
            "run_type": config.get("run_type", "backbone_training"),
            "model": model_name,
            "subset": subset,
            "split": config.get("split", "random_case_split"),
            "seed": seed,
            **config,
        },
        metrics,
        path=ledger_path,
    )
    return {"ledger_row": row, **run_payload}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a HemoMesh model from a config file.")
    parser.add_argument("--config", required=True, help="Path to a YAML training config.")
    parser.add_argument("--data-root", help="Override the dataset root in the config.")
    parser.add_argument("--epochs", type=int, help="Override the epoch count in the config.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_root:
        config.setdefault("data", {})["root"] = args.data_root
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    result = run_training(config)
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
