"""Run M3 conformal and MC-dropout uncertainty evaluation."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from hemomesh.data import SukHDF5Dataset, build_graph, fit_tensor_normalizer
from hemomesh.experiments import append_experiment
from hemomesh.metrics import ause, bootstrap_ci, regression_ece, risk_coverage_curve
from hemomesh.train import (
    _build_model,
    _forward,
    _normalize_graphs,
    evaluate_loss,
    load_config,
    set_seed,
    train_one_epoch,
)


def split_uq_indices(
    num_cases: int,
    train_cases: int,
    val_cases: int,
    calibration_cases: int,
    test_cases: int,
    seed: int,
) -> dict[str, list[int]]:
    """Create deterministic train/validation/calibration/test splits."""

    requested = train_cases + val_cases + calibration_cases + test_cases
    if requested > num_cases:
        raise ValueError(f"requested {requested} cases, but dataset contains {num_cases}")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_cases)[:requested].astype(int).tolist()
    train_end = train_cases
    val_end = train_end + val_cases
    calibration_end = val_end + calibration_cases
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "calibration": indices[val_end:calibration_end],
        "test": indices[calibration_end:],
    }


def conformal_radius(residual_norms: np.ndarray, nominal_coverage: float) -> float:
    """Return split-conformal radius for vector residual norms."""

    residuals = np.asarray(residual_norms, dtype=np.float64).reshape(-1)
    if len(residuals) == 0:
        raise ValueError("residual_norms must be non-empty")
    if not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage must lie in (0, 1)")
    quantile_level = min(float(np.ceil((len(residuals) + 1) * nominal_coverage) / len(residuals)), 1.0)
    return float(np.quantile(residuals, quantile_level, method="higher"))


def _target_weights(config: dict[str, Any], device: str):
    torch = __import__("torch")
    weights = config.get("training", {}).get("target_weights")
    if weights is None:
        return None
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _train_model(config: dict[str, Any], graphs_by_split: dict[str, list[Any]], device: str):
    torch = __import__("torch")
    train_config = config.get("training", {})
    model_name = config.get("model", "mesh_gnn")
    in_features = int(graphs_by_split["train"][0].x.shape[1])
    out_features = int(graphs_by_split["train"][0].y.shape[1])
    model = _build_model(config, in_features, out_features).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("lr", 1e-3)),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
    )
    weights = _target_weights(config, device=device)
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float]] = []

    for epoch in range(1, int(train_config.get("epochs", 80)) + 1):
        train_loss = train_one_epoch(
            model,
            graphs_by_split["train"],
            optimizer,
            model_name=model_name,
            device=device,
            target_weights=weights,
        )
        val_loss = evaluate_loss(
            model,
            graphs_by_split["val"],
            model_name=model_name,
            device=device,
            target_weights=weights,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    model.load_state_dict(best_state)
    return model, history, best_epoch, best_val_loss


def _predict_graph(model, graph, model_name: str, y_normalizer, device: str, stochastic_passes: int = 1):
    torch = __import__("torch")
    graph = graph.to(device)
    predictions = []
    with torch.no_grad():
        for _ in range(stochastic_passes):
            pred = _forward(model, graph, model_name)
            predictions.append(y_normalizer.inverse_transform(pred).detach().cpu().numpy())
    samples = np.stack(predictions, axis=0)
    reference = graph.y_raw.detach().cpu().numpy()
    return {
        "sample_id": graph.sample_id,
        "mean": samples.mean(axis=0),
        "std": samples.std(axis=0, ddof=1) if stochastic_passes > 1 else np.zeros_like(samples[0]),
        "reference": reference,
    }


def _predict_split(
    model,
    graphs: list[Any],
    model_name: str,
    y_normalizer,
    device: str,
    stochastic_passes: int,
):
    if stochastic_passes > 1:
        model.train()
    else:
        model.eval()
    return [
        _predict_graph(
            model,
            graph,
            model_name=model_name,
            y_normalizer=y_normalizer,
            device=device,
            stochastic_passes=stochastic_passes,
        )
        for graph in graphs
    ]


def _wss_residual_norm(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.linalg.norm(prediction[:, :3] - reference[:, :3], axis=1)


def _wss_uncertainty_norm(std: np.ndarray) -> np.ndarray:
    return np.linalg.norm(std[:, :3], axis=1)


def _flatten_metric(items: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for item in items:
        if key == "error":
            values.append(_wss_residual_norm(item["mean"], item["reference"]))
        elif key == "uncertainty":
            values.append(_wss_uncertainty_norm(item["std"]))
        else:
            raise ValueError(f"unknown metric key: {key}")
    return np.concatenate(values)


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _risk_svg(path: str | Path, rows: list[dict[str, float]]) -> None:
    width, height = 580, 360
    left, right, top, bottom = 75, 30, 50, 60
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_risk = max(row["risk"] for row in rows) * 1.08

    def x_pos(coverage: float) -> float:
        return left + (coverage - 0.5) / 0.5 * plot_width

    def y_pos(risk: float) -> float:
        return top + plot_height - risk / max_risk * plot_height

    points = " ".join(f"{x_pos(row['coverage']):.1f},{y_pos(row['risk']):.1f}" for row in rows)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="290" y="28" text-anchor="middle" font-family="Arial, sans-serif" font-size="18">M3 MC-dropout risk-coverage</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" stroke="#111827"/>',
        f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="3"/>',
    ]
    for row in rows:
        svg.append(
            f'<circle cx="{x_pos(row["coverage"]):.1f}" cy="{y_pos(row["risk"]):.1f}" r="4" fill="#2563eb"/>'
        )
    svg.extend(
        [
            f'<text x="{left + plot_width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">Retained coverage</text>',
            f'<text x="20" y="{top + plot_height / 2}" transform="rotate(-90 20 {top + plot_height / 2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">Retained WSS error</text>',
            "</svg>",
        ]
    )
    Path(path).write_text("\n".join(svg) + "\n", encoding="utf-8")


def _scatter_svg(path: str | Path, uncertainty: np.ndarray, error: np.ndarray) -> None:
    width, height = 580, 360
    left, right, top, bottom = 75, 30, 50, 60
    plot_width = width - left - right
    plot_height = height - top - bottom
    # Deterministic downsample keeps the SVG small and reproducible.
    step = max(1, len(error) // 500)
    uncertainty = uncertainty[::step]
    error = error[::step]
    max_unc = max(float(np.max(uncertainty)), 1e-12)
    max_error = max(float(np.max(error)), 1e-12)

    def x_pos(value: float) -> float:
        return left + value / max_unc * plot_width

    def y_pos(value: float) -> float:
        return top + plot_height - value / max_error * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="290" y="28" text-anchor="middle" font-family="Arial, sans-serif" font-size="18">M3 uncertainty vs WSS error</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" stroke="#111827"/>',
    ]
    for unc, err in zip(uncertainty, error, strict=True):
        svg.append(
            f'<circle cx="{x_pos(float(unc)):.1f}" cy="{y_pos(float(err)):.1f}" r="2" fill="#2563eb" opacity="0.35"/>'
        )
    svg.extend(
        [
            f'<text x="{left + plot_width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">MC-dropout WSS std norm</text>',
            f'<text x="20" y="{top + plot_height / 2}" transform="rotate(-90 20 {top + plot_height / 2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">WSS error norm</text>',
            "</svg>",
        ]
    )
    Path(path).write_text("\n".join(svg) + "\n", encoding="utf-8")


def run_uq(config: dict[str, Any]) -> dict[str, Any]:
    """Run M3 UQ calibration and write configured outputs."""

    torch = __import__("torch")
    seed = int(config.get("seed", 0))
    set_seed(seed)
    data_config = config.get("data", {})
    split_config = config.get("split_params", {})
    output_config = config.get("outputs", {})
    uq_config = config.get("uq", {})
    model_name = config.get("model", "mesh_gnn")
    subset = data_config.get("subset", config.get("subset", "single"))
    device = config.get("training", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")

    dataset = SukHDF5Dataset(data_config.get("root", "."), subset=subset)
    splits = split_uq_indices(
        num_cases=len(dataset),
        train_cases=int(split_config.get("train_cases", 48)),
        val_cases=int(split_config.get("val_cases", 16)),
        calibration_cases=int(split_config.get("calibration_cases", 16)),
        test_cases=int(split_config.get("test_cases", 16)),
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

    model, history, best_epoch, best_val_loss = _train_model(config, graphs_by_split, device=device)
    calibration_predictions = _predict_split(
        model,
        graphs_by_split["calibration"],
        model_name=model_name,
        y_normalizer=y_normalizer,
        device=device,
        stochastic_passes=1,
    )
    nominal = float(uq_config.get("nominal_coverage", 0.9))
    calibration_error = _flatten_metric(calibration_predictions, "error")
    radius = conformal_radius(calibration_error, nominal_coverage=nominal)

    mc_passes = int(uq_config.get("mc_dropout_passes", 20))
    test_predictions = _predict_split(
        model,
        graphs_by_split["test"],
        model_name=model_name,
        y_normalizer=y_normalizer,
        device=device,
        stochastic_passes=mc_passes,
    )
    test_error = _flatten_metric(test_predictions, "error")
    test_uncertainty = _flatten_metric(test_predictions, "uncertainty")
    coverage = float(np.mean(test_error <= radius))
    width = float(2.0 * radius)
    correlation = (
        float(np.corrcoef(test_uncertainty, test_error)[0, 1])
        if float(np.std(test_uncertainty)) > 0
        else float("nan")
    )
    risk_rows = risk_coverage_curve(
        test_uncertainty,
        test_error,
        coverages=np.asarray(uq_config.get("risk_coverages", [1.0, 0.9, 0.8, 0.7, 0.5])),
    )
    metrics = {
        "best_epoch": best_epoch,
        "val_loss_best": best_val_loss,
        "nominal_coverage": nominal,
        "conformal_radius": radius,
        "conformal_width": width,
        "test_conformal_coverage": coverage,
        "test_error_mean": float(np.mean(test_error)),
        "test_error_ci_low": bootstrap_ci(test_error, seed=seed)["ci_low"],
        "test_error_ci_high": bootstrap_ci(test_error, seed=seed)["ci_high"],
        "mc_dropout_passes": mc_passes,
        "mc_error_uncertainty_corr": correlation,
        "mc_regression_ece": regression_ece(test_uncertainty, test_error),
        "mc_ause": ause(test_uncertainty, test_error),
        "risk_at_50pct_coverage": next(row["risk"] for row in risk_rows if row["coverage"] <= 0.5),
    }
    payload = {
        "config": config,
        "metrics": metrics,
        "splits": splits,
        "history": history,
    }

    summary_rows = [
        {
            "method": "split_conformal_wss_radius",
            "nominal_coverage": nominal,
            "empirical_coverage": coverage,
            "radius": radius,
            "width": width,
            "mean_wss_error": metrics["test_error_mean"],
        },
        {
            "method": "mc_dropout_uncertainty",
            "nominal_coverage": "",
            "empirical_coverage": "",
            "radius": "",
            "width": "",
            "mean_wss_error": metrics["test_error_mean"],
            "error_uncertainty_corr": correlation,
            "regression_ece": metrics["mc_regression_ece"],
            "ause": metrics["mc_ause"],
        },
    ]
    _write_csv(output_config["summary_table"], summary_rows)
    _write_csv(output_config["risk_table"], risk_rows)
    _write_json(output_config["summary_json"], payload)
    _risk_svg(output_config["risk_figure"], risk_rows)
    _scatter_svg(output_config["scatter_figure"], test_uncertainty, test_error)

    summary_md = "\n".join(
        [
            "# M3 UQ Calibration",
            "",
            f"MeshGNN trained on {len(splits['train'])} cases, validated on {len(splits['val'])}, calibrated on {len(splits['calibration'])}, and tested on {len(splits['test'])}.",
            "",
            f"- Split-conformal WSS residual radius: `{radius:.4f}` at nominal `{nominal:.2f}` coverage.",
            f"- Test conformal coverage: `{coverage:.4f}`.",
            f"- Mean test WSS residual norm: `{metrics['test_error_mean']:.4f}`.",
            f"- MC-dropout error/uncertainty correlation: `{correlation:.4f}`.",
            f"- MC-dropout AUSE: `{metrics['mc_ause']:.4f}`.",
            "",
            "Interpretation: conformal prediction provides the calibrated coverage artifact for M3, while MC dropout provides a spatial uncertainty score for deferral and risk-coverage analysis.",
            "",
        ]
    )
    Path(output_config["summary_md"]).write_text(summary_md, encoding="utf-8")
    append_experiment(
        {
            "run_type": config.get("run_type", "uq_calibration"),
            "model": model_name,
            "subset": subset,
            "split": config.get("split", "train_val_calibration_test"),
            "seed": seed,
            **config,
        },
        metrics,
        path=output_config.get("experiment_log", "results/experiments.csv"),
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M3 UQ calibration for HemoMesh.")
    parser.add_argument("--config", default="configs/m3_uq_mesh_gnn.yaml")
    parser.add_argument("--data-root", help="Override dataset root.")
    parser.add_argument("--epochs", type=int, help="Override training epochs.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_root:
        config.setdefault("data", {})["root"] = args.data_root
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    result = run_uq(config)
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
