#!/usr/bin/env python3
"""Generate M1 dataset EDA artifacts for the Suk coronary mesh dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import h5py
import numpy as np

from hemomesh.data import file_md5

SUBSETS = ("single", "bifurcating")


def database_path(root: Path, subset: str) -> Path:
    return root / "vessel-datasets" / "stead" / subset / "raw" / "database.hdf5"


def summarize_sample(sample_id: str, group: h5py.Group, subset: str) -> dict[str, Any]:
    pos = np.asarray(group["pos"])
    face = np.asarray(group["face"])
    wss = np.asarray(group["wss"], dtype=np.float64)
    pressure = np.asarray(group["pressure"], dtype=np.float64).reshape(-1)
    inlet_idcs = np.asarray(group["inlet_idcs"])
    wss_mag = np.linalg.norm(wss, axis=1)

    return {
        "subset": subset,
        "sample_id": sample_id,
        "num_nodes": int(pos.shape[0]),
        "num_faces": int(face.shape[0]),
        "inlet_count": int(inlet_idcs.shape[0]),
        "wss_mag_mean": float(np.mean(wss_mag)),
        "wss_mag_std": float(np.std(wss_mag)),
        "wss_mag_max": float(np.max(wss_mag)),
        "pressure_mean": float(np.mean(pressure)),
        "pressure_min": float(np.min(pressure)),
        "pressure_max": float(np.max(pressure)),
        "pressure_range": float(np.max(pressure) - np.min(pressure)),
    }


def inspect_subset(root: Path, subset: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = database_path(root, subset)
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        for sample_id in sorted(handle.keys()):
            rows.append(summarize_sample(sample_id, handle[sample_id], subset))

    summary = {
        "subset": subset,
        "path": str(path.relative_to(root)),
        "md5": file_md5(path),
        "num_samples": len(rows),
        "node_count_min": min(row["num_nodes"] for row in rows),
        "node_count_mean": mean(row["num_nodes"] for row in rows),
        "node_count_max": max(row["num_nodes"] for row in rows),
        "face_count_min": min(row["num_faces"] for row in rows),
        "face_count_mean": mean(row["num_faces"] for row in rows),
        "face_count_max": max(row["num_faces"] for row in rows),
        "inlet_count_mean": mean(row["inlet_count"] for row in rows),
        "wss_mag_mean": mean(row["wss_mag_mean"] for row in rows),
        "wss_mag_max": max(row["wss_mag_max"] for row in rows),
        "pressure_range_mean": mean(row["pressure_range"] for row in rows),
        "pressure_range_max": max(row["pressure_range"] for row in rows),
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# M1 Dataset EDA Summary",
        "",
        "The Suk coronary mesh dataset is present in the expected steady-flow layout.",
        (
            "Raw HDF5 files remain outside version control; this artifact records "
            "reproducible metadata."
        ),
        "",
        (
            "| Subset | Samples | Nodes Mean | Nodes Range | Faces Mean | "
            "WSS Magnitude Mean | Max WSS Magnitude | Pressure Range Mean | MD5 |"
        ),
        "|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| {subset} | {num_samples} | {node_mean:.2f} | {node_min}-{node_max} | "
            "{face_mean:.2f} | {wss_mean:.4f} | {wss_max:.4f} | "
            "{pressure_range:.4f} | `{md5}` |".format(
                subset=row["subset"],
                num_samples=row["num_samples"],
                node_mean=row["node_count_mean"],
                node_min=row["node_count_min"],
                node_max=row["node_count_max"],
                face_mean=row["face_count_mean"],
                wss_mean=row["wss_mag_mean"],
                wss_max=row["wss_mag_max"],
                pressure_range=row["pressure_range_mean"],
                md5=row["md5"],
            )
        )
    lines.extend(
        [
            "",
            "Report use:",
            "",
            "- Use sample counts and checksums in the data section.",
            "- Use node and face ranges to justify mesh-size handling.",
            "- Use WSS and pressure summaries to sanity-check target scale before training.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_histogram_svg(
    path: Path,
    rows: list[dict[str, Any]],
    value_key: str,
    title: str,
    x_label: str,
    bins: int = 24,
) -> None:
    """Write a lightweight grouped histogram as SVG."""

    path.parent.mkdir(parents=True, exist_ok=True)
    all_values = np.asarray([float(row[value_key]) for row in rows], dtype=np.float64)
    bin_edges = np.linspace(float(np.min(all_values)), float(np.max(all_values)), bins + 1)
    max_count = 1
    histograms: dict[str, np.ndarray] = {}
    for subset in SUBSETS:
        values = np.asarray(
            [float(row[value_key]) for row in rows if row["subset"] == subset],
            dtype=np.float64,
        )
        counts, _ = np.histogram(values, bins=bin_edges)
        histograms[subset] = counts
        max_count = max(max_count, int(np.max(counts)))

    width, height = 900, 420
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 50, 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    group_width = plot_width / bins
    bar_width = group_width / (len(SUBSETS) + 1)
    colors = {"single": "#3867d6", "bifurcating": "#20bf6b"}

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" '
        'font-family="Arial" font-size="18" font-weight="bold">'
        f"{title}</text>",
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" '
        f'x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
        f'y2="{height - margin_bottom}" stroke="#333"/>',
    ]

    for subset_index, subset in enumerate(SUBSETS):
        for bin_index, count in enumerate(histograms[subset]):
            bar_height = (int(count) / max_count) * plot_height
            x = margin_left + bin_index * group_width + subset_index * bar_width
            y = height - margin_bottom - bar_height
            lines.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                f'height="{bar_height:.2f}" fill="{colors[subset]}" opacity="0.82"/>'
            )

    tick_values = np.linspace(bin_edges[0], bin_edges[-1], 5)
    for value in tick_values:
        x = margin_left + ((value - bin_edges[0]) / (bin_edges[-1] - bin_edges[0])) * plot_width
        lines.extend(
            [
                f'<line x1="{x:.2f}" y1="{height - margin_bottom}" '
                f'x2="{x:.2f}" y2="{height - margin_bottom + 5}" stroke="#333"/>',
                f'<text x="{x:.2f}" y="{height - margin_bottom + 22}" '
                'text-anchor="middle" font-family="Arial" font-size="11">'
                f"{value:.0f}</text>",
            ]
        )

    for fraction in (0.0, 0.5, 1.0):
        count = max_count * fraction
        y = height - margin_bottom - fraction * plot_height
        lines.extend(
            [
                f'<line x1="{margin_left - 5}" y1="{y:.2f}" x2="{margin_left}" '
                'y2="{y:.2f}" stroke="#333"/>',
                f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" '
                f'font-family="Arial" font-size="11">{count:.0f}</text>',
            ]
        )

    lines.extend(
        [
            f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" '
            f'font-family="Arial" font-size="13">{x_label}</text>',
            f'<text x="18" y="{height / 2}" text-anchor="middle" '
            'font-family="Arial" font-size="13" transform="rotate(-90 18 '
            f'{height / 2})">Cases</text>',
        ]
    )

    legend_x = width - 205
    for idx, subset in enumerate(SUBSETS):
        y = 58 + idx * 22
        lines.extend(
            [
                f'<rect x="{legend_x}" y="{y}" width="14" height="14" '
                f'fill="{colors[subset]}" opacity="0.82"/>',
                f'<text x="{legend_x + 22}" y="{y + 12}" '
                f'font-family="Arial" font-size="12">{subset}</text>',
            ]
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate M1 dataset EDA artifacts.")
    parser.add_argument("--root", default=".", help="Project root containing vessel-datasets.")
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for generated artifacts.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    artifacts_dir = output_dir / "artifacts"
    figures_dir = output_dir / "figures"

    summaries: list[dict[str, Any]] = []
    per_case_rows: list[dict[str, Any]] = []
    for subset in SUBSETS:
        summary, rows = inspect_subset(root, subset)
        summaries.append(summary)
        per_case_rows.extend(rows)

    write_csv(tables_dir / "m1_dataset_eda_summary.csv", summaries)
    write_csv(tables_dir / "m1_dataset_per_case_stats.csv", per_case_rows)
    write_markdown(artifacts_dir / "m1_dataset_eda_summary.md", summaries)
    write_histogram_svg(
        figures_dir / "m1_node_count_histogram.svg",
        per_case_rows,
        "num_nodes",
        "Mesh Node Count Distribution",
        "Nodes per case",
    )
    write_histogram_svg(
        figures_dir / "m1_wss_magnitude_histogram.svg",
        per_case_rows,
        "wss_mag_mean",
        "Mean WSS Magnitude Distribution",
        "Per-case mean WSS magnitude",
    )
    write_histogram_svg(
        figures_dir / "m1_pressure_range_histogram.svg",
        per_case_rows,
        "pressure_range",
        "Pressure Range Distribution",
        "Per-case pressure range",
    )
    (artifacts_dir / "m1_dataset_eda_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(summaries)} subset summaries and {len(per_case_rows)} per-case rows.")


if __name__ == "__main__":
    main()
