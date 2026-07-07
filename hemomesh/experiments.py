"""Small experiment ledger for reproducible baseline and training runs."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_EXPERIMENTS_PATH = Path("results/experiments.csv")


def config_hash(config: dict[str, Any]) -> str:
    """Return a stable short hash for a JSON-serializable configuration."""

    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def append_experiment(
    config: dict[str, Any],
    metrics: dict[str, Any],
    path: str | Path = DEFAULT_EXPERIMENTS_PATH,
) -> dict[str, Any]:
    """Append one run to the experiment ledger and return the written row."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(config),
        "run_type": config.get("run_type", "unknown"),
        "model": config.get("model", "unknown"),
        "subset": config.get("subset", "unknown"),
        "split": config.get("split", "unknown"),
        "seed": config.get("seed", ""),
        "config_json": json.dumps(config, sort_keys=True),
        "metrics_json": json.dumps(metrics, sort_keys=True),
    }
    fieldnames = list(row.keys())
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return row
