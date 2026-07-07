"""Selective-prediction and deferral metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def risk_coverage_curve(
    uncertainty: ArrayLike,
    error: ArrayLike,
    coverages: ArrayLike | None = None,
) -> list[dict[str, float]]:
    """Return retained-set risk after deferring highest-uncertainty items."""

    unc = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    err = np.asarray(error, dtype=np.float64).reshape(-1)
    if unc.shape != err.shape:
        raise ValueError("uncertainty and error must have the same flattened length")
    if len(err) == 0:
        raise ValueError("arrays must be non-empty")

    requested = (
        np.asarray(coverages, dtype=np.float64)
        if coverages is not None
        else np.asarray([1.0, 0.95, 0.9, 0.8, 0.7, 0.5], dtype=np.float64)
    )
    if np.any((requested <= 0.0) | (requested > 1.0)):
        raise ValueError("coverages must lie in (0, 1]")

    keep_order = np.argsort(unc)
    rows: list[dict[str, float]] = []
    for coverage in requested:
        keep_n = max(1, int(np.ceil(coverage * len(err))))
        kept = keep_order[:keep_n]
        rows.append(
            {
                "coverage": float(keep_n / len(err)),
                "deferred_fraction": float(1.0 - keep_n / len(err)),
                "risk": float(np.mean(err[kept])),
            }
        )
    return rows
