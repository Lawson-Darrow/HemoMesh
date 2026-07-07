"""Calibration metrics for interval and uncertainty estimates."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def empirical_coverage(lower: ArrayLike, upper: ArrayLike, reference: ArrayLike) -> float:
    """Return the fraction of targets contained in prediction intervals."""

    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if lo.shape != hi.shape or lo.shape != ref.shape:
        raise ValueError("lower, upper, and reference must share a shape")
    return float(np.mean((lo <= ref) & (ref <= hi)))


def regression_ece(
    uncertainty: ArrayLike,
    absolute_error: ArrayLike,
    num_bins: int = 10,
) -> float:
    """Estimate calibration error by comparing binned uncertainty and error."""

    unc = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    err = np.asarray(absolute_error, dtype=np.float64).reshape(-1)
    if unc.shape != err.shape:
        raise ValueError("uncertainty and absolute_error must have the same flattened length")
    if num_bins < 1:
        raise ValueError("num_bins must be positive")

    edges = np.quantile(unc, np.linspace(0.0, 1.0, num_bins + 1))
    edges = np.unique(edges)
    if len(edges) <= 1:
        return float(abs(np.mean(unc) - np.mean(err)))

    ece = 0.0
    total = len(unc)
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        mask = (unc >= start) & (unc <= end if end == edges[-1] else unc < end)
        if not np.any(mask):
            continue
        weight = np.sum(mask) / total
        ece += weight * abs(float(np.mean(unc[mask]) - np.mean(err[mask])))
    return float(ece)


def ause(uncertainty: ArrayLike, absolute_error: ArrayLike) -> float:
    """Return area under sparsification error for uncertainty-based deferral."""

    unc = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    err = np.asarray(absolute_error, dtype=np.float64).reshape(-1)
    if unc.shape != err.shape:
        raise ValueError("uncertainty and absolute_error must have the same flattened length")
    if len(err) == 0:
        raise ValueError("arrays must be non-empty")

    order_unc = np.argsort(unc)[::-1]
    order_oracle = np.argsort(err)[::-1]
    coverages = np.linspace(1.0, 0.0, len(err), endpoint=False)
    unc_curve = _retained_error_curve(err, order_unc)
    oracle_curve = _retained_error_curve(err, order_oracle)
    return float(np.trapz(unc_curve - oracle_curve, coverages))


def _retained_error_curve(error: np.ndarray, removal_order: np.ndarray) -> np.ndarray:
    retained = np.ones(len(error), dtype=bool)
    values = []
    for index in removal_order:
        values.append(float(np.mean(error[retained])))
        retained[index] = False
    return np.asarray(values, dtype=np.float64)
