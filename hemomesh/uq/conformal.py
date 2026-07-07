"""Split-conformal intervals for per-node regression targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ConformalInterval:
    """Symmetric conformal interval calibrated from absolute residuals."""

    nominal_coverage: float
    radius: NDArray[np.float64]

    def predict(self, prediction: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        pred = np.asarray(prediction, dtype=np.float64)
        return pred - self.radius, pred + self.radius


def fit_absolute_residual_conformal(
    prediction: ArrayLike,
    reference: ArrayLike,
    nominal_coverage: float = 0.9,
    per_dimension: bool = True,
) -> ConformalInterval:
    """Fit a split-conformal radius from calibration predictions and targets."""

    if not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage must lie in (0, 1)")

    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError("prediction and reference must share a shape")

    residual = np.abs(pred - ref)
    if not per_dimension:
        residual = np.linalg.norm(residual.reshape(-1, residual.shape[-1]), axis=1)

    n = residual.shape[0]
    quantile_level = np.ceil((n + 1) * nominal_coverage) / n
    quantile_level = min(float(quantile_level), 1.0)
    radius = np.quantile(residual, quantile_level, axis=0, method="higher")
    return ConformalInterval(nominal_coverage=nominal_coverage, radius=np.asarray(radius))
