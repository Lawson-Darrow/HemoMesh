"""Bootstrap confidence intervals for case-level metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def bootstrap_ci(
    values: ArrayLike,
    confidence: float = 0.95,
    num_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Return mean and percentile bootstrap confidence interval."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) == 0:
        raise ValueError("values must be non-empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(num_resamples, len(array)))
    means = array[indices].mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "mean": float(array.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": confidence,
        "num_cases": int(len(array)),
        "num_resamples": int(num_resamples),
    }
