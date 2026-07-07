"""Helpers for summarizing deep-ensemble predictions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def ensemble_mean_std(predictions: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return mean and sample standard deviation across ensemble members."""

    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("predictions must include an ensemble dimension")
    return np.mean(values, axis=0), np.std(values, axis=0, ddof=1)
