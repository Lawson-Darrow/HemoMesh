"""Helpers for summarizing MC-dropout predictions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def summarize_mc_predictions(samples: ArrayLike) -> dict[str, NDArray[np.float64]]:
    """Return mean, standard deviation, and variance from stochastic predictions."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("samples must include a stochastic-sample dimension")
    return {
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0, ddof=1),
        "var": np.var(values, axis=0, ddof=1),
    }
