"""Field-regression metrics used throughout HemoMesh."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_float_array(values: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


def approximation_error(prediction: ArrayLike, reference: ArrayLike, eps: float = 1e-12) -> float:
    """Return sqrt(sum(||prediction-reference||^2) / sum(||reference||^2))."""

    pred = _as_float_array(prediction)
    ref = _as_float_array(reference)
    numerator = np.sum(np.square(pred - ref))
    denominator = np.sum(np.square(ref))
    return float(np.sqrt(numerator / max(denominator, eps)))


def rmse(prediction: ArrayLike, reference: ArrayLike) -> float:
    """Return root mean squared error."""

    pred = _as_float_array(prediction)
    ref = _as_float_array(reference)
    return float(np.sqrt(np.mean(np.square(pred - ref))))


def nmae(prediction: ArrayLike, reference: ArrayLike, eps: float = 1e-12) -> float:
    """Return mean absolute error normalized by mean absolute reference magnitude."""

    pred = _as_float_array(prediction)
    ref = _as_float_array(reference)
    return float(np.mean(np.abs(pred - ref)) / max(np.mean(np.abs(ref)), eps))


def cosine_similarity(prediction: ArrayLike, reference: ArrayLike, eps: float = 1e-12) -> float:
    """Return mean per-node cosine similarity for vector-valued fields."""

    pred = _as_float_array(prediction)
    ref = _as_float_array(reference)
    if pred.ndim != 2 or ref.ndim != 2:
        raise ValueError("cosine_similarity expects two arrays with shape (N, D)")
    numerator = np.sum(pred * ref, axis=1)
    denominator = np.linalg.norm(pred, axis=1) * np.linalg.norm(ref, axis=1)
    return float(np.mean(numerator / np.maximum(denominator, eps)))


def pressure_r2(prediction: ArrayLike, reference: ArrayLike, eps: float = 1e-12) -> float:
    """Return coefficient of determination for scalar pressure predictions."""

    pred = _as_float_array(prediction).reshape(-1)
    ref = _as_float_array(reference).reshape(-1)
    residual = np.sum(np.square(ref - pred))
    total = np.sum(np.square(ref - np.mean(ref)))
    return float(1.0 - residual / max(total, eps))
