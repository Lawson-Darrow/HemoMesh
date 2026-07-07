"""Helpers for controlled in-data distribution shifts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class SeveritySplit:
    """Mild/severe split derived from a scalar per-case severity proxy."""

    mild_indices: list[int]
    severe_indices: list[int]
    mild_threshold: float
    severe_threshold: float
    proxy_name: str


def topology_shift_split(train_subset: str) -> tuple[str, str]:
    """Return train/test subsets for the single-vs-bifurcating topology shift."""

    if train_subset == "single":
        return "single", "bifurcating"
    if train_subset == "bifurcating":
        return "bifurcating", "single"
    raise ValueError("train_subset must be 'single' or 'bifurcating'")


def severity_quantile_split(
    severity_proxy: ArrayLike,
    proxy_name: str,
    mild_quantile: float = 0.4,
    severe_quantile: float = 0.6,
) -> SeveritySplit:
    """Split cases into mild and severe groups using proxy quantiles."""

    values = np.asarray(severity_proxy, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        raise ValueError("severity_proxy must be non-empty")
    if not 0.0 <= mild_quantile < severe_quantile <= 1.0:
        raise ValueError("expected 0 <= mild_quantile < severe_quantile <= 1")

    mild_threshold = float(np.quantile(values, mild_quantile))
    severe_threshold = float(np.quantile(values, severe_quantile))
    mild_indices = np.flatnonzero(values <= mild_threshold).astype(int).tolist()
    severe_indices = np.flatnonzero(values >= severe_threshold).astype(int).tolist()
    return SeveritySplit(
        mild_indices=mild_indices,
        severe_indices=severe_indices,
        mild_threshold=mild_threshold,
        severe_threshold=severe_threshold,
        proxy_name=proxy_name,
    )


def ordered_case_ids(case_ids: Sequence[str], indices: Sequence[int]) -> list[str]:
    """Return case IDs for a split while preserving deterministic index order."""

    return [case_ids[index] for index in indices]
