from __future__ import annotations

import numpy as np

from hemomesh.metrics import (
    approximation_error,
    cosine_similarity,
    empirical_coverage,
    nmae,
    pressure_r2,
    regression_ece,
    risk_coverage_curve,
    rmse,
)
from hemomesh.uq import fit_absolute_residual_conformal
from scripts.m3_uq_calibration import conformal_radius


def test_field_metrics_perfect_prediction() -> None:
    reference = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    prediction = reference.copy()

    assert approximation_error(prediction, reference) == 0.0
    assert rmse(prediction, reference) == 0.0
    assert nmae(prediction, reference) == 0.0
    assert cosine_similarity(prediction, reference) == 1.0


def test_pressure_r2() -> None:
    reference = np.asarray([1.0, 2.0, 3.0])
    prediction = reference.copy()

    assert pressure_r2(prediction, reference) == 1.0


def test_conformal_interval_and_coverage() -> None:
    prediction = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    reference = np.asarray([[0.1], [1.1], [2.1], [3.1]])

    interval = fit_absolute_residual_conformal(prediction, reference, nominal_coverage=0.9)
    lower, upper = interval.predict(prediction)

    assert empirical_coverage(lower, upper, reference) == 1.0


def test_conformal_radius_for_residual_norms() -> None:
    residuals = np.asarray([0.1, 0.2, 0.3, 0.4])

    assert conformal_radius(residuals, nominal_coverage=0.75) == 0.4


def test_regression_ece_and_risk_coverage_curve() -> None:
    uncertainty = np.asarray([0.1, 0.2, 0.8, 0.9])
    error = np.asarray([0.1, 0.2, 0.7, 0.8])

    assert regression_ece(uncertainty, error, num_bins=2) >= 0.0
    curve = risk_coverage_curve(uncertainty, error, coverages=[1.0, 0.5])

    assert curve[0]["coverage"] == 1.0
    assert curve[1]["coverage"] == 0.5
    assert curve[1]["risk"] < curve[0]["risk"]
