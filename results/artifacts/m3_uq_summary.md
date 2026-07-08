# M3 UQ Calibration

MeshGNN trained on 48 cases, validated on 16, calibrated on 16, and tested on 16.

- Split-conformal WSS residual radius: `52.1315` at nominal `0.90` coverage.
- Test conformal coverage: `0.9447`.
- Mean test WSS residual norm: `18.1565`.
- MC-dropout error/uncertainty correlation: `0.3548`.
- MC-dropout AUSE: `6.0215`.

Interpretation: conformal prediction provides the calibrated coverage artifact for M3, while MC dropout provides a spatial uncertainty score for deferral and risk-coverage analysis.
