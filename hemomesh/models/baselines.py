"""Baseline field regressors."""

from __future__ import annotations

from collections.abc import Sequence

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None


if nn is not None:

    class MLPFieldRegressor(nn.Module):
        """Per-node dense baseline for WSS and pressure regression."""

        def __init__(
            self,
            in_features: int,
            out_features: int = 4,
            hidden_features: Sequence[int] = (128, 128, 128),
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev = in_features
            for width in hidden_features:
                layers.append(nn.Linear(prev, width))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev = width
            layers.append(nn.Linear(prev, out_features))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Predict per-node targets from node features only."""

            return self.net(x)

else:

    class MLPFieldRegressor:
        """Placeholder that explains how to enable the optional training stack."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Install the `torch` extra to use model definitions.")


class PointwiseMLP(MLPFieldRegressor):
    """Small per-node MLP baseline for WSS/pressure regression."""
