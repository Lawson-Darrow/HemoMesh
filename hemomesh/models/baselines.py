"""Baseline field regressors."""

from __future__ import annotations

from collections.abc import Sequence


class MLPFieldRegressor:
    """Small per-node MLP baseline for WSS/pressure regression."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Sequence[int] = (128, 128, 128),
        dropout: float = 0.0,
    ) -> None:
        try:
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install the `torch` extra to use model definitions.") from exc

        layers: list[nn.Module] = []
        prev = in_features
        for width in hidden_features:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, out_features))
        self.module = nn.Sequential(*layers)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self

    def eval(self):
        self.module.eval()
        return self

    def parameters(self):
        return self.module.parameters()
