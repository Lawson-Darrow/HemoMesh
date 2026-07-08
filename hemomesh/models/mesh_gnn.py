"""Message-passing mesh GNN backbones."""

from __future__ import annotations

from collections.abc import Sequence

try:
    import torch
    import torch.nn as nn
    from torch_geometric.nn import SAGEConv
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    SAGEConv = None


if nn is not None and SAGEConv is not None:

    class MeshGNN(nn.Module):
        """GraphSAGE-style mesh regressor over face-derived edges."""

        def __init__(
            self,
            in_features: int,
            out_features: int = 4,
            hidden_features: Sequence[int] = (128, 128, 128),
            dropout: float = 0.0,
            output_smoothing_steps: int = 0,
            output_smoothing_alpha: float = 0.25,
        ) -> None:
            super().__init__()
            if not hidden_features:
                raise ValueError("hidden_features must contain at least one width")
            if output_smoothing_steps < 0:
                raise ValueError("output_smoothing_steps must be non-negative")
            if not 0.0 <= output_smoothing_alpha <= 1.0:
                raise ValueError("output_smoothing_alpha must be between 0 and 1")

            self.dropout = nn.Dropout(dropout)
            self.output_smoothing_steps = output_smoothing_steps
            self.output_smoothing_alpha = output_smoothing_alpha
            widths = [in_features, *hidden_features]
            self.convs = nn.ModuleList(
                SAGEConv(widths[index], widths[index + 1]) for index in range(len(widths) - 1)
            )
            self.norms = nn.ModuleList(nn.LayerNorm(width) for width in hidden_features)
            self.residuals = nn.ModuleList(
                nn.Identity()
                if widths[index] == widths[index + 1]
                else nn.Linear(widths[index], widths[index + 1], bias=False)
                for index in range(len(widths) - 1)
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_features[-1], hidden_features[-1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_features[-1], out_features),
            )

        def _smooth_output(self, values: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            source, target = edge_index
            degree = values.new_zeros(values.shape[0])
            degree.index_add_(0, target, values.new_ones(target.shape[0]))
            degree = degree.clamp_min(1.0).unsqueeze(1)
            smoothed = values
            for _ in range(self.output_smoothing_steps):
                neighbor_sum = values.new_zeros(values.shape)
                neighbor_sum.index_add_(0, target, smoothed[source])
                neighbor_mean = neighbor_sum / degree
                smoothed = (
                    (1.0 - self.output_smoothing_alpha) * smoothed
                    + self.output_smoothing_alpha * neighbor_mean
                )
            return smoothed

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            """Predict per-node targets using node features and mesh connectivity."""

            for conv, norm, residual in zip(self.convs, self.norms, self.residuals, strict=True):
                message = conv(x, edge_index)
                x = norm(torch.relu(message) + residual(x))
                x = self.dropout(x)
            output = self.head(x)
            return self._smooth_output(output, edge_index)

else:

    class MeshGNN:
        """Placeholder that explains how to enable the optional training stack."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Install the `torch` extra to use MeshGNN.")
