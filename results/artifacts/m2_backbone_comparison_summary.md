# M2 Backbone Comparison

Matched 120-epoch run on 64 train / 24 validation / 24 test `single` cases with WSS-focused target weights `[1.0, 1.0, 1.0, 0.2]`.

| Model | Best epoch | Test WSS AE mean | 95% CI | Test WSS RMSE | WSS cosine | Pressure RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Dense MLP | 120 | 0.8119 | [0.7935, 0.8293] | 19.3029 | 0.7977 | 818.3430 |
| MeshGNN | 82 | 0.7611 | [0.7343, 0.7899] | 17.9676 | 0.8154 | 886.5049 |

Current interpretation: the smoothed MeshGNN beats the dense MLP baseline on WSS approximation error with non-overlapping bootstrap confidence intervals. This satisfies the M2 course requirement for a geometric backbone outperforming a dense baseline on the primary WSS metric.
