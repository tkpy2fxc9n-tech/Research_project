# Conv1d network: instead of flattening the whole stencil into independent
# scalars, (lag x field) is treated as channels and the stencil points as a
# spatial axis, giving the network an explicit notion of neighborhood.
# forward(X) -> Y : X has shape (n_nodes*b, n_features), Y has shape
# (n_nodes*b, n_fwd) -- same convention as Tests/Archives/solver_gnn/.../model.py.
import torch
from torch import nn


def reshape_to_channels(X: torch.Tensor, n_lags: int, n_points: int, n_fields: int) -> torch.Tensor:
    # Factored out (rather than inlined in forward) so test_reshape.py can
    # verify this exact code path directly, with no risk of testing a copy
    # that drifts from what forward() actually runs.
    B = X.shape[0]
    x = X.view(B, n_lags, n_points, n_fields)
    return x.permute(0, 1, 3, 2).reshape(B, n_lags * n_fields, n_points)


class ConvNet(nn.Module):
    def __init__(self, n_lags: int, n_points: int, n_fields: int, n_outputs: int,
                 conv_channels: tuple = (16, 32), kernel_size: int = 5, hidden_size: int = 32):
        super().__init__()
        self.n_lags, self.n_points, self.n_fields = n_lags, n_points, n_fields
        in_channels = n_lags * n_fields

        conv_layers = []
        in_size = in_channels
        for out_size in conv_channels:
            conv_layers.append(nn.Conv1d(in_size, out_size, kernel_size, padding=kernel_size // 2))
            conv_layers.append(nn.ReLU())
            in_size = out_size
        self.conv = nn.Sequential(*conv_layers)

        self.head = nn.Sequential(
            nn.Linear(conv_channels[-1] * n_points, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, n_outputs))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = reshape_to_channels(X, self.n_lags, self.n_points, self.n_fields)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.head(x)
