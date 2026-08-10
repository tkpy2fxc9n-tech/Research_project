# Conv1d network: (lag x field) treated as channels, stencil points as a
# spatial axis. Ported from Tests/Model_in_tests/full_rollout_training_conv1d/
# training/code/model.py. Hyperparameters (kernel_size=5, channels=(16,32))
# are kept fixed across runs by deliberate choice -- not derived from SS, not
# equalized to the MLP's parameter count (see configs/runs/*p4_cnn*.yaml).
from __future__ import annotations

import torch
from torch import nn


def reshape_to_channels(X: torch.Tensor, n_lags: int, n_points: int, n_fields: int) -> torch.Tensor:
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


def build_cnn(n_inputs: int, n_outputs: int, cfg) -> ConvNet:
    # n_inputs is unused directly (the conv reshape derives its own layout
    # from cfg.M_BACK/SS/features) but kept in the signature so registry.MODELS
    # entries are interchangeable from cli.py's point of view.
    return ConvNet(n_lags=cfg.M_BACK, n_points=2 * cfg.SS + 1, n_fields=len(cfg.features),
                    n_outputs=n_outputs, conv_channels=cfg.CNN_CHANNELS, kernel_size=cfg.CNN_KERNEL_SIZE)
