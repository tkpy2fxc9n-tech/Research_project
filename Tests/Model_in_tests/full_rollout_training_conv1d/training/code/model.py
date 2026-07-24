# Reseau conv1d : au lieu d'aplatir tout le stencil en scalaires
# independants, on traite (lag x field) comme des canaux et les points du
# stencil comme un axe spatial, pour donner au reseau une notion explicite
# de voisinage. forward(X) -> Y : X de forme (n_nodes*b, n_features), Y de
# forme (n_nodes*b, n_fwd) -- meme convention que
# Model_in_tests/solver_gnn/.../model.py.
import torch
from torch import nn


def reshape_to_channels(X: torch.Tensor, n_lags: int, n_points: int, n_fields: int) -> torch.Tensor:
    # Factored out (rather than inlined in forward) so test_reshape.py can
    # verify this exact code path directly, with no risk of testing a copy
    # that drifts from what forward() actually runs.
    B = X.shape[0]
    x = X.view(B, n_lags, n_points, n_fields)
    return x.permute(0, 1, 3, 2).reshape(B, n_lags * n_fields, n_points)


class ReseauConv(nn.Module):
    def __init__(self, n_lags: int, n_points: int, n_fields: int, n_outputs: int,
                 conv_channels: tuple = (16, 32), kernel_size: int = 5, hidden_size: int = 32):
        super().__init__()
        self.n_lags, self.n_points, self.n_fields = n_lags, n_points, n_fields
        in_channels = n_lags * n_fields

        couches_conv = []
        taille_entree = in_channels
        for taille in conv_channels:
            couches_conv.append(nn.Conv1d(taille_entree, taille, kernel_size, padding=kernel_size // 2))
            couches_conv.append(nn.ReLU())
            taille_entree = taille
        self.conv = nn.Sequential(*couches_conv)

        self.head = nn.Sequential(
            nn.Linear(conv_channels[-1] * n_points, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, n_outputs))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = reshape_to_channels(X, self.n_lags, self.n_points, self.n_fields)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.head(x)
