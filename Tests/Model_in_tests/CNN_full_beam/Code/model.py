# Full-beam Conv1d network: the SAME conv trunk as full_rollout_training_
# conv1d's ReseauConv (kernel_size=5, channels 16->32, ReLU), but applied to
# the whole beam's Nx interior nodes at once instead of a 21-point window
# around a single node. The old Linear head (which was hard-wired to a
# fixed-size window and could only emit ONE node's outputs per forward call)
# is replaced by a pointwise Conv1d(kernel_size=1) head -- the same small
# read-out computation applied identically at every beam position, so one
# forward pass emits the whole field's outputs. No reshape helper needed:
# the input is assembled channels-first (see physics.build_field_history),
# so forward() is literally head(conv(X)).
#
# Input channels are the M_BACK past displacement snapshots (U only -- no
# velocity/curvature channels in this project), so in_channels = n_lags.
import torch
from torch import nn


class ReseauConvFullBeam(nn.Module):
    def __init__(self, n_lags: int, n_outputs: int,
                 conv_channels: tuple = (16, 32), kernel_size: int = 5):
        super().__init__()
        couches_conv = []
        taille_entree = n_lags
        for taille in conv_channels:
            couches_conv.append(nn.Conv1d(taille_entree, taille, kernel_size, padding=kernel_size // 2))
            couches_conv.append(nn.ReLU())
            taille_entree = taille
        self.conv = nn.Sequential(*couches_conv)

        self.head = nn.Conv1d(conv_channels[-1], n_outputs, kernel_size=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, n_lags, Nx) -> (B, n_outputs, Nx)
        return self.head(self.conv(X))
