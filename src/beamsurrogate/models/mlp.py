# Plain feed-forward MLP, GELU activations. Ported from
# Beam_surrogate_model/training/code/commun.py's `Reseau` (renamed to
# English: this repo's new code is English throughout).
from __future__ import annotations

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        layers = []
        in_size = n_inputs
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.GELU())
            in_size = h
        layers.append(nn.Linear(in_size, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_mlp(n_inputs: int, n_outputs: int, cfg) -> MLP:
    return MLP(n_inputs, n_outputs, cfg.HIDDEN_SIZES)
