# Single source of truth for every run parameter of this project. Defaults
# below are the values THIS project actually trains with (plain one-step
# teacher forcing, no input noise) -- not a generic 4-project default
# silently overridden elsewhere (main.py used to carry a separate
# CONFIG_OVERRIDES dict on top of a shared commun.Config() with different
# defaults; that indirection is gone, this is the only Config now). Only
# N_EPOCHS/BATCH_SIZE still vary at the call site (main.py's
# --epochs/--batch-size CLI flags) -- that's a real per-run knob, not
# leftover cross-project indirection.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Config:
    E: float = 1
    rho: float = 2
    L: float = 1

    Nt: int = 500
    Nx: int = 100
    SS: int = 10
    t_end: float = 5

    # M_BACK past levels -> N_FWD future horizons, spaced by ndt steps.
    ndt: int = 2
    M_BACK: int = 2
    N_FWD: int = 2

    AMP_MIN: float = 0.005
    AMP_MAX: float = 0.1
    OMEGA_MIN: float = 3
    OMEGA_MAX: float = 10

    LEARNING_RATE: float = 1e-3
    N_EPOCHS: int = 50
    BATCH_SIZE: int = 512

    NOISE_STD: float = 0.0
    SMOOTH_ALPHA: float = 0.20   # must stay < 0.25 (smoothing stability)

    SEED: int = 0
    SPLIT_SEED: int = 42

    def __post_init__(self):
        self.Ntot = self.Nx + 2 * self.SS
        self.i_left = self.SS
        self.i_right = self.Ntot - self.SS
        self.nodes = np.arange(self.i_left, self.i_right)
        self.dt = self.t_end / self.Nt
        self.dx = self.L / (self.Nx - 1)
        self.CFL = self.dt / self.dx * np.sqrt(self.E / self.rho)
        if self.CFL > 1:
            print(f"WARNING: CFL={self.CFL:.3f} > 1 -- the explicit scheme is numerically "
                  f"unstable with these Nt/Nx/t_end/L (the simulation will diverge). Increase Nt "
                  f"and/or reduce Nx to bring CFL back to <= 1.")


def set_seeds(cfg: Config) -> None:
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
