# Every tunable parameter for this project's one method
# (U_only_teacher_forcing_multiwave) lives here, and nothing else --
# commun.py only consumes Config, it never defines defaults for it, and
# main.py/test*.py import INPUT_FIELDS/METHOD_NAME/N_SCENARIOS/PATIENCE from
# here instead of re-declaring their own copies. This is the one file to
# edit to change an experiment setting.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# U only: the network only ever sees past displacement at each node, no
# velocity (Ut) or curvature (Uxx) features.
INPUT_FIELDS = ["U"]
METHOD_NAME = "U_only_teacher_forcing_multiwave"

N_SCENARIOS = 400  # number of randomly-sampled (left, right) boundary scenarios
PATIENCE = 20       # epochs without a val improvement before stopping early


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
    # M_BACK=2/N_FWD=2 and a wider network (HIDDEN_SIZES below) -- plain
    # teacher-forcing has no rollout to help it, so more temporal context
    # and capacity are given to make up for it.
    ndt: int = 2
    M_BACK: int = 2
    N_FWD: int = 2

    N_GRID: int = 10
    AMP_MIN: float = 0.005
    AMP_MAX: float = 0.15
    OMEGA_MIN: float = 1.0
    OMEGA_MAX: float = 10

    # None -> grid center (see __post_init__), not the corner (A_MIN, OMEGA_MIN).
    ROLLOUT_A_IDX: int | None = None
    ROLLOUT_OMEGA_IDX: int | None = None

    HIDDEN_SIZES: tuple = (512, 256, 64)

    LEARNING_RATE: float = 1e-3
    N_EPOCHS: int = 18
    BATCH_SIZE: int = 512

    # 0: plain teacher-forcing is meant to be exactly "real M_BACK window in
    # -> predict -> average the horizons' error -> update", with no input
    # noise and no pushforward augmentation (removed from this project's
    # vendored commun.py entirely -- see commun.train_model).
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
        self.AMPLITUDES = np.linspace(self.AMP_MIN, self.AMP_MAX, self.N_GRID).round(3).tolist()
        self.PULSATIONS = np.linspace(self.OMEGA_MIN, self.OMEGA_MAX, self.N_GRID).round(1).tolist()
        if self.ROLLOUT_A_IDX is None:
            self.ROLLOUT_A_IDX = self.N_GRID // 2
        if self.ROLLOUT_OMEGA_IDX is None:
            self.ROLLOUT_OMEGA_IDX = self.N_GRID // 2
