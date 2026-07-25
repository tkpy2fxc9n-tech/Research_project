# Every tunable parameter for this project's one method
# (full_rollout_U_Ut_Uxx) lives here, and nothing else -- commun.py only
# consumes Config, it never defines defaults for it, and main.py imports
# INPUT_FIELDS/METHOD_NAME from here instead of re-declaring its own copies.
# This is the one file to edit to change an experiment setting.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INPUT_FIELDS = ["U", "Ut", "Uxx"]
METHOD_NAME = "full_rollout_U_Ut_Uxx"

# Number of randomly-sampled (left, right) BC scenarios to simulate --
# replaces the old dense N_GRID x N_GRID (A, omega) grid now that each end
# is independently drawn from scenarios.ALLOWED_FAMILIES instead of always
# being a single Gaussian pulse. 100 keeps the dataset the same size as the
# old 10x10 grid it replaces.
N_SCENARIOS = 100


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

    # Range each end's "gaussian" family is sampled from (see scenarios.py).
    AMP_MIN: float = 0.005
    AMP_MAX: float = 0.1
    OMEGA_MIN: float = 3
    OMEGA_MAX: float = 10

    HIDDEN_SIZES: tuple = (256, 128, 32)

    LEARNING_RATE: float = 1e-3
    N_EPOCHS: int = 300

    # Full-rollout TBPTT training (train.py): simulations rolled out in
    # parallel per weight update, and hops between corrections. Calibrated
    # at ~17s/epoch on the full grid so that 300 epochs x 23 groups x 17
    # corrections/group (82 hops / TBPTT_HOPS=5) = ~117,300 weight
    # corrections -- deliberately at least as many as the ~114,620 updates
    # of the pushforward method this project is compared against, to avoid
    # concluding too early from insufficient training (see exchange with
    # Claude on 2026-07-20 -- readjust N_EPOCHS if a future config change
    # alters this per-epoch time).
    GROUP_SIZE: int = 4
    TBPTT_HOPS: int = 5

    # Clips the gradient norm at each weight correction (train.py) -- without
    # it, a single TBPTT segment with an unlucky combination of BC scenario
    # and network state can produce a huge loss (e.g. early in training, on a
    # "gaussian" pulse the untrained network mis-predicts badly for several
    # uncorrected hops in a row) whose gradient then permanently destroys the
    # weights in one Adam step. Standard safeguard for this kind of
    # autoregressive-rollout training.
    GRAD_CLIP_NORM: float = 1.0

    # Clips normalized network inputs to +/-INPUT_CLIP standard deviations
    # (train.py's TBPTT loop and commun._autoregressive_rollout_general).
    # u_xx features are a sharp second difference right at a driven
    # boundary: most rows are quiet (~0), so their pooled std is tiny, and
    # the rare rows where a pulse is actually at the edge can be tens of
    # thousands of std away -- feeding that straight into the network
    # produces an immediate runaway autoregressive blow-up (verified: this
    # is what caused loss to explode to ~1e17/NaN within a handful of hops
    # after generalizing boundary conditions to allow "gaussian" on either
    # end). Clipping the input is standard practice for this failure mode
    # in autoregressive/rollout training.
    INPUT_CLIP: float = 10.0

    NOISE_STD: float = 0.10
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
