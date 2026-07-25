# Composes the 3 requested boundary conditions -- Gaussian push, free end,
# fixed-at-0 end -- independently on each end of the beam, to cover the
# maximum number of distinct (left, right) combinations.
#
# The Gaussian push counts as 2 options (a displacement push, bc_type=
# dirichlet, and a flux/velocity push, bc_type=neumann), so there are 4
# possible settings per end:
#   ("dirichlet", "gaussian")  -- displacement follows a Gaussian bump
#   ("neumann",   "gaussian")  -- flux/velocity follows a Gaussian bump
#   ("neumann",   "rest")      -- free end (no push, no constraint)
#   ("dirichlet", "rest")      -- fixed end (clamped at 0 forever)
# Composed independently on the left and right ends: 4x4 = 16 possible
# scenario types.
#
# Unlike full_rollout_training_conv1d/training/code/scenarios.py, this
# drops the "correlated two-end phase" mechanism (only meaningful for the
# sinusoid/fourier/chirp families, which aren't part of the 3 requested
# conditions) and the "free_evolution" random-initial-condition case (not
# one of the 3 requested conditions either) -- every scenario here is a
# plain two-end forced/free/fixed BC pair, no random initial displacement.
import numpy as np

from waves import BC_WAVEFORMS

END_OPTIONS = [
    ("dirichlet", "gaussian"),
    ("neumann", "gaussian"),
    ("neumann", "rest"),
    ("dirichlet", "rest"),
]

END_TAGS = {
    ("dirichlet", "gaussian"): "gaussian_disp",
    ("neumann", "gaussian"): "gaussian_flux",
    ("neumann", "rest"): "free",
    ("dirichlet", "rest"): "fixed",
}


def end_tag(bc) -> str:
    bc_type, family, _ = bc
    return END_TAGS[(bc_type, family)]


def sample_one_end(rng, cfg):
    bc_type, family = END_OPTIONS[rng.integers(len(END_OPTIONS))]
    sampler, _ = BC_WAVEFORMS[family]
    return (bc_type, family, sampler(rng, cfg))


def sample_scenario(rng, cfg):
    return sample_one_end(rng, cfg), sample_one_end(rng, cfg)


def sample_scenarios(cfg, n_samples, rng=None):
    rng = rng if rng is not None else np.random.default_rng(cfg.SEED)
    return [sample_scenario(rng, cfg) for _ in range(n_samples)]


def all_combo_scenarios(cfg, rng):
    # One deterministic scenario per (left, right) combination (16 total),
    # for main.py's showcase gifs -- covers every distinct configuration at
    # least once, rather than relying on random sampling to eventually hit
    # all 16.
    scenarios = []
    for bc_left_opt in END_OPTIONS:
        for bc_right_opt in END_OPTIONS:
            bc_left = (bc_left_opt[0], bc_left_opt[1], BC_WAVEFORMS[bc_left_opt[1]][0](rng, cfg))
            bc_right = (bc_right_opt[0], bc_right_opt[1], BC_WAVEFORMS[bc_right_opt[1]][0](rng, cfg))
            scenarios.append((bc_left, bc_right))
    return scenarios
