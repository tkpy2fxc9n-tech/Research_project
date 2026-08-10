# Boundary-condition "waveforms" for this project's 3 requested conditions,
# composed independently on each end of the beam (see scenarios.py):
#   - a Gaussian push (as a displacement OR as a flux/velocity kick --
#     both bc_types are kept for this one family, to maximize the number of
#     distinct combinations, as requested)
#   - a free end (Neumann, family "rest" -- zero prescribed flux, the end is
#     not driven at all and just reflects the wave)
#   - a fixed-at-0 end (Dirichlet, family "rest" -- zero prescribed
#     displacement, the end never moves)
#
# Trimmed from full_rollout_training_conv1d/training/code/waves.py: that
# project's sinusoid/fourier/chirp/shock/filtered_random families and its
# flip() (correlated-phase) helper are dropped entirely, not just unused --
# they don't serve any of the 3 requested conditions.
#
# A boundary condition is (bc_type, waveform_family, params):
#   bc_type in {"dirichlet", "neumann"} -- Dirichlet prescribes a displacement,
#     Neumann prescribes a slope/flux (0 = a true free end).
#   waveform_family is a key into BC_WAVEFORMS below.
#   params is a dict of that family's sampled numeric parameters.
from __future__ import annotations

import numpy as np

BCSpec = tuple  # (bc_type: str, waveform_family: str, params: dict)

BC_TYPES = ("dirichlet", "neumann")


def sample_gaussian_params(rng, cfg) -> dict:
    return {
        "A": float(rng.uniform(cfg.AMP_MIN, cfg.AMP_MAX)),
        "sigma": float(rng.uniform(cfg.OMEGA_MIN, cfg.OMEGA_MAX)),
    }


def gaussian_value(p: dict, t: float) -> float:
    # A single bump, centered so it has mostly risen to zero by t=0
    # (t0 = 4*sigma puts the peak a few sigma in).
    sigma = p["sigma"]
    t0 = 4.0 * sigma
    return p["A"] * np.exp(-((t - t0) / sigma) ** 2)


def sample_rest_params(rng, cfg) -> dict:
    return {}


def rest_value(p: dict, t: float) -> float:
    # Nothing imposed -- with bc_type=dirichlet this is "fixed at 0"; with
    # bc_type=neumann this is "free to move" (zero prescribed flux).
    return 0.0


BC_WAVEFORMS = {
    "gaussian": (sample_gaussian_params, gaussian_value),
    "rest": (sample_rest_params, rest_value),
}


def bc_value(bc: BCSpec, t: float) -> float:
    _, family, params = bc
    _, value_fn = BC_WAVEFORMS[family]
    return value_fn(params, t)


def bc_describe(bc: BCSpec) -> str:
    bc_type, family, params = bc
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    return f"{bc_type}/{family}({param_str})"


def apply_boundary(u: np.ndarray, side: str, bc_type: str, value: float, cfg) -> np.ndarray:
    # Dirichlet: fill the whole ghost band + boundary node with `value`.
    #
    # Neumann: mirror the ghost band across the boundary with a linear
    # correction so the central-difference slope estimate at the boundary
    # equals `value` (0 -> true free end, reflects without flipping sign).
    # Deliberately does NOT overwrite the boundary node itself: the interior
    # leapfrog stencil already computes a value there using exactly one ghost
    # neighbor, so as long as that neighbor is correctly mirrored, the
    # boundary node comes out right on its own -- overwriting it too (like
    # Dirichlet does) would double-impose a condition that isn't a prescribed
    # value here, only a prescribed slope.
    i_left, i_right, SS, dx = cfg.i_left, cfg.i_right, cfg.SS, cfg.dx
    if side == "left":
        if bc_type == "dirichlet":
            u[:i_left + 1] = value
        else:
            for k in range(1, SS + 1):
                u[i_left - k] = u[i_left + k] - 2 * k * dx * value
    else:
        if bc_type == "dirichlet":
            u[i_right:] = value
        else:
            for k in range(1, SS):  # SS-1 pure ghost points beyond i_right
                u[i_right + k] = u[i_right - k] + 2 * k * dx * value
    return u


def apply_boundary_conditions(u: np.ndarray, t: float, bc_left: BCSpec, bc_right: BCSpec, cfg) -> np.ndarray:
    apply_boundary(u, "left", bc_left[0], bc_value(bc_left, t), cfg)
    apply_boundary(u, "right", bc_right[0], bc_value(bc_right, t), cfg)
    return u
