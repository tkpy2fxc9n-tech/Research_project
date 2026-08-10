# Boundary-condition "waveform" library: the (bc_type, waveform_family,
# params) spec, the signal families sampled by data/generate.py, and how a
# BC value/spec gets applied to the simulation grid.
#
# A boundary condition is (bc_type, waveform_family, params):
#   bc_type in {"dirichlet", "neumann"} -- Dirichlet prescribes a displacement,
#     Neumann prescribes a slope/flux (0 = a true free end).
#   waveform_family is a key into BC_WAVEFORMS below.
#   params is a dict of that family's sampled numeric parameters.
# Ported from Tests/Model_in_tests/full_rollout_training_conv1d/training/code/waves.py
# (the canonical version: gaussian pulse width is a direct SIGMA_MIN/MAX draw,
# not the omega->sigma interpolation hack in Beam_surrogate_model/commun.py).
from __future__ import annotations

import numpy as np

BCSpec = tuple  # (bc_type: str, waveform_family: str, params: dict)

BC_TYPES = ("dirichlet", "neumann")


# ---------------------------------------------------------------------------
# gaussian / sinusoid / rest
# ---------------------------------------------------------------------------
def sample_gaussian_params(rng, cfg) -> dict:
    return {
        "A": float(rng.uniform(cfg.AMP_MIN, cfg.AMP_MAX)),
        "sigma": float(rng.uniform(cfg.SIGMA_MIN, cfg.SIGMA_MAX)),
    }


def gaussian_value(p: dict, t: float) -> float:
    # A single bump, centered so it has mostly risen to zero by t=0
    # (t0 = 4*sigma puts the peak a few sigma in).
    sigma = p["sigma"]
    t0 = 4.0 * sigma
    return p["A"] * np.exp(-((t - t0) / sigma) ** 2)


def sample_sinusoid_params(rng, cfg) -> dict:
    return {
        "A": float(rng.uniform(cfg.AMP_MIN, cfg.AMP_MAX)),
        "omega": float(rng.uniform(cfg.OMEGA_MIN, cfg.OMEGA_MAX)),
        "phase": float(rng.uniform(0.0, 2 * np.pi)),
    }


def sinusoid_value(p: dict, t: float) -> float:
    return p["A"] * np.sin(p["omega"] * t + p["phase"])


def sample_rest_params(rng, cfg) -> dict:
    return {}


def rest_value(p: dict, t: float) -> float:
    return 0.0


# ---------------------------------------------------------------------------
# fourier (random Fourier sums, 4-8 tones)
# ---------------------------------------------------------------------------
def sample_fourier_params(rng, cfg) -> dict:
    n_tones = int(rng.integers(4, 9))
    return {
        "A": rng.uniform(cfg.AMP_MIN, cfg.AMP_MAX, size=n_tones).tolist(),
        "omega": rng.uniform(cfg.OMEGA_MIN, cfg.OMEGA_MAX, size=n_tones).tolist(),
        "phase": rng.uniform(0.0, 2 * np.pi, size=n_tones).tolist(),
    }


def fourier_value(p: dict, t: float) -> float:
    tones = zip(p["A"], p["omega"], p["phase"])
    return sum(A * np.sin(om * t + ph) for A, om, ph in tones) / len(p["A"])


# ---------------------------------------------------------------------------
# chirp (linear frequency sweep over the whole run)
# ---------------------------------------------------------------------------
def sample_chirp_params(rng, cfg) -> dict:
    return {
        "A": float(rng.uniform(cfg.AMP_MIN, cfg.AMP_MAX)),
        "omega0": float(rng.uniform(cfg.OMEGA_MIN, cfg.OMEGA_MAX)),
        "omega1": float(rng.uniform(cfg.OMEGA_MIN, cfg.OMEGA_MAX)),
        "phase0": float(rng.uniform(0.0, 2 * np.pi)),
        "t_end": float(cfg.t_end),
    }


def chirp_value(p: dict, t: float) -> float:
    omega0, omega1, t_end = p["omega0"], p["omega1"], p["t_end"]
    phase = p["phase0"] + omega0 * t + 0.5 * (omega1 - omega0) / t_end * t ** 2
    return p["A"] * np.sin(phase)


# ---------------------------------------------------------------------------
# shock (tanh-smoothed step)
# ---------------------------------------------------------------------------
def sample_shock_params(rng, cfg) -> dict:
    return {
        "A": float(rng.uniform(cfg.AMP_MIN, cfg.AMP_MAX)),
        "t_onset": float(rng.uniform(0.1 * cfg.t_end, 0.5 * cfg.t_end)),
        "tau": float(rng.uniform(0.01 * cfg.t_end, 0.05 * cfg.t_end)),
    }


def shock_value(p: dict, t: float) -> float:
    return p["A"] * 0.5 * (1.0 + np.tanh((t - p["t_onset"]) / p["tau"]))


# ---------------------------------------------------------------------------
# filtered_random (smoothed noise at control points, linearly interpolated)
# ---------------------------------------------------------------------------
def sample_filtered_random_params(rng, cfg) -> dict:
    n_ctrl = 12
    t_ctrl = np.linspace(0.0, cfg.t_end, n_ctrl)
    raw = rng.normal(size=n_ctrl)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    smoothed = np.convolve(raw, kernel, mode="same")
    smoothed = smoothed - smoothed[0]  # starts at 0, consistent with a forced-from-rest run
    A = float(rng.uniform(cfg.AMP_MIN, cfg.AMP_MAX))
    peak = np.abs(smoothed).max()
    scale = A / peak if peak > 1e-12 else 0.0
    return {"t_ctrl": t_ctrl.tolist(), "values": (smoothed * scale).tolist()}


def filtered_random_value(p: dict, t: float) -> float:
    return float(np.interp(t, p["t_ctrl"], p["values"]))


# ---------------------------------------------------------------------------
# table (pre-computed value series, e.g. loaded from an HDF5 dataset).
# No sampler: the values come from real data, not this module's own RNG.
# ---------------------------------------------------------------------------
def _no_sampler(rng, cfg) -> dict:
    raise NotImplementedError("'table' is a load-only family (pre-computed BC value series) -- "
                               "its params come from data/split.py's HDF5 loader, not random sampling.")


BC_WAVEFORMS = {
    "gaussian": (sample_gaussian_params, gaussian_value),
    "sinusoid": (sample_sinusoid_params, sinusoid_value),
    "rest": (sample_rest_params, rest_value),
    "fourier": (sample_fourier_params, fourier_value),
    "chirp": (sample_chirp_params, chirp_value),
    "shock": (sample_shock_params, shock_value),
    "filtered_random": (sample_filtered_random_params, filtered_random_value),
    "table": (_no_sampler, filtered_random_value),
}


def flip(family: str, params: dict) -> dict:
    # Returns the "opposite" version of a signal for anti-phase two-end loading.
    if family in ("sinusoid", "chirp"):
        key = "phase" if family == "sinusoid" else "phase0"
        q = dict(params)
        q[key] = params[key] + np.pi
        return q
    if family == "fourier":
        q = dict(params)
        q["phase"] = [ph + np.pi for ph in params["phase"]]
        return q
    if family in ("gaussian", "shock"):
        q = dict(params)
        q["A"] = -params["A"]
        return q
    if family == "filtered_random":
        q = dict(params)
        q["values"] = [-v for v in params["values"]]
        return q
    raise ValueError(f"No flip rule for family {family!r}")


def bc_value(bc: BCSpec, t: float) -> float:
    _, family, params = bc
    _, value_fn = BC_WAVEFORMS[family]
    return value_fn(params, t)


def bc_describe(bc: BCSpec) -> str:
    bc_type, family, params = bc
    if "source_label" in params:
        return f"{bc_type}/{family}[{params['source_label']}]"
    param_str = ", ".join(
        f"{k}={v}" if not isinstance(v, (list, tuple)) or len(v) <= 6 else f"{k}=[{len(v)} values]"
        for k, v in params.items()
    )
    return f"{bc_type}/{family}({param_str})"


def apply_boundary(u: np.ndarray, side: str, bc_type: str, value: float, cfg) -> np.ndarray:
    # Dirichlet: fill the whole ghost band + boundary node with `value`.
    # Neumann: mirror the ghost band with a linear correction so the
    # central-difference slope estimate at the boundary equals `value` (0 =
    # true free end). Deliberately does NOT overwrite the boundary node
    # itself -- the interior leapfrog stencil already computes it correctly
    # given one correctly-mirrored ghost neighbor.
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
