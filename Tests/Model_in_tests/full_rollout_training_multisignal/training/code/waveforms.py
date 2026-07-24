# New signal families for this project, on top of the generic
# (bc_type, waveform_family, params) boundary-condition machinery already
# validated in commun.py (BCSpec, BC_TYPES, BC_WAVEFORMS, bc_value,
# apply_boundary_conditions, run_fd_simulation_general, reconstruct_general,
# ...). commun.py already has "sinusoid" and "gaussian" -- reused as-is
# below, no reason to reimplement well-tested math. The 4 families the table
# calls for that commun.py doesn't have (Random Fourier sums, Chirps,
# Smoothed steps/shocks, Filtered random histories) are implemented here and
# registered into commun.BC_WAVEFORMS at import time: this mutates the
# already-imported module object for the lifetime of THIS process only
# (commun.py's file on disk is never touched), so the existing generic
# physics functions can dispatch these new families exactly like any
# built-in one via their normal `BC_WAVEFORMS[family]` lookup -- no need to
# duplicate the ~150 lines of validated PDE/reconstruction code for the sake
# of a few new family names.
import numpy as np


# ---------------------------------------------------------------------------
# Random Fourier sums (table: "Broad spectral coverage") -- like commun.py's
# built-in "random_multitone" but with more tones (4-8 instead of 2-4), since
# this family is meant to be the richest/busiest one (30% share).
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
# Chirps (table: "Frequency sweep in one run") -- linear sweep of the
# instantaneous angular frequency from omega0 to omega1 over the whole run;
# phase(t) is the integral of omega(t), so the sweep is smooth (no
# discontinuity at t=0). t_end is captured into params at sample time so
# value_fn keeps the 2-arg (params, t) signature every other family uses.
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
# Smoothed steps/shocks (table: "Sharp but resolved transients") -- a
# tanh-smoothed step: rises from 0 to A over a short but resolved timescale
# tau, then holds. Unlike commun.py's existing "step" (an instantaneous
# jump), this is deliberately smoothed so it stays numerically resolved by
# the grid instead of being a true discontinuity.
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
# Filtered random histories (table: "Coverage of generic smooth functions")
# -- iid noise at a handful of control points over [0, t_end], smoothed with
# a small moving-average kernel, rescaled to the amplitude range, then
# linearly interpolated for any t. Band-limited and non-parametric, unlike
# the fixed-frequency "fourier"/"chirp"/"sinusoid" families above.
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
# Registration + "anti-phase/antisymmetric" flip helper for correlated
# two-end sampling (see scenarios.py).
# ---------------------------------------------------------------------------
NEW_WAVEFORMS = {
    "fourier": (sample_fourier_params, fourier_value),
    "chirp": (sample_chirp_params, chirp_value),
    "shock": (sample_shock_params, shock_value),
    "filtered_random": (sample_filtered_random_params, filtered_random_value),
}


def register(C) -> None:
    for family, entry in NEW_WAVEFORMS.items():
        C.BC_WAVEFORMS.setdefault(family, entry)


def flip(family: str, params: dict) -> dict:
    # Returns the "opposite" version of a signal for anti-phase/antisymmetric
    # two-end loading: for oscillatory families a phase shift of pi has the
    # same effect as negating the amplitude; for one-signed pulses/histories,
    # negating the amplitude/values directly is the natural opposite.
    if family in ("sinusoid", "chirp"):
        key = "phase" if family == "sinusoid" else "phase0"
        q = dict(params)
        q[key] = params[key] + np.pi
        return q
    if family == "fourier":
        q = dict(params)
        q["phase"] = [ph + np.pi for ph in params["phase"]]
        return q
    if family == "gaussian" or family == "shock":
        q = dict(params)
        q["A"] = -params["A"]
        return q
    if family == "filtered_random":
        q = dict(params)
        q["values"] = [-v for v in params["values"]]
        return q
    raise ValueError(f"No flip rule for family {family!r}")
