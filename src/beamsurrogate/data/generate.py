# Generates a dataset of 1D beam wave simulations and saves it as one HDF5
# file, in the schema data/split.py's load_hdf5_dataset expects. Ported from
# Dataset/Creation_dataset.py (the "complex" profile, unchanged sampling),
# generalized with a `profile` argument so the same generator also produces
# the "simple" profile (Beam_surrogate_model/training/code/scenarios.py's
# gaussian-right/rest-left setup, generalized here to draw amplitude AND
# width from an interval, per the report). Never called automatically by a
# training run -- see scripts/make_dataset.py.
from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py

from ..physics.waves import BC_WAVEFORMS, apply_boundary


# ---------------------------------------------------------------------------
# Profiles: what varies between the "complex" and "simple" datasets. Adding
# a third profile costs one more dict here, not a new generator.
# ---------------------------------------------------------------------------
# (label, bc_type, integrate): "dirichlet" prescribes a displacement,
# "neumann" a slope/flux; integrate=True means the driving family's raw
# output is a velocity, numerically integrated into the imposed displacement.
BC_OPTIONS_COMPLEX = [
    ("Displacement", "dirichlet", False),
    ("Force (du/dx)", "neumann", False),
    ("Velocity (integrated)", "dirichlet", True),
]
BC_OPTIONS_SIMPLE = [("Displacement", "dirichlet", False)]

REST_SHARES_COMPLEX = {"both_driven": 0.65, "left_rest": 0.15, "right_rest": 0.15, "both_rest": 0.05}
REST_SHARES_SIMPLE = {"left_rest": 1.0}   # left always at rest, right always driven
REST_PATTERNS = {  # pattern name -> (left_driven, right_driven)
    "both_driven": (True, True), "left_rest": (False, True),
    "right_rest": (True, False), "both_rest": (False, False),
}

FAMILY_SHARES_COMPLEX = {
    "fourier": 0.30, "sinusoid": 0.15, "chirp": 0.15,
    "gaussian": 0.15, "shock": 0.10, "filtered_random": 0.15,
}
FAMILY_SHARES_SIMPLE = {"gaussian": 1.0}   # the only family the simple dataset ever uses

SPLIT_SHARES = {"train": 0.90, "val": 0.05, "test": 0.05}

INITIAL_STATE_SHARES_COMPLEX = {
    "rest": 0.20, "modal": 0.20, "gaussian_packet": 0.20,
    "travelling_wave": 0.15, "smooth_random": 0.15, "from_simulation": 0.10,
}
INITIAL_STATE_SHARES_SIMPLE = {"rest": 1.0}   # simple dataset never starts pre-excited

PROFILES = {
    "complex": dict(bc_options=BC_OPTIONS_COMPLEX, rest_shares=REST_SHARES_COMPLEX,
                     family_shares=FAMILY_SHARES_COMPLEX, initial_state_shares=INITIAL_STATE_SHARES_COMPLEX),
    "simple": dict(bc_options=BC_OPTIONS_SIMPLE, rest_shares=REST_SHARES_SIMPLE,
                    family_shares=FAMILY_SHARES_SIMPLE, initial_state_shares=INITIAL_STATE_SHARES_SIMPLE),
}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def run_simulation(bc_left, bc_right, cfg, u0=None, v0=None):
    # bc_left/bc_right: (bc_type, integrate, driven, family, params).
    bc_type_l, integrate_l, driven_l, family_l, params_l = bc_left
    bc_type_r, integrate_r, driven_r, family_r, params_r = bc_right
    if u0 is None:
        u0 = np.zeros(cfg.Nx)
    if v0 is None:
        v0 = np.zeros(cfg.Nx)

    i_left, i_right = cfg.i_left, cfg.i_right
    u = np.zeros(cfg.Ntot)
    u[i_left:i_right] = u0
    u_1 = np.zeros(cfg.Ntot)
    u_1[i_left:i_right] = u0 - cfg.dt * v0   # u(-dt) ~= u(0) - dt*v(0)
    u_storage = np.zeros((cfg.Nt + 1, cfg.Ntot))
    u_storage[0] = u
    left_values = np.zeros(cfg.Nt + 1)
    right_values = np.zeros(cfg.Nt + 1)
    left_integral = right_integral = 0.0

    for n in range(cfg.Nt):
        t = (n + 1) * cfg.dt

        if driven_l:
            raw = BC_WAVEFORMS[family_l][1](params_l, t)
            if integrate_l:
                left_integral += raw * cfg.dt
                left_value = left_integral
            else:
                left_value = raw
        else:
            left_value = 0.0

        if driven_r:
            raw = BC_WAVEFORMS[family_r][1](params_r, t)
            if integrate_r:
                right_integral += raw * cfg.dt
                right_value = right_integral
            else:
                right_value = raw
        else:
            right_value = 0.0

        left_values[n + 1] = left_value
        right_values[n + 1] = right_value

        u_new = np.zeros(cfg.Ntot)
        u_new[i_left:i_right + 1] = (
            2.0 * u[i_left:i_right + 1] - u_1[i_left:i_right + 1]
            + cfg.CFL ** 2 * (u[i_left - 1:i_right] - 2.0 * u[i_left:i_right + 1] + u[i_left + 1:i_right + 2])
        )
        apply_boundary(u_new, "left", bc_type_l, left_value, cfg)
        apply_boundary(u_new, "right", bc_type_r, right_value, cfg)
        u_1, u = u.copy(), u_new
        u_storage[n + 1] = u

    return u_storage[:, i_left:i_right], left_values, right_values


def sample_boundary_pair(rng, cfg, profile: dict):
    bc_options = profile["bc_options"]
    rest_shares = profile["rest_shares"]
    family_shares = profile["family_shares"]

    left_opt = bc_options[rng.integers(len(bc_options))]
    right_opt = bc_options[rng.integers(len(bc_options))]
    rest_pattern = rng.choice(list(rest_shares), p=list(rest_shares.values()))
    left_driven, right_driven = REST_PATTERNS[rest_pattern]
    left_family = rng.choice(list(family_shares), p=list(family_shares.values()))
    right_family = rng.choice(list(family_shares), p=list(family_shares.values()))

    left_params = BC_WAVEFORMS[left_family][0](rng, cfg) if left_driven else None
    right_params = BC_WAVEFORMS[right_family][0](rng, cfg) if right_driven else None

    bc_left = (left_opt[1], left_opt[2], left_driven, left_family, left_params)
    bc_right = (right_opt[1], right_opt[2], right_driven, right_family, right_params)
    return bc_left, bc_right, left_opt[0], right_opt[0], left_family, right_family


# ---------------------------------------------------------------------------
# Initial states (used to seed u(x,0), v(x,0)) -- only "rest" is reachable
# under the simple profile, but all 6 stay available for "complex".
# ---------------------------------------------------------------------------
def sample_rest_state(rng, cfg, profile):
    return np.zeros(cfg.Nx), np.zeros(cfg.Nx)


def sample_modal_state(rng, cfg, profile):
    x = np.linspace(0, cfg.L, cfg.Nx)
    n_modes = int(rng.integers(2, 6))
    u0 = np.zeros(cfg.Nx)
    for _ in range(n_modes):
        k = int(rng.integers(1, 6))
        A = rng.uniform(cfg.AMP_MIN, cfg.AMP_MAX) * rng.choice([-1, 1])
        u0 += A * np.sin(k * np.pi * x / cfg.L)
    return u0, np.zeros(cfg.Nx)


def sample_gaussian_packet_state(rng, cfg, profile):
    x = np.linspace(0, cfg.L, cfg.Nx)
    x0 = rng.uniform(0.2 * cfg.L, 0.8 * cfg.L)
    sigma = rng.uniform(cfg.SIGMA_MIN, cfg.SIGMA_MAX)
    A = rng.uniform(cfg.AMP_MIN, cfg.AMP_MAX) * rng.choice([-1, 1])
    packet = A * np.exp(-((x - x0) / sigma) ** 2)
    if rng.random() < 0.5:
        return packet, np.zeros(cfg.Nx)
    return np.zeros(cfg.Nx), packet


def sample_travelling_wave_state(rng, cfg, profile):
    x = np.linspace(0, cfg.L, cfg.Nx)
    x0 = rng.uniform(0.2 * cfg.L, 0.8 * cfg.L)
    sigma = rng.uniform(cfg.SIGMA_MIN, cfg.SIGMA_MAX)
    A = rng.uniform(cfg.AMP_MIN, cfg.AMP_MAX)
    u0 = A * np.exp(-((x - x0) / sigma) ** 2)
    wave_speed = np.sqrt(cfg.E / cfg.rho)
    direction = rng.choice([-1, 1])
    v0 = -direction * wave_speed * np.gradient(u0, cfg.dx)
    return u0, v0


def sample_smooth_random_state(rng, cfg, profile):
    n_ctrl = 12
    x_ctrl = np.linspace(0.0, cfg.L, n_ctrl)
    raw = rng.normal(size=n_ctrl)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    smoothed = np.convolve(raw, kernel, mode="same")
    A = rng.uniform(cfg.AMP_MIN, cfg.AMP_MAX)
    peak = np.abs(smoothed).max()
    scale = A / peak if peak > 1e-12 else 0.0
    x = np.linspace(0.0, cfg.L, cfg.Nx)
    u0 = np.interp(x, x_ctrl, smoothed * scale)
    return u0, np.zeros(cfg.Nx)


def sample_from_simulation_state(rng, cfg, profile):
    bc_left, bc_right, *_ = sample_boundary_pair(rng, cfg, profile)
    donor_u, _, _ = run_simulation(bc_left, bc_right, cfg)
    n_pick = int(rng.integers(cfg.Nt // 10, cfg.Nt + 1))
    u0 = donor_u[n_pick]
    v0 = (donor_u[n_pick] - donor_u[n_pick - 1]) / cfg.dt
    return u0, v0


INITIAL_STATE_FUNCTIONS = {
    "rest": sample_rest_state,
    "modal": sample_modal_state,
    "gaussian_packet": sample_gaussian_packet_state,
    "travelling_wave": sample_travelling_wave_state,
    "smooth_random": sample_smooth_random_state,
    "from_simulation": sample_from_simulation_state,
}


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------
def generate_dataset(cfg, profile_name: str, n_trajectories: int, output_path: Path,
                      seed: int | None = None, progress_every: int = 200) -> Path:
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile {profile_name!r}, expected one of {sorted(PROFILES)}")
    profile = PROFILES[profile_name]
    seed = cfg.SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    n = n_trajectories

    split_names, split_weights = list(SPLIT_SHARES), list(SPLIT_SHARES.values())
    initial_state_names = list(profile["initial_state_shares"])
    initial_state_weights = list(profile["initial_state_shares"].values())

    print(f"Generating {n} simulations ({profile_name} profile, {cfg.Nt + 1} timesteps x {cfg.Nx} points each)...")
    u = np.zeros((n, cfg.Nt + 1, cfg.Nx), dtype=np.float32)
    left_bc_value = np.zeros((n, cfg.Nt + 1), dtype=np.float32)
    right_bc_value = np.zeros((n, cfg.Nt + 1), dtype=np.float32)
    left_label, right_label = [], []
    left_driven_flags, right_driven_flags = [], []
    left_family_used, right_family_used = [], []
    split_label, initial_state_used = [], []

    for i in range(n):
        bc_left, bc_right, left_name, right_name, left_family, right_family = sample_boundary_pair(rng, cfg, profile)
        initial_state_name = rng.choice(initial_state_names, p=initial_state_weights)
        split_name = rng.choice(split_names, p=split_weights)

        u0, v0 = INITIAL_STATE_FUNCTIONS[initial_state_name](rng, cfg, profile)
        u[i], left_bc_value[i], right_bc_value[i] = run_simulation(bc_left, bc_right, cfg, u0, v0)

        left_label.append(left_name)
        right_label.append(right_name)
        left_driven_flags.append(bc_left[2])
        right_driven_flags.append(bc_right[2])
        left_family_used.append(str(left_family))
        right_family_used.append(str(right_family))
        split_label.append(str(split_name))
        initial_state_used.append(str(initial_state_name))

        if (i + 1) % progress_every == 0 or i + 1 == n:
            print(f"  {i + 1}/{n} simulations done")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_path} ...")
    with h5py.File(output_path, "w") as f:
        f.attrs.update(E=cfg.E, rho=cfg.rho, L=cfg.L, Nx=cfg.Nx, Nt=cfg.Nt, t_end=cfg.t_end,
                        dt=cfg.dt, dx=cfg.dx, CFL=cfg.CFL, seed=seed, profile=profile_name,
                        created=datetime.now().isoformat(timespec="seconds"))
        f.create_dataset("x", data=np.linspace(0, cfg.L, cfg.Nx, dtype=np.float32))
        f.create_dataset("t", data=(np.arange(cfg.Nt + 1) * cfg.dt).astype(np.float32))
        f.create_dataset("u", data=u)
        f.create_dataset("left_bc_value", data=left_bc_value)
        f.create_dataset("right_bc_value", data=right_bc_value)
        f.create_dataset("left_label", data=left_label)
        f.create_dataset("right_label", data=right_label)
        f.create_dataset("left_driven", data=left_driven_flags)
        f.create_dataset("right_driven", data=right_driven_flags)
        f.create_dataset("left_family", data=left_family_used)
        f.create_dataset("right_family", data=right_family_used)
        f.create_dataset("split", data=split_label)
        f.create_dataset("initial_state", data=initial_state_used)

    print(f"\nSaved {n} trajectories to {output_path}\n")
    print("Breakdown by BC pair:")
    for (l, r), count in Counter(zip(left_label, right_label)).items():
        print(f"  {l} (L) / {r} (R):  {count:4d}  ({100 * count / n:4.1f}%)")
    print("\nBreakdown by rest pattern:")
    for (ld, rd), count in Counter(zip(left_driven_flags, right_driven_flags)).items():
        print(f"  left {'driven' if ld else 'rest'} / right {'driven' if rd else 'rest'}:"
              f"  {count:4d}  ({100 * count / n:4.1f}%)")
    print("\nBreakdown by split:")
    for name, count in Counter(split_label).items():
        print(f"  {name:20s}{count:4d}  ({100 * count / n:4.1f}%)")

    return output_path
