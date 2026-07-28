# Creates a dataset of 1D beam wave simulations and saves it as one HDF5
# file. Just run this file (`python3 Creation_dataset.py`) -- no arguments
# needed.
#
# Physics/solver: same leapfrog finite-difference scheme and Dirichlet/
# Neumann ghost-band boundary conditions as Tests/Model_in_tests/
# CNN_full_beam/Code/{physics,waves}.py. The 6 boundary-condition
# "functions" (fourier/sinusoid/chirp/gaussian/shock/filtered_random) are
# the same ones as Tests/Model_in_tests/full_rollout_training_conv1d/
# training/code/waves.py.
#
# Every trajectory independently picks, by simple weighted random draw:
#   1. a (left, right) boundary-condition TYPE pair (Displacement / Force
#      du-dx / Velocity-integrated), uniformly,
#   2. a REST pattern: which end(s), if any, stay at 0 the whole run
#      instead of being driven,
#   3. a FUNCTION family used to drive whichever end(s) end up driven,
#   4. a train/val/test SPLIT label.
# "Velocity (integrated)" means the family's raw output is treated as a
# velocity: it's summed up over time (running total x dt, the simplest
# possible numerical integration) into the displacement that's actually
# imposed at that end.
from dataclasses import dataclass
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py


# =============================================================================
# DATASET CONFIGURATION -- the only part of this file you need to touch to
# change the dataset. Everything under FUNCTIONS below is implementation and
# doesn't need editing.
# =============================================================================
@dataclass
class Config:
    output_path: Path = Path(__file__).resolve().parent / "beam_dataset.h5"
    n_trajectories: int = 3000    # total number of simulations
    seed: int = 0

    E: float = 1.0      # beam physical properties
    rho: float = 2.0
    L: float = 1.0
    Nx: int = 100       # number of physical grid points along the beam
    SS: int = 10        # half-width of the ghost band used to apply boundary conditions
    Nt: int = 500        # number of timesteps
    t_end: float = 5.0   # simulated duration

    AMP_MIN: float = 0.0005    # amplitude range (all families)
    AMP_MAX: float = 0.005
    OMEGA_MIN: float = 0.05    # angular frequency range (sinusoid/fourier/chirp)
    OMEGA_MAX: float = 0.5
    SIGMA_MIN: float = 0.05    # pulse-width range (gaussian only)
    SIGMA_MAX: float = 0.5

    # -- computed automatically from the fields above, don't edit directly --
    def __post_init__(self):
        self.Ntot = self.Nx + 2 * self.SS
        self.i_left = self.SS
        self.i_right = self.Ntot - self.SS
        self.dt = self.t_end / self.Nt
        self.dx = self.L / (self.Nx - 1)
        self.CFL = self.dt / self.dx * np.sqrt(self.E / self.rho)
        if self.CFL > 1:
            print(f"WARNING: CFL={self.CFL:.3f} > 1 -- the simulation will be numerically "
                  f"unstable. Increase Nt and/or reduce Nx.")


# 1. Boundary-condition type pair -- sampled uniformly and independently at
# each end. (label, bc_type, integrate): "dirichlet" prescribes a
# displacement, "neumann" prescribes a slope/flux; "integrate"=True means
# the driving family's raw output is a velocity, summed over time into a
# displacement (see run_simulation).
BC_OPTIONS = [
    ("Displacement",           "dirichlet", False),
    ("Force (du/dx)",          "neumann",   False),
    ("Velocity (integrated)",  "dirichlet", True),
]

# 2. Rest pattern -- which end(s) stay at 0 the whole run, independently of
# the type pair above.
REST_SHARES = {
    "both_driven": 0.65,
    "left_rest":   0.15,
    "right_rest":  0.15,
    "both_rest":   0.05,
}
REST_PATTERNS = {  # pattern name -> (left_driven, right_driven)
    "both_driven": (True, True),
    "left_rest":   (False, True),
    "right_rest":  (True, False),
    "both_rest":   (False, False),
}

# 3. Boundary condition functions -- which shape drives a driven end. One
# family per trajectory, used at whichever end(s) end up driven.
FAMILY_SHARES = {
    "fourier":          0.30,
    "sinusoid":         0.15,
    "chirp":            0.15,
    "gaussian":         0.15,
    "shock":            0.10,
    "filtered_random":  0.15,
}

# 4. Split for training.
SPLIT_SHARES = {
    "train": 0.90,
    "val":   0.05,
    "test":  0.05,
}


# =============================================================================
# FUNCTIONS -- implementation, no need to edit below this line.
# =============================================================================

# --- boundary condition shape families (used by driven ends) ---------------
def sample_gaussian_params(rng, cfg):
    return {"A": float(rng.uniform(cfg.AMP_MIN, cfg.AMP_MAX)),
            "sigma": float(rng.uniform(cfg.SIGMA_MIN, cfg.SIGMA_MAX))}


def gaussian_value(p, t):
    sigma = p["sigma"]
    t0 = 4.0 * sigma
    return p["A"] * np.exp(-((t - t0) / sigma) ** 2)


def sample_sinusoid_params(rng, cfg):
    return {"A": float(rng.uniform(cfg.AMP_MIN, cfg.AMP_MAX)),
            "omega": float(rng.uniform(cfg.OMEGA_MIN, cfg.OMEGA_MAX)),
            "phase": float(rng.uniform(0.0, 2 * np.pi))}


def sinusoid_value(p, t):
    return p["A"] * np.sin(p["omega"] * t + p["phase"])


def sample_fourier_params(rng, cfg):
    n_tones = int(rng.integers(4, 9))
    return {"A": rng.uniform(cfg.AMP_MIN, cfg.AMP_MAX, size=n_tones).tolist(),
            "omega": rng.uniform(cfg.OMEGA_MIN, cfg.OMEGA_MAX, size=n_tones).tolist(),
            "phase": rng.uniform(0.0, 2 * np.pi, size=n_tones).tolist()}


def fourier_value(p, t):
    tones = zip(p["A"], p["omega"], p["phase"])
    return sum(A * np.sin(om * t + ph) for A, om, ph in tones) / len(p["A"])


def sample_chirp_params(rng, cfg):
    return {"A": float(rng.uniform(cfg.AMP_MIN, cfg.AMP_MAX)),
            "omega0": float(rng.uniform(cfg.OMEGA_MIN, cfg.OMEGA_MAX)),
            "omega1": float(rng.uniform(cfg.OMEGA_MIN, cfg.OMEGA_MAX)),
            "phase0": float(rng.uniform(0.0, 2 * np.pi)),
            "t_end": float(cfg.t_end)}


def chirp_value(p, t):
    omega0, omega1, t_end = p["omega0"], p["omega1"], p["t_end"]
    phase = p["phase0"] + omega0 * t + 0.5 * (omega1 - omega0) / t_end * t ** 2
    return p["A"] * np.sin(phase)


def sample_shock_params(rng, cfg):
    return {"A": float(rng.uniform(cfg.AMP_MIN, cfg.AMP_MAX)),
            "t_onset": float(rng.uniform(0.1 * cfg.t_end, 0.5 * cfg.t_end)),
            "tau": float(rng.uniform(0.01 * cfg.t_end, 0.05 * cfg.t_end))}


def shock_value(p, t):
    return p["A"] * 0.5 * (1.0 + np.tanh((t - p["t_onset"]) / p["tau"]))


def sample_filtered_random_params(rng, cfg):
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


def filtered_random_value(p, t):
    return float(np.interp(t, p["t_ctrl"], p["values"]))


BC_WAVEFORMS = {
    "gaussian": (sample_gaussian_params, gaussian_value),
    "sinusoid": (sample_sinusoid_params, sinusoid_value),
    "fourier": (sample_fourier_params, fourier_value),
    "chirp": (sample_chirp_params, chirp_value),
    "shock": (sample_shock_params, shock_value),
    "filtered_random": (sample_filtered_random_params, filtered_random_value),
}


def apply_boundary(u, side, bc_type, value, cfg):
    if side == "left":
        if bc_type == "dirichlet":
            u[:cfg.i_left + 1] = value
        else:
            for k in range(1, cfg.SS + 1):
                u[cfg.i_left - k] = u[cfg.i_left + k] - 2 * k * cfg.dx * value
    else:
        if bc_type == "dirichlet":
            u[cfg.i_right:] = value
        else:
            for k in range(1, cfg.SS):
                u[cfg.i_right + k] = u[cfg.i_right - k] + 2 * k * cfg.dx * value


def run_simulation(bc_left, bc_right, cfg):
    # bc_left/bc_right: (bc_type, integrate, driven, family, params).
    bc_type_l, integrate_l, driven_l, family_l, params_l = bc_left
    bc_type_r, integrate_r, driven_r, family_r, params_r = bc_right

    i_left, i_right = cfg.i_left, cfg.i_right
    u = np.zeros(cfg.Ntot)
    u_1 = np.zeros(cfg.Ntot)
    u_storage = np.zeros((cfg.Nt + 1, cfg.Ntot))
    left_values = np.zeros(cfg.Nt + 1)
    right_values = np.zeros(cfg.Nt + 1)
    left_integral = 0.0
    right_integral = 0.0

    for n in range(cfg.Nt):
        t = (n + 1) * cfg.dt

        if driven_l:
            raw = BC_WAVEFORMS[family_l][1](params_l, t)
            if integrate_l:
                left_integral += raw * cfg.dt   # running total x dt = simple numerical integration
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

    return u_storage[:, i_left:i_right], left_values, right_values  # u_storage sliced to the Nx physical beam nodes


def main():
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_trajectories

    rest_names, rest_weights = list(REST_SHARES), list(REST_SHARES.values())
    family_names, family_weights = list(FAMILY_SHARES), list(FAMILY_SHARES.values())
    split_names, split_weights = list(SPLIT_SHARES), list(SPLIT_SHARES.values())

    print(f"Generating {n} simulations ({cfg.Nt + 1} timesteps x {cfg.Nx} points each)...")
    u = np.zeros((n, cfg.Nt + 1, cfg.Nx), dtype=np.float32)
    left_bc_value = np.zeros((n, cfg.Nt + 1), dtype=np.float32)
    right_bc_value = np.zeros((n, cfg.Nt + 1), dtype=np.float32)
    left_label, right_label = [], []
    left_driven_flags, right_driven_flags = [], []
    family_used, split_label = [], []

    for i in range(n):
        left_opt = BC_OPTIONS[rng.integers(len(BC_OPTIONS))]
        right_opt = BC_OPTIONS[rng.integers(len(BC_OPTIONS))]
        rest_pattern = rng.choice(rest_names, p=rest_weights)
        left_driven, right_driven = REST_PATTERNS[rest_pattern]
        family = rng.choice(family_names, p=family_weights)
        split_name = rng.choice(split_names, p=split_weights)

        left_params = BC_WAVEFORMS[family][0](rng, cfg) if left_driven else None
        right_params = BC_WAVEFORMS[family][0](rng, cfg) if right_driven else None

        bc_left = (left_opt[1], left_opt[2], left_driven, family, left_params)
        bc_right = (right_opt[1], right_opt[2], right_driven, family, right_params)
        u[i], left_bc_value[i], right_bc_value[i] = run_simulation(bc_left, bc_right, cfg)

        left_label.append(left_opt[0])
        right_label.append(right_opt[0])
        left_driven_flags.append(left_driven)
        right_driven_flags.append(right_driven)
        family_used.append(str(family))
        split_label.append(str(split_name))

        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n} simulations done")

    print(f"Saving to {cfg.output_path} ...")
    with h5py.File(cfg.output_path, "w") as f:
        f.attrs.update(E=cfg.E, rho=cfg.rho, L=cfg.L, Nx=cfg.Nx, Nt=cfg.Nt, t_end=cfg.t_end,
                        dt=cfg.dt, dx=cfg.dx, CFL=cfg.CFL, seed=cfg.seed,
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
        f.create_dataset("family", data=family_used)
        f.create_dataset("split", data=split_label)

    print(f"\nSaved {n} trajectories to {cfg.output_path}\n")

    print("Breakdown by BC pair:")
    for (l, r), count in Counter(zip(left_label, right_label)).items():
        print(f"  {l} (L) / {r} (R):  {count:4d}  ({100 * count / n:4.1f}%)")

    print("\nBreakdown by rest pattern:")
    for (ld, rd), count in Counter(zip(left_driven_flags, right_driven_flags)).items():
        print(f"  left {'driven' if ld else 'rest'} / right {'driven' if rd else 'rest'}:"
              f"  {count:4d}  ({100 * count / n:4.1f}%)")

    print("\nBreakdown by family:")
    for name, count in Counter(family_used).items():
        print(f"  {name:20s}{count:4d}  ({100 * count / n:4.1f}%)")

    print("\nBreakdown by split:")
    for name, count in Counter(split_label).items():
        print(f"  {name:20s}{count:4d}  ({100 * count / n:4.1f}%)")


if __name__ == "__main__":
    main()
