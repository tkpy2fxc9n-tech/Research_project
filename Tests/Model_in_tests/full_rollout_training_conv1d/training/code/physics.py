# The PDE solver (leapfrog finite differences, generalized arbitrary
# Dirichlet/Neumann boundary conditions via waves.py) and the feature
# windowing shared by dataset generation, training, and rollout.
from __future__ import annotations

import numpy as np
import torch

from waves import BCSpec, apply_boundary_conditions

FIELD_LABELS = {"U": "u", "Ut": "u_dot", "Uxx": "u_xx"}


# ---------------------------------------------------------------------------
# Feature windowing (stencil x lag -> flat feature vector)
# ---------------------------------------------------------------------------
def jlabel(k: int) -> str:
    return "j" if k == 0 else f"j{k:+d}"


def lag_label(lag: int) -> str:
    return "t" if lag == 0 else f"t-{lag}ndt"


def uxx_field(u: np.ndarray, cfg) -> np.ndarray:
    out = np.zeros(cfg.Ntot)
    i_left, i_right = cfg.i_left, cfg.i_right
    out[i_left:i_right+1] = (u[i_left-1:i_right] - 2*u[i_left:i_right+1] + u[i_left+1:i_right+2]) / cfg.dx**2
    return out


def field_value(field_name: str, get_u, m: int, cfg) -> np.ndarray:
    if field_name == "U":
        return get_u(m)
    if field_name == "Ut":
        return (get_u(m) - get_u(m - cfg.ndt)) / (cfg.ndt * cfg.dt)
    if field_name == "Uxx":
        return uxx_field(get_u(m), cfg)
    raise ValueError(f"Unknown input field: {field_name!r}")


def make_feature_columns(input_fields: list[str], cfg) -> list[str]:
    cols = []
    for lag in range(cfg.M_BACK):
        lab = lag_label(lag)
        for k in range(-cfg.SS, cfg.SS + 1):
            for f in input_fields:
                cols.append(f"{FIELD_LABELS[f]}({lab},{jlabel(k)})")
    return cols


def make_output_columns(cfg) -> list[str]:
    return [f"delta_u@{h}ndt" for h in range(1, cfg.N_FWD + 1)]


def build_window(m_list, get_u, input_fields: list[str], cfg) -> np.ndarray:
    # Column order must stay synchronized with make_feature_columns.
    nodes = cfg.nodes
    n_features = cfg.M_BACK * (2*cfg.SS + 1) * len(input_fields)
    X = np.zeros((len(nodes), n_features), dtype=np.float32)
    col = 0
    for m in m_list:
        field_arrays = {f: field_value(f, get_u, m, cfg) for f in input_fields}
        for k in range(-cfg.SS, cfg.SS + 1):
            for f in input_fields:
                X[:, col] = field_arrays[f][nodes + k]
                col += 1
    return X


# ---------------------------------------------------------------------------
# Finite-difference solver
# ---------------------------------------------------------------------------
def run_fd_simulation_general(bc_left: BCSpec, bc_right: BCSpec, cfg) -> np.ndarray:
    # Leapfrog scheme, ghost-filled every step via apply_boundary_conditions
    # so each side can independently be Dirichlet or Neumann, any family.
    i_left, i_right, Ntot = cfg.i_left, cfg.i_right, cfg.Ntot
    u_storage = np.zeros((cfg.Nt + 1, Ntot))
    u = np.zeros(Ntot)
    u_1 = np.zeros(Ntot)
    for n in range(cfg.Nt):
        t = n * cfg.dt
        u_new = np.zeros(Ntot)
        u_new[i_left:i_right+1] = (
            2.0 * u[i_left:i_right+1] - u_1[i_left:i_right+1]
            + cfg.CFL**2 * (u[i_left-1:i_right] - 2.0*u[i_left:i_right+1] + u[i_left+1:i_right+2])
        )
        apply_boundary_conditions(u_new, t + cfg.dt, bc_left, bc_right, cfg)
        u_1, u = u.copy(), u_new
        u_storage[n+1] = u.copy()
    return u_storage


def run_fd_simulation_free(bc_left: BCSpec, bc_right: BCSpec, u0_profile: np.ndarray, cfg) -> np.ndarray:
    # "Free evolution" / random initial state (table: "avoid dependence on
    # forced-from-rest trajectories"): both ends stay at rest (no push at
    # all); instead the bar *starts* bent into a random smooth shape
    # (scenarios.sample_random_ic) and is simply let go. Same leapfrog scheme
    # as run_fd_simulation_general, except u/u_1 start from u0_profile (zero
    # initial velocity) instead of zero displacement.
    i_left, i_right, Ntot = cfg.i_left, cfg.i_right, cfg.Ntot
    u_storage = np.zeros((cfg.Nt + 1, Ntot))

    u = np.zeros(Ntot)
    u[cfg.nodes] = u0_profile
    apply_boundary_conditions(u, 0.0, bc_left, bc_right, cfg)
    u_1 = u.copy()
    u_storage[0] = u.copy()

    for n in range(cfg.Nt):
        t = n * cfg.dt
        u_new = np.zeros(Ntot)
        u_new[i_left:i_right + 1] = (
            2.0 * u[i_left:i_right + 1] - u_1[i_left:i_right + 1]
            + cfg.CFL ** 2 * (u[i_left - 1:i_right] - 2.0 * u[i_left:i_right + 1] + u[i_left + 1:i_right + 2])
        )
        apply_boundary_conditions(u_new, t + cfg.dt, bc_left, bc_right, cfg)
        u_1, u = u.copy(), u_new
        u_storage[n + 1] = u.copy()

    return u_storage


def reconstruct_general(u_curr, n_curr, pred_norm, bc_left: BCSpec, bc_right: BCSpec, mu_out, sd_out,
                         cfg, biais_repos: np.ndarray | None = None) -> dict:
    # Numpy reference reconstruction -- used directly by check_equivalence.py
    # to validate rollout_torch.py's differentiable torch port.
    deltas = pred_norm * sd_out + mu_out
    if biais_repos is not None:
        deltas = deltas - biais_repos
    nodes = cfg.nodes
    champs = {}
    for h in range(1, cfg.N_FWD + 1):
        s = n_curr + h * cfg.ndt
        t = s * cfg.dt
        u = np.zeros(cfg.Ntot)
        u[nodes] = u_curr[nodes] + deltas[:, h-1]
        apply_boundary_conditions(u, t, bc_left, bc_right, cfg)
        if cfg.SMOOTH_ALPHA > 0:
            j0, j1 = cfg.i_left + 1, cfg.i_right
            lap = u[j0-1:j1-1] - 2*u[j0:j1] + u[j0+1:j1+1]
            u[j0:j1] += cfg.SMOOTH_ALPHA * lap
        champs[s] = u
    return champs


def biais_repos(modele, mu_in, sd_in, mu_out, sd_out, cfg):
    # Network output for a zero input, subtracted from the rollout so the
    # resting zone stays at 0.
    Xz = (np.zeros((len(cfg.nodes), len(mu_in)), dtype=np.float32) - mu_in) / sd_in
    with torch.no_grad():
        return (modele(torch.tensor(Xz)).numpy() * sd_out + mu_out)[0]


def autoregressive_rollout(modele, U_reel, input_fields, mu_in, sd_in, mu_out, sd_out,
                            biais_repos, bc_left: BCSpec, bc_right: BCSpec, cfg) -> np.ndarray:
    history_needed = cfg.M_BACK * cfg.ndt
    U = np.zeros((cfg.Nt + 1, cfg.Ntot))
    for m in range(history_needed + 1):
        U[m] = U_reel[m]

    for n in range(history_needed, cfg.Nt - cfg.N_FWD*cfg.ndt + 1, cfg.N_FWD*cfg.ndt):
        m_list = [n - lag*cfg.ndt for lag in range(cfg.M_BACK)]
        X = (build_window(m_list, lambda m: U[m], input_fields, cfg) - mu_in) / sd_in
        with torch.no_grad():
            sortie = modele(torch.tensor(X)).numpy()
        deltas = sortie * sd_out + mu_out - biais_repos

        for h in range(1, cfg.N_FWD + 1):
            s = n + h*cfg.ndt
            t = s * cfg.dt
            U[s, cfg.nodes] = U[n, cfg.nodes] + deltas[:, h-1]
            apply_boundary_conditions(U[s], t, bc_left, bc_right, cfg)

            if cfg.SMOOTH_ALPHA > 0:
                j0, j1 = cfg.i_left + 1, cfg.i_right
                lap = U[s, j0-1:j1-1] - 2*U[s, j0:j1] + U[s, j0+1:j1+1]
                U[s, j0:j1] += cfg.SMOOTH_ALPHA * lap

    return U
