# The PDE solver (leapfrog finite differences, generalized arbitrary
# Dirichlet/Neumann boundary conditions via waves.py) and the feature
# assembly shared by dataset generation, TBPTT training, and rollout.
#
# Unlike full_rollout_training_conv1d/training/code/physics.py, this project
# only ever feeds the network past displacement (U) -- no velocity (Ut) or
# curvature (Uxx) channels -- so there's no per-field dispatch machinery
# here, and it builds a channels-first (n_lags, Nx) array for the WHOLE beam
# per call instead of a per-node (n_nodes, n_features) window. The FD solver
# itself (run_fd_simulation_general) and the boundary-condition application
# (waves.py) are unchanged.
from __future__ import annotations

import numpy as np
import torch

from waves import BCSpec, apply_boundary_conditions


def lag_label(lag: int) -> str:
    return "t" if lag == 0 else f"t-{lag}ndt"


def make_input_channel_labels(cfg) -> list[str]:
    return [f"u({lag_label(lag)})" for lag in range(cfg.M_BACK)]


def make_output_channel_labels(cfg) -> list[str]:
    return [f"delta_u@{h}ndt" for h in range(1, cfg.N_FWD + 1)]


def build_field_history(m_list, get_u, cfg) -> np.ndarray:
    # Returns (n_lags, Nx): for each requested past instant, the beam's
    # displacement at its Nx interior nodes (cfg.nodes) -- no ghost/BC band,
    # "the beam" is only ever the physical points, see the plan's point 4.
    nodes = cfg.nodes
    X = np.zeros((len(m_list), len(nodes)), dtype=np.float32)
    for i, m in enumerate(m_list):
        X[i] = get_u(m)[nodes]
    return X


# ---------------------------------------------------------------------------
# Finite-difference solver (unchanged from the reference project)
# ---------------------------------------------------------------------------
def run_fd_simulation_general(bc_left: BCSpec, bc_right: BCSpec, cfg) -> np.ndarray:
    # Leapfrog scheme, ghost-filled every step via apply_boundary_conditions
    # so each side can independently be Dirichlet or Neumann.
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


def reconstruct_general(u_curr, n_curr, pred_norm, bc_left: BCSpec, bc_right: BCSpec, mu_out, sd_out,
                         cfg, biais_repos: np.ndarray | None = None) -> dict:
    # Numpy reference reconstruction -- used directly by check_equivalence.py
    # to validate rollout_torch.py's differentiable torch port.
    # pred_norm/deltas/biais_repos: (N_FWD, Nx). mu_out/sd_out: (N_FWD,).
    deltas = pred_norm * sd_out[:, None] + mu_out[:, None]
    if biais_repos is not None:
        deltas = deltas - biais_repos
    nodes = cfg.nodes
    champs = {}
    for h in range(1, cfg.N_FWD + 1):
        s = n_curr + h * cfg.ndt
        t = s * cfg.dt
        u = np.zeros(cfg.Ntot)
        u[nodes] = u_curr[nodes] + deltas[h-1]
        apply_boundary_conditions(u, t, bc_left, bc_right, cfg)
        if cfg.SMOOTH_ALPHA > 0:
            j0, j1 = cfg.i_left + 1, cfg.i_right
            lap = u[j0-1:j1-1] - 2*u[j0:j1] + u[j0+1:j1+1]
            u[j0:j1] += cfg.SMOOTH_ALPHA * lap
        champs[s] = u
    return champs


def biais_repos(modele, mu_in, sd_in, mu_out, sd_out, cfg):
    # Network output for an all-at-rest beam, subtracted from the rollout so
    # the resting zone stays at 0. Full (N_FWD, Nx) field, not a scalar per
    # horizon -- see the plan's point 7: the pointwise head's 2 conv layers
    # still see the two ends differently from the middle (zero-padding),
    # even on an all-zero input.
    Xz = (np.zeros((len(mu_in), len(cfg.nodes)), dtype=np.float32) - mu_in[:, None]) / sd_in[:, None]
    with torch.no_grad():
        out = modele(torch.tensor(Xz[None])).numpy()[0]  # (N_FWD, Nx)
    return out * sd_out[:, None] + mu_out[:, None]


def autoregressive_rollout(modele, U_reel, mu_in, sd_in, mu_out, sd_out,
                            biais_repos, bc_left: BCSpec, bc_right: BCSpec, cfg) -> np.ndarray:
    history_needed = cfg.M_BACK * cfg.ndt
    U = np.zeros((cfg.Nt + 1, cfg.Ntot))
    for m in range(history_needed + 1):
        U[m] = U_reel[m]

    for n in range(history_needed, cfg.Nt - cfg.N_FWD*cfg.ndt + 1, cfg.N_FWD*cfg.ndt):
        m_list = [n - lag*cfg.ndt for lag in range(cfg.M_BACK)]
        Xc = build_field_history(m_list, lambda m: U[m], cfg)  # (n_lags, Nx)
        Xn = (Xc - mu_in[:, None]) / sd_in[:, None]
        with torch.no_grad():
            pred_norm = modele(torch.tensor(Xn[None])).numpy()[0]  # (N_FWD, Nx)

        champs = reconstruct_general(U[n], n, pred_norm, bc_left, bc_right, mu_out, sd_out, cfg,
                                      biais_repos=biais_repos)
        for s, u in champs.items():
            U[s] = u

    return U
