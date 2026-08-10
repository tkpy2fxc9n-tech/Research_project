# Torch port (differentiable, never falling back to numpy) of the
# reconstruction physics already present and validated in
# Code_comparaison_des_inputs/commun.py (uxx_field, field_value, build_window,
# reconstruct). No in-place operations: every step rebuilds a new tensor,
# to never disturb the gradient computation path.
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR

sys.path.insert(0, str(COMMUN_DIR))
import commun as C


def uxx_field_torch(u: torch.Tensor, cfg: "C.Config") -> torch.Tensor:
    # u : (G, Ntot). Direct port of C.uxx_field.
    i_left, i_right = cfg.i_left, cfg.i_right
    G = u.shape[0]
    interior = (u[:, i_left - 1:i_right] - 2 * u[:, i_left:i_right + 1]
                + u[:, i_left + 1:i_right + 2]) / cfg.dx ** 2
    left_pad = torch.zeros(G, i_left, dtype=u.dtype)
    right_pad = torch.zeros(G, cfg.Ntot - (i_right + 1), dtype=u.dtype)
    return torch.cat([left_pad, interior, right_pad], dim=1)


def build_window_torch(history: list[torch.Tensor], input_fields: list[str], cfg: "C.Config") -> torch.Tensor:
    # history: list of M_BACK+1 tensors (G, Ntot), from the oldest
    # (history[0]) to the most recent (history[-1] = current state n).
    # Returns X of shape (G*Nx, n_features), same column order as
    # C.make_feature_columns (lag, then k, then field).
    nodes = cfg.nodes
    G = history[-1].shape[0]
    Nx = len(nodes)

    cols = []
    for lag in range(cfg.M_BACK):
        u_m = history[-(lag + 1)]
        field_arrays = {}
        for f in input_fields:
            if f == "U":
                field_arrays[f] = u_m
            elif f == "Ut":
                u_prev = history[-(lag + 2)]
                field_arrays[f] = (u_m - u_prev) / (cfg.ndt * cfg.dt)
            elif f == "Uxx":
                field_arrays[f] = uxx_field_torch(u_m, cfg)
            else:
                raise ValueError(f"Unknown input field: {f!r}")
        for k in range(-cfg.SS, cfg.SS + 1):
            idx = nodes + k
            for f in input_fields:
                cols.append(field_arrays[f][:, idx])  # (G, Nx)

    X = torch.stack(cols, dim=-1)  # (G, Nx, n_features)
    return X.reshape(G * Nx, -1)


def reconstruct_torch(baseline: torch.Tensor, pred_norm: torch.Tensor,
                       A_list: list[float], omega_list: list[float], n_curr: int,
                       mu_out_t: torch.Tensor, sd_out_t: torch.Tensor,
                       biais_repos_t: torch.Tensor | None, cfg: "C.Config") -> tuple[list[torch.Tensor], list[int]]:
    # baseline: (G, Ntot), state BEFORE this hop -- fixed for all h (don't
    # chain h=1 into h=2, like C.reconstruct).
    # pred_norm: (G*Nx, N_FWD), raw network output for this hop.
    # Returns (list of N_FWD tensors (G, Ntot), list of corresponding time indices s).
    nodes = cfg.nodes
    i_left, i_right, Ntot = cfg.i_left, cfg.i_right, cfg.Ntot
    G = baseline.shape[0]
    Nx = len(nodes)

    pred_norm = pred_norm.reshape(G, Nx, cfg.N_FWD)
    deltas = pred_norm * sd_out_t + mu_out_t
    if biais_repos_t is not None:
        deltas = deltas - biais_repos_t

    new_states, s_list = [], []
    for h in range(1, cfg.N_FWD + 1):
        s = n_curr + h * cfg.ndt
        t = s * cfg.dt
        interior_nodes = baseline[:, nodes] + deltas[:, :, h - 1]  # (G, Nx), physical values at the nodes

        right_vals = torch.tensor([C.u_right_val(A, omega, t) for A, omega in zip(A_list, omega_list)],
                                   dtype=baseline.dtype)
        right_block = right_vals.unsqueeze(1).expand(G, Ntot - i_right)

        # left clamping (indices [0, i_left] included, so it also overwrites
        # the physical value computed at node i_left -- like C.reconstruct)
        left_block = torch.zeros(G, i_left + 1, dtype=baseline.dtype)
        u_full = torch.cat([left_block, interior_nodes[:, 1:], right_block], dim=1)

        if cfg.SMOOTH_ALPHA > 0:
            j0, j1 = i_left + 1, i_right
            lap = u_full[:, j0 - 1:j1 - 1] - 2 * u_full[:, j0:j1] + u_full[:, j0 + 1:j1 + 1]
            smoothed_middle = u_full[:, j0:j1] + cfg.SMOOTH_ALPHA * lap
            u_full = torch.cat([u_full[:, :j0], smoothed_middle, u_full[:, j1:]], dim=1)

        new_states.append(u_full)
        s_list.append(s)

    return new_states, s_list
