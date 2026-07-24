# Torch port (differentiable, never falling back to numpy) of the
# generalized-boundary-condition physics validated in physics.py/waves.py
# (apply_boundary_conditions, run_fd_simulation_general, reconstruct_general).
# No in-place operations: every step rebuilds a new tensor, to never disturb
# the gradient computation path.
#
# uxx_field_torch/build_window_torch are byte-for-byte the same as in the
# other full_rollout_training projects (confirmed boundary-condition-agnostic:
# they only compute interior second differences / assemble feature windows
# from whatever ghost bands are already filled). reconstruct_torch_general is
# the new piece: unlike the fixed left=0/right=Gaussian-pulse case, each
# simulation in a training group can have its OWN boundary type (Dirichlet or
# Neumann) on each end, so the boundary fill can't be vectorized uniformly
# across the batch the way the old reconstruct_torch did -- it's applied one
# row (one simulation) at a time, each row still fully differentiable.
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from config import Config
from waves import bc_value


def uxx_field_torch(u: torch.Tensor, cfg: "Config") -> torch.Tensor:
    # u : (G, Ntot). Direct port of physics.uxx_field.
    i_left, i_right = cfg.i_left, cfg.i_right
    G = u.shape[0]
    interior = (u[:, i_left - 1:i_right] - 2 * u[:, i_left:i_right + 1]
                + u[:, i_left + 1:i_right + 2]) / cfg.dx ** 2
    left_pad = torch.zeros(G, i_left, dtype=u.dtype)
    right_pad = torch.zeros(G, cfg.Ntot - (i_right + 1), dtype=u.dtype)
    return torch.cat([left_pad, interior, right_pad], dim=1)


def build_window_torch(history: list[torch.Tensor], input_fields: list[str], cfg: "Config") -> torch.Tensor:
    # history: list of M_BACK+1 tensors (G, Ntot), from the oldest
    # (history[0]) to the most recent (history[-1] = current state n).
    # Returns X of shape (G*Nx, n_features), same column order as
    # physics.make_feature_columns (lag, then k, then field).
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


def apply_boundary_torch(u_row: torch.Tensor, side: str, bc_type: str, value: float, cfg: "Config") -> torch.Tensor:
    # u_row: 1D tensor (Ntot,). Returns a NEW tensor (no in-place ops) --
    # exact torch counterpart of waves.apply_boundary, same index conventions
    # (Dirichlet overwrites the boundary node too; Neumann leaves it alone
    # since the interior update/network prediction already computed it
    # correctly once its one ghost neighbor is mirrored).
    i_left, i_right, SS, dx = cfg.i_left, cfg.i_right, cfg.SS, cfg.dx
    Ntot = cfg.Ntot
    if side == "left":
        if bc_type == "dirichlet":
            left_part = torch.full((i_left + 1,), float(value), dtype=u_row.dtype)
            return torch.cat([left_part, u_row[i_left + 1:]])
        else:
            mirrored = torch.stack([u_row[i_left + k] - 2 * k * dx * value for k in range(1, SS + 1)])
            ghost = torch.flip(mirrored, dims=[0])  # ascending indices 0..i_left-1
            return torch.cat([ghost, u_row[i_left:]])
    else:
        if bc_type == "dirichlet":
            right_part = torch.full((Ntot - i_right,), float(value), dtype=u_row.dtype)
            return torch.cat([u_row[:i_right], right_part])
        else:
            ghost = torch.stack([u_row[i_right - k] + 2 * k * dx * value for k in range(1, SS)])
            return torch.cat([u_row[:i_right + 1], ghost])


def apply_boundary_conditions_torch(u: torch.Tensor, t: float, bc_left_list, bc_right_list,
                                     cfg: "Config") -> torch.Tensor:
    # u: (G, Ntot). Each row can have a different BC type/family, so this
    # loops over the (small, group-sized) batch rather than vectorizing.
    rows = []
    for g in range(u.shape[0]):
        row = u[g]
        left_val = bc_value(bc_left_list[g], t)
        right_val = bc_value(bc_right_list[g], t)
        row = apply_boundary_torch(row, "left", bc_left_list[g][0], left_val, cfg)
        row = apply_boundary_torch(row, "right", bc_right_list[g][0], right_val, cfg)
        rows.append(row)
    return torch.stack(rows, dim=0)


def reconstruct_torch_general(baseline: torch.Tensor, pred_norm: torch.Tensor,
                               bc_left_list, bc_right_list, n_curr: int,
                               mu_out_t: torch.Tensor, sd_out_t: torch.Tensor,
                               biais_repos_t: torch.Tensor | None, cfg: "Config") -> tuple[list[torch.Tensor], list[int]]:
    # baseline: (G, Ntot), state BEFORE this hop -- fixed for all h (don't
    # chain h=1 into h=2, like physics.reconstruct_general).
    # pred_norm: (G*Nx, N_FWD), raw network output for this hop.
    # Returns (list of N_FWD tensors (G, Ntot), list of corresponding time indices s).
    nodes = cfg.nodes
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
        interior_nodes = baseline[:, nodes] + deltas[:, :, h - 1]  # (G, Nx)

        # Place the predicted node values at indices [i_left, i_right) --
        # everything else is a placeholder, immediately replaced by
        # apply_boundary_conditions_torch below. Unlike the old
        # Gaussian-only reconstruct_torch, we can't drop the i_left column
        # here: some rows may be Neumann on the left, which needs that
        # network-predicted value kept (only Dirichlet rows discard it).
        pad_left = torch.zeros(G, cfg.i_left, dtype=baseline.dtype)
        pad_right = torch.zeros(G, cfg.Ntot - cfg.i_right, dtype=baseline.dtype)
        u_full = torch.cat([pad_left, interior_nodes, pad_right], dim=1)

        u_full = apply_boundary_conditions_torch(u_full, t, bc_left_list, bc_right_list, cfg)

        if cfg.SMOOTH_ALPHA > 0:
            j0, j1 = cfg.i_left + 1, cfg.i_right
            lap = u_full[:, j0 - 1:j1 - 1] - 2 * u_full[:, j0:j1] + u_full[:, j0 + 1:j1 + 1]
            smoothed_middle = u_full[:, j0:j1] + cfg.SMOOTH_ALPHA * lap
            u_full = torch.cat([u_full[:, :j0], smoothed_middle, u_full[:, j1:]], dim=1)

        new_states.append(u_full)
        s_list.append(s)

    return new_states, s_list
