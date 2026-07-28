# Torch port (differentiable, never falling back to numpy) of the physics
# validated in physics.py/waves.py (apply_boundary_conditions,
# reconstruct_general). No in-place operations: every step rebuilds a new
# tensor, to never disturb the gradient computation path. This is what the
# TBPTT training loop (train.py) runs at every hop.
#
# Simpler than full_rollout_training_conv1d's rollout_torch.py: the old
# per-node windowing (with its (G, Nx, n_features) -> (G*Nx, n_features)
# flatten/reshape) is gone entirely -- build_field_history_torch just stacks
# the history tensors' interior nodes into a channels-first (G, n_lags, Nx)
# tensor, and the model's output (G, N_FWD, Nx) is used as-is.
# apply_boundary_torch/apply_boundary_conditions_torch are unchanged from
# the reference project (per-row, since each simulation in a training group
# can have its own BC type on each end).
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from config import Config
from waves import bc_value


def build_field_history_torch(history: list[torch.Tensor], cfg: "Config") -> torch.Tensor:
    # history: list of at least M_BACK tensors (G, Ntot), from the oldest
    # (history[0]) to the most recent (history[-1] = current state n).
    # Returns X of shape (G, n_lags, Nx), channel order = lag 0 (current)
    # first -- same convention as physics.build_field_history's m_list
    # ([n, n-ndt, ...]).
    nodes = cfg.nodes
    chans = [history[-(lag + 1)][:, nodes] for lag in range(cfg.M_BACK)]  # each (G, Nx)
    return torch.stack(chans, dim=1)


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
    # pred_norm: (G, N_FWD, Nx), raw network output for this hop.
    # biais_repos_t: (N_FWD, Nx) full field (see physics.biais_repos).
    # Returns (list of N_FWD tensors (G, Ntot), list of corresponding time indices s).
    nodes = cfg.nodes
    G = baseline.shape[0]

    deltas = pred_norm * sd_out_t[None, :, None] + mu_out_t[None, :, None]
    if biais_repos_t is not None:
        deltas = deltas - biais_repos_t[None]

    new_states, s_list = [], []
    for h in range(1, cfg.N_FWD + 1):
        s = n_curr + h * cfg.ndt
        t = s * cfg.dt
        interior_nodes = baseline[:, nodes] + deltas[:, h - 1, :]  # (G, Nx)

        # Place the predicted node values at indices [i_left, i_right) --
        # everything else is a placeholder, immediately replaced by
        # apply_boundary_conditions_torch below. The i_left column must be
        # kept: rows with a Neumann left end need that network-predicted
        # value (only Dirichlet rows discard it).
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
