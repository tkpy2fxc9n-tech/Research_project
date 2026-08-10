# Differentiable (never falling back to numpy) torch port of the
# generalized-boundary-condition physics validated in physics/solver.py and
# physics/waves.py, plus the PDE residual (H6/PINN term) and the composite
# loss combination shared by the bptt and pushforward regimes. No in-place
# tensor ops: every step rebuilds a new tensor, so the gradient path is never
# disturbed. Ported from Tests/Model_in_tests/full_rollout_training_conv1d/
# training/code/rollout_torch.py.
from __future__ import annotations

import torch

from ..physics.waves import bc_value


def uxx_field_torch(u: torch.Tensor, cfg) -> torch.Tensor:
    # u : (G, Ntot).
    i_left, i_right = cfg.i_left, cfg.i_right
    G = u.shape[0]
    interior = (u[:, i_left - 1:i_right] - 2 * u[:, i_left:i_right + 1]
                + u[:, i_left + 1:i_right + 2]) / cfg.dx ** 2
    left_pad = torch.zeros(G, i_left, dtype=u.dtype)
    right_pad = torch.zeros(G, cfg.Ntot - (i_right + 1), dtype=u.dtype)
    return torch.cat([left_pad, interior, right_pad], dim=1)


def build_window_torch(history: list[torch.Tensor], input_fields: list[str], cfg) -> torch.Tensor:
    # history: list of M_BACK+1 tensors (G, Ntot), oldest to most recent.
    # Returns X of shape (G*Nx, n_features), same column order as
    # data.windows.make_feature_columns (lag, then k, then field).
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


def apply_boundary_torch(u_row: torch.Tensor, side: str, bc_type: str, value: float, cfg) -> torch.Tensor:
    # u_row: 1D tensor (Ntot,). Returns a NEW tensor (no in-place ops).
    i_left, i_right, SS, dx = cfg.i_left, cfg.i_right, cfg.SS, cfg.dx
    Ntot = cfg.Ntot
    if side == "left":
        if bc_type == "dirichlet":
            left_part = torch.full((i_left + 1,), float(value), dtype=u_row.dtype)
            return torch.cat([left_part, u_row[i_left + 1:]])
        else:
            mirrored = torch.stack([u_row[i_left + k] - 2 * k * dx * value for k in range(1, SS + 1)])
            ghost = torch.flip(mirrored, dims=[0])
            return torch.cat([ghost, u_row[i_left:]])
    else:
        if bc_type == "dirichlet":
            right_part = torch.full((Ntot - i_right,), float(value), dtype=u_row.dtype)
            return torch.cat([u_row[:i_right], right_part])
        else:
            ghost = torch.stack([u_row[i_right - k] + 2 * k * dx * value for k in range(1, SS)])
            return torch.cat([u_row[:i_right + 1], ghost])


def apply_boundary_conditions_torch(u: torch.Tensor, t: float, bc_left_list, bc_right_list, cfg) -> torch.Tensor:
    # u: (G, Ntot). Each row can have a different BC type/family.
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
                               rest_bias_t: torch.Tensor | None, cfg) -> tuple[list[torch.Tensor], list[int]]:
    # baseline: (G, Ntot), state BEFORE this hop -- fixed for all h.
    # pred_norm: (G*Nx, N_FWD), raw network output for this hop.
    nodes = cfg.nodes
    G = baseline.shape[0]
    Nx = len(nodes)

    pred_norm = pred_norm.reshape(G, Nx, cfg.N_FWD)
    deltas = pred_norm * sd_out_t + mu_out_t
    if rest_bias_t is not None:
        deltas = deltas - rest_bias_t

    new_states, s_list = [], []
    for h in range(1, cfg.N_FWD + 1):
        s = n_curr + h * cfg.ndt
        t = s * cfg.dt
        interior_nodes = baseline[:, nodes] + deltas[:, :, h - 1]  # (G, Nx)

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


def utt_uxx_torch(u_prev: torch.Tensor, u_curr: torch.Tensor, u_next: torch.Tensor,
                   dt_eff: float, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    # u_prev/u_curr/u_next: (G, Ntot), three consecutive snapshots dt_eff
    # apart. dt_eff = cfg.dt for raw ground-truth steps, cfg.ndt*cfg.dt for
    # ndt-spaced rollout-hop steps.
    i_left, i_right = cfg.i_left, cfg.i_right
    G = u_curr.shape[0]
    interior = (u_next[:, i_left:i_right + 1] - 2 * u_curr[:, i_left:i_right + 1]
                + u_prev[:, i_left:i_right + 1]) / dt_eff ** 2
    left_pad = torch.zeros(G, i_left, dtype=u_curr.dtype)
    right_pad = torch.zeros(G, cfg.Ntot - (i_right + 1), dtype=u_curr.dtype)
    u_tt = torch.cat([left_pad, interior, right_pad], dim=1)
    u_xx = uxx_field_torch(u_curr, cfg)
    return u_tt, u_xx


def pde_residual_torch(u_prev: torch.Tensor, u_curr: torch.Tensor, u_next: torch.Tensor,
                        dt_eff: float, cfg) -> torch.Tensor:
    # u_tt - (E/rho)*u_xx : ~0 wherever the triple is a genuine,
    # physically-consistent wave-equation solution. Callers must slice to
    # i_left+1:i_right (not cfg.nodes) before reducing to a scalar: a
    # Dirichlet BC can overwrite u at i_left/i_right directly, making u_tt
    # there reflect the forcing function, not the PDE.
    u_tt, u_xx = utt_uxx_torch(u_prev, u_curr, u_next, dt_eff, cfg)
    return u_tt - (cfg.E / cfg.rho) * u_xx


def combine_losses(cfg, *, rollout=None, physics=None, data=None) -> torch.Tensor:
    # Weighted sum of whichever terms a regime actually computed -- the same
    # LAMBDA_ROLLOUT/LAMBDA_PHYSICS/LAMBDA_DATA combination used by both the
    # bptt and pushforward regimes (LAMBDA_PHYSICS=0 disables the H6/PINN
    # term without the caller needing a separate code path).
    total = torch.zeros(())
    if rollout is not None:
        total = total + cfg.LAMBDA_ROLLOUT * rollout
    if physics is not None and cfg.LAMBDA_PHYSICS > 0:
        total = total + cfg.LAMBDA_PHYSICS * physics
    if data is not None:
        total = total + cfg.LAMBDA_DATA * data
    return total
