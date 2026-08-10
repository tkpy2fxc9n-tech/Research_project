# The pushforward trick (Brandstetter et al. 2022): PF_HOPS autoregressive
# hops chained together, each hop fed by its OWN previous prediction (not
# ground truth) -- gradient is detached on every hop except the last, so
# training sees a realistic, slightly-off-distribution input without paying
# for backprop through the whole chain. Re-extracted and generalized from
# Archives/Archives/Useless/solver_gnn/training/code/train.py's
# `pushforward_loss_gnn` (removed from every currently-used project's
# commun.py, GNN-only until now): the GNN version built its window with a
# per-node `build_node_features` (position/time/A/omega); this version
# reuses the same stencil `build_window_torch`/`reconstruct_torch_general`
# every other regime uses, batched over a GROUP of trajectories exactly like
# training/bptt.py, not over graph nodes.
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import TrainResult
from .bptt import infinite_batches, evaluate_val_rollout
from .losses import build_window_torch, reconstruct_torch_general
from ..physics.solver import compute_rest_bias
from ..data.norm import norm_stats_arrays


def pushforward_loss(model, FIELDS, bc_pairs, group_indices, start_n, input_fields,
                      mu_in_t, sd_in_t, mu_out_t, sd_out_t, rest_bias_t, criterion, cfg) -> torch.Tensor:
    nodes = cfg.nodes
    bc_left_list = [bc_pairs[idx][0] for idx in group_indices]
    bc_right_list = [bc_pairs[idx][1] for idx in group_indices]
    history_needed = cfg.M_BACK * cfg.ndt

    history = []
    for lag in range(cfg.M_BACK, -1, -1):
        m = start_n - lag * cfg.ndt
        arr = np.stack([FIELDS[idx][m] for idx in group_indices], axis=0)
        history.append(torch.tensor(arr, dtype=torch.float32))

    n = start_n
    pred_norm = None
    for hop in range(cfg.PF_HOPS):
        last = hop == cfg.PF_HOPS - 1
        X = (build_window_torch(history, input_fields, cfg) - mu_in_t) / sd_in_t

        if last:
            pred_norm = model(X)
            pred_for_state = pred_norm.detach()
        else:
            with torch.no_grad():
                pred_for_state = model(X)

        baseline = history[-1]
        new_states, s_list = reconstruct_torch_general(baseline, pred_for_state, bc_left_list, bc_right_list, n,
                                                         mu_out_t, sd_out_t, rest_bias_t, cfg)
        if last:
            baseline_nodes = baseline[:, nodes]
        history = history[cfg.N_FWD:] + new_states
        n = n + cfg.N_FWD * cfg.ndt

    # True targets, ground-truth trajectory indexed by real simulation time
    # (well defined regardless of whether the self-rolled baseline matches
    # ground truth), relative to the SELF-ROLLED baseline entering the last
    # hop -- not the ground-truth baseline. This is the pushforward trick:
    # the network learns to correct from a realistic, imperfect input.
    n_before_last_hop = n - cfg.N_FWD * cfg.ndt
    target_list = [
        torch.tensor(np.stack([FIELDS[idx][n_before_last_hop + h * cfg.ndt][nodes] for idx in group_indices], axis=0),
                      dtype=torch.float32) - baseline_nodes
        for h in range(1, cfg.N_FWD + 1)
    ]
    target = torch.stack(target_list, dim=-1)  # (G, Nx, N_FWD)
    G, Nx = len(group_indices), len(nodes)
    target_norm = ((target - mu_out_t) / sd_out_t).reshape(G * Nx, cfg.N_FWD)

    return criterion(pred_norm, target_norm)


def run(model, FIELDS, bc_pairs, indices_train, indices_val, input_fields,
        norm_stats, INPUTS, OUTPUTS, cfg, train_loader, model_path: Path, patience: int | None = None) -> TrainResult:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    mu_in, sd_in, mu_out, sd_out = norm_stats_arrays(norm_stats, INPUTS, OUTPUTS)
    mu_in_t, sd_in_t = torch.tensor(mu_in), torch.tensor(sd_in)
    mu_out_t, sd_out_t = torch.tensor(mu_out), torch.tensor(sd_out)
    data_iter = infinite_batches(train_loader)

    rng = np.random.default_rng(cfg.SEED)
    history_needed = cfg.M_BACK * cfg.ndt
    last_valid_start = cfg.Nt - cfg.PF_HOPS * cfg.N_FWD * cfg.ndt
    if last_valid_start < history_needed:
        raise ValueError("PF_HOPS too large for Nt/N_FWD/ndt: no valid pushforward start step exists.")
    valid_starts = list(range(history_needed, last_valid_start + 1, cfg.N_FWD * cfg.ndt))

    train_history, val_history, pf_loss_history = [], [], []
    best_val = float("inf")
    epochs_without_improvement = 0
    n_batches_per_epoch = max(1, len(train_loader))

    t0 = time.perf_counter()
    for epoch in range(1, cfg.N_EPOCHS + 1):
        rest_bias_t = torch.tensor(compute_rest_bias(model, mu_in, sd_in, mu_out, sd_out, cfg))
        # The model is near-zero at the start of training -- ramp the
        # pushforward weight in linearly over PF_WARMUP epochs instead of
        # applying it at full strength from epoch 1.
        lam_pf = cfg.LAMBDA_PF * (min(1.0, epoch / cfg.PF_WARMUP) if cfg.PF_WARMUP > 0 else 1.0)

        model.train()
        epoch_data = epoch_pf = 0.0
        for _ in range(n_batches_per_epoch):
            X_batch, y_batch = next(data_iter)
            optimizer.zero_grad()
            data_loss = criterion(model(X_batch), y_batch)

            if lam_pf > 0:
                group_size = min(cfg.N_PF_GROUPS, len(indices_train))
                group_indices = rng.choice(indices_train, size=group_size, replace=False).tolist()
                start_n = int(rng.choice(valid_starts))
                pf_loss = pushforward_loss(model, FIELDS, bc_pairs, group_indices, start_n, input_fields,
                                            mu_in_t, sd_in_t, mu_out_t, sd_out_t, rest_bias_t, criterion, cfg)
            else:
                pf_loss = torch.tensor(0.0)

            total = data_loss + lam_pf * pf_loss
            total.backward()
            optimizer.step()

            epoch_data += data_loss.item()
            epoch_pf += pf_loss.item()
        epoch_data /= n_batches_per_epoch
        epoch_pf /= n_batches_per_epoch

        val_err = evaluate_val_rollout(model, FIELDS, bc_pairs, indices_val, input_fields, norm_stats,
                                        INPUTS, OUTPUTS, cfg)

        train_history.append(epoch_data)
        val_history.append(val_err)
        pf_loss_history.append(epoch_pf)

        print(f"Epoch {epoch:4d}/{cfg.N_EPOCHS}  --  data: {epoch_data:.4f}  |  "
              f"pushforward: {epoch_pf:.4f}  |  L2 rel error (val): {val_err:.4f}")

        if val_err < best_val:
            best_val = val_err
            torch.save(model.state_dict(), model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience is not None and epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}: val loss hasn't improved for "
                  f"{patience} epochs (best={best_val:.6f}).")
            break

    train_time_s = time.perf_counter() - t0
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Best model reloaded -- minimum L2 rel error (val): {best_val:.6f}")

    n_params = sum(p.numel() for p in model.parameters())
    return TrainResult(train_history, val_history, best_val, train_time_s, n_params,
                        extra_history={"pushforward": pf_loss_history})
