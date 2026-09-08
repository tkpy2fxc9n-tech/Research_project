# TBPTT composite training: rolls out complete trajectories hop by hop
# without ever resetting to ground truth, combining rollout + physics + data
# loss terms into one weighted scalar every TBPTT_HOPS hops (the gradient
# thread is cut there, the rollout itself stays continuous). Ported from
# Tests/Model_in_tests/full_rollout_training_conv1d/training/code/train.py.
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import TrainResult
from .losses import build_window_torch, reconstruct_torch_general, pde_residual_torch, combine_losses
from ..physics.solver import compute_rest_bias, autoregressive_rollout
from ..data.norm import norm_stats_arrays


def make_epoch_groups(indices: list[int], group_size: int, rng: np.random.Generator) -> list[list[int]]:
    order = rng.permutation(len(indices))
    shuffled = [indices[i] for i in order]
    return [shuffled[i:i + group_size] for i in range(0, len(shuffled), group_size)]


def infinite_batches(loader):
    # Re-calls iter(loader) on every wrap-around, which reshuffles (loader
    # has shuffle=True) -- unlike itertools.cycle, which would replay the
    # first pass forever.
    while True:
        for batch in loader:
            yield batch


def rollout_group_tbptt(model, group_indices, FIELDS, bc_pairs, input_fields,
                         mu_in_t, sd_in_t, mu_out_t, sd_out_t, rest_bias_t,
                         criterion, optimizer, cfg, data_iter) -> tuple[float, float, float, int]:
    nodes = cfg.nodes
    G, Nx = len(group_indices), len(nodes)
    bc_left_list = [bc_pairs[idx][0] for idx in group_indices]
    bc_right_list = [bc_pairs[idx][1] for idx in group_indices]
    history_needed = cfg.M_BACK * cfg.ndt

    history = []
    for lag in range(cfg.M_BACK, -1, -1):
        m = history_needed - lag * cfg.ndt
        arr = np.stack([FIELDS[idx][m] for idx in group_indices], axis=0)
        history.append(torch.tensor(arr, dtype=torch.float32))

    hops = list(range(history_needed, cfg.Nt - cfg.N_FWD * cfg.ndt + 1, cfg.N_FWD * cfg.ndt))
    dt_eff = cfg.ndt * cfg.dt
    total_rollout_log = total_physics_log = total_data_log = 0.0
    n_updates = 0
    segment_rollout, segment_physics, segment_hops = torch.zeros(()), torch.zeros(()), 0

    for i, n in enumerate(hops):
        X = (build_window_torch(history, input_fields, cfg) - mu_in_t) / sd_in_t
        pred_norm = model(X)  # (G*Nx, N_FWD)

        baseline = history[-1]
        new_states, s_list = reconstruct_torch_general(baseline, pred_norm, bc_left_list, bc_right_list, n,
                                                         mu_out_t, sd_out_t, rest_bias_t, cfg)

        baseline_nodes = baseline[:, nodes]
        target_list = [
            torch.tensor(np.stack([FIELDS[idx][s][nodes] for idx in group_indices], axis=0), dtype=torch.float32)
            - baseline_nodes
            for s in s_list
        ]
        target = torch.stack(target_list, dim=-1)  # (G, Nx, N_FWD)
        target_norm = ((target - mu_out_t) / sd_out_t).reshape(G * Nx, cfg.N_FWD)

        rollout_loss_hop = criterion(pred_norm, target_norm)

        if cfg.LAMBDA_PHYSICS > 0:
            # Physics residual: every consecutive triple in [history[-2],
            # history[-1]] + new_states is dt_eff apart, all from tensors
            # already computed above (no extra forward pass). Sliced to
            # i_left+1:i_right (not cfg.nodes): a Dirichlet BC can overwrite u
            # at i_left/i_right directly, which would make u_tt there reflect
            # the forcing function, not the PDE. Gated on LAMBDA_PHYSICS>0 --
            # this residual only ever matters in phase 3 (PINN sweep); computing
            # it unconditionally wasted compute and populated extra_history/
            # plots with a curve that never affected training everywhere else.
            seq = [history[-2], history[-1]] + new_states
            physics_terms = [
                (pde_residual_torch(seq[k - 1], seq[k], seq[k + 1], dt_eff, cfg)[:, cfg.i_left + 1:cfg.i_right] ** 2).mean()
                for k in range(1, len(seq) - 1)
            ]
            physics_loss_hop = torch.stack(physics_terms).mean()
        else:
            physics_loss_hop = torch.zeros(())

        segment_rollout = segment_rollout + rollout_loss_hop
        segment_physics = segment_physics + physics_loss_hop
        segment_hops += 1
        total_rollout_log += rollout_loss_hop.item()
        total_physics_log += physics_loss_hop.item()

        history = history[cfg.N_FWD:] + new_states

        if segment_hops == cfg.TBPTT_HOPS or i == len(hops) - 1:
            X_batch, y_batch = next(data_iter)
            X_in = X_batch + cfg.NOISE_STD * torch.randn_like(X_batch) if cfg.NOISE_STD > 0 else X_batch
            data_loss = criterion(model(X_in), y_batch)

            avg_rollout = segment_rollout / segment_hops
            avg_physics = segment_physics / segment_hops
            total = combine_losses(cfg, rollout=avg_rollout, physics=avg_physics, data=data_loss)

            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            n_updates += 1
            total_data_log += data_loss.item()
            history = [h.detach() for h in history]
            segment_rollout, segment_physics, segment_hops = torch.zeros(()), torch.zeros(()), 0

    return (total_rollout_log / len(hops), total_physics_log / len(hops),
            total_data_log / n_updates, n_updates)


def evaluate_val_rollout(model, FIELDS, bc_pairs, indices_val, input_fields, norm_stats, INPUTS, OUTPUTS, cfg) -> float:
    from ..evaluate.metrics import l2_rel

    mu_in, sd_in, mu_out, sd_out = norm_stats_arrays(norm_stats, INPUTS, OUTPUTS)
    rest_bias = compute_rest_bias(model, mu_in, sd_in, mu_out, sd_out, cfg)

    model.eval()
    errs = []
    with torch.no_grad():
        for idx in indices_val:
            bc_left, bc_right = bc_pairs[idx]
            U_reel = FIELDS[idx]
            U_pred = autoregressive_rollout(model, U_reel, input_fields, mu_in, sd_in, mu_out, sd_out,
                                             rest_bias, bc_left, bc_right, cfg)
            errs.append(l2_rel(U_pred[:, cfg.nodes], U_reel[:, cfg.nodes]))
    model.train()
    return float(np.mean(errs))


def evaluate_val_loss(model, FIELDS, bc_pairs, indices_val, input_fields, X_val, y_val,
                       mu_in_t, sd_in_t, mu_out_t, sd_out_t, rest_bias_t, criterion, cfg, rng) -> tuple[float, float, float]:
    # VAL_METRIC="loss": the same combine_losses(rollout, physics, data)
    # formula run() uses for the training total, evaluated on a group of
    # validation trajectories with no gradient (mirrors rollout_group_tbptt's
    # math, minus the TBPTT segmenting/optimizer step) -- so val_history
    # becomes directly comparable to train_history, unlike VAL_METRIC=
    # "rollout" (evaluate_val_rollout, a full autoregressive replay in
    # different units). Returns (total, rollout, data) so the caller can
    # track val curves per active component, same as the train side's
    # extra_history -- physics is left out here since LAMBDA_PHYSICS is 0
    # for every phase-1 run (it only becomes relevant in phase 3).
    nodes = cfg.nodes
    group_size = min(cfg.GROUP_SIZE, len(indices_val))
    group_indices = rng.choice(indices_val, size=group_size, replace=False).tolist()
    bc_left_list = [bc_pairs[idx][0] for idx in group_indices]
    bc_right_list = [bc_pairs[idx][1] for idx in group_indices]
    history_needed = cfg.M_BACK * cfg.ndt

    model.eval()
    with torch.no_grad():
        history = []
        for lag in range(cfg.M_BACK, -1, -1):
            m = history_needed - lag * cfg.ndt
            arr = np.stack([FIELDS[idx][m] for idx in group_indices], axis=0)
            history.append(torch.tensor(arr, dtype=torch.float32))

        hops = list(range(history_needed, cfg.Nt - cfg.N_FWD * cfg.ndt + 1, cfg.N_FWD * cfg.ndt))
        rollout_sum = physics_sum = 0.0
        for n in hops:
            X = (build_window_torch(history, input_fields, cfg) - mu_in_t) / sd_in_t
            pred_norm = model(X)

            baseline = history[-1]
            new_states, s_list = reconstruct_torch_general(baseline, pred_norm, bc_left_list, bc_right_list, n,
                                                             mu_out_t, sd_out_t, rest_bias_t, cfg)

            baseline_nodes = baseline[:, nodes]
            target_list = [
                torch.tensor(np.stack([FIELDS[idx][s][nodes] for idx in group_indices], axis=0), dtype=torch.float32)
                - baseline_nodes
                for s in s_list
            ]
            target = torch.stack(target_list, dim=-1)
            G, Nx = len(group_indices), len(nodes)
            target_norm = ((target - mu_out_t) / sd_out_t).reshape(G * Nx, cfg.N_FWD)
            rollout_sum += criterion(pred_norm, target_norm).item()

            if cfg.LAMBDA_PHYSICS > 0:
                # Gated same as rollout_group_tbptt's own physics residual --
                # only matters in phase 3, see the comment there.
                seq = [history[-2], history[-1]] + new_states
                physics_terms = [
                    (pde_residual_torch(seq[k - 1], seq[k], seq[k + 1], cfg.ndt * cfg.dt, cfg)[:, cfg.i_left + 1:cfg.i_right] ** 2).mean()
                    for k in range(1, len(seq) - 1)
                ]
                physics_sum += torch.stack(physics_terms).mean().item()

            history = history[cfg.N_FWD:] + new_states

        avg_rollout = rollout_sum / len(hops)
        avg_physics = physics_sum / len(hops)
        data_loss = criterion(model(torch.tensor(X_val)), torch.tensor(y_val)).item()
        total = combine_losses(cfg, rollout=torch.tensor(avg_rollout), physics=torch.tensor(avg_physics),
                                data=torch.tensor(data_loss)).item()
    model.train()
    return total, avg_rollout, data_loss


def run(model, FIELDS, bc_pairs, indices_train, indices_val, input_fields,
        norm_stats, INPUTS, OUTPUTS, cfg, train_loader, X_val, y_val,
        model_path: Path, patience: int | None = None) -> TrainResult:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    mu_in, sd_in, mu_out, sd_out = norm_stats_arrays(norm_stats, INPUTS, OUTPUTS)
    mu_in_t, sd_in_t = torch.tensor(mu_in), torch.tensor(sd_in)
    mu_out_t, sd_out_t = torch.tensor(mu_out), torch.tensor(sd_out)
    data_iter = infinite_batches(train_loader)

    rng = np.random.default_rng(cfg.SEED)
    train_history, val_history = [], []
    data_loss_history, physics_loss_history, rollout_loss_history = [], [], []
    val_data_history, val_rollout_history = [], []
    best_val = float("inf")
    epochs_without_improvement = 0

    t0 = time.perf_counter()
    for epoch in range(1, cfg.N_EPOCHS + 1):
        rest_bias_t = torch.tensor(compute_rest_bias(model, mu_in, sd_in, mu_out, sd_out, cfg))
        groups = make_epoch_groups(indices_train, cfg.GROUP_SIZE, rng)

        model.train()
        t_epoch0 = time.perf_counter()
        epoch_rollout = epoch_physics = epoch_data = 0.0
        for i, group_indices in enumerate(groups):
            avg_rollout, avg_physics, avg_data, n_updates = rollout_group_tbptt(
                model, group_indices, FIELDS, bc_pairs, input_fields,
                mu_in_t, sd_in_t, mu_out_t, sd_out_t, rest_bias_t,
                criterion, optimizer, cfg, data_iter)
            epoch_rollout += avg_rollout
            epoch_physics += avg_physics
            epoch_data += avg_data
        epoch_rollout /= len(groups)
        epoch_physics /= len(groups)
        epoch_data /= len(groups)
        epoch_loss = combine_losses(cfg, rollout=torch.tensor(epoch_rollout), physics=torch.tensor(epoch_physics),
                                     data=torch.tensor(epoch_data)).item()

        if cfg.VAL_METRIC == "loss":
            val_err, val_rollout, val_data = evaluate_val_loss(
                model, FIELDS, bc_pairs, indices_val, input_fields, X_val, y_val,
                mu_in_t, sd_in_t, mu_out_t, sd_out_t, rest_bias_t, criterion, cfg, rng)
            val_data_history.append(val_data)
            val_rollout_history.append(val_rollout)
        else:
            val_err = evaluate_val_rollout(model, FIELDS, bc_pairs, indices_val, input_fields, norm_stats,
                                            INPUTS, OUTPUTS, cfg)
        train_history.append(epoch_loss)
        val_history.append(val_err)
        rollout_loss_history.append(epoch_rollout)
        physics_loss_history.append(epoch_physics)
        data_loss_history.append(epoch_data)

        val_label = "L2 rel error (val)" if cfg.VAL_METRIC == "rollout" else "comparable loss (val)"
        print(f"Epoch {epoch:4d}/{cfg.N_EPOCHS} -- combined loss (train): {epoch_loss:.4f}  |  "
              f"{val_label}: {val_err:.4f} -- {time.perf_counter()-t_epoch0:.1f}s")

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
    best_val_label = "L2 rel error (val)" if cfg.VAL_METRIC == "rollout" else "comparable loss (val)"
    print(f"Best model reloaded -- minimum {best_val_label}: {best_val:.6f}")

    n_params = sum(p.numel() for p in model.parameters())
    extra_history = {"data": data_loss_history, "physics": physics_loss_history,
                      "rollout": rollout_loss_history}
    if cfg.VAL_METRIC == "loss":
        extra_history["val_data"] = val_data_history
        extra_history["val_rollout"] = val_rollout_history
    return TrainResult(train_history, val_history, best_val, train_time_s, n_params,
                        extra_history=extra_history)
