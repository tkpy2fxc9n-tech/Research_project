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
from .losses import build_window_torch, reconstruct_torch_general, pde_residual_torch
from ..physics.solver import compute_rest_bias
from ..data.norm import norm_stats_arrays


def pushforward_loss(model, FIELDS, bc_pairs, group_indices, start_n, input_fields,
                      mu_in_t, sd_in_t, mu_out_t, sd_out_t, rest_bias_t, criterion, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    nodes = cfg.nodes
    bc_left_list = [bc_pairs[idx][0] for idx in group_indices]
    bc_right_list = [bc_pairs[idx][1] for idx in group_indices]

    history = []
    for lag in range(cfg.M_BACK, -1, -1):
        m = start_n - lag * cfg.ndt
        arr = np.stack([FIELDS[idx][m] for idx in group_indices], axis=0)
        history.append(torch.tensor(arr, dtype=torch.float32))

    n = start_n
    pred_norm = None
    physics_loss = torch.zeros(())
    for hop in range(cfg.PF_HOPS):
        last = hop == cfg.PF_HOPS - 1
        X = (build_window_torch(history, input_fields, cfg) - mu_in_t) / sd_in_t

        if last:
            # Undetached on purpose (unlike every earlier hop): this is the
            # only transition in the whole PF_HOPS chain still connected to
            # the graph (every prior hop ran under torch.no_grad(), that's
            # the pushforward trick), so it's the only one a PINN residual
            # can be computed on here -- phase-3/PINN residual_torch needs THREE
            # consecutive states with a live gradient path, and only this
            # last one qualifies. See training/bptt.py's rollout_group_tbptt
            # for the full-sequence version, where every hop is live.
            pred_norm = model(X)
            pred_for_state = pred_norm
        else:
            with torch.no_grad():
                pred_for_state = model(X)

        baseline = history[-1]
        new_states, s_list = reconstruct_torch_general(baseline, pred_for_state, bc_left_list, bc_right_list, n,
                                                         mu_out_t, sd_out_t, rest_bias_t, cfg)
        if last:
            baseline_nodes = baseline[:, nodes]
            if cfg.LAMBDA_PHYSICS > 0:
                # history[-2]: the state one ndt before `baseline`, still needed
                # as u_prev for the first triple below (kept around from the
                # window-building loop above -- never dropped since
                # history[cfg.N_FWD:] only trims from the front by N_FWD, and
                # M_BACK>=1 guarantees at least 2 entries survive). Gated on
                # LAMBDA_PHYSICS>0 -- only matters in phase 3 (PINN sweep).
                seq = [history[-2], baseline] + new_states
                physics_terms = [
                    (pde_residual_torch(seq[k - 1], seq[k], seq[k + 1], cfg.ndt * cfg.dt, cfg)[:, cfg.i_left + 1:cfg.i_right] ** 2).mean()
                    for k in range(1, len(seq) - 1)
                ]
                physics_loss = torch.stack(physics_terms).mean()
            else:
                physics_loss = torch.zeros(())
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

    return criterion(pred_norm, target_norm), physics_loss


def evaluate_val_loss(model, FIELDS, bc_pairs, indices_val, input_fields, X_val, y_val,
                       mu_in_t, sd_in_t, mu_out_t, sd_out_t, rest_bias_t, criterion, cfg,
                       rng, valid_starts, lam_pf) -> tuple[float, float, float, float]:
    # VAL_METRIC="loss": the SAME formula run() uses for the training total
    # (data + lam_pf*pushforward + LAMBDA_PHYSICS*physics), evaluated on
    # validation data with no gradient -- so val_history becomes directly
    # comparable to train_history, unlike VAL_METRIC="rollout"
    # (evaluate_val_rollout, a full autoregressive replay in different
    # units). lam_pf is the CALLER's current-epoch ramped value, not
    # recomputed here, so the val total uses the same weight the train
    # total used this same epoch. Returns (total, data, pushforward,
    # physics) so the caller can track val curves per active component,
    # same as the train side's extra_history -- physics_loss is already
    # computed above as a byproduct of pushforward_loss whenever lam_pf>0,
    # just never returned before; phase 3 needs it, unlike phase 1 where
    # LAMBDA_PHYSICS was always 0 anyway.
    model.eval()
    with torch.no_grad():
        data_loss = criterion(model(torch.tensor(X_val)), torch.tensor(y_val))
        if lam_pf > 0:
            group_size = min(cfg.N_PF_GROUPS, len(indices_val))
            group_indices = rng.choice(indices_val, size=group_size, replace=False).tolist()
            start_n = int(rng.choice(valid_starts))
            pf_loss, physics_loss = pushforward_loss(model, FIELDS, bc_pairs, group_indices, start_n, input_fields,
                                                       mu_in_t, sd_in_t, mu_out_t, sd_out_t, rest_bias_t, criterion, cfg)
        else:
            pf_loss = torch.tensor(0.0)
            physics_loss = torch.tensor(0.0)
        total = data_loss + lam_pf * pf_loss + cfg.LAMBDA_PHYSICS * physics_loss
    model.train()
    return float(total.item()), float(data_loss.item()), float(pf_loss.item()), float(physics_loss.item())


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
    history_needed = cfg.M_BACK * cfg.ndt
    last_valid_start = cfg.Nt - cfg.PF_HOPS * cfg.N_FWD * cfg.ndt
    if last_valid_start < history_needed:
        raise ValueError("PF_HOPS too large for Nt/N_FWD/ndt: no valid pushforward start step exists.")
    valid_starts = list(range(history_needed, last_valid_start + 1, cfg.N_FWD * cfg.ndt))

    train_history, val_history, pf_loss_history, physics_loss_history = [], [], [], []
    val_data_history, val_pushforward_history, val_physics_history = [], [], []
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
        epoch_data = epoch_pf = epoch_physics = 0.0
        for _ in range(n_batches_per_epoch):
            X_batch, y_batch = next(data_iter)
            optimizer.zero_grad()
            # Phase 10 noise-injection stabilizer (cfg.NOISE_STD, see
            # training/stabilizers.py) -- was only wired into teacher_forcing.py
            # and bptt.py's batch loops, never here, so every "noise"/"both"
            # p10_* run (all regime=pushforward) silently trained with noise
            # injection inert. Same pattern as bptt.py's own X_in construction.
            X_in = X_batch + cfg.NOISE_STD * torch.randn_like(X_batch) if cfg.NOISE_STD > 0 else X_batch
            data_loss = criterion(model(X_in), y_batch)

            if lam_pf > 0:
                # physics_loss piggybacks on this same rollout rather than a
                # separate call: the last hop's reconstructed state is the
                # only differentiable one available (see pushforward_loss),
                # so LAMBDA_PHYSICS has no effect while lam_pf==0 (PF_WARMUP
                # ramp, or LAMBDA_PF==0) -- there's no rollout to read it from.
                group_size = min(cfg.N_PF_GROUPS, len(indices_train))
                group_indices = rng.choice(indices_train, size=group_size, replace=False).tolist()
                start_n = int(rng.choice(valid_starts))
                pf_loss, physics_loss = pushforward_loss(model, FIELDS, bc_pairs, group_indices, start_n, input_fields,
                                                           mu_in_t, sd_in_t, mu_out_t, sd_out_t, rest_bias_t, criterion, cfg)
            else:
                pf_loss = torch.tensor(0.0)
                physics_loss = torch.tensor(0.0)

            total = data_loss + lam_pf * pf_loss + cfg.LAMBDA_PHYSICS * physics_loss
            total.backward()
            optimizer.step()

            epoch_data += data_loss.item()
            epoch_pf += pf_loss.item()
            epoch_physics += physics_loss.item()
        epoch_data /= n_batches_per_epoch
        epoch_pf /= n_batches_per_epoch
        epoch_physics /= n_batches_per_epoch

        if cfg.VAL_METRIC == "loss":
            val_err, val_data, val_pf, val_physics = evaluate_val_loss(
                model, FIELDS, bc_pairs, indices_val, input_fields, X_val, y_val,
                mu_in_t, sd_in_t, mu_out_t, sd_out_t, rest_bias_t, criterion, cfg,
                rng, valid_starts, lam_pf)
            val_data_history.append(val_data)
            val_pushforward_history.append(val_pf)
            val_physics_history.append(val_physics)
        else:
            val_err = evaluate_val_rollout(model, FIELDS, bc_pairs, indices_val, input_fields, norm_stats,
                                            INPUTS, OUTPUTS, cfg)

        train_history.append(epoch_data)
        val_history.append(val_err)
        pf_loss_history.append(epoch_pf)
        physics_loss_history.append(epoch_physics)

        val_label = "L2 rel error (val)" if cfg.VAL_METRIC == "rollout" else "comparable loss (val)"
        print(f"Epoch {epoch:4d}/{cfg.N_EPOCHS}  --  data: {epoch_data:.4f}  |  "
              f"pushforward: {epoch_pf:.4f}  |  physics: {epoch_physics:.4f}  |  {val_label}: {val_err:.4f}")

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
    extra_history = {"pushforward": pf_loss_history, "physics": physics_loss_history}
    if cfg.VAL_METRIC == "loss":
        extra_history["val_data"] = val_data_history
        extra_history["val_pushforward"] = val_pushforward_history
        extra_history["val_physics"] = val_physics_history
    return TrainResult(train_history, val_history, best_val, train_time_s, n_params,
                        extra_history=extra_history)
