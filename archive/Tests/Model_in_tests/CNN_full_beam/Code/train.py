# "Full rollout" (TBPTT) training loop -- the training scheme this project
# ACTUALLY uses (called from main.py, unlike the reference project where the
# equivalent file was ported but never called): for each group of
# simulations, rolls out the complete trajectory without ever resetting to
# ground truth, with a weight correction every `tbptt_hops` hops (the
# gradient thread is cut at that point, but not the state -- the rollout
# stays continuous and autonomous end to end).
#
# Adapted from full_rollout_training_conv1d/training/code/train.py: the
# per-node windowing (build_window_torch + (G*Nx, ...) reshapes) is replaced
# by whole-beam tensors ((G, n_lags, Nx) in, (G, N_FWD, Nx) out) -- the
# TBPTT mechanics themselves (epoch groups, correction cadence, detach) are
# unchanged. Early stopping on the validation rollout error is added, driven
# by main.py's --early-stop-patience.
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from config import Config
from rollout_torch import build_field_history_torch, reconstruct_torch_general
from physics import biais_repos as compute_biais_repos, autoregressive_rollout
from evaluation import l2_rel
from training import TrainResult, norm_values


def make_epoch_groups(indices: list[int], group_size: int, rng: np.random.Generator) -> list[list[int]]:
    order = rng.permutation(len(indices))
    shuffled = [indices[i] for i in order]
    return [shuffled[i:i + group_size] for i in range(0, len(shuffled), group_size)]


def rollout_group_tbptt(modele, group_indices, FIELDS, bc_pairs,
                         mu_in_t, sd_in_t, mu_out_t, sd_out_t, biais_repos_t,
                         criterion, optimiseur, cfg: "Config", tbptt_hops: int) -> tuple[float, int]:
    # Full rollout for a group of simulations, WITHOUT ever resetting to
    # ground truth -- but with a weight correction every `tbptt_hops` hops
    # rather than a single one at the very end. Between two corrections, the
    # gap between predicted state and true state can become huge, which
    # drowns out the useful signal if we wait for the whole trajectory to
    # correct. Here, as soon as `tbptt_hops` hops have passed, we correct
    # then detach the state (the gradient thread is cut, but the rollout
    # keeps going on the PREDICTED state, never reset to ground truth).
    nodes = cfg.nodes
    G = len(group_indices)
    bc_left_list = [bc_pairs[idx][0] for idx in group_indices]
    bc_right_list = [bc_pairs[idx][1] for idx in group_indices]
    history_needed = cfg.M_BACK * cfg.ndt

    history = []
    for lag in range(cfg.M_BACK, -1, -1):
        m = history_needed - lag * cfg.ndt
        arr = np.stack([FIELDS[idx][m] for idx in group_indices], axis=0)
        history.append(torch.tensor(arr, dtype=torch.float32))

    hops = list(range(history_needed, cfg.Nt - cfg.N_FWD * cfg.ndt + 1, cfg.N_FWD * cfg.ndt))
    total_loss_log, n_updates = 0.0, 0
    segment_loss, segment_hops = torch.zeros(()), 0

    for i, n in enumerate(hops):
        X = (build_field_history_torch(history, cfg) - mu_in_t[None, :, None]) / sd_in_t[None, :, None]
        pred_norm = modele(X)  # (G, N_FWD, Nx)

        baseline = history[-1]
        new_states, s_list = reconstruct_torch_general(baseline, pred_norm, bc_left_list, bc_right_list, n,
                                                         mu_out_t, sd_out_t, biais_repos_t, cfg)

        baseline_nodes = baseline[:, nodes]
        target_list = [
            torch.tensor(np.stack([FIELDS[idx][s][nodes] for idx in group_indices], axis=0), dtype=torch.float32)
            - baseline_nodes
            for s in s_list
        ]
        target = torch.stack(target_list, dim=1)  # (G, N_FWD, Nx)
        target_norm = (target - mu_out_t[None, :, None]) / sd_out_t[None, :, None]

        hop_loss = criterion(pred_norm, target_norm)
        segment_loss = segment_loss + hop_loss
        segment_hops += 1
        total_loss_log += hop_loss.item()

        history = history[cfg.N_FWD:] + new_states

        if segment_hops == tbptt_hops or i == len(hops) - 1:
            optimiseur.zero_grad()
            (segment_loss / segment_hops).backward()
            optimiseur.step()
            n_updates += 1
            history = [h.detach() for h in history]
            segment_loss, segment_hops = torch.zeros(()), 0

    return total_loss_log / len(hops), n_updates


def evaluate_val_rollout(modele, FIELDS, bc_pairs, indices_val, norm_stats, INPUT_CHANNELS, OUTPUT_CHANNELS,
                          cfg: "Config") -> float:
    # Reuses the existing numpy/no_grad evaluation rollout
    # (autoregressive_rollout) -- only the training step needs to stay
    # differentiable, not the monitoring metric.
    mu_in, sd_in = norm_values(norm_stats, INPUT_CHANNELS)
    mu_out, sd_out = norm_values(norm_stats, OUTPUT_CHANNELS)
    biais_repos = compute_biais_repos(modele, mu_in, sd_in, mu_out, sd_out, cfg)

    modele.eval()
    errs = []
    with torch.no_grad():
        for idx in indices_val:
            bc_left, bc_right = bc_pairs[idx]
            U_reel = FIELDS[idx]
            U_pred = autoregressive_rollout(modele, U_reel, mu_in, sd_in, mu_out, sd_out,
                                             biais_repos, bc_left, bc_right, cfg)
            errs.append(l2_rel(U_pred[:, cfg.nodes], U_reel[:, cfg.nodes]))
    modele.train()
    return float(np.mean(errs))


def train_full_rollout(modele, FIELDS, bc_pairs, indices_train, indices_val,
                        norm_stats, INPUT_CHANNELS, OUTPUT_CHANNELS, cfg: "Config", group_size: int,
                        n_epochs: int, model_path: Path, tbptt_hops: int = 10,
                        patience: int | None = None) -> "TrainResult":
    criterion = nn.MSELoss()
    optimiseur = torch.optim.Adam(modele.parameters(), lr=cfg.LEARNING_RATE)

    mu_in, sd_in = norm_values(norm_stats, INPUT_CHANNELS)
    mu_out, sd_out = norm_values(norm_stats, OUTPUT_CHANNELS)
    mu_in_t, sd_in_t = torch.tensor(mu_in), torch.tensor(sd_in)
    mu_out_t, sd_out_t = torch.tensor(mu_out), torch.tensor(sd_out)

    rng = np.random.default_rng(cfg.SEED)
    historique_train, historique_val = [], []
    meilleure_val = float("inf")
    epochs_sans_amelioration = 0

    t0 = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        biais_repos_t = torch.tensor(compute_biais_repos(modele, mu_in, sd_in, mu_out, sd_out, cfg))
        groups = make_epoch_groups(indices_train, group_size, rng)

        modele.train()
        t_epoch0 = time.perf_counter()
        epoch_loss = 0.0
        for i, group_indices in enumerate(groups):
            t_g0 = time.perf_counter()
            avg_loss, n_updates = rollout_group_tbptt(modele, group_indices, FIELDS, bc_pairs,
                                                        mu_in_t, sd_in_t, mu_out_t, sd_out_t, biais_repos_t,
                                                        criterion, optimiseur, cfg, tbptt_hops)
            epoch_loss += avg_loss
            print(f"  epoch {epoch:3d}  group {i+1:3d}/{len(groups)} "
                  f"({len(group_indices)} sims) -- avg loss/hop={avg_loss:.4f} -- "
                  f"{n_updates} corrections -- {time.perf_counter()-t_g0:.2f}s/group")
        epoch_loss /= len(groups)

        val_err = evaluate_val_rollout(modele, FIELDS, bc_pairs, indices_val, norm_stats,
                                        INPUT_CHANNELS, OUTPUT_CHANNELS, cfg)
        historique_train.append(epoch_loss)
        historique_val.append(val_err)

        print(f"Epoch {epoch:4d}/{n_epochs} -- rollout loss (train): {epoch_loss:.4f}  |  "
              f"L2 rel error (val): {val_err:.4f} -- {time.perf_counter()-t_epoch0:.1f}s")

        if val_err < meilleure_val:
            meilleure_val = val_err
            torch.save(modele.state_dict(), model_path)
            epochs_sans_amelioration = 0
        else:
            epochs_sans_amelioration += 1

        if patience is not None and epochs_sans_amelioration >= patience:
            print(f"Early stopping at epoch {epoch}: val rollout error hasn't improved for "
                  f"{patience} epochs (best={meilleure_val:.6f}).")
            break

    train_time_s = time.perf_counter() - t0
    modele.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Best model reloaded -- minimum L2 rel error (val): {meilleure_val:.6f}")

    n_params = sum(p.numel() for p in modele.parameters())
    return TrainResult(historique_train, historique_val, meilleure_val, train_time_s, n_params)


def plot_rollout_training_curve(result: "TrainResult", output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.historique_train, label="Rollout loss (train)")
    ax.plot(result.historique_val, label="Relative L2 error (val)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Value")
    ax.set_title("Learning curve (differentiable full rollout)")
    ax.set_yscale("log"); ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "courbe_apprentissage.png", dpi=150, bbox_inches="tight")
    plt.close()
