# Post-training analysis: rollout error metrics, inference-speed
# benchmarking, plotting, and the resume.txt run-summary export.
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import os
import platform
import socket
import subprocess
import time

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from waves import BCSpec, bc_describe
from physics import biais_repos as compute_biais_repos, autoregressive_rollout, run_fd_simulation_general, FIELD_LABELS
from training import TrainResult


def l2_rel(pred, true, eps=1e-12):
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + eps)


def smape(pred, true):
    m = true != 0
    return np.mean(2*np.abs(true[m] - pred[m]) / (np.abs(true[m]) + np.abs(pred[m])))


@dataclass
class RolloutResult:
    U: np.ndarray
    U_reel: np.ndarray
    left_bc: "BCSpec"
    right_bc: "BCSpec"


def run_rollout(modele, FIELDS: dict, bc_pairs: list[tuple[BCSpec, BCSpec]], rollout_idx: int,
                input_fields, norm_stats, INPUTS, OUTPUTS, cfg) -> RolloutResult:
    left_bc, right_bc = bc_pairs[rollout_idx]
    U_reel = FIELDS[rollout_idx]

    mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
    sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)
    mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
    sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype(np.float32)

    biais_repos = compute_biais_repos(modele, mu_in, sd_in, mu_out, sd_out, cfg)
    U = autoregressive_rollout(modele, U_reel, input_fields, mu_in, sd_in, mu_out, sd_out,
                                biais_repos, left_bc, right_bc, cfg)
    return RolloutResult(U=U, U_reel=U_reel, left_bc=left_bc, right_bc=right_bc)


def compute_errors(rollout: RolloutResult, cfg):
    steps = np.arange(2*cfg.ndt, cfg.Nt + 1, cfg.ndt)
    t_axis = steps * cfg.dt
    nodes = cfg.nodes
    U, U_reel = rollout.U, rollout.U_reel
    l2_list = [l2_rel(U[k, nodes], U_reel[k, nodes]) for k in steps]
    linf_list = [np.max(np.abs(U[k, nodes] - U_reel[k, nodes])) for k in steps]
    smape_list = [100.0 * smape(U[k, nodes], U_reel[k, nodes]) for k in steps]
    return t_axis, l2_list, linf_list, smape_list


def chrono(fonction, n_repeat=15, n_warmup=3):
    for _ in range(n_warmup):
        fonction()
    durations = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fonction()
        durations.append(time.perf_counter() - t0)
    d = np.array(durations)
    return d.mean(), d.std(), float(np.median(d))


@dataclass
class BenchmarkResult:
    fd_time_med: float
    fd_time_std: float
    nn_time_med: float
    nn_time_std: float
    flops_per_call: float
    n_calls: int


def benchmark_inference(modele, FIELDS, input_fields, norm_stats, INPUTS, OUTPUTS,
                         rollout: RolloutResult, cfg) -> BenchmarkResult:
    left_bc, right_bc, U_reel = rollout.left_bc, rollout.right_bc, rollout.U_reel

    mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
    sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)
    mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
    sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype(np.float32)
    biais_repos = compute_biais_repos(modele, mu_in, sd_in, mu_out, sd_out, cfg)

    def fd_once():
        return run_fd_simulation_general(left_bc, right_bc, cfg)

    def rollout_once():
        return autoregressive_rollout(modele, U_reel, input_fields, mu_in, sd_in, mu_out, sd_out,
                                       biais_repos, left_bc, right_bc, cfg)

    fd_mean, fd_std, fd_med = chrono(fd_once)
    nn_mean, nn_std, nn_med = chrono(rollout_once)

    n_calls = len(range(cfg.M_BACK*cfg.ndt, cfg.Nt - cfg.N_FWD*cfg.ndt + 1, cfg.N_FWD*cfg.ndt))
    n_features = cfg.M_BACK * (2*cfg.SS + 1) * len(input_fields)

    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        modele(torch.zeros((len(cfg.nodes), n_features)))
    flops_per_call = fc.get_total_flops() * n_calls

    print(f"FD (real)   : {fd_med*1e3:7.3f} ms")
    print(f"NN (rollout): {nn_med*1e3:7.3f} ms  (±{nn_std*1e3:.3f})")
    print(f"speedup FD/NN = {fd_med/nn_med:.2f}x   (>1 = the network is faster)")

    return BenchmarkResult(
        fd_time_med=fd_med, fd_time_std=float(fd_std),
        nn_time_med=nn_med, nn_time_std=float(nn_std),
        flops_per_call=flops_per_call, n_calls=n_calls,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_rollout_error(t_axis, l2_list, linf_list, output_dir: Path):
    plt.figure(figsize=(9, 5))
    plt.plot(t_axis, l2_list, "o-", ms=3, label="relative L2 error")
    plt.plot(t_axis, linf_list, "s-", ms=3, label="max absolute error (Linf)")
    plt.yscale("log")
    plt.xlabel("t"); plt.ylabel("error"); plt.grid(True, which="both"); plt.legend()
    plt.title("Rollout error over time")
    plt.savefig(output_dir / "erreur_temps.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_smape(t_axis, smape_list, output_dir: Path):
    plt.figure(figsize=(9, 5))
    plt.plot(t_axis, smape_list, "s-", ms=3, label="sMAPE")
    plt.xlabel("t"); plt.ylabel("error (%)"); plt.grid(True); plt.legend()
    plt.title("Rollout sMAPE over time")
    plt.savefig(output_dir / "smape_temps.png", dpi=150, bbox_inches="tight")
    plt.close()


def make_rollout_animation(rollout: RolloutResult, cfg, output_dir: Path):
    U, U_reel = rollout.U, rollout.U_reel
    nodes = cfg.nodes
    x = np.linspace(0, cfg.L, cfg.Nx)
    frames = np.arange(0, cfg.Nt + 1, cfg.ndt)

    fig_anim, (axA, axB) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ligne_reel, = axA.plot([], [], "r", lw=2, label="real")
    ligne_pred, = axA.plot([], [], "b--", lw=2, label="predicted")
    ymax = np.abs(U_reel[:, nodes]).max() * 1.2
    axA.set_xlim(0, cfg.L); axA.set_ylim(-ymax, ymax)
    axA.set_ylabel("u"); axA.legend(loc="upper right"); axA.grid(True)

    ligne_err, = axB.plot([], [], "k", lw=1.5, label="|predicted - real|")
    err_max = max(np.max([np.abs(U[m, nodes] - U_reel[m, nodes]).max() for m in frames]) * 1.2, 1e-9)
    axB.set_xlim(0, cfg.L); axB.set_ylim(0, err_max)
    axB.set_xlabel("x"); axB.set_ylabel("absolute error"); axB.legend(loc="upper right"); axB.grid(True)

    titre = fig_anim.suptitle("")

    def maj(m):
        ligne_reel.set_data(x, U_reel[m, nodes])
        ligne_pred.set_data(x, U[m, nodes])
        ligne_err.set_data(x, np.abs(U[m, nodes] - U_reel[m, nodes]))
        titre.set_text(f"Wave propagation — t = {m*cfg.dt:.3f}  (step {m})")
        return ligne_reel, ligne_pred, ligne_err, titre

    anim = animation.FuncAnimation(fig_anim, maj, frames=frames, interval=50, blit=False)
    anim.save(output_dir / "propagation_onde.gif", writer="pillow", fps=20, dpi=110)
    plt.close(fig_anim)


def plot_utt_uxx(rollout: RolloutResult, cfg, output_dir: Path):
    # PDE check: u_tt as a function of u_xx, real then predicted.
    U, U_reel = rollout.U, rollout.U_reel
    i_left, i_right = cfg.i_left, cfg.i_right
    dt, dx, ndt, Nt, Ntot = cfg.dt, cfg.dx, cfg.ndt, cfg.Nt, cfg.Ntot

    ureel_tt = np.zeros((Nt, Ntot)); ureel_xx = np.zeros((Nt, Ntot))
    for n in range(1, Nt):
        u_prev, u_curr, u_next = U_reel[n-1], U_reel[n], U_reel[n+1]
        ureel_tt[n, i_left:i_right+1] = (u_next[i_left:i_right+1] - 2*u_curr[i_left:i_right+1] + u_prev[i_left:i_right+1]) / dt**2
        ureel_xx[n, i_left:i_right+1] = (u_curr[i_left-1:i_right] - 2*u_curr[i_left:i_right+1] + u_curr[i_left+1:i_right+2]) / dx**2

    snaps_reel = sorted({int(np.clip(round(frac * Nt), 1, Nt - 1)) for frac in (0.1, 0.2, 0.4)})
    plt.figure()
    for n in snaps_reel:
        plt.scatter(ureel_xx[n, i_left+1:i_right], ureel_tt[n, i_left+1:i_right], s=10, label=f"n = {n}")
    plt.xlabel("u_xx (real)"); plt.ylabel("u_tt (real)")
    plt.legend(); plt.grid(); plt.xlim(-10, 10); plt.ylim(-10, 10)
    plt.title("u_tt as a function of u_xx (real)")
    plt.savefig(output_dir / "utt_uxx_reel.png", dpi=150, bbox_inches="tight")
    plt.close()

    upred_tt = np.zeros((Nt, Ntot)); upred_xx = np.zeros((Nt, Ntot))
    for n in range(ndt, Nt - ndt + 1, ndt):
        u_prev, u_curr, u_next = U[n-ndt], U[n], U[n+ndt]
        upred_tt[n, i_left:i_right+1] = (u_next[i_left:i_right+1] - 2*u_curr[i_left:i_right+1] + u_prev[i_left:i_right+1]) / (ndt*dt)**2
        upred_xx[n, i_left:i_right+1] = (u_curr[i_left-1:i_right] - 2*u_curr[i_left:i_right+1] + u_curr[i_left+1:i_right+2]) / dx**2

    n_max_pred = ((Nt - ndt) // ndt) * ndt
    snaps_pred = sorted({int(np.clip(round(frac * Nt / ndt) * ndt, ndt, n_max_pred)) for frac in (0.01, 0.2, 0.3)})
    plt.figure()
    for n in snaps_pred:
        plt.scatter(upred_xx[n, i_left+1:i_right], upred_tt[n, i_left+1:i_right], s=10, label=f"n = {n}")
    plt.xlabel("u_xx (predicted)"); plt.ylabel("u_tt (predicted)")
    plt.grid(); plt.xlim(-10, 10); plt.ylim(-10, 10); plt.legend()
    plt.title("u_tt as a function of u_xx (prediction)")
    plt.savefig(output_dir / "utt_uxx_predit.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# resume.txt export
# ---------------------------------------------------------------------------
def _git_commit() -> str:
    # Best-effort: resume.txt should never fail to write just because git
    # is unavailable (e.g. code copied outside the repo).
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent,
                              capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            dirty = subprocess.run(["git", "diff", "--quiet"], cwd=Path(__file__).resolve().parent).returncode != 0
            return out.stdout.strip() + (" (+ uncommitted changes)" if dirty else "")
    except OSError:
        pass
    return "unknown (not a git repo or git unavailable)"


def _input_fields_from_columns(INPUTS) -> list[str]:
    # INPUTS columns look like "u(t,j-3)" / "u_dot(t-1ndt,j+2)" -- the part
    # before "(" is the FIELD_LABELS value, decoded back to "U"/"Ut"/"Uxx" so
    # resume.txt can name the fields without needing input_fields as a
    # separate argument.
    reverse_labels = {v: k for k, v in FIELD_LABELS.items()}
    seen = []
    for col in INPUTS:
        label = col.split("(")[0]
        if label not in seen:
            seen.append(label)
    return [reverse_labels.get(lab, lab) for lab in seen]


def export_resume(output_dir: Path, cfg, method_name: str, df: pd.DataFrame, INPUTS, OUTPUTS,
                   train_result: TrainResult, tf_metrics: dict, rollout: RolloutResult,
                   bench: BenchmarkResult, errors, extra_info: dict | None = None):
    t_axis, l2_list, linf_list, smape_list = errors
    l2_final, l2_max = l2_list[-1], max(l2_list)
    linf_final, linf_max = linf_list[-1], max(linf_list)
    smape_final, smape_max = smape_list[-1], max(smape_list)

    with open(output_dir / "resume.txt", "w") as f:
        f.write("=====  RUN SUMMARY  =====\n\n")
        f.write(f"Method          : {method_name}\n\n")

        f.write("--- Environment ---\n")
        f.write(f"Run timestamp   : {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Git commit      : {_git_commit()}\n")
        job_id = os.environ.get("SLURM_JOB_ID")
        f.write(f"Host / job      : {socket.gethostname()} / "
                f"{'SLURM job ' + job_id if job_id else 'no SLURM job id (local run)'}\n")
        f.write(f"Python / torch  : {platform.python_version()} / {torch.__version__} "
                f"({'cuda' if torch.cuda.is_available() else 'cpu'})\n\n")

        f.write("--- Configuration ---\n")
        f.write(f"Grid            : Nt={cfg.Nt}, Nx={cfg.Nx}, SS={cfg.SS}, ndt={cfg.ndt} "
                f"(dt={cfg.dt:.4g}, dx={cfg.dx:.4g}, CFL={cfg.CFL:.3f})\n")
        f.write(f"Physical        : E={cfg.E}, rho={cfg.rho}, L={cfg.L}, t_end={cfg.t_end}\n")
        f.write(f"M / N           : M_BACK={cfg.M_BACK}, N_FWD={cfg.N_FWD}\n")
        f.write(f"Input fields    : {', '.join(_input_fields_from_columns(INPUTS))}\n")
        f.write(f"Stencil         : {2*cfg.SS+1} points (SS={cfg.SS} each side)\n")
        f.write(f"Sampling range  : A in [{cfg.AMP_MIN}, {cfg.AMP_MAX}], omega in [{cfg.OMEGA_MIN}, {cfg.OMEGA_MAX}]\n")
        f.write(f"Rollout BC      : left={bc_describe(rollout.left_bc)}  right={bc_describe(rollout.right_bc)}\n")
        f.write(f"Smoothing/noise : SMOOTH_ALPHA={cfg.SMOOTH_ALPHA}, NOISE_STD={cfg.NOISE_STD}\n")
        f.write(f"Seeds           : SEED={cfg.SEED}, SPLIT_SEED={cfg.SPLIT_SEED}\n")
        f.write(f"Dataset         : {len(df):,} rows\n")
        f.write(f"Features        : {len(INPUTS)} inputs, {len(OUTPUTS)} output(s)\n")
        for s in ["train", "val", "test"]:
            n = (df["split"] == s).sum()
            f.write(f"  split {s:5s}   : {n:>8,} rows ({100*n/len(df):.1f} %)\n")
        f.write(f"NN parameters   : {train_result.n_params:,}\n")
        if extra_info:
            for k, v in extra_info.items():
                f.write(f"{k:15s} : {v}\n")
        f.write("\n")

        f.write("--- Training ---\n")
        f.write(f"Learning rate   : {cfg.LEARNING_RATE:g}\n")
        f.write(f"Minimum val     : {train_result.meilleure_val:.6e}\n")
        for col, m in tf_metrics.items():
            f.write(f"{col:15s} : MSE (norm) = {m['mse_norm']:.4e} | R2 = {m['r2']:.4f}\n")
        f.write("\n")

        f.write("--- Execution time ---\n")
        n_epochs_run = len(train_result.historique_train)
        epochs_label = (f"{n_epochs_run} epochs" if n_epochs_run == cfg.N_EPOCHS
                        else f"{n_epochs_run}/{cfg.N_EPOCHS} epochs, early stopped")
        f.write(f"Training ({epochs_label})         : {train_result.train_time_s:.3f} s\n")
        f.write(f"Real simulation (FD, median)         : {bench.fd_time_med*1e3:.3f} ms\n")
        f.write(f"Predicted rollout (NN, median)       : {bench.nn_time_med*1e3:.3f} ms  (+/-{bench.nn_time_std*1e3:.3f})\n")
        f.write(f"Speedup FD/NN                        : {bench.fd_time_med/bench.nn_time_med:.2f}\n")
        f.write(f"Network FLOPs (full rollout)          : {bench.flops_per_call:,.0f}\n\n")

        f.write("--- Rollout errors ---\n")
        f.write(f"L2 relative  : final = {l2_final:.4e}  |  max = {l2_max:.4e}\n")
        f.write(f"Linf absolute: final = {linf_final:.4e}  |  max = {linf_max:.4e}\n")
        f.write(f"sMAPE (%)    : final = {smape_final:.3f}  |  max = {smape_max:.3f}\n")

    print(f"Summary saved: {output_dir / 'resume.txt'}")
