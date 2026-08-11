# Per-run diagnostic figures (runs/<run_id>/figures/) -- distinct from
# analysis/*.py, which reads runs/*/metrics.json across many runs to build
# the comparative figures the report expects (figures/ at the repo root).
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .rollout import RolloutResult


def plot_training_curve(train_result, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_result.train_history, label="train")
    ax.plot(train_result.val_history, label="val")
    for name, hist in train_result.extra_history.items():
        ax.plot(hist, "--", label=f"{name} (unweighted)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Learning curve")
    ax.set_yscale("log"); ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "training_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_rollout_error(curves: dict, output_dir: Path):
    plt.figure(figsize=(9, 5))
    plt.plot(curves["t"], curves["err_rel_mean"], "o-", ms=3, label="relative L2 error")
    plt.plot(curves["t"], curves["err_max"], "s-", ms=3, label="max absolute error (Linf)")
    plt.yscale("log")
    plt.xlabel("t"); plt.ylabel("error"); plt.grid(True, which="both"); plt.legend()
    plt.title("Rollout error over time")
    plt.savefig(output_dir / "error_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_amplitude_and_energy(curves: dict, output_dir: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(curves["t"], curves["amp_max"])
    ax1.set_ylabel("max |u|"); ax1.grid(True); ax1.set_title("Peak amplitude over time")
    ax2.plot(curves["t"], curves["energy"])
    ax2.set_ylabel("mechanical energy"); ax2.set_xlabel("t"); ax2.grid(True)
    ax2.set_title("Mechanical energy over time (should stay ~constant)")
    plt.tight_layout()
    plt.savefig(output_dir / "amplitude_energy_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_one_step_predictions(y_true: np.ndarray, y_pred: np.ndarray, OUTPUTS: list[str],
                               one_step_metrics: dict, output_dir: Path,
                               filename: str = "one_step_predictions.png"):
    fig, axes = plt.subplots(1, len(OUTPUTS), figsize=(6 * len(OUTPUTS), 6), squeeze=False)
    for i, (ax, col) in enumerate(zip(axes.flatten(), OUTPUTS)):
        y_r, y_p = y_true[:, i], y_pred[:, i]
        ax.scatter(y_r, y_p, alpha=0.4, s=8)
        lim = max(abs(y_r).max(), abs(y_p).max())
        ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="perfect prediction")
        ax.set_xlabel(f"{col} real (physical)"); ax.set_ylabel(f"{col} predicted (physical)")
        m = one_step_metrics[col]
        ax.set_title(f"{col}\nMSE (norm)={m['mse_norm']:.2e}  |  R²={m['r2']:.3f}")
        ax.legend(); ax.grid(True)
    fig.suptitle("One-step prediction over the test split", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()


def make_rollout_animation(rollout: RolloutResult, cfg, output_dir: Path, filename: str = "rollout.gif"):
    U, U_reel = rollout.U, rollout.U_reel
    nodes = cfg.nodes
    x = np.linspace(0, cfg.L, cfg.Nx)
    frames = np.arange(0, cfg.Nt + 1, cfg.ndt)

    fig_anim, (axA, axB) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    line_real, = axA.plot([], [], "r", lw=2, label="real")
    line_pred, = axA.plot([], [], "b--", lw=2, label="predicted")
    ymax = max(np.abs(U_reel[:, nodes]).max() * 1.2, 1e-9)
    axA.set_xlim(0, cfg.L); axA.set_ylim(-ymax, ymax)
    axA.set_ylabel("u"); axA.legend(loc="upper right"); axA.grid(True)

    line_err, = axB.plot([], [], "k", lw=1.5, label="|predicted - real|")
    err_max = max(np.max([np.abs(U[m, nodes] - U_reel[m, nodes]).max() for m in frames]) * 1.2, 1e-9)
    if not np.isfinite(err_max):
        err_max = ymax
    axB.set_xlim(0, cfg.L); axB.set_ylim(0, err_max)
    axB.set_xlabel("x"); axB.set_ylabel("absolute error"); axB.legend(loc="upper right"); axB.grid(True)

    title_obj = fig_anim.suptitle("")

    def update(m):
        line_real.set_data(x, U_reel[m, nodes])
        line_pred.set_data(x, U[m, nodes])
        line_err.set_data(x, np.abs(U[m, nodes] - U_reel[m, nodes]))
        title_obj.set_text(f"Wave propagation -- t = {m*cfg.dt:.3f}  (step {m})")
        return line_real, line_pred, line_err, title_obj

    anim = animation.FuncAnimation(fig_anim, update, frames=frames, interval=50, blit=False)
    anim.save(output_dir / filename, writer="pillow", fps=20, dpi=110)
    plt.close(fig_anim)


def plot_spectrum(spectrum: dict, output_dir: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(spectrum["k"], spectrum["power_ref"], label="reference")
    plt.plot(spectrum["k"], spectrum["power_pred"], "--", label="predicted")
    plt.yscale("log")
    plt.xlabel("wavenumber k"); plt.ylabel("power"); plt.grid(True); plt.legend()
    plt.title("Final-step wavenumber spectrum")
    plt.savefig(output_dir / "spectrum.png", dpi=150, bbox_inches="tight")
    plt.close()
