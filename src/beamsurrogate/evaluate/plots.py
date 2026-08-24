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


def _tagged(filename: str, cfg) -> str:
    # Every figure gets the run_id worked into its filename (e.g.
    # "training_curves.png" -> "training_curves_p3_pinn_0.01.png") -- without
    # this, every run's figures/ folder has the exact same filenames, so
    # downloading a handful of runs' figures into one local folder silently
    # overwrites same-named files from different runs.
    stem, dot, ext = filename.rpartition(".")
    return f"{stem}_{cfg.run_id}.{ext}" if dot else f"{filename}_{cfg.run_id}"


def reconstruct_train_total(cfg, train_history: list, extra_history: dict) -> list:
    # pushforward's own train_history is the DATA component alone (see
    # training/pushforward.py's run(): train_history.append(epoch_data)) --
    # the actual weighted loss the optimizer minimized also adds the
    # pushforward term (ramped 0 -> LAMBDA_PF over PF_WARMUP epochs) and the
    # physics term. bptt/teacher_forcing's train_history is already the true
    # combined total (bptt: combine_losses's own return value; teacher_forcing:
    # nothing else to combine), so it's returned as-is. Same reconstruction
    # analysis/p1_diag_training_curve_total.py used to do per-run, now the
    # single source both the live pipeline and any post-hoc backfill call.
    if cfg.regime != "pushforward":
        return list(train_history)
    n_epochs = len(train_history)
    lam_pf = [cfg.LAMBDA_PF * min(1.0, e / cfg.PF_WARMUP) if cfg.PF_WARMUP > 0 else cfg.LAMBDA_PF
              for e in range(1, n_epochs + 1)]
    total = [d + p * pf for d, pf, p in zip(train_history, extra_history["pushforward"], lam_pf)]
    return [t + cfg.LAMBDA_PHYSICS * ph for t, ph in zip(total, extra_history["physics"])]


def plot_training_curve(train_result, output_dir: Path, cfg):
    total_train = reconstruct_train_total(cfg, train_result.train_history, train_result.extra_history)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(total_train, "-", label="train")
    ax.plot(train_result.val_history, "--", label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Learning curve")
    ax.set_yscale("log"); ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / _tagged("training_curves.png", cfg), dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curve_active_components(train_result, output_dir: Path, cfg):
    # Solid=train, dashed=val, one line per loss component with nonzero
    # weight at some point in training (a component pinned to 0 the whole
    # run -- e.g. physics before phase 3 -- is dropped rather than plotted
    # flat at zero). Moved from analysis/p1_diag_training_curves_active.py
    # so every run gets this automatically instead of it being a manual
    # post-hoc step; that script is now a thin wrapper around this.
    regime = cfg.regime
    if regime not in ("pushforward", "bptt"):
        return  # teacher_forcing has no component breakdown -- extra_history is empty

    train_history = train_result.train_history
    extra_history = train_result.extra_history
    n_epochs = len(train_history)
    epochs = list(range(1, n_epochs + 1))

    if regime == "pushforward":
        lam_pf = [cfg.LAMBDA_PF * min(1.0, e / cfg.PF_WARMUP) if cfg.PF_WARMUP > 0 else cfg.LAMBDA_PF
                  for e in epochs]
        components = {
            "data": ([1.0] * n_epochs, train_history, "val_data"),
            "pushforward": (lam_pf, extra_history["pushforward"], "val_pushforward"),
            "physics": ([cfg.LAMBDA_PHYSICS] * n_epochs, extra_history["physics"], "val_physics"),
        }
    else:  # bptt
        components = {
            "data": ([cfg.LAMBDA_DATA] * n_epochs, extra_history["data"], "val_data"),
            "rollout": ([cfg.LAMBDA_ROLLOUT] * n_epochs, extra_history["rollout"], "val_rollout"),
            "physics": ([cfg.LAMBDA_PHYSICS] * n_epochs, extra_history["physics"], None),
        }

    total_train = reconstruct_train_total(cfg, train_history, extra_history)
    active = {name: v for name, v in components.items() if max(v[0]) > 0}
    dropped = [name for name in components if name not in active]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(epochs, total_train, "-", color="black", lw=2, label="total (train)")
    ax.plot(epochs, train_result.val_history, "--", color="black", lw=2, label="total (val)")
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for (name, (w, raw, val_key)), color in zip(active.items(), colors):
        weighted = [wi * ri for wi, ri in zip(w, raw)]
        ax.plot(epochs, weighted, "-", color=color, lw=1.4, label=f"{name} (train, weighted)")
        val_raw = extra_history.get(val_key) if val_key else None
        if val_raw:
            val_weighted = [wi * ri for wi, ri in zip(w, val_raw)]
            ax.plot(epochs, val_weighted, "--", color=color, lw=1.4, label=f"{name} (val, weighted)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (weighted)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    title_extra = f" -- {', '.join(dropped)} excluded (weight=0)" if dropped else ""
    ax.set_title(f"Learning curve by component -- solid=train, dashed=val{title_extra}")
    plt.tight_layout()
    plt.savefig(output_dir / _tagged("training_curve_active_components.png", cfg), dpi=150, bbox_inches="tight")
    plt.close()


def plot_rollout_error(curves: dict, output_dir: Path, cfg, t_div: float | None = None):
    plt.figure(figsize=(9, 5))
    plt.plot(curves["t"], curves["err_rel_mean"], "o-", ms=3, label="relative L2 error")
    plt.plot(curves["t"], curves["err_max"], "s-", ms=3, label="max absolute error")
    plt.yscale("log")
    if t_div is not None:
        # Single divergence-time marker for both curves at once, rather than
        # a separate threshold recomputed from the noisy relative curve --
        # see compute_t_div in metrics.py for why it's err_max-based.
        plt.axvline(t_div, color="red", linestyle="--", label=f"t_div = {t_div:.2f}")
    plt.xlabel("t"); plt.ylabel("error"); plt.grid(True, which="both"); plt.legend()
    plt.title("Rollout error over time")
    plt.savefig(output_dir / _tagged("error_vs_time.png", cfg), dpi=150, bbox_inches="tight")
    plt.close()


def plot_relative_error(curves: dict, output_dir: Path, cfg):
    # err_rel_mean alone, split out of plot_rollout_error's combined figure --
    # for a single run's own report figure (e.g. "the retained architecture"),
    # not next to err_max.
    plt.figure(figsize=(8, 5))
    plt.plot(curves["t"], curves["err_rel_mean"], "o-", ms=3)
    plt.yscale("log")
    plt.xlabel("t"); plt.ylabel("relative L2 error"); plt.grid(True, which="both")
    plt.title("Relative L2 rollout error over time")
    plt.savefig(output_dir / _tagged("relative_error_vs_time.png", cfg), dpi=150, bbox_inches="tight")
    plt.close()


def plot_amplitude_and_energy(curves: dict, output_dir: Path, cfg):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(curves["t"], curves["amp_max"])
    ax1.set_ylabel("max |u|"); ax1.grid(True); ax1.set_title("Peak amplitude over time")
    ax2.plot(curves["t"], curves["energy_ref"], "r-", label="FD reference")
    ax2.plot(curves["t"], curves["energy"], "b--", label="predicted")
    ax2.set_ylabel("mechanical energy"); ax2.set_xlabel("t"); ax2.grid(True); ax2.legend()
    ax2.set_title("Mechanical energy over time (should stay ~constant for both)")
    plt.tight_layout()
    plt.savefig(output_dir / _tagged("amplitude_energy_vs_time.png", cfg), dpi=150, bbox_inches="tight")
    plt.close()


def plot_shape_vs_amplitude_error(curves: dict, output_dir: Path, cfg):
    # Pearson r (shape agreement, amplitude-blind) plotted directly above
    # err_mean_abs (amplitude-sensitive) on a shared time axis -- pearson_r
    # alone can be misleadingly high (right shape, wrong size still scores
    # well), so it must always be read next to this second panel, never in
    # isolation.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(curves["t"], curves["pearson_r"], "o-", ms=3)
    ax1.set_ylabel("Pearson r (shape)"); ax1.set_ylim(-1.05, 1.05); ax1.grid(True)
    ax1.set_title("Predicted-vs-reference shape agreement (amplitude-blind -- see panel below)")
    ax2.plot(curves["t"], curves["err_mean_abs"], "o-", ms=3, color="#eb6834")
    ax2.set_yscale("log")
    ax2.set_ylabel("mean absolute error (log)"); ax2.set_xlabel("t"); ax2.grid(True, which="both")
    ax2.set_title("Amplitude error over time")
    plt.tight_layout()
    plt.savefig(output_dir / _tagged("shape_vs_amplitude_error.png", cfg), dpi=150, bbox_inches="tight")
    plt.close()


def plot_one_step_predictions(y_true: np.ndarray, y_pred: np.ndarray, OUTPUTS: list[str],
                               one_step_metrics: dict, output_dir: Path, cfg,
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
    plt.savefig(output_dir / _tagged(filename, cfg), dpi=150, bbox_inches="tight")
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
    anim.save(output_dir / _tagged(filename, cfg), writer="pillow", fps=20, dpi=110)
    plt.close(fig_anim)


def plot_spectrum(spectrum: dict, output_dir: Path, cfg):
    # One color per snapshot time, solid=reference vs dashed=predicted at
    # that SAME time -- shows whether power shifts toward high k (the
    # jagged/checkerboard end) as the rollout progresses, not just a single
    # final-step snapshot.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    n = len(spectrum["t_snapshots"])
    colors = plt.cm.viridis(np.linspace(0, 0.85, max(n, 1)))
    for i, (t, color) in enumerate(zip(spectrum["t_snapshots"], colors)):
        ax.plot(spectrum["k"], spectrum["power_ref"][i], "-", color=color, label=f"t={t:.2f} (ref)")
        ax.plot(spectrum["k"], spectrum["power_pred"][i], "--", color=color, label=f"t={t:.2f} (pred)")
    ax.set_yscale("log")
    ax.set_xlabel("wavenumber k"); ax.set_ylabel("power (windowed)")
    ax.grid(True); ax.legend(fontsize=8, ncol=2)
    ax.set_title("Wavenumber spectrum over the rollout (Hann-windowed)")
    plt.tight_layout()
    plt.savefig(output_dir / _tagged("spectrum.png", cfg), dpi=150, bbox_inches="tight")
    plt.close()
