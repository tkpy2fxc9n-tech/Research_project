#!/usr/bin/env python3
# Phase 0: the report's opening argument, in two complementary halves --
# merged from p0_diag_r2_vs_rollout.py + p0_hist_deltau.py, both writing into
# the single shared runs/p0/p0_analysis/ output folder (ANALYSIS_RUN_ID):
#
#   r2_vs_rollout(): the key figure -- a high one-step R^2 next to that SAME
#     run's rollout error curve (log y-axis) diverging over time. Pure
#     post-processing over runs/p0/p0_baseline/results.yaml.
#   hist_deltau(): distribution of delta_u straight from the raw dataset, no
#     model, no training -- shows the near-zero concentration that inflates
#     the one-step R^2 the other half plots.
#
# Together: the one-step R^2 looks excellent mostly because most delta_u
# targets are near zero (easy to predict), which is exactly what falls apart
# under autoregressive rollout.
#
# Usage: python analysis/p0/p0_analysis_r2_vs_deltau.py
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]   # this file -> analysis/p0/ -> analysis/ -> repo root
sys.path.insert(0, str(REPO_ROOT / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "src"))
from _common import run_dir as _run_dir  # noqa: E402
from beamsurrogate.config import load_config  # noqa: E402
from beamsurrogate.registry import DATASETS  # noqa: E402

# No runs/<phase>/*/config.yaml for either half on purpose -- neither tests a
# hyperparameter (pure post-processing of an already-finished run / the raw
# dataset), so a config.yaml would only ever carry this identity and
# nothing else, and its presence there once made it possible to pass it to
# scripts/run.job by mistake (accidentally training a full, redundant copy
# of the baseline for ~18h instead of running this analysis).
ANALYSIS_RUN_ID = "p0_analysis"   # single shared output folder for both halves below
R2_SOURCE_RUN_ID = "p0_baseline"
# The last couple of rollout steps spike/collapse (end-of-horizon artifact,
# not representative of the divergence trend) and dominate the log-scale
# plot's y-range -- drop them for r2_vs_rollout() display only.
R2_DROP_LAST_N = 2

AMPLITUDE_THRESHOLD_PCT = 1.0   # "part des delta_u proches de zero (ex. % en dessous de 1% de l'amplitude max)"


def r2_vs_rollout() -> bool:
    source_metrics_path = _run_dir(REPO_ROOT, R2_SOURCE_RUN_ID) / "results.yaml"
    if not source_metrics_path.exists():
        print(f"WARNING: {source_metrics_path} not found -- run {R2_SOURCE_RUN_ID} first, skipping.",
              file=sys.stderr)
        return False

    with open(source_metrics_path) as f:
        source_metrics = yaml.safe_load(f)

    r2 = source_metrics["scalars"]["r2_onestep"]
    t_div = source_metrics["scalars"].get("t_div")   # 10%-of-peak-amplitude divergence time, see compute_t_div
    curves = source_metrics["curves"]
    if r2 is None or not curves["t"]:
        print(f"WARNING: {source_metrics_path} is missing r2_onestep or curves.t -- "
              f"{R2_SOURCE_RUN_ID} may not have finished successfully, skipping.", file=sys.stderr)
        return False

    run_dir = _run_dir(REPO_ROOT, ANALYSIS_RUN_ID)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    n = len(curves["t"]) - R2_DROP_LAST_N if R2_DROP_LAST_N else len(curves["t"])

    for key, ylabel, color, marker, filename in (
        ("err_rel_mean", "relative L2 error (rollout)", "tab:blue", "o", "r2_vs_rollout_relerr.png"),
        ("err_max", "max absolute error (rollout)", "tab:orange", "s", "r2_vs_rollout_maxerr.png"),
    ):
        is_maxerr = key == "err_max"   # r2_vs_rollout_maxerr.png only: no title/R^2 box, growth rate in legend instead
        label = ylabel
        if is_maxerr:
            # err_max blows up several orders of magnitude over the rollout (unstabilized
            # autoregressive divergence) and eventually overflows to NaN -- a plain linear
            # slope is meaningless (dominated by the last finite points) and can itself be
            # NaN. Fit log(err_max) vs t instead: its slope k is the exponential growth
            # rate (err_max(t) ~= err_max(0) * exp(k*t)), reported here via its doubling time.
            t_arr = np.asarray(curves["t"][:n])
            e_arr = np.asarray(curves[key][:n])
            mask = np.isfinite(e_arr) & (e_arr > 0)
            k, _ = np.polyfit(t_arr[mask], np.log(e_arr[mask]), 1)
            label = f"{ylabel} (growth rate k ≈ {k:.2f}/s)"

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(curves["t"][:n], curves[key][:n], marker + "-", ms=3, color=color, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("time (s)" if is_maxerr else "t", fontsize=14)
        ax.set_ylabel("error (log)", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, which="both")
        if is_maxerr and t_div is not None:
            ax.axvline(t_div, color="red", linestyle="--", label=f"Thr10% (t={t_div:.2f})")
        ax.legend(loc="upper left")
        if not is_maxerr:
            ax.set_title(f"one-step R² = {r2:.3f} vs {ylabel} over time ({R2_SOURCE_RUN_ID})")
            ax.annotate(f"one-step R² = {r2:.3f}\n(looks excellent -- see how it holds up in rollout)",
                        xy=(0.98, 0.05), xycoords="axes fraction", ha="right", va="bottom",
                        fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="gray"))
        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    summary = (
        f"{ANALYSIS_RUN_ID} -- one-step R^2 vs rollout error contrast "
        f"(source: {R2_SOURCE_RUN_ID})\n"
        f"One-step R^2: {r2:.6f}\n"
        f"Rollout relative L2 error: final={curves['err_rel_mean'][-1]:.4e}  "
        f"max={max(curves['err_rel_mean']):.4e}\n"
    )
    (run_dir / "summary_r2_vs_rollout.txt").write_text(summary)
    print(summary)
    print(f"Done -- outputs in {run_dir}")
    return True


def hist_deltau() -> bool:
    # base.yaml directly: this analysis only needs the shared dataset/ndt/dt
    # fields, which should track base.yaml if it ever changes -- not a
    # frozen, hardcoded copy of them.
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    dataset_path = REPO_ROOT / "data" / DATASETS[cfg.dataset]
    if not dataset_path.exists():
        print(f"WARNING: dataset not found at {dataset_path}. Generate it first with "
              f"dataset/make_dataset.py --profile {cfg.dataset}, skipping.", file=sys.stderr)
        return False

    # u: (n_traj, Nt+1, Nx), physical nodes only -- delta_u@1ndt (the
    # horizon the one-step R^2 in r2_vs_rollout() is computed on) needs no
    # boundary/ghost-band reconstruction, unlike training's stencil features.
    with h5py.File(dataset_path, "r") as f:
        u = f["u"][:]
    delta_u = (u[:, cfg.ndt:, :] - u[:, :-cfg.ndt, :]).ravel()

    amp_max = float(np.abs(u).max())
    threshold = (AMPLITUDE_THRESHOLD_PCT / 100) * amp_max
    pct_near_zero = 100 * float(np.mean(np.abs(delta_u) < threshold))

    run_dir = _run_dir(REPO_ROOT, ANALYSIS_RUN_ID)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(delta_u, bins=200)
    plt.yscale("log")
    plt.xlabel(f"delta_u ({cfg.ndt} step{'s' if cfg.ndt != 1 else ''} ahead, all nodes/trajectories)")
    plt.ylabel("count (log)")
    plt.title(f"Distribution of delta_u -- {dataset_path.name}\n"
              f"{pct_near_zero:.1f}% within {AMPLITUDE_THRESHOLD_PCT:.0f}% of max amplitude ({amp_max:.3e})")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(figures_dir / "hist_deltau.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary = (
        f"{ANALYSIS_RUN_ID} -- delta_u distribution ({dataset_path.name})\n"
        f"delta_u horizon: {cfg.ndt} step(s) ({cfg.dt * cfg.ndt:.4f} time units)\n"
        f"Max |u| amplitude in dataset: {amp_max:.6e}\n"
        f"Threshold ({AMPLITUDE_THRESHOLD_PCT:.0f}% of max amplitude): {threshold:.6e}\n"
        f"Share of delta_u within threshold: {pct_near_zero:.2f}%\n"
        f"Total samples: {delta_u.size:,}\n"
    )
    (run_dir / "summary_hist_deltau.txt").write_text(summary)
    print(summary)
    print(f"Done -- outputs in {run_dir}")
    return True


def main():
    ok_r2 = r2_vs_rollout()
    ok_deltau = hist_deltau()
    if not (ok_r2 or ok_deltau):
        sys.exit(1)


if __name__ == "__main__":
    main()
