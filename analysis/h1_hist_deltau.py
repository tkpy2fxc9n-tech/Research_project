#!/usr/bin/env python3
# H1 (phase 0): distribution of delta_u straight from the raw dataset, no
# model, no training. Shows the near-zero concentration that inflates the
# one-step R^2 -- see h1_diag_r2_vs_rollout.py for the figure this sets up
# (same run's rollout error curve, annotated with that R^2, so the two
# together make the contrast the report's H1 argues).
# Usage: python analysis/h1_hist_deltau.py
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from beamsurrogate.config import load_config
from beamsurrogate.registry import DATASETS

# No configs/runs/*.yaml for this one on purpose -- it doesn't test any
# hyperparameter, so a run YAML would only ever carry this identity and
# nothing else, and its presence in configs/runs/ once made it possible to
# pass it to scripts/run.job by mistake (accidentally training a full,
# redundant copy of the baseline for ~18h instead of running this analysis).
RUN_ID = "p0_hist_deltau"
PHASE = 0
HYPOTHESIS = "H1"
AMPLITUDE_THRESHOLD_PCT = 1.0   # "part des delta_u proches de zero (ex. % en dessous de 1% de l'amplitude max)"


def main():
    # base.yaml directly: this analysis only needs the shared dataset/ndt/dt
    # fields, which should track base.yaml if it ever changes -- not a
    # frozen, hardcoded copy of them.
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    dataset_path = REPO_ROOT / "data" / DATASETS[cfg.dataset]
    if not dataset_path.exists():
        print(f"ERROR: dataset not found at {dataset_path}. Generate it first with "
              f"scripts/make_dataset.py --profile {cfg.dataset}.", file=sys.stderr)
        sys.exit(1)

    # u: (n_traj, Nt+1, Nx), physical nodes only -- delta_u@1ndt (the
    # horizon the one-step R^2 in h1_diag_r2_vs_rollout.py is computed on)
    # needs no boundary/ghost-band reconstruction, unlike training's
    # stencil features.
    with h5py.File(dataset_path, "r") as f:
        u = f["u"][:]
    delta_u = (u[:, cfg.ndt:, :] - u[:, :-cfg.ndt, :]).ravel()

    amp_max = float(np.abs(u).max())
    threshold = (AMPLITUDE_THRESHOLD_PCT / 100) * amp_max
    pct_near_zero = 100 * float(np.mean(np.abs(delta_u) < threshold))

    run_dir = REPO_ROOT / "runs" / RUN_ID
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
        f"{RUN_ID} (phase {PHASE}, {HYPOTHESIS}) -- delta_u distribution ({dataset_path.name})\n"
        f"delta_u horizon: {cfg.ndt} step(s) ({cfg.dt * cfg.ndt:.4f} time units)\n"
        f"Max |u| amplitude in dataset: {amp_max:.6e}\n"
        f"Threshold ({AMPLITUDE_THRESHOLD_PCT:.0f}% of max amplitude): {threshold:.6e}\n"
        f"Share of delta_u within threshold: {pct_near_zero:.2f}%\n"
        f"Total samples: {delta_u.size:,}\n"
    )
    (run_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"Done -- outputs in {run_dir}")


if __name__ == "__main__":
    main()
