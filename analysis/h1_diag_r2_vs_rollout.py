#!/usr/bin/env python3
# H1 (phase 0): the key report figure -- a high one-step R^2 next to that
# SAME run's rollout error curve (log y-axis) diverging over time. Pure
# post-processing over runs/20260810_p0_baseline/metrics.json (see
# h1_hist_deltau.py for the companion figure explaining WHY the one-step R^2
# is inflated in the first place). No model is loaded, no training happens.
# Usage: python analysis/h1_diag_r2_vs_rollout.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

# No configs/runs/*.yaml for this one on purpose -- it doesn't test any
# hyperparameter (pure post-processing of an already-finished run's
# metrics.json), so a run YAML would only ever carry this identity and
# nothing else, and its presence in configs/runs/ once made it possible to
# pass it to scripts/run.job by mistake (accidentally training a full,
# redundant copy of the baseline for ~18h instead of running this analysis).
RUN_ID = "20260810_p0_diag_R2_vs_rollout"
PHASE = 0
HYPOTHESIS = "H1"
SOURCE_RUN_ID = "20260810_p0_baseline"


def main():
    source_metrics_path = REPO_ROOT / "runs" / SOURCE_RUN_ID / "metrics.json"
    if not source_metrics_path.exists():
        print(f"ERROR: {source_metrics_path} not found -- run {SOURCE_RUN_ID} first "
              f"(sbatch scripts/run.job configs/runs/{SOURCE_RUN_ID}.yaml).", file=sys.stderr)
        sys.exit(1)

    with open(source_metrics_path) as f:
        source_metrics = json.load(f)

    r2 = source_metrics["scalars"]["r2_onestep"]
    curves = source_metrics["curves"]
    if r2 is None or not curves["t"]:
        print(f"ERROR: {source_metrics_path} is missing r2_onestep or curves.t -- "
              f"{SOURCE_RUN_ID} may not have finished successfully.", file=sys.stderr)
        sys.exit(1)

    run_dir = REPO_ROOT / "runs" / RUN_ID
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curves["t"], curves["err_rel_mean"], "o-", ms=3, label="relative L2 error (rollout)")
    ax.plot(curves["t"], curves["err_max"], "s-", ms=3, label="max absolute error (rollout)")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("error (log)")
    ax.grid(True, which="both")
    ax.legend(loc="upper left")
    ax.set_title(f"H1 -- one-step R² = {r2:.3f} vs rollout error over time ({SOURCE_RUN_ID})")
    ax.annotate(f"one-step R² = {r2:.3f}\n(looks excellent -- see how it holds up in rollout)",
                xy=(0.98, 0.05), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="gray"))
    plt.tight_layout()
    plt.savefig(figures_dir / "r2_vs_rollout.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary = (
        f"{RUN_ID} (phase {PHASE}, {HYPOTHESIS}) -- one-step R^2 vs rollout error contrast "
        f"(source: {SOURCE_RUN_ID})\n"
        f"One-step R^2: {r2:.6f}\n"
        f"Rollout relative L2 error: final={curves['err_rel_mean'][-1]:.4e}  "
        f"max={max(curves['err_rel_mean']):.4e}\n"
    )
    (run_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"Done -- outputs in {run_dir}")


if __name__ == "__main__":
    main()
