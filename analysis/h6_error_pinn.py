#!/usr/bin/env python3
# H6 (phase 3): max absolute rollout error along the beam vs simulation
# time, across the LAMBDA_PHYSICS (PINN residual weight) sweep {0, 0.001,
# 0.01, 0.1} -- see configs/runs/p3_pinn_*.yaml. Same construction as
# h2_error_mback.py/h3_error_nfwd.py/error_ndt.py -- pure post-processing
# over each source run's metrics.json, no t_div markers.
#
# NOT RUNNABLE YET as of 2026-08-13: none of the 4 source runs exist (the
# p3_pinn_*.yaml configs still have `features` TBD, pending the H4 feature
# ablation currently in progress -- see configs/runs/p3_pinn_0.yaml's own
# comment). Launch those 4 runs first.
#
# TAIL ARTIFACT -- CHECK BEFORE TRUSTING THE PLOT: every pushforward-regime
# comparison so far on this dataset/reference trajectory (idx_test[0]) has
# needed a harmonized T_CUTOFF near the end of the rollout, because whichever
# run(s) fully damp their prediction to ~0 (energy_drift_pct == -100%) get an
# err_max that converges to the reference trajectory's own amplitude instead
# of measuring model quality -- not necessarily all 4 curves, and the
# affected one(s) aren't known ahead of time. Once these runs exist, check
# each metrics.json's curves.err_max tail the same way (compare against the
# other h*_error_*.py scripts' T_CUTOFF derivation) before treating this
# figure as final -- don't assume no cutoff is needed just because this
# regime is bptt-free/PINN-related rather than a regime comparison.
# Usage: python analysis/h6_error_pinn.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

RUN_ID = "p3_diag_error_pinn"
PHASE = 3
HYPOTHESIS = "H6"
SOURCES = {
    "PINN=0": "p3_pinn_0",
    "PINN=0.001": "p3_pinn_0.001",
    "PINN=0.01": "p3_pinn_0.01",
    "PINN=0.1": "p3_pinn_0.1",
}
MARKERS = ["s-", "o-", "^-", "d-", "v-"]

# Set once the 4 runs exist and the tail has been checked (see module
# docstring) -- None means "no cutoff applied yet", NOT "verified clean".
T_CUTOFF = None


def main():
    curves = {}
    for label, source_run_id in SOURCES.items():
        source_metrics_path = REPO_ROOT / "runs" / source_run_id / "metrics.json"
        if not source_metrics_path.exists():
            print(f"ERROR: {source_metrics_path} not found -- run {source_run_id} first "
                  f"(sbatch scripts/run.job configs/runs/{source_run_id}.yaml).", file=sys.stderr)
            sys.exit(1)
        with open(source_metrics_path) as f:
            metrics = json.load(f)
        if not metrics["curves"]["t"]:
            print(f"ERROR: {source_metrics_path} has no curves -- {source_run_id} may not have "
                  f"finished successfully.", file=sys.stderr)
            sys.exit(1)
        curves[label] = metrics["curves"]

    run_dir = REPO_ROOT / "runs" / RUN_ID
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for (label, c), marker in zip(curves.items(), MARKERS):
        if T_CUTOFF is not None:
            t_plot = [tt for tt in c["t"] if tt <= T_CUTOFF]
            e_plot = c["err_max"][:len(t_plot)]
        else:
            t_plot, e_plot = c["t"], c["err_max"]
        ax.plot(t_plot, e_plot, marker, ms=3, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("max absolute error along the beam (log)")
    ax.grid(True, which="both")
    ax.legend()
    title_suffix = "" if T_CUTOFF is None else " (tail artifact removed)"
    ax.set_title(f"H6 -- max rollout error over time vs PINN residual weight{title_suffix}")
    plt.tight_layout()
    plt.savefig(figures_dir / "error_pinn.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary_lines = [f"{RUN_ID} (phase {PHASE}, {HYPOTHESIS}) -- max rollout error vs LAMBDA_PHYSICS",
                      f"(T_CUTOFF={T_CUTOFF} -- see module docstring; None means the tail artifact "
                      f"hasn't been checked yet)\n"]
    for label, source_run_id in SOURCES.items():
        c = curves[label]
        if T_CUTOFF is not None:
            e_stats = c["err_max"][:len([tt for tt in c["t"] if tt <= T_CUTOFF])]
        else:
            e_stats = c["err_max"]
        summary_lines.append(f"{label} (source: {source_run_id}): "
                              f"final err_max={e_stats[-1]:.4e}  peak err_max={max(e_stats):.4e}")
    summary = "\n".join(summary_lines) + "\n"
    (run_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"Done -- outputs in {run_dir}")


if __name__ == "__main__":
    main()
