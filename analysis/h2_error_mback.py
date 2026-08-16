#!/usr/bin/env python3
# H2 (phase 1): max absolute rollout error along the beam vs simulation
# time, across M_BACK in {1, 2, 3}. Same construction as
# h5_error_pushforward_vs_bptt.py -- pure post-processing over each source
# run's metrics.json, no t_div markers. M_BACK=2 has no dedicated run of its
# own: it's the base.yaml default, i.e. exactly p2_pushforward's config
# (runs/p1_mback2/ exists but was cancelled before finishing, see
# [[project_repo_restructure]]-adjacent session notes) -- reused here rather
# than re-run, same logic the user's own tracking spreadsheet already uses
# for this row.
# Usage: python analysis/h2_error_mback.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

RUN_ID = "p1_diag_error_mback"
PHASE = 1
HYPOTHESIS = "H2"
SOURCES = {
    "M_BACK=1": "p1_mback1",
    "M_BACK=2 (reference)": "p2_pushforward",
    "M_BACK=3": "p1_mback3",
}
MARKERS = ["s-", "o-", "^-", "d-", "v-"]

# By t=4.95-4.98, M_BACK=2 and M_BACK=3 have both fully damped their
# prediction to ~0 (energy_drift_pct == -100%), so err_max there converges
# to the reference trajectory's own amplitude instead of measuring model
# quality (same artifact as h5_error_pushforward_vs_bptt.py). M_BACK=1
# doesn't hit this (it diverges by blowing up instead of decaying, so its
# own tail is genuine) -- but every curve on one plot must share the same
# t range, so the harmonized cutoff is set by the earliest-affected curve
# (M_BACK=2, artifact from t=4.95) and applied to all three.
T_CUTOFF = 4.92


def main():
    curves = {}
    for label, source_run_id in SOURCES.items():
        source_metrics_path = REPO_ROOT / "runs" / source_run_id / "metrics.json"
        if not source_metrics_path.exists():
            print(f"ERROR: {source_metrics_path} not found -- run {source_run_id} first.", file=sys.stderr)
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
        t_cut = [tt for tt in c["t"] if tt <= T_CUTOFF]
        e_cut = c["err_max"][:len(t_cut)]
        ax.plot(t_cut, e_cut, marker, ms=3, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("max absolute error along the beam (log)")
    ax.grid(True, which="both")
    ax.legend()
    ax.set_title("H2 -- max rollout error over time vs M_BACK")
    plt.tight_layout()
    plt.savefig(figures_dir / "error_mback.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary_lines = [f"{RUN_ID} (phase {PHASE}, {HYPOTHESIS}) -- max rollout error vs M_BACK",
                      f"(curves truncated to t<={T_CUTOFF} -- see T_CUTOFF comment in this script)\n"]
    for label, source_run_id in SOURCES.items():
        c = curves[label]
        e_cut = c["err_max"][:len([tt for tt in c["t"] if tt <= T_CUTOFF])]
        summary_lines.append(f"{label} (source: {source_run_id}): "
                              f"final err_max={e_cut[-1]:.4e}  peak err_max={max(e_cut):.4e}")
    summary = "\n".join(summary_lines) + "\n"
    (run_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"Done -- outputs in {run_dir}")


if __name__ == "__main__":
    main()
