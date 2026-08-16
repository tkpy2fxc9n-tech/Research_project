#!/usr/bin/env python3
# ndt sweep (phase 1): max absolute rollout error along the beam vs
# simulation time, across ndt in {2, 3, 4, 5}. Same construction as
# h5_error_pushforward_vs_bptt.py -- pure post-processing over each source
# run's metrics.json, no t_div markers. No H-number: this sweep isn't in the
# original H1-H9 plan (see configs/runs/p1_ndt2.yaml's own `hypothesis:
# null` comment) -- filename left without an "hN_" prefix for the same
# reason, unlike its H2/H3/H4 siblings. ndt=3 has no dedicated run of its
# own: it's the base.yaml default, i.e. exactly p2_pushforward's config
# (no runs/p1_ndt3/ was ever created) -- reused here rather than re-run,
# same logic the user's own tracking spreadsheet already uses for this row.
# Usage: python analysis/error_ndt.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

RUN_ID = "p1_diag_error_ndt"
PHASE = 1
HYPOTHESIS = None
SOURCES = {
    "ndt=2": "p1_ndt2",
    "ndt=3 (reference)": "p2_pushforward",
    "ndt=4": "p1_ndt4",
    "ndt=5": "p1_ndt5",
}
MARKERS = ["s-", "o-", "^-", "d-", "v-"]

# By the end of the rollout, ndt=2/3/5 have all fully damped their
# prediction to ~0 (energy_drift_pct == -100%), so err_max there converges
# to the reference trajectory's own amplitude instead of measuring model
# quality (same artifact as h5_error_pushforward_vs_bptt.py). ndt=4 doesn't
# hit this -- but every curve on one plot must share the same t range, and
# different ndt values step on different time grids (coarser ndt = fewer,
# wider-spaced points, all still spanning ~[0, t_end]), so this is a time
# cutoff, not a fixed number of points dropped. Harmonized cutoff set by the
# earliest-affected curve (ndt=5, artifact from t=4.95) and applied to all four.
T_CUTOFF = 4.90


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
    ax.set_title("max rollout error over time vs ndt")
    plt.tight_layout()
    plt.savefig(figures_dir / "error_ndt.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary_lines = [f"{RUN_ID} (phase {PHASE}, hypothesis={HYPOTHESIS}) -- max rollout error vs ndt",
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
