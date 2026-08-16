#!/usr/bin/env python3
# H3 (phase 1): max absolute rollout error along the beam vs simulation
# time, across N_FWD in {1, 2, 3, 4, 5}. Same construction as
# h5_error_pushforward_vs_bptt.py -- pure post-processing over each source
# run's metrics.json, no t_div markers. N_FWD=3 has no dedicated run of its
# own: it's the base.yaml default, i.e. exactly p2_pushforward's config
# (no runs/p1_nfwd3/ was ever created) -- reused here rather than re-run,
# same logic the user's own tracking spreadsheet already uses for this row.
# Usage: python analysis/h3_error_nfwd.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

RUN_ID = "p1_diag_error_nfwd"
PHASE = 1
HYPOTHESIS = "H3"
SOURCES = {
    "N_FWD=1": "p1_nfwd1",
    "N_FWD=2": "p1_nfwd2",
    "N_FWD=3 (reference)": "p2_pushforward",
    "N_FWD=4": "p1_nfwd4",
    "N_FWD=5": "p1_nfwd5",
}
MARKERS = ["s-", "o-", "^-", "d-", "v-"]

# By t=4.95-4.98, N_FWD=3 and N_FWD=5 have both fully damped their
# prediction to ~0 (energy_drift_pct == -100%), so err_max there converges
# to the reference trajectory's own amplitude instead of measuring model
# quality (same artifact as h5_error_pushforward_vs_bptt.py). N_FWD=1/2/4
# don't hit this -- but every curve on one plot must share the same t
# range, so the harmonized cutoff is set by the earliest-affected curve
# (N_FWD=3, artifact from t=4.95) and applied to all five.
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
    ax.set_title("H3 -- max rollout error over time vs N_FWD")
    plt.tight_layout()
    plt.savefig(figures_dir / "error_nfwd.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary_lines = [f"{RUN_ID} (phase {PHASE}, {HYPOTHESIS}) -- max rollout error vs N_FWD",
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
