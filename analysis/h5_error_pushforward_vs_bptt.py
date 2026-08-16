#!/usr/bin/env python3
# H5 (phase 2): max absolute rollout error along the beam vs simulation
# time, pushforward against full TBPTT -- the two curves that
# training/pushforward.py's and training/bptt.py's own error_vs_time.png
# each plot separately, overlaid here for a direct regime comparison. No
# t_div markers on purpose (see plot_rollout_error in evaluate/plots.py for
# the version with them). Pure post-processing over both runs' metrics.json.
# Usage: python analysis/h5_error_pushforward_vs_bptt.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

# No configs/runs/*.yaml for this one on purpose, same reasoning as
# h1_diag_r2_vs_rollout.py: pure post-processing, nothing here should ever
# be passed to scripts/run.job.
RUN_ID = "p2_diag_error_pushforward_vs_bptt"
PHASE = 2
HYPOTHESIS = "H5"
SOURCES = {
    "pushforward": "p2_pushforward",
    "full BPTT (TBPTT)": "p2_bptt_hops5_group4",
}


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

    # Drop the last 2 points of every curve: by the end of the rollout both
    # regimes have fully damped their prediction to ~0 (energy_drift_pct ==
    # -100% for both runs), so err_max there no longer measures model
    # quality -- it just converges to the reference trajectory's own
    # amplitude (|0 - U_reel| ~= |U_reel|), identical for both since it's
    # the same reference trajectory. Confirmed bit-identical between the two
    # runs' raw curves.json at exactly the last 2 samples (t=4.95, t=4.98),
    # not a plotting artifact -- and not just 1 point: t=4.95 was also where
    # each curve's reported peak err_max came from.
    N_TAIL_DROP = 2
    fig, ax = plt.subplots(figsize=(9, 5))
    markers = {"pushforward": "s-", "full BPTT (TBPTT)": "o-"}
    for label, c in curves.items():
        ax.plot(c["t"][:-N_TAIL_DROP], c["err_max"][:-N_TAIL_DROP], markers[label], ms=3, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("max absolute error along the beam (log)")
    ax.grid(True, which="both")
    ax.legend()
    ax.set_title("H5 -- max rollout error over time: pushforward vs full BPTT")
    plt.tight_layout()
    plt.savefig(figures_dir / "error_pushforward_vs_bptt.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary_lines = [f"{RUN_ID} (phase {PHASE}, {HYPOTHESIS}) -- max rollout error, pushforward vs full BPTT",
                      f"(last {N_TAIL_DROP} samples excluded from stats below -- both regimes fully damp to "
                      f"~0 by then, see comment in this script)\n"]
    for label, source_run_id in SOURCES.items():
        err_max_kept = curves[label]["err_max"][:-N_TAIL_DROP]
        summary_lines.append(f"{label} (source: {source_run_id}): "
                              f"final err_max={err_max_kept[-1]:.4e}  peak err_max={max(err_max_kept):.4e}")
    summary = "\n".join(summary_lines) + "\n"
    (run_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"Done -- outputs in {run_dir}")


if __name__ == "__main__":
    main()
