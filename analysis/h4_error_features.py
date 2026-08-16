#!/usr/bin/env python3
# H4 (phase 1): max absolute rollout error along the beam vs simulation
# time, across the 4 input-feature subsets {U}, {U,Ut}, {Ut,Uxx}, {U,Ut,Uxx}.
# Same construction as h5_error_pushforward_vs_bptt.py -- pure
# post-processing over each source run's metrics.json, no t_div markers.
# Unlike H2/H3/ndt, none of these 4 configs equals the base.yaml reference
# (features=[U] IS the base default, but p1_feat_u is still its own run --
# it differs from p2_pushforward in nothing, which is fine, it's just not
# reused as a shortcut here since the run itself is cheap and already
# queued/running).
# Usage: python analysis/h4_error_features.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

RUN_ID = "p1_diag_error_features"
PHASE = 1
HYPOTHESIS = "H4"
SOURCES = {
    "features={U}": "p1_feat_u",
    "features={U,Ut}": "p1_feat_u_ut",
    "features={Ut,Uxx}": "p1_feat_ut_uxx",
    "features={U,Ut,Uxx}": "p1_feat_all",
}
MARKERS = ["s-", "o-", "^-", "d-", "v-"]


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
        # Drop the last 2 points: autoregressive_rollout's hop loop (N_FWD*ndt
        # per hop) doesn't evenly divide cfg.Nt - history_needed for this
        # config, so the tail few raw steps are never predicted and U stays at
        # its zero-init there -- err_max on those steps is just |U_reel|,
        # identical across every run regardless of model. Confirmed via
        # curves["amp_max"], which is exactly 0.0 at these two points for all
        # 4 source runs. This -2 is specific to M_BACK=2/N_FWD=3/ndt=3; other
        # hypotheses (H2/H3/ndt sweeps) would need a different trim count.
        curves[label] = {k: v[:-2] for k, v in metrics["curves"].items()}

    run_dir = REPO_ROOT / "runs" / RUN_ID
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for (label, c), marker in zip(curves.items(), MARKERS):
        ax.plot(c["t"], c["err_max"], marker, ms=3, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("max absolute error along the beam (log)")
    ax.grid(True, which="both")
    ax.legend()
    ax.set_title("H4 -- max rollout error over time vs input features")
    plt.tight_layout()
    plt.savefig(figures_dir / "error_features.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary_lines = [f"{RUN_ID} (phase {PHASE}, {HYPOTHESIS}) -- max rollout error vs input features\n"]
    for label, source_run_id in SOURCES.items():
        c = curves[label]
        summary_lines.append(f"{label} (source: {source_run_id}): "
                              f"final err_max={c['err_max'][-1]:.4e}  peak err_max={max(c['err_max']):.4e}")
    summary = "\n".join(summary_lines) + "\n"
    (run_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"Done -- outputs in {run_dir}")


if __name__ == "__main__":
    main()
