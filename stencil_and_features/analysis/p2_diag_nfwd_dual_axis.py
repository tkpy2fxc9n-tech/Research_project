#!/usr/bin/env python3
# Phase 2 (N_FWD sweep): single dual-axis figure, T_max_10% (left) and E_max
# (right) vs N_FWD in {1..5}, both read from each source run's own
# multi_trajectory_summary.json (99 held-out test trajectories) -- the same
# numbers already reported in runs/p2_analysis/table_readable.txt's
# "Prediction horizon and temporal subsampling" block, N_FWD rows only, just
# as a single figure instead of a table.
#
# Usage: python analysis/p2_diag_nfwd_dual_axis.py
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "common"))
from _common import load_multi_summary  # noqa: E402

RUN_DIR = REPO_ROOT / "stencil_and_features" / "runs" / "p2_diag_error_nfwd"
FIGURES_DIR = RUN_DIR / "figures"

# (N_FWD value, source run_id) -- p2_reference doubles as N_FWD=3, same
# substitution p2_analysis_stencils_parameters.py's table already makes.
SOURCES = [
    (1, "p2_nfwd1"),
    (2, "p2_nfwd2"),
    (3, "p2_reference"),
    (4, "p2_nfwd4"),
    (5, "p2_nfwd5"),
]


def main():
    n_fwd, t_max_10, e_max = [], [], []
    for n, run_id in SOURCES:
        summary = load_multi_summary(REPO_ROOT, run_id)
        if summary is None:
            continue
        n_fwd.append(n)
        t_max_10.append(summary["T_max_10pct"]["median_reached"])
        e_max.append(summary["E_max"]["mean"])

    if len(n_fwd) < len(SOURCES):
        print(f"WARNING: only {len(n_fwd)}/{len(SOURCES)} N_FWD sources available -- "
              f"plotting those, see WARNINGs above for what's missing.", file=sys.stderr)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 8))   # taller than the default 8x5: at this fontsize
                                                # the rotated ylabels need the extra height or
                                                # bbox_inches="tight" clips their tail
    color1, color2 = "tab:blue", "tab:red"

    ax1.plot(n_fwd, t_max_10, "o-", color=color1)
    ax1.set_xlabel("N_FWD", fontsize=17)
    ax1.set_ylabel("T_max_10% (s, median over reached trajectories)", color=color1, fontsize=17)
    ax1.set_xticks(n_fwd)
    ax1.tick_params(axis="both", labelsize=17)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(n_fwd, e_max, "s--", color=color2)
    ax2.set_yscale("log")
    ax2.set_ylabel("E_max (log, mean over test trajectories)", color=color2, fontsize=17)
    ax2.tick_params(axis="both", labelsize=17)

    plt.tight_layout()
    out_path = FIGURES_DIR / "tmax10_emax_vs_nfwd.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
