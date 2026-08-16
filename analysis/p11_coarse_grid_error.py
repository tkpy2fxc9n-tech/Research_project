#!/usr/bin/env python3
# 11a (grille grossiere): max absolute rollout error along the beam
# (Linf) vs simulation time, r1 (fine grid, reference) vs r2 vs r4. Same
# construction as h3_error_nfwd.py -- pure post-processing over each
# source run's metrics.json. r1 has no dedicated run of its own: it's
# "aucun entrainement, reutilise le modele final de l'etape 10" per the
# planning table, which this campaign settled on as p3_pinn_0 (see the
# p11/p12/p13 launch discussion) -- reused here rather than re-run.
#
# Tail-zero artifact check (autoregressive_rollout leaves the last few raw
# steps un-predicted when (Nt-history_needed) isn't a multiple of
# N_FWD*ndt -- see h3/h4's own T_CUTOFF comments for the general bug):
# for M_BACK=3/N_FWD=2/ndt=3 (all 3 runs here), r1 (Nt=500) hits it at its
# very last point (t=4.98, amp_max==0.0 there); r2 (Nt=250) and r4
# (Nt=125) do NOT (verified empirically: neither ever has amp_max==0 at
# its own tail). T_CUTOFF below drops that one point from all 3 curves
# uniformly, even though only r1 needed it, so every series covers the
# same range.
#
# Usage: python analysis/p11_coarse_grid_error.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

RUN_ID = "p11_diag_error_coarse"
PHASE = 11
SOURCES = {
    "r1 = 1 (fine, reference)": "p3_pinn_0",
    "r = 2": "p11_coarse_r2",
    "r = 4": "p11_coarse_r4",
}
# Validated categorical palette (dataviz skill, palette.md slots 1-3: the
# only 3-slot subset clearing all-pairs CVD separation) + line style as the
# required secondary encoding.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
STYLES = ["-", "--", ":"]

T_CUTOFF = 4.95


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
    for (label, c), color, style in zip(curves.items(), COLORS, STYLES):
        t_cut = [tt for tt in c["t"] if tt <= T_CUTOFF]
        e_cut = c["err_max"][:len(t_cut)]
        ax.plot(t_cut, e_cut, style, color=color, lw=2, marker="o", ms=3, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("max absolute error along the beam -- Linf (log)")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    ax.set_title("11a -- grille grossiere: erreur de rollout (Linf) vs temps")
    plt.tight_layout()
    plt.savefig(figures_dir / "error_coarse_grid.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary_lines = [f"{RUN_ID} (phase {PHASE}) -- max rollout error (Linf) vs grid coarsening",
                      f"(curves truncated to t<={T_CUTOFF} -- r1 (p3_pinn_0) hits the known "
                      f"tail-zero rollout artifact at its very last point, t=4.98; r2/r4 don't, "
                      f"but the cutoff is applied to all 3 for a shared range)\n"]
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
