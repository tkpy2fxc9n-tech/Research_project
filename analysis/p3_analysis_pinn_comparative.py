#!/usr/bin/env python3
# Phase 3 (PINN): max absolute rollout error along the beam vs simulation
# time, across the LAMBDA_PHYSICS (PINN residual weight) sweep {0, 0.001,
# 0.01, 0.1} -- see configs/runs/p3_pinn_*.yaml. Same construction as
# p2_diag_error_mback.py/p2_diag_error_nfwd.py/p2_diag_error_ndt.py -- pure post-processing
# over each source run's metrics.json, no t_div markers.
#
# All 4 source runs exist since 2026-08-13.
#
# TAIL ARTIFACT -- FIXED AT THE SOURCE 2026-08-23. All 4 runs used to report
# err_max = 3.137447e-03 at t=4.98, bit-identical and 96% of the reference
# trajectory's own amplitude (3.26921348e-03) -- looked like every prediction
# had collapsed to ~0, but the real cause was an off-by-one in
# _rollout_steps (evaluate/metrics.py): autoregressive_rollout
# (physics/solver.py) only ever writes U in blocks of N_FWD*ndt, so when
# (Nt - history_needed) isn't a multiple of that block size its last block
# stops short of Nt and the trailing row of U is left at its
# np.zeros(...) init -- never written by the model. _rollout_steps sampled
# up to Nt regardless, so that unwritten row got read as err_max = |0 -
# U_reel| = |U_reel|, the same number for every model and not a real error.
# Fixed by capping _rollout_steps at the last step autoregressive_rollout
# actually writes, so every run's curves (and every field derived from them:
# E_max, t_E_max, T_max_*, T_mean_*, P_thr_5pct, E_short) are correct at
# generation time now -- no post-hoc truncation needed in this script
# anymore. The 4 p3_pinn_* runs' results.yaml were patched in place to drop
# the one stale contaminated sample and recompute their scalars accordingly.
#
# SAME REFERENCE TRAJECTORY -- verified, so overlaying the 4 curves is
# meaningful: the split is read from the h5 file itself (not reseeded per
# run) and data/split.py takes rollout_idx = idx_test[0], so every run on the
# `simple` dataset rolls out the same trajectory. Confirmed numerically: the
# recovered reference amplitude is bit-identical (3.26921348e-03) across all
# 4 runs, and the 4 configs differ only in run_id and LAMBDA_PHYSICS.
#
# Also writes table.tex: the PINN-loss comparison table (P_thr, T_5%^max,
# T_10%^max, T_5%^mean, T_10%^mean, E_short, E_max). All 4 runs' results.yaml
# now carry amp_ref/E_max/P_thr_5pct/T_max_*/T_mean_* directly -- _common.py's
# resolve_amp_ref/fill_missing_scalars are a no-op fallback here, kept only
# for whatever future run in this table predates those fields.
# Usage: python analysis/p3_analysis_pinn_comparative.py
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))
from _common import (  # noqa: E402
    fmt_pct_mean_std, fmt_sci_mean_std, fmt_time_censored, load_multi_summary, load_run_metrics,
    fmt_pct_mean_std_readable, fmt_sci_mean_std_readable, fmt_time_censored_readable,
    readable_metric_notes, render_latex_table, render_readable_table, MISSING_ROW,
)

RUN_ID = "p3_pinn_comparative"
SOURCES = {
    "PINN=0": "p3_pinn_0",
    "PINN=0.001": "p3_pinn_0.001",
    "PINN=0.01": "p3_pinn_0.01",
    "PINN=0.1": "p3_pinn_0.1",
}
TABLE_ROW_LABELS = {
    "PINN=0": "No PINN loss",
    "PINN=0.001": "$\\lambda = 0.001$",
    "PINN=0.01": "$\\lambda = 0.01$",
    "PINN=0.1": "$\\lambda = 0.1$",
}
MARKERS = ["s-", "o-", "^-", "d-", "v-"]
# Fixed per-label color/marker, rather than matplotlib's automatic cycling --
# plot_loss_split() skips PINN=0 entirely in its second panel (R_phys), which
# would otherwise shift every subsequent label's auto-assigned color and make
# e.g. PINN=0.001 blue in one panel of the report figure but orange in the
# other.
STYLE = {"PINN=0": ("#2a78d6", "s-"), "PINN=0.001": ("#eb6834", "o-"),
         "PINN=0.01": ("#1baf7a", "^-"), "PINN=0.1": ("#d6272a", "d-")}


def build_table(summaries: dict) -> str:
    rows = []
    for label, run_id in SOURCES.items():
        s = summaries.get(run_id)
        row_label = TABLE_ROW_LABELS[label]
        if s is None:
            rows.append([row_label, *([MISSING_ROW] * 6)])
            continue
        rows.append([
            row_label,
            fmt_pct_mean_std(s["P_thr_5pct"]["mean"], s["P_thr_5pct"]["std"]),
            fmt_time_censored(s["T_max_5pct"]["median_reached"], s["T_max_5pct"]["pct_reached"]),
            fmt_time_censored(s["T_max_10pct"]["median_reached"], s["T_max_10pct"]["pct_reached"]),
            fmt_time_censored(s["T_mean_5pct"]["median_reached"], s["T_mean_5pct"]["pct_reached"]),
            fmt_time_censored(s["T_mean_10pct"]["median_reached"], s["T_mean_10pct"]["pct_reached"]),
            fmt_sci_mean_std(s["E_short"]["mean"], s["E_short"]["std"]),
            fmt_sci_mean_std(s["E_max"]["mean"], s["E_max"]["std"]),
        ])
    n = next((s["n_trajectories"] for s in summaries.values() if s), "?")
    return render_latex_table(
        caption=f"Comparison of performances depending on a PINN loss term, evaluated by "
                f"autoregressive rollout on $N={n}$ held-out test trajectories. "
                f"$P_{{\\mathrm{{thr}}}}$, $E_{{\\mathrm{{short}}}}$ and $E_{{\\max}}$ are reported "
                f"as mean $\\pm$ std across trajectories. $T_{{5\\%}}$/$T_{{10\\%}}$ columns are "
                f"right-censored (not every trajectory reaches the threshold within the simulated "
                f"window); each cell reports the median crossing time among trajectories that "
                f"reached it, with the fraction that reached it in parentheses.",
        label="tab:PINN_performances",
        col_spec="lccccccc",
        header_cells=["Strategy", "$P_{\\mathrm{thr}}$", "$T_{5\\%}^{\\max}$", "$T_{10\\%}^{\\max}$",
                      "$T_{5\\%}^{\\mathrm{mean}}$", "$T_{10\\%}^{\\mathrm{mean}}$",
                      "$E_{\\mathrm{short}}$", "$E_{\\max}$"],
        rows=rows,
    )


def build_readable_table(summaries: dict) -> str:
    rows = []
    for label, run_id in SOURCES.items():
        s = summaries.get(run_id)
        if s is None:
            rows.append([label, *(["--"] * 6)])
            continue
        rows.append([
            label,
            fmt_pct_mean_std_readable(s["P_thr_5pct"]["mean"], s["P_thr_5pct"]["std"]),
            fmt_time_censored_readable(s["T_max_5pct"]["median_reached"], s["T_max_5pct"]["pct_reached"]),
            fmt_time_censored_readable(s["T_max_10pct"]["median_reached"], s["T_max_10pct"]["pct_reached"]),
            fmt_time_censored_readable(s["T_mean_5pct"]["median_reached"], s["T_mean_5pct"]["pct_reached"]),
            fmt_time_censored_readable(s["T_mean_10pct"]["median_reached"], s["T_mean_10pct"]["pct_reached"]),
            fmt_sci_mean_std_readable(s["E_short"]["mean"], s["E_short"]["std"]),
            fmt_sci_mean_std_readable(s["E_max"]["mean"], s["E_max"]["std"]),
        ])
    table = render_readable_table(
        header_cells=["Strategy", "P_thr_5%", "T_max_5%", "T_max_10%",
                      "T_mean_5%", "T_mean_10%", "E_short", "E_max"],
        rows=rows,
    )
    notes = readable_metric_notes({"P_thr_5pct", "T_max_5pct", "T_mean_5pct", "E_short", "E_max"})
    n = next((s["n_trajectories"] for s in summaries.values() if s), "?")
    notes.append(f"(mean +/- std / median (%reached) across N={n} held-out test trajectories)")
    return table + "\n\n" + "\n".join(notes) + "\n"


def plot_loss_split(metrics_by_run: dict, figures_dir: Path) -> None:
    # Separated loss terms (validation) across the LAMBDA_PHYSICS sweep, for
    # the report's fig:pinn_loss_split. Both terms come straight from each
    # run's own extra_history (training/pushforward.py's val_data_history /
    # val_physics_history) -- unweighted, per-epoch, no re-training/re-eval.
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, run_id in SOURCES.items():
        val_data = metrics_by_run[run_id].get("extra_history", {}).get("val_data")
        if not val_data:
            print(f"WARNING: {run_id} has no val_data in extra_history -- skipping in loss-split plot.",
                  file=sys.stderr)
            continue
        color, marker = STYLE[label]
        ax.plot(range(1, len(val_data) + 1), val_data, marker, ms=3, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$L_{\mathrm{data}}$ (validation)")
    ax.grid(True, which="both")
    ax.legend()
    ax.set_title("Data loss (validation) vs epoch")
    plt.tight_layout()
    plt.savefig(figures_dir / "pinn_ldata_val.png", dpi=150, bbox_inches="tight")
    plt.close()

    # PINN=0 excluded here, not just left to plot as a flat line: with
    # LAMBDA_PHYSICS=0, pushforward.py's pushforward_loss() is never even
    # called for the physics term (`if cfg.LAMBDA_PHYSICS > 0` guard) -- its
    # val_physics is a literal 0.0 placeholder every epoch, not a measured
    # residual, and log-scale can't render a zero anyway.
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, run_id in SOURCES.items():
        if run_id == "p3_pinn_0":
            continue
        val_physics = metrics_by_run[run_id].get("extra_history", {}).get("val_physics")
        if not val_physics:
            print(f"WARNING: {run_id} has no val_physics in extra_history -- skipping in loss-split plot.",
                  file=sys.stderr)
            continue
        color, marker = STYLE[label]
        ax.plot(range(1, len(val_physics) + 1), val_physics, marker, ms=3, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$R_{\mathrm{phys}}$ (validation, unweighted)")
    ax.grid(True, which="both")
    ax.legend()
    ax.set_title("Unweighted PDE residual (validation) vs epoch -- PINN=0 excluded (never computed)")
    plt.tight_layout()
    plt.savefig(figures_dir / "pinn_rphys_val.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {figures_dir / 'pinn_ldata_val.png'} and {figures_dir / 'pinn_rphys_val.png'}")


def main():
    metrics_by_run = {run_id: load_run_metrics(REPO_ROOT, run_id) for run_id in SOURCES.values()}
    if any(m is None for m in metrics_by_run.values()):
        print("ERROR: all 4 p3_pinn_* runs are required for the plot below.", file=sys.stderr)
        sys.exit(1)
    curves = {label: metrics_by_run[run_id]["curves"] for label, run_id in SOURCES.items()}

    run_dir = REPO_ROOT / "runs" / RUN_ID
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summaries = {run_id: load_multi_summary(REPO_ROOT, run_id) for run_id in SOURCES.values()}

    table_tex = build_table(summaries)
    (run_dir / "table.tex").write_text(table_tex + "\n")
    print(table_tex)
    print(f"\nSaved {run_dir / 'table.tex'}")

    table_readable = build_readable_table(summaries)
    (run_dir / "table_readable.txt").write_text(table_readable)
    print(table_readable)
    print(f"Saved {run_dir / 'table_readable.txt'}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for (label, c), marker in zip(curves.items(), MARKERS):
        ax.plot(c["t"], c["err_max"], marker, ms=3, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("max absolute error along the beam (log)")
    ax.grid(True, which="both")
    ax.legend()
    ax.set_title("max rollout error over time vs PINN residual weight")
    plt.tight_layout()
    plt.savefig(figures_dir / "error_pinn.png", dpi=150, bbox_inches="tight")
    plt.close()

    plot_loss_split(metrics_by_run, figures_dir)

    summary_lines = [f"{RUN_ID} -- max rollout error vs LAMBDA_PHYSICS\n"]
    for label, source_run_id in SOURCES.items():
        e_stats = curves[label]["err_max"]
        summary_lines.append(f"{label} (source: {source_run_id}): "
                              f"final err_max={e_stats[-1]:.4e}  peak err_max={max(e_stats):.4e}")
    summary = "\n".join(summary_lines) + "\n"
    (run_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"Done -- outputs in {run_dir}")


if __name__ == "__main__":
    main()
