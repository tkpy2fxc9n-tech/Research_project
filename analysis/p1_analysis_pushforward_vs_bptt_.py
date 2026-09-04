#!/usr/bin/env python3
# Phase 1: 2-curve version of pushforward_vs_bptt.py, restricted to the
# "loss val" pair specifically retained for phase 1: pushforward (labelled
# "Pushforward") vs bptt (labelled "TBPTT"). Two separate figures: max
# absolute rollout error (err_max) and mean absolute rollout error
# (err_mean_abs, see evaluate/metrics.py's compute_error_curves --
# physical-unit mean over the beam, not the L2-ratio err_rel_mean). These
# still come from each run's own single showcase rollout (metrics.json's
# curves) -- a per-timestep curve isn't something you average over 99
# trajectories, only the scalars below are.
#
# Also writes table.tex: the rollout-error-metrics report table (T_5%^max,
# T_10%^max, T_5%^mean, T_10%^mean, E_max, t_E_max, P_thr) for the two
# regimes, now from runs/p1_analysis/multi_trajectory_summary.json (99 test
# trajectories, see scripts/eval_p1_multi_trajectory.py) instead of each
# run's single stored rollout_idx -- E_max/t_E_max/P_thr as mean +/- std
# across trajectories, T_* as median time among trajectories that actually
# crossed that threshold, with the % that did in parentheses (right-censored:
# a trajectory that never crosses is the BEST outcome, not a missing value).
#
# Usage: python analysis/p1_analysis_pushforward_vs_bptt_.py
#   (requires runs/p1_analysis/multi_trajectory_summary.json to exist --
#   regenerate via: sbatch scripts/run_analysis.job scripts/eval_p1_multi_trajectory.py)
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))
from _common import (  # noqa: E402
    fmt_duration_h, fmt_duration_h_readable, fmt_pct_mean_std, fmt_sci_mean_std, fmt_time_censored,
    fmt_time_mean_std, load_run_metrics, fmt_pct_mean_std_readable, fmt_sci_mean_std_readable,
    fmt_time_censored_readable, fmt_time_mean_std_readable, readable_metric_notes, render_latex_table,
    render_readable_table, MISSING_ROW,
)

RUN_DIR = REPO_ROOT / "runs" / "p1_analysis"
FIGURES_DIR = RUN_DIR / "figures"
MULTI_SUMMARY_PATH = RUN_DIR / "multi_trajectory_summary.json"

SOURCES = {
    "Pushforward": "p1_pushforward",
    "TBPTT": "p1_bptt",
}
N_TAIL_DROP = 2


def load_multi_summary() -> dict:
    if not MULTI_SUMMARY_PATH.exists():
        print(f"ERROR: {MULTI_SUMMARY_PATH} not found -- regenerate it first via:\n"
              f"  sbatch scripts/run_analysis.job scripts/eval_p1_multi_trajectory.py", file=sys.stderr)
        sys.exit(1)
    with open(MULTI_SUMMARY_PATH) as f:
        return json.load(f)


def build_table(multi_summary: dict, train_time_by_label: dict) -> str:
    # Transposed vs build_readable_table below (metrics as rows, regimes as
    # columns) -- table.tex only, kept as a separate function rather than a
    # shared one so table_readable.txt's own row/column layout is untouched.
    METRIC_ROWS = ["$T_{5\\%}^{\\max}$", "$T_{10\\%}^{\\max}$", "$E_{\\max}$",
                   "$t_{E_{\\max}}$", "$P_{\\mathrm{thr}}$", "$t_{\\mathrm{train}}$"]

    def metrics_for(label: str) -> list[str]:
        s = multi_summary.get(label)
        if s is None:
            return [MISSING_ROW] * 6
        return [
            fmt_time_censored(s["T_max_5pct"]["median_reached"], s["T_max_5pct"]["pct_reached"]),
            fmt_time_censored(s["T_max_10pct"]["median_reached"], s["T_max_10pct"]["pct_reached"]),
            fmt_sci_mean_std(s["E_max"]["mean"], s["E_max"]["std"]),
            fmt_time_mean_std(s["t_E_max"]["mean"], s["t_E_max"]["std"]),
            fmt_pct_mean_std(s["P_thr_5pct"]["mean"], s["P_thr_5pct"]["std"]),
            fmt_duration_h(train_time_by_label.get(label)),
        ]

    values_by_label = {label: metrics_for(label) for label in SOURCES}
    rows = [[metric_name, *(values_by_label[label][i] for label in SOURCES)]
            for i, metric_name in enumerate(METRIC_ROWS)]

    n = next((s["n_trajectories"] for s in multi_summary.values() if s), "?")
    return render_latex_table(
        caption=f"Rollout error metrics for the two training regimes, evaluated by autoregressive "
                f"rollout on $N={n}$ held-out test trajectories. $E_{{\\max}}$, $t_{{E_{{\\max}}}}$ and "
                f"$P_{{\\mathrm{{thr}}}}$ are reported as mean $\\pm$ std across trajectories. $T_{{5\\%}}^{{\\max}}/"
                f"T_{{10\\%}}^{{\\max}}$ are right-censored (not every trajectory reaches the threshold "
                f"within the simulated window); each cell reports the median crossing time among "
                f"trajectories that reached it, with the fraction that reached it in parentheses. "
                f"$t_{{\\mathrm{{train}}}}$ is the total wall-clock training time for that run.",
        label="tab:rollout_metrics",
        col_spec="l" + "c" * len(SOURCES),
        header_cells=["Metric", *SOURCES],
        rows=rows,
    )


def build_readable_table(multi_summary: dict, train_time_by_label: dict) -> str:
    rows = []
    for label in SOURCES:
        s = multi_summary.get(label)
        if s is None:
            rows.append([label, *(["--"] * 8)])
            continue
        rows.append([
            label,
            fmt_time_censored_readable(s["T_max_5pct"]["median_reached"], s["T_max_5pct"]["pct_reached"]),
            fmt_time_censored_readable(s["T_max_10pct"]["median_reached"], s["T_max_10pct"]["pct_reached"]),
            fmt_time_censored_readable(s["T_mean_5pct"]["median_reached"], s["T_mean_5pct"]["pct_reached"]),
            fmt_time_censored_readable(s["T_mean_10pct"]["median_reached"], s["T_mean_10pct"]["pct_reached"]),
            fmt_sci_mean_std_readable(s["E_max"]["mean"], s["E_max"]["std"]),
            fmt_time_mean_std_readable(s["t_E_max"]["mean"], s["t_E_max"]["std"]),
            fmt_pct_mean_std_readable(s["P_thr_5pct"]["mean"], s["P_thr_5pct"]["std"]),
            fmt_duration_h_readable(train_time_by_label.get(label)),
        ])
    n = next((s["n_trajectories"] for s in multi_summary.values() if s), "?")
    table = render_readable_table(
        header_cells=["Regime", "T_max_5%", "T_max_10%", "T_mean_5%", "T_mean_10%",
                      "E_max", "t_E_max", "P_thr_5%", "t_train"],
        rows=rows,
    )
    notes = readable_metric_notes({"E_max", "t_E_max", "T_max_5pct", "T_mean_5pct", "P_thr_5pct"})
    notes.append("t_train : total wall-clock training time for that run.")
    notes.append(f"(mean +/- std / median (%reached) across N={n} held-out test trajectories)")
    return table + "\n\n" + "\n".join(notes) + "\n"


def main():
    metrics_by_run = {run_id: load_run_metrics(REPO_ROOT, run_id) for run_id in SOURCES.values()}
    if any(m is None for m in metrics_by_run.values()):
        print("ERROR: both p1_pushforward and p1_bptt are required for the plots below.",
              file=sys.stderr)
        sys.exit(1)
    curves = {label: metrics_by_run[run_id]["curves"] for label, run_id in SOURCES.items()}

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    multi_summary = load_multi_summary()
    train_time_by_label = {label: metrics_by_run[run_id]["scalars"].get("train_time_s")
                            for label, run_id in SOURCES.items()}

    table_tex = build_table(multi_summary, train_time_by_label)
    (RUN_DIR / "table.tex").write_text(table_tex + "\n")
    print(table_tex)
    print(f"\nSaved {RUN_DIR / 'table.tex'}")

    table_readable = build_readable_table(multi_summary, train_time_by_label)
    (RUN_DIR / "table_readable.txt").write_text(table_readable)
    print(table_readable)
    print(f"Saved {RUN_DIR / 'table_readable.txt'}")

    for field, ylabel, fname in (
        ("err_max", "max absolute error along the beam (log)", "error_max_pushforward_vs_fbptt.png"),
        ("err_mean_abs", "mean absolute error over the beam (log)", "error_mean_pushforward_vs_fbptt.png"),
    ):
        fig, ax = plt.subplots(figsize=(9, 6.5))   # taller than the other p0/p1 figures: at this
                                                     # fontsize the rotated ylabel needs the extra
                                                     # height or bbox_inches="tight" clips its tail
        markers = {"Pushforward": "s-", "TBPTT": "o-"}
        for label, c in curves.items():
            ax.plot(c["t"][:-N_TAIL_DROP], c[field][:-N_TAIL_DROP], markers[label], ms=3, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("time (s)", fontsize=17)
        ax.set_ylabel(ylabel, fontsize=17)
        ax.tick_params(axis="both", labelsize=17)
        ax.grid(True, which="both")
        ax.legend(fontsize=17)
        plt.tight_layout()
        out_path = FIGURES_DIR / fname
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")

    # Same two fields, but TBPTT's error divided by Pushforward's at each timestep
    # instead of the two curves overlaid -- how many times worse/better TBPTT is,
    # directly. Both regimes share the exact same rollout time grid (see the `t`
    # arrays above), so this is a plain elementwise ratio, no interpolation needed.
    for field, ylabel, fname in (
        ("err_max", "TBPTT / Pushforward -- max absolute error ratio", "error_max_ratio_fbptt_over_pushforward.png"),
        ("err_mean_abs", "TBPTT / Pushforward -- mean absolute error ratio",
         "error_mean_ratio_fbptt_over_pushforward.png"),
    ):
        t = curves["Pushforward"]["t"][:-N_TAIL_DROP]
        pf = curves["Pushforward"][field][:-N_TAIL_DROP]
        bp = curves["TBPTT"][field][:-N_TAIL_DROP]
        ratio = [b / p if p != 0 else float("nan") for b, p in zip(bp, pf)]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t, ratio, "d-", ms=3, color="tab:purple")
        ax.axhline(1.0, color="gray", linestyle="--", lw=1, label="ratio = 1 (equal error)")
        ax.set_xlabel("t")
        ax.set_ylabel(ylabel)
        ax.minorticks_on()   # linear axes have no minor ticks by default -- unlike the log-scale
                              # plots above, "which='both'" below is a no-op without this first
        ax.grid(True, which="major")
        ax.grid(True, which="minor", alpha=0.3, linestyle=":")
        ax.legend()
        plt.tight_layout()
        out_path = FIGURES_DIR / fname
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
