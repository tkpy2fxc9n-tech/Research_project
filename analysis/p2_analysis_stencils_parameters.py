#!/usr/bin/env python3
# Phase 2 (stencil parameters): max absolute rollout error along the beam
# vs simulation time, one figure per swept stencil parameter -- input
# features, M_BACK (backward stencil size), ndt (timestep stride), N_FWD
# (forward prediction hops). Merged from
# p2_diag_error_{features,mback,ndt,nfwd}.py, which were four near-identical
# copies of the same SOURCES-loop/plot/summary structure; that structure is
# now the single plot_error_sweep() below, called once per sweep with its
# own sources/cutoff. Each sweep keeps its own original RUN_ID/output dir --
# only the .py files were combined.
#
# All 4 are pure post-processing over each source run's results.yaml, no
# t_div markers. A sweep whose source runs aren't all finished yet is
# skipped with a WARNING rather than aborting the others (e.g. while
# p2_feat_all is being retrained, mback/ndt/nfwd can still run).
#
# Tail-zero artifact -- FIXED AT THE SOURCE 2026-08-23. Every plot_error_sweep
# call here used to pass a hand-measured t_cutoff/tail_drop_n to hide 1-4
# trailing samples that were actually never written by autoregressive_rollout
# (an off-by-one in evaluate/metrics.py's _rollout_steps -- see
# p3_analysis_pinn_comparative.py's module docstring for the full diagnosis),
# misread here as the model "fully damping to ~0". Fixed at the source, and
# all 10 p2_* runs' results.yaml were patched to drop those stale samples --
# no truncation needed in this script anymore.
#
# Also writes runs/p2_analysis/table.tex: the single stencil-ablation report
# table (all 4 sweeps combined into one table, 3 blocks -- temporal input
# depth, prediction horizon/subsampling, input features -- exactly like the
# 4 plot sweeps above but as one merged table instead of 4 separate
# figures). M_BACK=2/N_FWD=3/ndt=3/features={u} all share the same reference
# config, so they all reuse p2_reference's own results, same substitution
# the plot sweeps above already make for M_BACK/N_FWD/ndt (the features
# sweep used to point its own reference row at "p2_feat_u", a run that was
# never actually trained since its config is identical to p2_reference --
# now points at p2_reference directly instead).
#
# Usage: python analysis/p2_analysis_stencils_parameters.py
from __future__ import annotations

import sys
from pathlib import Path

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))
from _common import run_dir as _run_dir  # noqa: E402
from _common import (  # noqa: E402
    fmt_pct_mean_std, fmt_sci_mean_std, fmt_time_censored, load_multi_summary,
    fmt_pct_mean_std_readable, fmt_sci_mean_std_readable, fmt_time_censored_readable,
    readable_metric_notes, render_latex_table, render_readable_table, MISSING_ROW,
)

MARKERS = ["s-", "o-", "^-", "d-", "v-"]

# (row label, source run_id) per block, in table order. The reference run_id
# (p2_reference) repeats across blocks -- load_run_metrics is only called
# once per distinct run_id (see build_stencil_table), not once per row.
STENCIL_TABLE_ROWS = [
    ("section", "Temporal input depth"),
    ("$M_{\\mathrm{BACK}} = 1$", "p2_mback1"),
    ("$M_{\\mathrm{BACK}} = 2$ (reference)", "p2_reference"),
    ("$M_{\\mathrm{BACK}} = 3$", "p2_mback3"),
    ("section", "Prediction horizon and temporal subsampling"),
    ("$N_{\\mathrm{FWD}} = 1$", "p2_nfwd1"),
    ("$N_{\\mathrm{FWD}} = 2$", "p2_nfwd2"),
    ("$N_{\\mathrm{FWD}} = 3$ (reference)", "p2_reference"),
    ("$N_{\\mathrm{FWD}} = 4$", "p2_nfwd4"),
    ("$N_{\\mathrm{FWD}} = 5$", "p2_nfwd5"),
    ("$n_{dt} = 2$", "p2_ndt2"),
    ("$n_{dt} = 3$ (reference)", "p2_reference"),
    ("$n_{dt} = 4$", "p2_ndt4"),
    ("$n_{dt} = 5$", "p2_ndt5"),
    ("section", "Input features"),
    ("$\\{u\\}$ (reference)", "p2_reference"),
    ("$\\{u, \\dot{u}\\}$", "p2_feat_u_ut"),
    ("$\\{\\dot{u}, u_{xx}\\}$", "p2_feat_ut_uxx"),
    ("$\\{u, \\dot{u}, u_{xx}\\}$", "p2_feat_all"),
]

# Plain-text row labels for table_readable.txt -- STENCIL_TABLE_ROWS' own
# labels are LaTeX math (table.tex is their only other consumer), unreadable
# verbatim in a text file.
READABLE_LABELS = {
    "$M_{\\mathrm{BACK}} = 1$": "M_BACK = 1",
    "$M_{\\mathrm{BACK}} = 2$ (reference)": "M_BACK = 2 (reference)",
    "$M_{\\mathrm{BACK}} = 3$": "M_BACK = 3",
    "$N_{\\mathrm{FWD}} = 1$": "N_FWD = 1",
    "$N_{\\mathrm{FWD}} = 2$": "N_FWD = 2",
    "$N_{\\mathrm{FWD}} = 3$ (reference)": "N_FWD = 3 (reference)",
    "$N_{\\mathrm{FWD}} = 4$": "N_FWD = 4",
    "$N_{\\mathrm{FWD}} = 5$": "N_FWD = 5",
    "$n_{dt} = 2$": "n_dt = 2",
    "$n_{dt} = 3$ (reference)": "n_dt = 3 (reference)",
    "$n_{dt} = 4$": "n_dt = 4",
    "$n_{dt} = 5$": "n_dt = 5",
    "$\\{u\\}$ (reference)": "{u} (reference)",
    "$\\{u, \\dot{u}\\}$": "{u, u_dot}",
    "$\\{\\dot{u}, u_{xx}\\}$": "{u_dot, u_xx}",
    "$\\{u, \\dot{u}, u_{xx}\\}$": "{u, u_dot, u_xx}",
}


def load_stencil_summaries() -> dict:
    run_ids = {run_id for row in STENCIL_TABLE_ROWS if row[0] != "section" for run_id in (row[1],)}
    return {run_id: load_multi_summary(REPO_ROOT, run_id) for run_id in run_ids}


def build_stencil_table(summaries: dict) -> str:
    rows = []
    for label, run_id in STENCIL_TABLE_ROWS:
        if label == "section":
            rows.append(("section", run_id))
            continue
        s = summaries[run_id]
        if s is None:
            rows.append([label, *([MISSING_ROW] * 4)])
            continue
        rows.append([
            label,
            fmt_pct_mean_std(s["P_thr_5pct"]["mean"], s["P_thr_5pct"]["std"]),
            fmt_sci_mean_std(s["E_max"]["mean"], s["E_max"]["std"]),
            fmt_time_censored(s["T_max_5pct"]["median_reached"], s["T_max_5pct"]["pct_reached"]),
            fmt_time_censored(s["T_max_10pct"]["median_reached"], s["T_max_10pct"]["pct_reached"]),
        ])

    n = next((s["n_trajectories"] for s in summaries.values() if s), "?")
    return render_latex_table(
        caption=f"Ablation of the stencil parameters against a single fixed reference "
                f"configuration ($M_{{\\mathrm{{BACK}}}}=2$, $N_{{\\mathrm{{FWD}}}}=3$, $n_{{dt}}=3$, "
                f"features $=\\{{u\\}}$). Each block varies one parameter at a time; all other "
                f"parameters are held at the reference values shown in each block. Each "
                f"configuration is evaluated by autoregressive rollout on $N={n}$ held-out test "
                f"trajectories. $P_{{\\mathrm{{thr}}}}$ and $E_{{\\max}}$ are reported as mean "
                f"$\\pm$ std across trajectories. $T_{{5\\%}}^{{\\max}}/T_{{10\\%}}^{{\\max}}$ are "
                f"right-censored (not every trajectory reaches the threshold within the simulated "
                f"window); each cell reports the median crossing time among trajectories that "
                f"reached it, with the fraction that reached it in parentheses.",
        label="tab:stencil_ablation",
        col_spec="lcccc",
        header_cells=["Configuration", "$P_{\\mathrm{thr}}$", "$E_{\\max}$",
                      "$T_{5\\%}^{\\max}$", "$T_{10\\%}^{\\max}$"],
        rows=rows,
    )


def build_stencil_readable_table(summaries: dict) -> str:
    rows = []
    for label, run_id in STENCIL_TABLE_ROWS:
        if label == "section":
            rows.append(("section", run_id))
            continue
        readable_label = READABLE_LABELS.get(label, label)
        s = summaries[run_id]
        if s is None:
            rows.append([readable_label, *(["--"] * 7)])
            continue
        rows.append([
            readable_label,
            fmt_sci_mean_std_readable(s["E_short"]["mean"], s["E_short"]["std"]),
            fmt_pct_mean_std_readable(s["P_thr_5pct"]["mean"], s["P_thr_5pct"]["std"]),
            fmt_sci_mean_std_readable(s["E_max"]["mean"], s["E_max"]["std"]),
            fmt_time_censored_readable(s["T_max_5pct"]["median_reached"], s["T_max_5pct"]["pct_reached"]),
            fmt_time_censored_readable(s["T_max_10pct"]["median_reached"], s["T_max_10pct"]["pct_reached"]),
            fmt_time_censored_readable(s["T_mean_5pct"]["median_reached"], s["T_mean_5pct"]["pct_reached"]),
            fmt_time_censored_readable(s["T_mean_10pct"]["median_reached"], s["T_mean_10pct"]["pct_reached"]),
        ])
    table = render_readable_table(
        header_cells=["Configuration", "E_short", "P_thr_5%", "E_max",
                      "T_max_5%", "T_max_10%", "T_mean_5%", "T_mean_10%"],
        rows=rows,
    )
    notes = readable_metric_notes({"E_short", "P_thr_5pct", "E_max", "T_max_5pct", "T_mean_5pct"})
    n = next((s["n_trajectories"] for s in summaries.values() if s), "?")
    notes.append(f"(mean +/- std / median (%reached) across N={n} held-out test trajectories)")
    return table + "\n\n" + "\n".join(notes) + "\n"


def plot_error_sweep(run_id: str, sources: dict, title: str, figure_filename: str) -> bool:
    curves = {}
    for label, source_run_id in sources.items():
        source_metrics_path = _run_dir(REPO_ROOT, source_run_id) / "results.yaml"
        if not source_metrics_path.exists():
            print(f"WARNING: {source_metrics_path} not found -- run {source_run_id} first, "
                  f"skipping {run_id}.", file=sys.stderr)
            return False
        with open(source_metrics_path) as f:
            metrics = yaml.safe_load(f)
        if not metrics["curves"]["t"]:
            print(f"WARNING: {source_metrics_path} has no curves -- {source_run_id} may not have "
                  f"finished successfully, skipping {run_id}.", file=sys.stderr)
            return False
        curves[label] = {"t": metrics["curves"]["t"], "err_max": metrics["curves"]["err_max"]}

    run_dir = REPO_ROOT / "runs" / run_id
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6.5))   # taller than the default 9x5: at this fontsize the
                                                 # rotated ylabel needs the extra height or
                                                 # bbox_inches="tight" clips its tail
    for (label, c), marker in zip(curves.items(), MARKERS):
        ax.plot(c["t"], c["err_max"], marker, ms=3, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("time (s)", fontsize=17)
    ax.set_ylabel("max absolute error along the beam (log)", fontsize=17)
    ax.tick_params(axis="both", labelsize=17)
    ax.grid(True, which="both")
    ax.legend(fontsize=17)
    plt.tight_layout()
    plt.savefig(figures_dir / figure_filename, dpi=150, bbox_inches="tight")
    plt.close()

    summary_lines = [f"{run_id} -- {title}\n"]
    for label, source_run_id in sources.items():
        c = curves[label]
        summary_lines.append(f"{label} (source: {source_run_id}): "
                              f"final err_max={c['err_max'][-1]:.4e}  peak err_max={max(c['err_max']):.4e}")
    summary = "\n".join(summary_lines) + "\n"
    (run_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"Done -- outputs in {run_dir}")
    return True


def main():
    ran_any = False

    # features={U} IS the base default, same reference config as M_BACK/
    # N_FWD/ndt below -- p2_feat_u (a dedicated run for it) was never
    # actually trained since it'd be identical to p2_reference, so this row
    # reuses p2_reference too, same substitution as the other 3 sweeps.
    ran_any |= plot_error_sweep(
        "p2_diag_error_features",
        {"features={U}": "p2_reference", "features={U,Ut}": "p2_feat_u_ut",
         "features={Ut,Uxx}": "p2_feat_ut_uxx", "features={U,Ut,Uxx}": "p2_feat_all"},
        "max rollout error over time vs input features", "error_features.png")

    # M_BACK=2 has no dedicated run of its own: it's the base.yaml default,
    # i.e. exactly p2_reference's config -- reused here rather than
    # re-run, same logic the user's own tracking spreadsheet already uses
    # for this row.
    ran_any |= plot_error_sweep(
        "p2_diag_error_mback",
        {"M_BACK=1": "p2_mback1", "M_BACK=2 (reference)": "p2_reference", "M_BACK=3": "p2_mback3"},
        "max rollout error over time vs M_BACK", "error_mback.png")

    # ndt=3 has no dedicated run of its own: it's the base.yaml default,
    # i.e. exactly p2_reference's config -- reused here rather than re-run.
    ran_any |= plot_error_sweep(
        "p2_diag_error_ndt",
        {"ndt=2": "p2_ndt2", "ndt=3 (reference)": "p2_reference", "ndt=4": "p2_ndt4", "ndt=5": "p2_ndt5"},
        "max rollout error over time vs ndt", "error_ndt.png")

    # N_FWD=3 has no dedicated run of its own: it's the base.yaml default,
    # i.e. exactly p2_reference's config -- reused here rather than re-run.
    ran_any |= plot_error_sweep(
        "p2_diag_error_nfwd",
        {"N_FWD=1": "p2_nfwd1", "N_FWD=2": "p2_nfwd2", "N_FWD=3 (reference)": "p2_reference",
         "N_FWD=4": "p2_nfwd4", "N_FWD=5": "p2_nfwd5"},
        "max rollout error over time vs N_FWD", "error_nfwd.png")

    if not ran_any:
        sys.exit(1)

    # -- merged stencil-ablation table, separate output folder from the 4
    # plot sweeps above (those stay untouched) --
    table_dir = REPO_ROOT / "runs" / "p2_analysis"
    table_dir.mkdir(parents=True, exist_ok=True)
    summaries = load_stencil_summaries()
    table_tex = build_stencil_table(summaries)
    (table_dir / "table.tex").write_text(table_tex + "\n")
    print(table_tex)
    print(f"\nSaved {table_dir / 'table.tex'}")

    table_readable = build_stencil_readable_table(summaries)
    (table_dir / "table_readable.txt").write_text(table_readable)
    print(table_readable)
    print(f"Saved {table_dir / 'table_readable.txt'}")


if __name__ == "__main__":
    main()
