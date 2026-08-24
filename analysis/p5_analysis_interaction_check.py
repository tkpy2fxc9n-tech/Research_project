#!/usr/bin/env python3
# Phase 5 (parameter-interaction confirmation): checks whether M_BACK and
# features interact, or whether their effects on rollout error are
# (approximately) additive -- the two single-factor sweeps that varied them
# independently in phase 2 (p2_mback3, p2_feat_u_ut, both against
# p2_reference) never tested the combination M_BACK=3 + features={U,Ut}
# together. p5_mback3_feat_u_ut fills that missing cell of the 2x2 grid.
#
# Same plot/table approach as p2_analysis_stencils_parameters.py, reused via
# analysis/_common.py, but for a single 2x2 grid instead of 4 independent
# sweeps, plus an explicit additive-effect check: does the combined run's
# headline metrics match reference + (mback3 - reference) +
# (feat_u_ut - reference), or diverge from it?
#
# Usage: python analysis/p5_analysis_interaction_check.py
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

# (label, run_id) for the 2x2 grid, in table/plot order.
GRID_ROWS = [
    ("$M_{\\mathrm{BACK}}=2$, $\\{u\\}$ (reference)", "p2_reference"),
    ("$M_{\\mathrm{BACK}}=3$, $\\{u\\}$", "p2_mback3"),
    ("$M_{\\mathrm{BACK}}=2$, $\\{u,\\dot{u}\\}$", "p2_feat_u_ut"),
    ("$M_{\\mathrm{BACK}}=3$, $\\{u,\\dot{u}\\}$ (combined)", "p5_mback3_feat_u_ut"),
]

READABLE_LABELS = {
    "$M_{\\mathrm{BACK}}=2$, $\\{u\\}$ (reference)": "M_BACK=2, {u} (reference)",
    "$M_{\\mathrm{BACK}}=3$, $\\{u\\}$": "M_BACK=3, {u}",
    "$M_{\\mathrm{BACK}}=2$, $\\{u,\\dot{u}\\}$": "M_BACK=2, {u, u_dot}",
    "$M_{\\mathrm{BACK}}=3$, $\\{u,\\dot{u}\\}$ (combined)": "M_BACK=3, {u, u_dot} (combined)",
}

MARKERS = ["s-", "o-", "^-", "d-"]

DIAG_RUN_ID = "p5_diag_interaction"
COMBO_RUN_ID = "p5_mback3_feat_u_ut"


def load_grid_summaries() -> dict:
    run_ids = {run_id for _, run_id in GRID_ROWS}
    return {run_id: load_multi_summary(REPO_ROOT, run_id) for run_id in run_ids}


def build_grid_table(summaries: dict) -> str:
    rows = []
    for label, run_id in GRID_ROWS:
        s = summaries[run_id]
        if s is None:
            rows.append([label, *([MISSING_ROW] * 7)])
            continue
        rows.append([
            label,
            fmt_sci_mean_std(s["E_short"]["mean"], s["E_short"]["std"]),
            fmt_pct_mean_std(s["P_thr_5pct"]["mean"], s["P_thr_5pct"]["std"]),
            fmt_sci_mean_std(s["E_max"]["mean"], s["E_max"]["std"]),
            fmt_time_censored(s["T_max_5pct"]["median_reached"], s["T_max_5pct"]["pct_reached"]),
            fmt_time_censored(s["T_max_10pct"]["median_reached"], s["T_max_10pct"]["pct_reached"]),
            fmt_time_censored(s["T_mean_5pct"]["median_reached"], s["T_mean_5pct"]["pct_reached"]),
            fmt_time_censored(s["T_mean_10pct"]["median_reached"], s["T_mean_10pct"]["pct_reached"]),
        ])
    n = next((s["n_trajectories"] for s in summaries.values() if s), "?")
    return render_latex_table(
        caption=f"Confirmation of the weak-interaction assumption: the two stencil "
                f"parameters swept independently in phase 2 ($M_{{\\mathrm{{BACK}}}}$ and "
                f"input features), now tested together. Evaluated by autoregressive rollout on "
                f"$N={n}$ held-out test trajectories; see Table~\\ref{{tab:stencil_ablation}} for "
                f"the mean/std and censored-median conventions used here.",
        label="tab:interaction_confirmation",
        col_spec="lccccccc",
        header_cells=["Configuration", "$E_{\\mathrm{short}}$", "$P_{\\mathrm{thr}}$", "$E_{\\max}$",
                      "$T_{5\\%}^{\\max}$", "$T_{10\\%}^{\\max}$", "$T_{5\\%}^{\\mathrm{mean}}$",
                      "$T_{10\\%}^{\\mathrm{mean}}$"],
        rows=rows,
    )


def build_grid_readable_table(summaries: dict) -> str:
    rows = []
    for label, run_id in GRID_ROWS:
        readable_label = READABLE_LABELS[label]
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


def plot_interaction(metrics_by_run: dict) -> None:
    run_dir = REPO_ROOT / "runs" / "p5" / COMBO_RUN_ID
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for (label, run_id), marker in zip(GRID_ROWS, MARKERS):
        m = metrics_by_run[run_id]
        if m is None:
            continue
        ax.plot(m["curves"]["t"], m["curves"]["err_max"], marker, ms=3, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("max absolute error along the beam (log)")
    ax.grid(True, which="both")
    ax.legend()
    ax.set_title("max rollout error over time -- M_BACK x features interaction check")
    plt.tight_layout()
    plt.savefig(figures_dir / "error_interaction.png", dpi=150, bbox_inches="tight")
    plt.close()


# Additive-effect check on the two headline scalars (E_short: lower is
# better; T_max_5pct: higher is better, capped at the rollout horizon). If
# the two parameters' individual effects were purely additive, the combined
# run's value would equal reference + (mback3 - reference) +
# (feat_u_ut - reference) = mback3 + feat_u_ut - reference. Comparing that
# prediction to what was actually measured is the concrete evidence for the
# thesis's weak-interaction-assumption paragraph.
def additive_check(summaries: dict) -> str:
    for _, run_id in GRID_ROWS:
        if summaries[run_id] is None:
            return (f"Cannot compute the additive-effect check -- {run_id} has no multi-trajectory "
                     f"results yet (run still training/queued, or not backfilled via "
                     f"scripts/eval_multi_trajectory.py?).\n")

    # Point estimate per scalar: mean for E_short (continuous, unbiased),
    # median_reached for T_max_5pct (censored -- see
    # aggregate_multi_rollout_metrics). Using the multi-trajectory summary
    # instead of a single stored rollout, same source as build_grid_table.
    def point(run_id, key):
        s = summaries[run_id][key]
        return s["mean"] if "mean" in s else s["median_reached"]

    ref, mback3, feat, combo = ("p2_reference", "p2_mback3", "p2_feat_u_ut", COMBO_RUN_ID)

    lines = ["Additive-effect check (weak-interaction assumption)\n"]
    for key, higher_is_better in (("E_short", False), ("T_max_5pct", True)):
        r, b, f, c = point(ref, key), point(mback3, key), point(feat, key), point(combo, key)
        if None in (r, b, f, c):
            lines.append(f"{key}: cannot compute (a required value is missing/None, e.g. "
                         f"one config never reaches this threshold).")
            continue
        predicted = b + f - r
        diff = c - predicted
        rel = abs(diff) / abs(predicted) if predicted else float("inf")
        direction = "better than" if (diff < 0) == higher_is_better else "worse than"
        verdict = "holds (roughly additive)" if rel < 0.15 else "does NOT hold (real interaction)"
        lines.append(
            f"{key}: reference={r:.4g}, M_BACK=3 alone={b:.4g}, features={{U,Ut}} alone={f:.4g}, "
            f"combined (measured)={c:.4g}, combined (additive prediction)={predicted:.4g} "
            f"-- measured is {rel*100:.1f}% {direction} the additive prediction. "
            f"Weak-interaction assumption {verdict}."
        )
    return "\n".join(lines) + "\n"


def main():
    run_ids = {run_id for _, run_id in GRID_ROWS}
    metrics_by_run = {run_id: load_run_metrics(REPO_ROOT, run_id) for run_id in run_ids}
    summaries = load_grid_summaries()

    if any(m is None for m in metrics_by_run.values()):
        missing = [run_id for run_id, m in metrics_by_run.items() if m is None]
        print(f"WARNING: missing results for {missing} -- run "
              f"'sbatch scripts/run.job runs/p5/{COMBO_RUN_ID}/config.yaml' first if it's "
              f"not finished yet.", file=sys.stderr)

    plot_interaction(metrics_by_run)

    out_dir = REPO_ROOT / "runs" / DIAG_RUN_ID
    out_dir.mkdir(parents=True, exist_ok=True)

    table_tex = build_grid_table(summaries)
    (out_dir / "table.tex").write_text(table_tex + "\n")
    print(table_tex)

    table_readable = build_grid_readable_table(summaries)
    (out_dir / "table_readable.txt").write_text(table_readable)
    print(table_readable)

    summary = additive_check(summaries)
    (out_dir / "summary.txt").write_text(summary)
    print(summary)

    print(f"Done -- outputs in {out_dir} and "
          f"{REPO_ROOT / 'runs' / 'p5' / COMBO_RUN_ID / 'figures'}")


if __name__ == "__main__":
    main()
