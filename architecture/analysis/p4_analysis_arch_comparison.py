#!/usr/bin/env python3
# Phase 4 (architecture-size ablation): compares the p4_arch_* runs (10
# defined in SOURCES below -- 9 finished as of 2026-08-20, the
# 512+256+64 reference not yet trained, see _load()), which differ only in
# HIDDEN_SIZES (everything else inherits phase2's winner -- pushforward,
# M_BACK=3, N_FWD=2, features=[U]). Pure post-processing over each source
# run's results.yaml, no re-training/re-inference.
#
# Two outputs:
#  1. table.csv -- one row per architecture: n_params, the rollout-error-
#     metrics report table's own columns (E_short, P_thr_5pct, E_max,
#     t_E_max, T_max_5pct, T_max_10pct, T_mean_5pct, T_mean_10pct -- see
#     evaluate/metrics.py's build_metrics), plus nn_flops/nn wall-clock (the
#     surrogate's own rollout inference) next to fd_flops/fd wall-clock (the
#     SAME quantities for the ground-truth finite-difference solver -- both
#     already measured per-run by evaluate/rollout.py's benchmark, no new
#     computation needed here), plus a derived fd/nn wall-clock speedup ratio.
#  2. error_max_vs_time_all_archs.png / error_mean_vs_time_all_archs.png --
#     err_max (max absolute error) and err_mean_abs (beam-averaged), both log scale, vs t
#     for every available architecture overlaid, same construction/metric as
#     every other error-vs-time figure in this campaign (h2/h3/h5/p11).
#
# Tail-zero artifact -- FIXED AT THE SOURCE 2026-08-23. All 9 MLP runs used
# to hit it at their very last point (t=4.98, amp_max==0.0 there, err_max
# collapsed to the reference's own amplitude -- identical across all 9 runs
# at that point): traced to an off-by-one in evaluate/metrics.py's
# _rollout_steps sampling past the last step autoregressive_rollout actually
# writes. Fixed there, and the affected p4_arch_* runs' results.yaml were
# patched to drop the one stale sample and recompute their scalars -- no
# T_CUTOFF truncation needed in this script anymore.
#
# Color/style encoding: 9 series exceeds the validated categorical palette's
# fixed-order cap (dataviz skill, palette.md: 8 hues), so per the skill's own
# rule ("never generate a 9th hue") linear does NOT get a 9th hue -- it's a
# qualitatively different baseline (linear model, no hidden layer) styled as
# a neutral black dotted reference line, same pattern this campaign already
# uses for reference curves (p11/h2/h3's "(reference)" entries). The 8 MLP
# architectures get the palette's 8 hues in their documented fixed order
# (validated for adjacent pairs on line charts), grouped/ordered by depth;
# linestyle is a second, depth-family encoding (solid=1 hidden layer,
# dashed=2, dash-dot=3) so the figure stays legible in grayscale too.
#
# Usage: python analysis/p4_analysis_arch_comparison.py
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "common"))
from _common import run_dir as _run_dir  # noqa: E402
from _common import (  # noqa: E402
    fill_missing_scalars, fmt_pct_mean_std, fmt_pct_mean_std_readable, fmt_sci_mean_std,
    fmt_sci_mean_std_readable, fmt_time_censored, fmt_time_censored_readable, fmt_time_mean_std,
    fmt_time_mean_std_readable, load_multi_summary, readable_metric_notes, render_latex_table,
    render_readable_table, resolve_amp_ref, MISSING_ROW,
)

RUN_ID = "p4_diag_arch_comparison"

# label, source run_id, color, linestyle
SOURCES = [
    ("64",            "p4_arch_64",          "#2a78d6", "-"),
    ("128",           "p4_arch_128",         "#eb6834", "-"),
    ("256",           "p4_arch_256",         "#1baf7a", "-"),
    ("64+16",         "p4_arch_64_16",       "#eda100", "--"),
    ("128+32",        "p4_arch_128_32",      "#e87ba4", "--"),
    ("256+16",        "p4_arch_256_16",      "#008300", "--"),
    ("128+64+16",     "p4_arch_128_64_16",   "#4a3aa7", "-."),
    ("256+128+32",    "p4_arch_256_128_32",  "#e34948", "-."),
    # Reuses the 128+64+16 hue (#4a3aa7) rather than a 9th hue (dataviz
    # skill, palette.md: 8-hue cap) -- still distinguishable from it via a
    # denser dash-dot-dot pattern, since both are 3-hidden-layer nets and
    # would otherwise collide on the "-." depth-family linestyle too.
    # p4_arch_512_256_64 hasn't been trained yet (config.yaml only, no
    # results.yaml as of 2026-08-20) -- this entry makes it show up in the
    # table/figures automatically once it has, no script edit needed then.
    ("512+256+64 (reference)", "p4_arch_512_256_64", "#4a3aa7", (0, (3, 1, 1, 1))),
    ("linear", "p4_arch_ridge",      "#0b0b0b", ":"),
]

def _load(run_id: str) -> dict | None:
    # Returns None (with a warning) rather than exiting, so a not-yet-trained
    # SOURCES entry (e.g. p4_arch_512_256_64 as of 2026-08-20) doesn't block
    # the other 9 -- it just starts appearing in the table/figures on its own
    # the first time this script runs after that run finishes.
    path = _run_dir(REPO_ROOT, run_id) / "results.yaml"
    if not path.exists():
        print(f"WARNING: {path} not found -- {run_id} hasn't been run yet, skipping it.", file=sys.stderr)
        return None
    with open(path) as f:
        metrics = yaml.safe_load(f)
    if not metrics["curves"]["t"]:
        print(f"WARNING: {path} has no curves -- {run_id} may not have finished successfully, skipping it.",
              file=sys.stderr)
        return None
    return metrics


def _fmt_flops(x):
    if x is None:
        return "--"
    for unit, div in (("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= div:
            return f"{x / div:.2f}{unit}"
    return f"{x:.0f}"


def _fmt_ms(med, std):
    if med is None:
        return "--"
    return f"{med * 1000:.2f}+-{std * 1000:.2f}"


def _fmt_t_div(x):
    return "never" if x is None else f"{x:.2f}s"


def _fmt_sci(x):
    return "--" if x is None else f"{x:.3e}"


CSV_HEADERS = ["architecture", "run_id", "n_params",
               "E_short_mean", "E_short_std", "P_thr_5pct_mean", "P_thr_5pct_std",
               "E_max_mean", "E_max_std", "t_E_max_mean", "t_E_max_std",
               "T_max_5pct_median", "T_max_5pct_pct_reached",
               "T_max_10pct_median", "T_max_10pct_pct_reached",
               "T_mean_5pct_median", "T_mean_5pct_pct_reached",
               "T_mean_10pct_median", "T_mean_10pct_pct_reached",
               "nn_flops", "nn_time_med_ms", "nn_time_std_ms",
               "fd_flops", "fd_time_med_ms", "fd_time_std_ms", "speedup_fd_over_nn"]


def build_rows(sources: list, metrics_by_run: dict, summaries: dict) -> list[list]:
    # Raw numeric values (not human-formatted strings) -- CSV is meant to be
    # re-parsed (pandas, Excel, another script), so units are in the header
    # instead of baked into cells like "73.61M" or "40.54+-0.24". Accuracy
    # columns (E_short..T_mean_10pct) come from the multi-trajectory summary
    # (mean/std for continuous metrics, median/pct_reached for the censored
    # T_* family -- see evaluate/metrics.py's aggregate_multi_rollout_metrics)
    # rather than a single stored rollout; n_params/flops/timing are still
    # single-run benchmark measurements (evaluate/rollout.py's
    # benchmark_inference), not something a test trajectory varies.
    rows = []
    for label, run_id, _, _ in sources:
        s = metrics_by_run[run_id]["scalars"]
        m = summaries.get(run_id)

        def cont(key):
            return (m[key]["mean"], m[key]["std"]) if m else (None, None)

        def cens(key):
            return (m[key]["median_reached"], m[key]["pct_reached"]) if m else (None, None)

        speedup = (s["fd_time_med_s"] / s["nn_time_med_s"]) if s.get("nn_time_med_s") else None
        rows.append([
            label, run_id, s.get("n_params"),
            *cont("E_short"), *cont("P_thr_5pct"), *cont("E_max"), *cont("t_E_max"),
            *cens("T_max_5pct"), *cens("T_max_10pct"), *cens("T_mean_5pct"), *cens("T_mean_10pct"),
            s.get("nn_flops"),
            s["nn_time_med_s"] * 1000 if s.get("nn_time_med_s") is not None else None,
            s["nn_time_std_s"] * 1000 if s.get("nn_time_std_s") is not None else None,
            s.get("fd_flops"),
            s["fd_time_med_s"] * 1000 if s.get("fd_time_med_s") is not None else None,
            s["fd_time_std_s"] * 1000 if s.get("fd_time_std_s") is not None else None,
            speedup,
        ])
    return rows


def write_csv(path: Path, rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)
        writer.writerows(rows)


def _fmt_pct(x):
    return "--" if x is None else f"{x:.1f}%"


def render_readable_tables(rows: list[list]) -> str:
    # Human-readable sibling of table.csv/table.tex, split into two
    # wide-spaced tables (accuracy/stability, then compute/speed) rather
    # than one 15-column wall of text -- table.csv itself stays raw/numeric
    # (see build_rows), this is purely for a human staring at a terminal or
    # an editor. E_max/t_E_max used to be left out of table A: every MLP row
    # was bit-identical due to the tail-zero artifact (see module docstring),
    # so ranking/eyeballing them was actively misleading. Now fixed at the
    # source, they're back in table A below.
    table_a_rows, table_b_rows = [], []
    for (label, run_id, n_params,
         e_short_mean, e_short_std, p_thr_mean, p_thr_std, e_max_mean, e_max_std,
         t_e_max_mean, t_e_max_std,
         t_max_5_med, t_max_5_pct, t_max_10_med, t_max_10_pct,
         t_mean_5_med, t_mean_5_pct, t_mean_10_med, t_mean_10_pct,
         nn_flops, nn_med, nn_std, fd_flops, fd_med, fd_std, speedup) in rows:
        table_a_rows.append([
            label,
            f"{n_params:,}" if n_params is not None else "--",
            fmt_sci_mean_std_readable(e_short_mean, e_short_std),
            fmt_pct_mean_std_readable(p_thr_mean, p_thr_std),
            fmt_sci_mean_std_readable(e_max_mean, e_max_std),
            fmt_time_mean_std_readable(t_e_max_mean, t_e_max_std),
            fmt_time_censored_readable(t_max_5_med, t_max_5_pct),
            fmt_time_censored_readable(t_max_10_med, t_max_10_pct),
            fmt_time_censored_readable(t_mean_5_med, t_mean_5_pct),
            fmt_time_censored_readable(t_mean_10_med, t_mean_10_pct),
        ])
        table_b_rows.append([
            label,
            _fmt_flops(nn_flops),
            f"{nn_med:.1f}" if nn_med is not None else "--",
            _fmt_flops(fd_flops),
            f"{fd_med:.1f}" if fd_med is not None else "--",
            f"{speedup:.2f}x" if speedup is not None else "--",
        ])

    table_a = render_readable_table(
        header_cells=["Architecture", "Params", "E_short", "P_thr_5%", "E_max", "t_E_max",
                      "T_max_5%", "T_max_10%", "T_mean_5%", "T_mean_10%"],
        rows=table_a_rows,
    )
    table_b = render_readable_table(
        header_cells=["Architecture", "NN FLOPs/step", "NN ms/step",
                      "FD FLOPs/step", "FD ms/step", "Speedup (FD/NN)"],
        rows=table_b_rows,
    )
    notes_a = readable_metric_notes({"E_short", "P_thr_5pct", "E_max", "t_E_max", "T_max_5pct", "T_mean_5pct"})
    note_b = "Speedup (FD/NN): FD time / NN time per rollout step. Above 1.00x the surrogate beats FD; below it, FD wins."

    return (
        "TABLE A -- Accuracy & rollout stability\n\n"
        + table_a + "\n\n" + "\n".join(notes_a) + "\n\n"
        + "TABLE B -- Compute cost & wall-clock speed\n\n"
        + table_b + "\n\n" + note_b + "\n"
    )


def _plot_error_all_archs(sources: list, metrics_by_run: dict, figures_dir: Path, curve_key: str,
                           ylabel: str, filename: str) -> None:
    # Same construction for both error metrics: err_max (max absolute error, worst node on
    # the beam) and err_mean_abs (beam-averaged, physical units) -- see
    # metrics.py's compute_error_curves for how each is derived. Kept as one
    # parametrized function rather than two copies so SOURCES/
    # color-linestyle encoding can't drift apart between the two figures.
    fig, ax = plt.subplots(figsize=(13, 6.5))
    n_plotted = 0
    for label, run_id, color, style in sources:
        c = metrics_by_run[run_id]["curves"]
        if curve_key not in c:
            # Same cause as a missing scalar (see build_rows): this run's
            # results.yaml predates compute_error_curves growing this key --
            # skip it here instead of a KeyError, rather than block the
            # other architectures that do have it.
            print(f"WARNING: {run_id}'s results.yaml has no {curve_key!r} curve "
                  f"(predates that metric) -- leaving it off {filename}.", file=sys.stderr)
            continue
        lw = 2.4 if run_id == "p4_arch_ridge" else 1.8
        ax.plot(c["t"], c[curve_key], linestyle=style, color=color, lw=lw, label=f"{label}")
        n_plotted += 1
    if n_plotted == 0:
        print(f"WARNING: no source has a {curve_key!r} curve -- skipping {filename} entirely "
              f"(relaunch the runs to get it).", file=sys.stderr)
        plt.close(fig)
        return False
    ax.set_yscale("log")
    ax.set_xlabel("time (s)", fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)
    ax.tick_params(axis="both", labelsize=17)
    ax.grid(True, which="both", alpha=0.4)
    legend = ax.legend(loc="upper left", borderaxespad=0,
                        title="HIDDEN_SIZES", fontsize=13)
    plt.setp(legend.get_title(), fontsize=13)
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _math(s: str) -> str:
    # fmt_pct_mean_std/fmt_time_mean_std return plain text ("78.3\% \pm
    # 13.1\%", "3.98 \pm 0.78") -- table.tex wants those in math mode ($...$)
    # to match the report's style. "--" (a genuinely missing value) is left
    # alone: math mode collapses "--" into a double-minus glyph, not the
    # en-dash the plain-text convention intends.
    return s if s == "--" else f"${s}$"


def build_latex_rows(sources: list, summaries: dict) -> list[list]:
    # 6 rollout-error-metrics columns (E_short, P_thr, E_max, T_5%^max,
    # T_10%^max, t_E_max) -- T_mean_5pct/10pct dropped from table.tex only
    # (table_readable.txt's render_readable_tables keeps them). Distinct
    # from build_rows/CSV_HEADERS, which also carries the FLOPs/wall-clock
    # columns the report table doesn't. Sourced from the multi-trajectory
    # summary now, not a single stored rollout -- see load_multi_summary/
    # aggregate_multi_rollout_metrics.
    rows = []
    for label, run_id, _, _ in sources:
        s = summaries.get(run_id)
        if s is None:
            rows.append([label, *([MISSING_ROW] * 6)])
            continue
        rows.append([
            label,
            fmt_sci_mean_std(s["E_short"]["mean"], s["E_short"]["std"]),
            _math(fmt_pct_mean_std(s["P_thr_5pct"]["mean"], s["P_thr_5pct"]["std"])),
            fmt_sci_mean_std(s["E_max"]["mean"], s["E_max"]["std"]),
            fmt_time_censored(s["T_max_5pct"]["median_reached"], s["T_max_5pct"]["pct_reached"]),
            fmt_time_censored(s["T_max_10pct"]["median_reached"], s["T_max_10pct"]["pct_reached"]),
            _math(fmt_time_mean_std(s["t_E_max"]["mean"], s["t_E_max"]["std"])),
        ])
    return rows


def main():
    metrics_by_run = {}
    for _, run_id, _, _ in SOURCES:
        m = _load(run_id)
        if m is not None:
            metrics_by_run[run_id] = m
    available = [s for s in SOURCES if s[1] in metrics_by_run]

    # Fill in any scalar a run's results.yaml predates (E_max, t_E_max,
    # P_thr_5pct, T_max_*, T_mean_*) from that run's own curves -- see
    # _common.py's fill_missing_scalars. amp_ref is shared across all 10
    # architectures (it only depends on the eval trajectory, not the model),
    # resolved once from whichever of p4_arch_64/p4_arch_128 has it.
    amp_ref = resolve_amp_ref(REPO_ROOT, metrics_by_run)
    for run_id, m in metrics_by_run.items():
        m["scalars"] = fill_missing_scalars(m["scalars"], m["curves"], amp_ref)

    summaries = {run_id: load_multi_summary(REPO_ROOT, run_id) for run_id in metrics_by_run}

    run_dir = REPO_ROOT / "architecture" / "runs" / RUN_ID
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # -- table --
    rows = build_rows(available, metrics_by_run, summaries)
    write_csv(run_dir / "table.csv", rows)
    header = (f"# {RUN_ID} -- architecture-size ablation ({len(available)}/{len(SOURCES)} runs available)\n\n"
              f"fd_flops/fd_time are the ground-truth finite-difference solver's own rollout cost\n"
              f"(measured independently per run -- fd_flops is identical across all 9 by construction,\n"
              f"353500.0; fd_time varies only by machine noise). speedup fd/nn > 1 means the trained\n"
              f"surrogate's rollout inference is faster wall-clock than just solving the FD reference.\n\n")
    table_readable = header + render_readable_tables(rows)
    (run_dir / "table_readable.txt").write_text(table_readable)
    print(table_readable)
    print(f"Saved {run_dir / 'table_readable.txt'}")
    print(f"CSV: {run_dir / 'table.csv'}")

    latex_rows = build_latex_rows(SOURCES, summaries)
    table_tex = render_latex_table(
        caption="Rollout error metrics for each architecture",
        label="tab:architecture_metrics",
        col_spec="lcccccc",
        header_cells=["Architecture", "$E_{\\mathrm{short}}$", "$P_{\\mathrm{thr}}$", "$E_{\\max}$",
                      "$T_{5\\%}^{\\max}$", "$T_{10\\%}^{\\max}$", "$t_{E_{\\max}}$"],
        rows=latex_rows,
        resizebox=True,
    )
    (run_dir / "table.tex").write_text(table_tex + "\n")
    print(table_tex)
    print(f"\nSaved {run_dir / 'table.tex'}")

    # -- plots -- (only actually written if listed here; see the return value)
    figure_names = []
    if _plot_error_all_archs(available, metrics_by_run, figures_dir, "err_max",
                              "max absolute error along the beam (log)",
                              "error_max_vs_time_all_archs.png"):
        figure_names.append("error_max_vs_time_all_archs.png")
    if _plot_error_all_archs(available, metrics_by_run, figures_dir, "err_mean_abs",
                              "mean absolute error along the beam (log)",
                              "error_mean_vs_time_all_archs.png"):
        figure_names.append("error_mean_vs_time_all_archs.png")

    # -- narrative summary --
    # "Best" t_div is now a ranking over censored multi-trajectory data, not
    # a single number: rank first by survival rate (100 - pct_reached, i.e.
    # the fraction of the 99 test trajectories that DIDN'T diverge -- more
    # is better), then by median_reached as a tiebreak among architectures
    # with similar survival. An architecture missing its summary (not yet
    # backfilled via eval_multi_trajectory.py) sorts last.
    def _t_div_rank_key(run_id):
        s = summaries.get(run_id)
        if s is None:
            return (-1.0, -1.0)
        td = s["T_max_10pct"]
        return (100.0 - (td["pct_reached"] or 0.0), td["median_reached"] or 0.0)

    best_t_div_label, best_t_div_run, *_ = max(available, key=lambda r: _t_div_rank_key(r[1]))
    fastest_nn_label, fastest_nn_run, *_ = min(
        available, key=lambda r: metrics_by_run[r[1]]["scalars"]["nn_time_med_s"])
    best_td = summaries.get(best_t_div_run)
    best_td_desc = (f"{best_td['T_max_10pct']['median_reached']:.2f}s median among "
                     f"{best_td['T_max_10pct']['pct_reached']:.0f}% of trajectories that diverged"
                     if best_td and best_td["T_max_10pct"]["median_reached"] is not None
                     else "never diverged" if best_td else "not backfilled")
    n_traj = next((s["n_trajectories"] for s in summaries.values() if s), "?")
    summary = (
        f"{RUN_ID} -- architecture-size ablation, {len(available)}/{len(SOURCES)} runs available\n"
        f"Most stable rollout across N={n_traj} test trajectories (t_div @ 10%): "
        f"{best_t_div_label} ({best_td_desc})\n"
        f"Fastest surrogate rollout (nn wall-clock): {fastest_nn_label} "
        f"({metrics_by_run[fastest_nn_run]['scalars']['nn_time_med_s']*1000:.2f}ms)\n"
        f"Every MLP variant has fd/nn speedup < 1 (the trained surrogate is slower wall-clock "
        f"than the FD solver it replaces at this problem size); only the linear baseline "
        f"beats it.\n"
        f"Full table: {run_dir / 'table.csv'}\n"
        f"Figures: {', '.join(str(figures_dir / n) for n in figure_names) or '(none written)'}\n"
    )
    (run_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"Done -- outputs in {run_dir}")


if __name__ == "__main__":
    main()
