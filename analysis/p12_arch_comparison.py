#!/usr/bin/env python3
# P12 (phase 12, hypothesis: null -- architecture-size ablation, not in the
# original H1-H9 plan): compares all 9 p12_arch_* runs, which differ only in
# HIDDEN_SIZES (everything else inherits phase2's winner -- pushforward,
# M_BACK=3, N_FWD=2, features=[U]). Pure post-processing over each source
# run's metrics.json, no re-training/re-inference.
#
# Two outputs:
#  1. table.csv -- one row per architecture: n_params, t_div, E_short,
#     nn_flops/nn wall-clock (the surrogate's own rollout inference) next to
#     fd_flops/fd wall-clock (the SAME quantities for the ground-truth finite-
#     difference solver -- both already measured per-run by evaluate/rollout.py's
#     benchmark, no new computation needed here), plus a derived fd/nn
#     wall-clock speedup ratio.
#  2. error_vs_time_all_archs.png -- err_max (Linf, log scale) vs t for all 9
#     architectures overlaid, same construction/metric as every other
#     error-vs-time figure in this campaign (h2/h3/h5/p11).
#
# Tail-zero artifact (see p11_coarse_grid_error.py's own comment for the
# general cause): verified empirically that ALL 9 p12 runs hit it at their
# very last point (t=4.98, amp_max==0.0 there, err_max collapses to the
# reference's own amplitude -- identical across all 9 runs at that point).
# T_CUTOFF drops that one point from every curve.
#
# Color/style encoding: 9 series exceeds the validated categorical palette's
# fixed-order cap (dataviz skill, palette.md: 8 hues), so per the skill's own
# rule ("never generate a 9th hue") ridge does NOT get a 9th hue -- it's a
# qualitatively different baseline (linear model, no hidden layer) styled as
# a neutral black dotted reference line, same pattern this campaign already
# uses for reference curves (p11/h2/h3's "(reference)" entries). The 8 MLP
# architectures get the palette's 8 hues in their documented fixed order
# (validated for adjacent pairs on line charts), grouped/ordered by depth;
# linestyle is a second, depth-family encoding (solid=1 hidden layer,
# dashed=2, dash-dot=3) so the figure stays legible in grayscale too.
#
# Usage: python analysis/p12_arch_comparison.py
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

RUN_ID = "p12_diag_arch_comparison"
PHASE = 12

# label, source run_id, color, linestyle
SOURCES = [
    ("64",            "p12_arch_64",          "#2a78d6", "-"),
    ("128",           "p12_arch_128",         "#eb6834", "-"),
    ("256",           "p12_arch_256",         "#1baf7a", "-"),
    ("64+16",         "p12_arch_64_16",       "#eda100", "--"),
    ("128+32",        "p12_arch_128_32",      "#e87ba4", "--"),
    ("256+16",        "p12_arch_256_16",      "#008300", "--"),
    ("128+64+16",     "p12_arch_128_64_16",   "#4a3aa7", "-."),
    ("256+128+32",    "p12_arch_256_128_32",  "#e34948", "-."),
    ("ridge (linear)", "p12_arch_ridge",      "#0b0b0b", ":"),
]

T_CUTOFF = 4.95


def _load(run_id: str) -> dict:
    path = REPO_ROOT / "runs" / run_id / "metrics.json"
    if not path.exists():
        print(f"ERROR: {path} not found -- run {run_id} first.", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        metrics = json.load(f)
    if not metrics["curves"]["t"]:
        print(f"ERROR: {path} has no curves -- {run_id} may not have finished successfully.", file=sys.stderr)
        sys.exit(1)
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
    return "never" if x is None else f"{x:.2f}"


def _fmt_sci(x):
    return "--" if x is None else f"{x:.3e}"


CSV_HEADERS = ["architecture", "run_id", "n_params", "t_div_s", "E_short",
               "nn_flops", "nn_time_med_ms", "nn_time_std_ms",
               "fd_flops", "fd_time_med_ms", "fd_time_std_ms", "speedup_fd_over_nn"]


def build_rows(metrics_by_run: dict) -> list[list]:
    # Raw numeric values (not human-formatted strings) -- CSV is meant to be
    # re-parsed (pandas, Excel, another script), so units are in the header
    # instead of baked into cells like "73.61M" or "40.54+-0.24".
    rows = []
    for label, run_id, _, _ in SOURCES:
        s = metrics_by_run[run_id]["scalars"]
        speedup = (s["fd_time_med_s"] / s["nn_time_med_s"]) if s["nn_time_med_s"] else None
        rows.append([
            label, run_id, s["n_params"], s["t_div"], s["E_short"],
            s["nn_flops"],
            s["nn_time_med_s"] * 1000 if s["nn_time_med_s"] is not None else None,
            s["nn_time_std_s"] * 1000 if s["nn_time_std_s"] is not None else None,
            s["fd_flops"],
            s["fd_time_med_s"] * 1000 if s["fd_time_med_s"] is not None else None,
            s["fd_time_std_s"] * 1000 if s["fd_time_std_s"] is not None else None,
            speedup,
        ])
    return rows


def write_csv(path: Path, rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)
        writer.writerows(rows)


def render_table_text(rows: list[list]) -> str:
    # Human-readable rendering of the same rows, for the terminal/summary.txt
    # only -- table.csv itself stays raw/numeric (see build_rows).
    headers = ["architecture", "n_params", "t_div (s)", "E_short",
               "nn_flops", "nn_time (ms)", "fd_flops", "fd_time (ms)", "speedup fd/nn"]
    text_rows = []
    for label, run_id, n_params, t_div, e_short, nn_flops, nn_med, nn_std, fd_flops, fd_med, fd_std, speedup in rows:
        text_rows.append([
            f"{label} ({run_id})",
            f"{n_params:,}" if n_params is not None else "--",
            _fmt_t_div(t_div),
            _fmt_sci(e_short),
            _fmt_flops(nn_flops),
            f"{nn_med:.2f}+-{nn_std:.2f}" if nn_med is not None else "--",
            _fmt_flops(fd_flops),
            f"{fd_med:.2f}+-{fd_std:.2f}" if fd_med is not None else "--",
            f"{speedup:.2f}x" if speedup is not None else "--",
        ])
    widths = [max(len(h), max(len(r[i]) for r in text_rows)) for i, h in enumerate(headers)]
    def fmt_row(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    lines = [fmt_row(headers), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines += [fmt_row(r) for r in text_rows]
    return "\n".join(lines)


def main():
    metrics_by_run = {run_id: _load(run_id) for _, run_id, _, _ in SOURCES}

    run_dir = REPO_ROOT / "runs" / RUN_ID
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # -- table --
    rows = build_rows(metrics_by_run)
    write_csv(run_dir / "table.csv", rows)
    header = (f"# {RUN_ID} (phase {PHASE}) -- p12 architecture-size ablation\n\n"
              f"fd_flops/fd_time are the ground-truth finite-difference solver's own rollout cost\n"
              f"(measured independently per run -- fd_flops is identical across all 9 by construction,\n"
              f"353500.0; fd_time varies only by machine noise). speedup fd/nn > 1 means the trained\n"
              f"surrogate's rollout inference is faster wall-clock than just solving the FD reference.\n\n")
    table_text = render_table_text(rows)
    print(header + table_text)
    print(f"\nCSV: {run_dir / 'table.csv'}")

    # -- plot --
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for label, run_id, color, style in SOURCES:
        c = metrics_by_run[run_id]["curves"]
        t_cut = [tt for tt in c["t"] if tt <= T_CUTOFF]
        e_cut = c["err_max"][:len(t_cut)]
        lw = 2.4 if run_id == "p12_arch_ridge" else 1.8
        ax.plot(t_cut, e_cut, linestyle=style, color=color, lw=lw, label=f"{label}")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("max absolute error along the beam -- Linf (log)")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
              title="HIDDEN_SIZES", fontsize=9)
    ax.set_title("P12 -- erreur de rollout (Linf) vs temps, toutes architectures")
    plt.tight_layout()
    plt.savefig(figures_dir / "error_vs_time_all_archs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -- narrative summary --
    best_t_div_label, best_t_div_run, *_ = max(
        SOURCES, key=lambda r: (metrics_by_run[r[1]]["scalars"]["t_div"] or -1))
    fastest_nn_label, fastest_nn_run, *_ = min(
        SOURCES, key=lambda r: metrics_by_run[r[1]]["scalars"]["nn_time_med_s"])
    summary = (
        f"{RUN_ID} (phase {PHASE}) -- p12 architecture-size ablation, 9 runs\n"
        f"Longest stable rollout (t_div): {best_t_div_label} "
        f"({metrics_by_run[best_t_div_run]['scalars']['t_div']:.2f}s)\n"
        f"Fastest surrogate rollout (nn wall-clock): {fastest_nn_label} "
        f"({metrics_by_run[fastest_nn_run]['scalars']['nn_time_med_s']*1000:.2f}ms)\n"
        f"Every MLP variant has fd/nn speedup < 1 (the trained surrogate is slower wall-clock "
        f"than the FD solver it replaces at this problem size); only the linear ridge baseline "
        f"beats it.\n"
        f"Full table: {run_dir / 'table.csv'}\n"
        f"Figure: {figures_dir / 'error_vs_time_all_archs.png'}\n"
    )
    (run_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"Done -- outputs in {run_dir}")


if __name__ == "__main__":
    main()
