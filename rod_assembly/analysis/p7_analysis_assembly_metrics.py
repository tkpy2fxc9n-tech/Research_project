#!/usr/bin/env python3
# Fills the thesis's Table 17 (\label{tab:assembly_metrics}, Section 4.8
# "Extension to a network of rods"): one row per shortlisted p7 model,
# columns T_5%^max/T_10%^max/E_max/Delta E_max/t_{E_max}/P_thr. Follows the
# same read-cached-metrics-and-tabulate pattern as
# p2_analysis_stencils_parameters.py / p4_analysis_arch_comparison.py --
# reuses p7_compare_assembly_models.load_cached_metrics() and
# p7_structure_several_rods.first_crossing_times() rather than
# recomputing anything from scratch.
#
# Baseline (row 1) is p7_dataset_medium_arch_256_128_32_pinn_0.01: the only
# one of the 5 kept models combining every setting actually retained by the
# earlier single-rod ablation phases (architecture [256,128,32] from Phase 4,
# PINN weight lambda=0.01 from Phase 3, single-field input {U} from Phase 2)
# -- i.e. "the retained configuration applied unchanged" that Section 4.8's
# opening sentence refers to. The other 4 rows all use the original
# reference architecture [512,256,64] instead.
#
# Usage: python analysis/p7_analysis_assembly_metrics.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "common"))
import p7_structure_several_rods as m  # noqa: E402
import p7_compare_assembly_models as cmp  # noqa: E402

OUT_DIR = REPO_ROOT / "rod_assembly" / "runs" / "analysis"

BASELINE_RUN_ID = "p7_dataset_medium_arch_256_128_32_pinn_0.01"

# (run_id, configuration label) -- baseline first.
ROWS = [
    (BASELINE_RUN_ID, "Retained config (arch [256,128,32], PINN $\\lambda=0.01$)"),
    ("p7_dataset_medium", "Reference architecture, no PINN"),
    ("p7_dataset_medium_ut", "Reference architecture, $+\\dot u$ feature"),
    ("p7_dataset_medium_pinn_0.01", "Reference architecture, $+\\dot u$ feature, PINN"),
    ("p7_dataset_medium_bidir", "Reference architecture, bidirectional dataset (C)"),
]


def first_crossing_for(run_id: str) -> tuple[float | None, float | None]:
    # load_cached_metrics() gives the scalar metrics (abs_max, t_of_max_error,
    # pct_time_above_threshold, ...) but not T_5%/T_10%, which need the raw
    # err_U curve -- rebuild it the same overlap-aligned way compute_metrics()
    # and p7_export_latex_report.py both do.
    out_dir = cmp.MODELS_DIR / run_id
    cached = np.load(out_dir / f"{run_id}_frames.npz")
    fd_steps, fd_U = cached["fd_steps"], cached["fd_U"]
    nn_steps, nn_U = cached["nn_steps"], cached["nn_U"]
    cfg, _ = m.load_source(run_id)
    err_steps, fd_idx, nn_idx = np.intersect1d(fd_steps, nn_steps, assume_unique=True, return_indices=True)
    err_U = fd_U[fd_idx] - nn_U[nn_idx]
    err_t = err_steps * cfg.dt
    fd_peak = float(np.abs(fd_U).max())
    return m.first_crossing_times(err_U, err_t, fd_peak)


def fmt_t(t: float | None) -> str:
    return f"{t:.2f}s" if t is not None else "--"


def main():
    rows = []
    for run_id, label in ROWS:
        metrics = cmp.load_cached_metrics(run_id)
        t5, t10 = first_crossing_for(run_id)
        rows.append({"run_id": run_id, "label": label, "t5": t5, "t10": t10, **metrics})

    baseline_emax = rows[0]["abs_max"]
    for r in rows:
        r["delta_emax"] = r["abs_max"] / baseline_emax

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- plain-text, human-checkable ---
    lines = [
        "p17 rectangular assembly -- rollout error metrics (Table 17 / tab:assembly_metrics)",
        f"Baseline: {BASELINE_RUN_ID}",
        "",
        f"{'Configuration':<55}{'T_5%^max':>10}{'T_10%^max':>10}{'E_max':>12}{'DeltaE_max':>12}{'t_Emax':>9}{'P_thr%':>8}",
    ]
    for r in rows:
        lines.append(
            f"{r['label']:<55}{fmt_t(r['t5']):>10}{fmt_t(r['t10']):>10}{r['abs_max']:>12.6f}"
            f"{r['delta_emax']:>12.2f}{r['t_of_max_error']:>8.2f}s{r['pct_time_above_threshold']:>7.1f}%"
        )
    readable = "\n".join(lines) + "\n"
    (OUT_DIR / "table_readable.txt").write_text(readable)
    print(readable)

    # --- LaTeX fragment, drop-in for tab:assembly_metrics's tabular body ---
    tex_lines = [
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\small",
        "    \\caption{Rollout error metrics for the rod assembly. Row~1 is the baseline configuration;",
        "    subsequent rows vary one factor against it. $\\Delta E_{\\max}$ is the ratio of each",
        "    configuration's $E_{\\max}$ to the baseline value, so that $<1$ denotes an improvement.",
        "    A dash indicates that the threshold was not reached within the simulated window.}",
        "    \\label{tab:assembly_metrics}",
        "    \\begin{tabular}{l c c c c c c}",
        "        \\toprule",
        "        Configuration & $T_{5\\%}^{\\max}$ & $T_{10\\%}^{\\max}$ & $E_{\\max}$",
        "        & $\\Delta E_{\\max}$ & $t_{E_{\\max}}$ & $P_{\\mathrm{thr}}$ (\\%) \\\\",
        "        \\midrule",
    ]
    for i, r in enumerate(rows):
        prefix = "Baseline & " if i == 0 else f"{r['label']} & "
        tex_lines.append(
            f"        {prefix}{fmt_t(r['t5'])} & {fmt_t(r['t10'])} & {r['abs_max']:.6f} & "
            f"{r['delta_emax']:.2f} & {r['t_of_max_error']:.2f}s & {r['pct_time_above_threshold']:.1f} \\\\"
        )
    tex_lines += [
        "        \\bottomrule",
        "    \\end{tabular}",
        "\\end{table}",
    ]
    tex = "\n".join(tex_lines) + "\n"
    (OUT_DIR / "table.tex").write_text(tex)
    print(tex)
    print(f"Saved {OUT_DIR / 'table_readable.txt'} and {OUT_DIR / 'table.tex'}")


if __name__ == "__main__":
    main()
