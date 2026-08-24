#!/usr/bin/env python3
# Phase 10 (rollout stabilisation strategies): compares the 8
# p10_stabilisation_strategies/* runs -- none/noise/laplacian/both, each
# with and without the rest-bias correction (_bias_off) -- to answer
# Section 5.2's own open questions ("is Laplacian smoothing better? what
# about noise injection?"). Same construction as p2/p4/p5's comparison
# scripts (multi-trajectory summary -> mean+-std / censored-median table).
#
# IMPORTANT: as of writing, p10_noise/p10_both/p10_noise_bias_off/
# p10_both_bias_off were just relaunched (jobs 149704-149707) after fixing
# a real bug -- cfg.NOISE_STD was only ever read in teacher_forcing.py/
# bptt.py, never in pushforward.py, which every p10_* run actually uses, so
# noise injection was silently inert for the whole "noise"/"both" half of
# this campaign. Re-run this script (and scripts/eval_multi_trajectory.py
# for the 4 rerun run_ids, if their multi_trajectory_summary.json is still
# missing/stale) once those jobs finish -- see squeue/sacct.
#
# Usage: python analysis/p10_analysis_stabilisation.py
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))
from _common import (  # noqa: E402
    fmt_pct_mean_std, fmt_sci_mean_std, fmt_time_censored,
    fmt_pct_mean_std_readable, fmt_sci_mean_std_readable, fmt_time_censored_readable,
    readable_metric_notes, render_latex_table, render_readable_table, MISSING_ROW,
)

RUN_ID = "p10_analysis"

# Flat run_ids under runs/p10_stabilisation_strategies/<id>/ -- not phase-
# prefixed, so _common.py's regex-based run_dir()/load_multi_summary (which
# resolve to runs/<phase>/<run_id>/) don't apply here; same reasoning
# p6_analysis_dataset_comparison.py gives for its own hardcoded path.
STABILIZERS = ["none", "noise", "laplacian", "both"]
BIAS_VARIANTS = [("bias on (default)", ""), ("bias off", "_bias_off")]


def _run_dir(run_id: str) -> Path:
    return REPO_ROOT / "runs" / "p10_stabilisation_strategies" / run_id


def _load_multi_summary(run_id: str):
    fpath = _run_dir(run_id) / "multi_trajectory_summary.json"
    if not fpath.exists():
        print(f"WARNING: no multi_trajectory_summary.json for {run_id} -- run "
              f"scripts/eval_multi_trajectory.py for it first, skipping.", file=sys.stderr)
        return None
    import json
    with open(fpath) as f:
        return json.load(f)


def build_table(summaries: dict) -> tuple[list, list]:
    tex_rows, readable_rows = [], []
    for stabilizer in STABILIZERS:
        for bias_label, suffix in BIAS_VARIANTS:
            run_id = f"p10_{stabilizer}{suffix}"
            label = f"{stabilizer} ({bias_label})"
            s = summaries.get(run_id)
            if s is None:
                tex_rows.append([label, *([MISSING_ROW] * 5)])
                readable_rows.append([label, *(["--"] * 5)])
                continue
            tex_rows.append([
                label,
                fmt_sci_mean_std(s["E_short"]["mean"], s["E_short"]["std"]),
                fmt_pct_mean_std(s["P_thr_5pct"]["mean"], s["P_thr_5pct"]["std"]),
                fmt_sci_mean_std(s["E_max"]["mean"], s["E_max"]["std"]),
                fmt_time_censored(s["T_max_5pct"]["median_reached"], s["T_max_5pct"]["pct_reached"]),
                fmt_time_censored(s["T_max_10pct"]["median_reached"], s["T_max_10pct"]["pct_reached"]),
            ])
            readable_rows.append([
                label,
                fmt_sci_mean_std_readable(s["E_short"]["mean"], s["E_short"]["std"]),
                fmt_pct_mean_std_readable(s["P_thr_5pct"]["mean"], s["P_thr_5pct"]["std"]),
                fmt_sci_mean_std_readable(s["E_max"]["mean"], s["E_max"]["std"]),
                fmt_time_censored_readable(s["T_max_5pct"]["median_reached"], s["T_max_5pct"]["pct_reached"]),
                fmt_time_censored_readable(s["T_max_10pct"]["median_reached"], s["T_max_10pct"]["pct_reached"]),
            ])
    return tex_rows, readable_rows


def main():
    all_ids = [f"p10_{s}{suf}" for s in STABILIZERS for _, suf in BIAS_VARIANTS]
    summaries = {run_id: _load_multi_summary(run_id) for run_id in all_ids}
    n_available = sum(1 for v in summaries.values() if v is not None)

    run_dir = REPO_ROOT / "runs" / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)

    tex_rows, readable_rows = build_table(summaries)
    n = next((s["n_trajectories"] for s in summaries.values() if s), "?")

    table_tex = render_latex_table(
        caption=f"Comparison of rollout-stabilisation strategies (input-noise injection, "
                f"Laplacian smoothing, both combined, and neither -- each with and without the "
                f"rest-bias correction), evaluated by autoregressive rollout on $N={n}$ held-out "
                f"test trajectories. $E_{{\\mathrm{{short}}}}$, $P_{{\\mathrm{{thr}}}}$ and "
                f"$E_{{\\max}}$ are reported as mean $\\pm$ std across trajectories. "
                f"$T_{{5\\%}}^{{\\max}}/T_{{10\\%}}^{{\\max}}$ are right-censored (not every "
                f"trajectory reaches the threshold within the simulated window); each cell "
                f"reports the median crossing time among trajectories that reached it, with the "
                f"fraction that reached it in parentheses.",
        label="tab:stabilisation_strategies",
        col_spec="lccccc",
        header_cells=["Strategy", "$E_{\\mathrm{short}}$", "$P_{\\mathrm{thr}}$", "$E_{\\max}$",
                      "$T_{5\\%}^{\\max}$", "$T_{10\\%}^{\\max}$"],
        rows=tex_rows,
    )
    (run_dir / "table.tex").write_text(table_tex + "\n")
    print(table_tex)
    print(f"\nSaved {run_dir / 'table.tex'}")

    readable_table = render_readable_table(
        header_cells=["Strategy", "E_short", "P_thr_5%", "E_max", "T_max_5%", "T_max_10%"],
        rows=readable_rows,
    )
    notes = readable_metric_notes({"E_short", "P_thr_5pct", "E_max", "T_max_5pct"})
    notes.append(f"(mean +/- std / median (%reached) across N={n} held-out test trajectories)")
    table_readable = (
        f"p10_analysis -- rollout stabilisation strategies ({n_available}/{len(all_ids)} runs available)\n\n"
        + readable_table + "\n\n" + "\n".join(notes) + "\n"
    )
    (run_dir / "table_readable.txt").write_text(table_readable)
    print(table_readable)
    print(f"Saved {run_dir / 'table_readable.txt'}")


if __name__ == "__main__":
    main()
