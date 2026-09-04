# Shared by every analysis/*.py script: where a run's own results live, how
# to load them tolerantly, how to fill in any rollout-error scalar that a
# given run's results file predates, and how to render the filled table as
# LaTeX. Centralized here so all four p1-p4 table scripts stay consistent
# instead of drifting into four slightly-different copies.
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml


def run_dir(repo_root: Path, run_id: str) -> Path:
    m = re.match(r"p\d+", run_id)
    if not m:
        raise ValueError(f"can't extract a phase prefix from run_id {run_id!r}")
    return repo_root / "runs" / m.group() / run_id


def load_run_metrics(repo_root: Path, run_id: str) -> dict | None:
    # Two run-folder conventions coexist: runs/<phase>/<run_id>/ (p2/p3/p4)
    # and the flat runs/<run_id>/ (p1_pushforward, p1_bptt, which predate the
    # phase-subfolder convention). Try both. Two results-file formats also
    # coexist: results.yaml (current) and metrics.json (p1_pushforward/
    # p1_bptt only). Returns None (with a WARNING, never an exception) if the
    # run folder, its results file, or its curves don't exist yet -- a run
    # still training/queued shouldn't block the other rows.
    for candidate in (run_dir(repo_root, run_id), repo_root / "runs" / run_id):
        if candidate.exists():
            run_path = candidate
            break
    else:
        print(f"WARNING: no run folder found for {run_id!r} -- skipping.", file=sys.stderr)
        return None

    for fname, load in (("results.yaml", yaml.safe_load), ("metrics.json", json.load)):
        fpath = run_path / fname
        if fpath.exists():
            with open(fpath) as f:
                metrics = load(f)
            if not metrics.get("curves", {}).get("t"):
                print(f"WARNING: {fpath} has no curves -- {run_id} may not have finished "
                      f"successfully, skipping.", file=sys.stderr)
                return None
            return metrics

    print(f"WARNING: no results.yaml or metrics.json in {run_path} -- {run_id} hasn't been "
          f"run yet, skipping.", file=sys.stderr)
    return None


def load_multi_summary(repo_root: Path, run_id: str) -> dict | None:
    # aggregate_multi_rollout_metrics's output for one run (see
    # scripts/eval_multi_trajectory.py), read from
    # <run_dir>/multi_trajectory_summary.json -- the per-trajectory rollout
    # eval, as opposed to load_run_metrics's single stored rollout_idx.
    # Same tolerant-None convention as load_run_metrics: a run not yet
    # backfilled just gets a WARNING and a MISSING_ROW/-- in its table row,
    # never an exception that would block every other row.
    for candidate in (run_dir(repo_root, run_id), repo_root / "runs" / run_id):
        if candidate.exists():
            fpath = candidate / "multi_trajectory_summary.json"
            if fpath.exists():
                with open(fpath) as f:
                    return json.load(f)
            print(f"WARNING: no multi_trajectory_summary.json in {candidate} -- run "
                  f"scripts/eval_multi_trajectory.py for {run_id!r} first, skipping.", file=sys.stderr)
            return None
    print(f"WARNING: no run folder found for {run_id!r} -- skipping.", file=sys.stderr)
    return None


def resolve_amp_ref(repo_root: Path, metrics_by_run: dict[str, dict],
                     fallback_run_id: str = "p1_pushforward") -> float | None:
    # amp_ref (the FD reference trajectory's own global peak amplitude) is
    # not model-dependent -- it only depends on the shared eval dataset/test
    # trajectory, confirmed identical (0.0032692134846001863) across every
    # run inspected that has it. So a run whose results file predates this
    # field can safely borrow it from a sibling run in the same table, or --
    # if none of them have it either (e.g. every p3_pinn_* run) -- from
    # p1_pushforward, which always has it.
    for m in metrics_by_run.values():
        v = m.get("scalars", {}).get("amp_ref")
        if v is not None:
            return v
    fallback = load_run_metrics(repo_root, fallback_run_id)
    if fallback is not None:
        v = fallback.get("scalars", {}).get("amp_ref")
        if v is not None:
            return v
    print(f"WARNING: could not resolve amp_ref from any loaded run or fallback "
          f"{fallback_run_id!r}.", file=sys.stderr)
    return None


def _first_crossing(t: list, err: list, amp_ref: float, pct: float) -> float | None:
    # First time `err` exceeds pct*amp_ref -- same rule as metrics.py's
    # compute_t_div/compute_t_mean_threshold, applied to an already-saved
    # curve instead of the raw rollout tensor (which isn't persisted to
    # disk, only its derived per-step curves are).
    threshold = pct * amp_ref
    for tt, e in zip(t, err):
        if e > threshold:
            return float(tt)
    return None


def fill_missing_scalars(scalars: dict, curves: dict, amp_ref: float | None) -> dict:
    # Returns a copy of `scalars` with E_max, t_E_max, T_max_5pct/10pct,
    # T_mean_5pct/10pct, P_thr_5pct computed from `curves` whenever the
    # source results file predates that field (key missing or None).
    # E_short is left untouched -- it needs the raw per-step RMSE over the
    # first 50 rollout steps, not recoverable from the saved curves, but it
    # is present in every run's scalars in practice so this is never a gap.
    out = dict(scalars)
    t = curves.get("t") or []
    err_max = curves.get("err_max") or []
    err_mean_abs = curves.get("err_mean_abs") or []

    if out.get("E_max") is None and err_max:
        out["E_max"] = float(max(err_max))

    if out.get("t_E_max") is None and err_max and t:
        idx = err_max.index(max(err_max))
        out["t_E_max"] = float(t[idx])

    if amp_ref is not None and t:
        if out.get("T_max_5pct") is None and err_max:
            out["T_max_5pct"] = _first_crossing(t, err_max, amp_ref, 0.05)
        if out.get("T_max_10pct") is None and err_max:
            out["T_max_10pct"] = _first_crossing(t, err_max, amp_ref, 0.10)
        if out.get("T_mean_5pct") is None and err_mean_abs:
            out["T_mean_5pct"] = _first_crossing(t, err_mean_abs, amp_ref, 0.05)
        if out.get("T_mean_10pct") is None and err_mean_abs:
            out["T_mean_10pct"] = _first_crossing(t, err_mean_abs, amp_ref, 0.10)
        if out.get("P_thr_5pct") is None and err_max:
            n_above = sum(1 for v in err_max if v > 0.05 * amp_ref)
            out["P_thr_5pct"] = float(100.0 * n_above / len(err_max))

    return out


# -- LaTeX formatting -- "--" is a value that genuinely computed to None
# (e.g. the error never crossed that threshold, a real result); MISSING_ROW
# is for a source run that hasn't produced a results file at all yet.
MISSING_ROW = "\\todo{}"


def fmt_time(x: float | None) -> str:
    return "--" if x is None else f"{x:.2f}"


def fmt_pct(x: float | None) -> str:
    return "--" if x is None else f"{x:.1f}\\%"


def fmt_sci(x: float | None) -> str:
    if x is None:
        return "--"
    mantissa, exp = f"{x:.2e}".split("e")
    return f"${mantissa} \\times 10^{{{int(exp)}}}$"


# -- multi-trajectory formatting -- same "--" convention, but consuming the
# {"mean","std","median","n"} / {"pct_reached","median_reached","n_reached"}
# shapes evaluate/metrics.py's aggregate_multi_rollout_metrics returns,
# instead of a single value. CONTINUOUS_SCALAR_KEYS metrics (defined on
# every trajectory) render as mean +/- std; CENSORED_TIME_KEYS metrics (only
# defined on trajectories that actually crossed their threshold) render as
# median (% reached) -- see aggregate_multi_rollout_metrics's own docstring
# for why these can't be averaged the same way.
def fmt_time_mean_std(mean: float | None, std: float | None) -> str:
    return "--" if mean is None else f"{mean:.2f} \\pm {std:.2f}"


def fmt_pct_mean_std(mean: float | None, std: float | None) -> str:
    return "--" if mean is None else f"{mean:.1f}\\% \\pm {std:.1f}\\%"


def fmt_sci_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "--"
    m_mant, m_exp = f"{mean:.2e}".split("e")
    s_mant, s_exp = f"{std:.2e}".split("e")
    return f"${m_mant} \\times 10^{{{int(m_exp)}}} \\pm {s_mant} \\times 10^{{{int(s_exp)}}}$"


# -- median-only siblings: mean+/-std is misleading for metrics where a
# handful of outlier trajectories (e.g. a diverged rollout) can be orders of
# magnitude larger than the typical case -- the mean then reflects the
# outliers, not the typical trajectory. Median is robust to that.
def fmt_pct_median(median: float | None) -> str:
    return "--" if median is None else f"{median:.1f}\\%"


def fmt_sci_median(median: float | None) -> str:
    if median is None:
        return "--"
    mantissa, exp = f"{median:.2e}".split("e")
    return f"${mantissa} \\times 10^{{{int(exp)}}}$"


def fmt_time_censored(median_reached: float | None, pct_reached: float | None) -> str:
    return "--" if median_reached is None else f"{median_reached:.2f}\\ ({pct_reached:.0f}\\%)"


# -- plain-text ("readable") siblings of the four above.
def fmt_time_mean_std_readable(mean: float | None, std: float | None) -> str:
    return "--" if mean is None else f"{mean:.2f}s +/- {std:.2f}s"


def fmt_pct_mean_std_readable(mean: float | None, std: float | None) -> str:
    return "--" if mean is None else f"{mean:.1f}% +/- {std:.1f}%"


def fmt_sci_mean_std_readable(mean: float | None, std: float | None) -> str:
    return "--" if mean is None else f"{mean:.2e} +/- {std:.2e}"


def fmt_time_censored_readable(median_reached: float | None, pct_reached: float | None) -> str:
    return "--" if median_reached is None else f"{median_reached:.2f}s ({pct_reached:.0f}%)"


def fmt_pct_median_readable(median: float | None) -> str:
    return "--" if median is None else f"{median:.1f}%"


def fmt_sci_median_readable(median: float | None) -> str:
    return "--" if median is None else f"{median:.2e}"


# -- training duration -- scalars.train_time_s is wall-clock seconds for the
# WHOLE training run (not a per-trajectory rollout-eval quantity, so it comes
# straight from a run's own metrics.json/results.yaml, never from the
# multi-trajectory summary). Rendered in hours: every run here trains for
# hours, not seconds, so seconds would just force the reader to do this
# division themselves.
def fmt_duration_h(seconds: float | None) -> str:
    return "--" if seconds is None else f"{seconds / 3600:.1f}\\,h"


def fmt_duration_h_readable(seconds: float | None) -> str:
    return "--" if seconds is None else f"{seconds / 3600:.2f}h"


# -- plain-text ("readable") formatting -- same values as fmt_time/fmt_pct/
# fmt_sci above, without the LaTeX markup, for a human-facing table.tex
# sibling (table_readable.txt) rather than a paper draft. "--" mirrors the
# LaTeX formatters' convention for a genuinely-None value.
def fmt_time_readable(x: float | None) -> str:
    return "--" if x is None else f"{x:.2f}s"


def fmt_pct_readable(x: float | None) -> str:
    return "--" if x is None else f"{x:.1f}%"


def fmt_sci_readable(x: float | None) -> str:
    return "--" if x is None else f"{x:.2e}"


# One line per metric family, included only for the columns a given table
# actually has (see callers) -- the two facts every one of these tables gets
# misread on: P_thr is a badness fraction (lower is better, despite reading
# like a pass rate), and the T_* columns are time-to-threshold-crossing
# (higher is better, capped at the rollout horizon).
READABLE_METRIC_NOTES = {
    "E_short": "E_short  : mean RMSE over the first 50 rollout steps (short-horizon error). Lower is better.",
    "P_thr_5pct": "P_thr_5%: % of rollout timesteps where error exceeds 5% of reference amplitude -- a BADNESS fraction. Lower is better.",
    "E_max": "E_max    : peak max absolute error over the rollout. Lower is better.",
    "t_E_max": "t_E_max  : time at which E_max occurs.",
    "T_max": "T_max_*  : first time the WORST-NODE error crosses the 5%/10% threshold. Higher is better (capped at the rollout horizon).",
    "T_mean": "T_mean_* : first time the BEAM-MEAN error crosses the 5%/10% threshold. Higher is better (capped at the rollout horizon).",
}


def readable_metric_notes(fields: set[str]) -> list[str]:
    keys = []
    for key in ("E_short", "P_thr_5pct", "E_max", "t_E_max"):
        if key in fields:
            keys.append(key)
    if "T_max_5pct" in fields or "T_max_10pct" in fields:
        keys.append("T_max")
    if "T_mean_5pct" in fields or "T_mean_10pct" in fields:
        keys.append("T_mean")
    return [READABLE_METRIC_NOTES[k] for k in keys]


def render_readable_table(*, header_cells: list[str], rows: list) -> str:
    # Plain-text sibling of render_latex_table: same `rows` shape (a plain
    # row of pre-formatted cell strings, or a ("section", title) spanner --
    # only p2's stencil table uses spanners), rendered as a wide-spaced,
    # column-aligned table instead of a tabular environment. First column
    # left-aligned (it's a name), the rest right-aligned (they're numbers).
    data_rows = [r for r in rows if not (isinstance(r, tuple) and r and r[0] == "section")]
    widths = [len(h) for h in header_cells]
    for r in data_rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    gap = "   "

    def fmt_row(cells):
        return gap.join(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(cells))

    lines = [fmt_row(header_cells), gap.join("-" * w for w in widths)]
    for r in rows:
        if isinstance(r, tuple) and r and r[0] == "section":
            lines.append("")
            lines.append(r[1])
        else:
            lines.append(fmt_row(r))
    return "\n".join(lines)


def render_latex_table(*, caption: str, label: str, col_spec: str,
                        header_cells: list[str], rows: list, comment: str | None = None,
                        resizebox: bool = False) -> str:
    # `rows` is a list of either a plain row (list of pre-formatted cell
    # strings, one per header_cells entry) or a section-header row, the
    # tuple ("section", "title"), rendered as a \multicolumn{...}{l}{...}
    # spanner line preceded by \addlinespace (skipped before the first row)
    # -- only the p2 stencil-ablation table uses these, for its 3 blocks.
    # resizebox=True wraps the tabular in \resizebox{\textwidth}{!}{...} for
    # a wide table (opt-in per call, default off -- existing callers'
    # output is unaffected).
    lines = ["\\begin{table}[H]", "    \\centering"]
    if comment:
        lines.append(f"    % {comment}")
    lines += [
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
    ]
    if resizebox:
        lines.append("    \\resizebox{\\textwidth}{!}{%")
    lines += [
        f"    \\begin{{tabular}}{{{col_spec}}}",
        "        \\toprule",
        "        " + " & ".join(header_cells) + " \\\\",
        "        \\midrule",
    ]
    seen_any = False
    for row in rows:
        if isinstance(row, tuple) and row[0] == "section":
            if seen_any:
                lines.append("        \\addlinespace")
            lines.append(f"        \\multicolumn{{{len(header_cells)}}}{{l}}{{\\textit{{{row[1]}}}}} \\\\")
        else:
            lines.append("        " + " & ".join(row) + " \\\\")
        seen_any = True
    lines.append("        \\bottomrule")
    if resizebox:
        lines += ["    \\end{tabular}%", "    }"]
    else:
        lines.append("    \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)
