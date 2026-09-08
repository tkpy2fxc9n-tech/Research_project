#!/usr/bin/env python3
# Phase 6 (dataset-composition comparison): same fixed model architecture
# (pushforward, M_BACK=3, N_FWD=2, features=[U], HIDDEN_SIZES=[256,128,32],
# LAMBDA_PHYSICS=0.001 -- p4's best-stable-rollout architecture + p3's best
# PINN weight) trained once per dataset -- A=simple, B=medium,
# C=medium_bidir, D=complex (see runs/p6_performance_different_datasets/) --
# then rolled out against a common 6-wave test battery NOT drawn from any
# dataset's own held-out split (gaussian, chirp, shock="tanh shock",
# filtered_random="filtered random walk", sawtooth, sinusoid -- all already
# in physics/waves.py's BC_WAVEFORMS, no new physics code needed), each wave
# tested at a fixed seed so every model sees the identical sampled
# excitation. Chirp is R-only (matches the manuscript table); every other
# wave is tested both R (right end only) and R+L (both ends, anti-phase via
# waves.flip) -- 11 rows total, not the full 6x2=12.
#
# Two outputs into runs/p6_diag_dataset_comparison/ (flat comparative
# sibling, same convention as p3_pinn_comparative/p4_diag_arch_comparison --
# no config.yaml/training here, analysis only):
#   1. figures/training_curves_{A,B,C,D}.png -- copied (not regenerated)
#      from each run's own figures/training_curves.png.
#   2. table.csv / table.tex -- one row per (wave, excitation), one column
#      pair (T_5%, E_max) per model, via _common.render_latex_table.
#
# Deliberately does NOT use _common.py's run_dir()/load_run_metrics(): those
# extract the pN phase prefix from run_id via regex and would resolve
# p6_dataset_* to runs/p6/p6_dataset_*, colliding with an OLDER, unrelated
# zero-shot experiment that already used bare "p6"-prefixed run_ids
# (runs/p6/p6_complex_zeroshot/, see scripts/eval_zeroshot.py). Path
# resolution here is hardcoded to runs/p6_performance_different_datasets/
# instead. Only _common.py's pure formatting helpers are reused.
#
# Runnable at any point before all 4 runs finish -- a model with no
# model.pth yet is just skipped (MISSING_ROW in the LaTeX, blank in the
# CSV), so this can be launched the moment any subset of the 4 finishes and
# rerun as more land.
#
# Usage: python analysis/p6_analysis_dataset_comparison.py
from __future__ import annotations

import csv
import gc
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "common"))
sys.path.insert(0, str(REPO_ROOT / "src"))
from _common import (  # noqa: E402
    fmt_time_censored, fmt_sci_mean_std, fmt_time_censored_readable, fmt_sci_mean_std_readable,
    render_readable_table, readable_metric_notes, MISSING_ROW,
)

from beamsurrogate.config import load_config  # noqa: E402
from beamsurrogate.registry import MODELS, DATASETS  # noqa: E402
from beamsurrogate.data.split import load_hdf5_dataset, compute_norm_stats  # noqa: E402
from beamsurrogate.data.norm import norm_stats_arrays  # noqa: E402
from beamsurrogate.physics.solver import (  # noqa: E402
    compute_rest_bias, autoregressive_rollout, run_fd_simulation_general,
)
from beamsurrogate.physics.waves import BC_WAVEFORMS, flip  # noqa: E402
from beamsurrogate.evaluate.rollout import RolloutResult  # noqa: E402
from beamsurrogate.evaluate.metrics import build_metrics, aggregate_multi_rollout_metrics  # noqa: E402

# Repeats per (family, excitation) cell: N different random realizations of
# that family's own parameters (amplitude, frequency/pulse-width, ... --
# same AMP_MIN/MAX, OMEGA_MIN/MAX etc. ranges data/generate.py samples from
# for training), not N copies of the same wave. Each family's TEST_WAVES
# seed below spawns N_REPEATS child seeds via a dedicated RNG (see
# build_bcspec_repeats), so results stay reproducible while every repeat is
# still a genuinely different parameter draw -- same reason p1-p5 moved from
# one held-out trajectory to N=99: a single draw doesn't tell you the
# variance, and a marker can always ask "was that just a lucky wave?".
N_REPEATS = 20

RUN_GROUP = "p6_performance_different_datasets"
COMPARATIVE_ID = "p6_diag_dataset_comparison"

# letter -> run_id, per the plan's A/B/C/D <-> dataset mapping.
SOURCES = [
    ("A", "p6_dataset_simple"),
    ("B", "p6_dataset_medium"),
    ("C", "p6_dataset_medium_bidir"),
    ("D", "p6_dataset_complex"),
]

# What each letter's dataset actually contains, per src/beamsurrogate/data/
# generate.py's PROFILES dict -- NOT the thesis draft's Table 11, which
# lists the wrong wave families for B/C (a stray "sum of sinusoids" that
# only belongs to complex's "fourier" family, and "Both ends" overstating
# medium_bidir's actual 50/25/25 mixed driving pattern).
DATASET_DESCRIPTIONS = {
    "A": "Gaussian pulse only. Right end driven, left always at rest. "
         "Dirichlet (displacement) BC only.",
    "B": "5 wave families, evenly weighted -- Gaussian pulse, sine pulse, triangular, "
         "sawtooth, square. Right end driven, left always at rest. "
         "Dirichlet (displacement) BC only.",
    "C": "same 5 families as B. Driving pattern now mixed: 50% both ends driven "
         "(independent family per end), 25% right only, 25% left only. "
         "Dirichlet (displacement) BC only.",
    "D": "6 wave families -- Fourier sum (30%), sinusoid (15%), chirp (15%), Gaussian "
         "pulse (15%), tanh shock (10%), filtered random walk (15%). Driving pattern "
         "mixed: 65% both ends, 15% left only, 15% right only, 5% both at rest. BC type "
         "per end independently drawn from Displacement (Dirichlet), Force/slope "
         "(Neumann), or integrated Velocity (Dirichlet).",
}

# The 6-wave test battery. seed is fixed per wave (not per model) so every
# model is evaluated against the IDENTICAL sampled excitation -- apples to
# apples. Excitation modes per wave, read directly off the manuscript's
# table:dataset_results rows: chirp is R-only; every other wave gets both.
TEST_WAVES = [
    ("gaussian", 2001, ("R", "RL")),
    ("chirp", 2002, ("R",)),
    ("shock", 2003, ("R", "RL")),
    ("filtered_random", 2004, ("R", "RL")),
    ("sawtooth", 2005, ("R", "RL")),
    ("sinusoid", 2006, ("R", "RL")),
]

WAVE_LABELS = {
    "gaussian": "Gaussian", "chirp": "Chirp", "shock": "Tanh shock",
    "filtered_random": "Filtered rand. walk", "sawtooth": "Sawtooth",
    "sinusoid": "Sinusoids",
}

# LaTeX-escaped variant (the "\ " keeps the period from reading as an
# end-of-sentence space) -- only table.tex needs this, table.csv and
# table_readable.txt use the plain WAVE_LABELS.
LATEX_WAVE_LABELS = dict(WAVE_LABELS, **{"filtered_random": "Filtered rand.\\ walk"})


def run_dir(run_id: str) -> Path:
    # Hardcoded, not _common.run_dir() -- see module docstring.
    return REPO_ROOT / "dataset_comparison" / "runs" / run_id


def load_model_and_stats(run_id: str) -> dict | None:
    rd = run_dir(run_id)
    model_path = rd / "model.pth"
    if not model_path.exists():
        print(f"WARNING: {model_path} not found -- {run_id} hasn't finished training yet, "
              f"skipping.", file=sys.stderr)
        return None

    cfg = load_config(rd / "config.yaml")
    dataset_path = REPO_ROOT / "data" / DATASETS[cfg.dataset]
    (df, _FIELDS, INPUTS, OUTPUTS, *_rest) = load_hdf5_dataset(
        cfg.features, cfg, dataset_path, max_trajectories=None)
    norm_stats = compute_norm_stats(df, INPUTS, OUTPUTS, cfg)
    del df, _rest
    gc.collect()

    model = MODELS[cfg.model](len(INPUTS), len(OUTPUTS), cfg)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    mu_in, sd_in, mu_out, sd_out = norm_stats_arrays(norm_stats, INPUTS, OUTPUTS)
    rest_bias = compute_rest_bias(model, mu_in, sd_in, mu_out, sd_out, cfg)
    return dict(cfg=cfg, model=model, mu_in=mu_in, sd_in=sd_in,
                mu_out=mu_out, sd_out=sd_out, rest_bias=rest_bias)


def build_bcspec(family: str, seed: int, excitation: str, cfg):
    rng = np.random.default_rng(seed)
    sampler, _ = BC_WAVEFORMS[family]
    params = sampler(rng, cfg)
    bc_right = ("dirichlet", family, params)
    if excitation == "R":
        bc_left = ("dirichlet", "rest", {})
    else:  # "RL" -- both ends, anti-phase (same convention as medium_bidir)
        bc_left = ("dirichlet", family, flip(family, params))
    return bc_left, bc_right


def repeat_seeds(base_seed: int, n: int) -> list[int]:
    # n child seeds spawned from one base seed via a dedicated RNG, rather
    # than e.g. base_seed+i -- avoids any risk of accidentally overlapping
    # seed ranges between two different (family, excitation) cells while
    # staying fully deterministic/reproducible from TEST_WAVES alone.
    rng = np.random.default_rng(base_seed)
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n)]


def custom_rollout(loaded: dict, bc_left, bc_right) -> RolloutResult:
    # FD ground truth for this arbitrary, not-from-any-dataset BCSpec pair,
    # NN rollout warm-started off that same FD trajectory's own first
    # M_BACK*ndt steps -- same technique p7_structure_several_rods.py uses
    # for its own hand-built BCSpecs.
    cfg = loaded["cfg"]
    U_reel = run_fd_simulation_general(bc_left, bc_right, cfg)
    U = autoregressive_rollout(
        loaded["model"], U_reel, cfg.features,
        loaded["mu_in"], loaded["sd_in"], loaded["mu_out"], loaded["sd_out"],
        loaded["rest_bias"], bc_left, bc_right, cfg)
    return RolloutResult(U=U, U_reel=U_reel, left_bc=bc_left, right_bc=bc_right)


def main():
    comp_dir = REPO_ROOT / "dataset_comparison" / "runs" / COMPARATIVE_ID
    figures_dir = comp_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    n_curves_copied = 0
    for letter, run_id in SOURCES:
        src = run_dir(run_id) / "figures" / "training_curves.png"
        dst = figures_dir / f"training_curves_{letter}.png"
        if src.exists():
            shutil.copy2(src, dst)
            n_curves_copied += 1
        else:
            print(f"WARNING: {src} not found -- skipping training_curves_{letter}.png.",
                  file=sys.stderr)

    # One (dataset, model) at a time -- loading all 4 datasets' trajectory
    # data simultaneously (needed for norm stats) exceeds this box's
    # per-user memory cap. Each model + its norm stats are loaded, rolled
    # out against the full test battery, then freed before the next letter.
    grid: dict[tuple[str, str], dict[str, dict]] = {
        (family, excitation): {}
        for family, _seed, excitations in TEST_WAVES
        for excitation in excitations
    }
    available_letters = []
    for letter, run_id in SOURCES:
        loaded = load_model_and_stats(run_id)
        if loaded is None:
            continue
        available_letters.append(letter)
        for family, seed, excitations in TEST_WAVES:
            for excitation in excitations:
                # N_REPEATS different parameter draws for this family (not
                # N copies of the same wave -- see build_bcspec/repeat_seeds),
                # each run through FD + the model, then aggregated the same
                # way as the held-out-trajectory tables elsewhere in this
                # report (mean+/-std for continuous scalars, median/%reached
                # for the censored T_* family).
                records = []
                for child_seed in repeat_seeds(seed, N_REPEATS):
                    bc_left, bc_right = build_bcspec(family, child_seed, excitation, loaded["cfg"])
                    rollout = custom_rollout(loaded, bc_left, bc_right)
                    records.append(build_metrics(loaded["cfg"], rollout=rollout)["scalars"])
                grid.setdefault((family, excitation), {})[letter] = aggregate_multi_rollout_metrics(records)
        del loaded
        gc.collect()

    if not available_letters:
        print("WARNING: none of the 4 p6 runs have a model.pth yet -- writing an empty-skeleton "
              "table.tex/table.csv only.", file=sys.stderr)

    letters = [letter for letter, _ in SOURCES]

    # Row order matches the manuscript table: block of R rows for all 6
    # waves first, then a block of R+L rows for the 5 waves that have it
    # (chirp is R-only) -- NOT the per-wave (R, then its own RL) order
    # TEST_WAVES itself is written in. block_split marks where table.tex /
    # table_readable.txt insert their blank-row separator.
    ordered_rows = [(family, "R") for family, _seed, exc in TEST_WAVES if "R" in exc]
    block_split = len(ordered_rows)
    ordered_rows += [(family, "RL") for family, _seed, exc in TEST_WAVES if "RL" in exc]

    csv_headers = ["wave", "excitation"]
    for letter in letters:
        csv_headers += [f"{letter}_T_max_5pct_median", f"{letter}_T_max_5pct_std",
                         f"{letter}_T_max_5pct_pct_reached",
                         f"{letter}_E_max_mean", f"{letter}_E_max_std"]
    csv_rows = []
    for family, excitation in ordered_rows:
        row_metrics = grid[(family, excitation)]
        row = [WAVE_LABELS[family], excitation]
        for letter in letters:
            s = row_metrics.get(letter)
            if s is None:
                row += [None, None, None, None, None]
            else:
                row += [s["T_max_5pct"]["median_reached"], s["T_max_5pct"]["std_reached"],
                        s["T_max_5pct"]["pct_reached"],
                        s["E_max"]["mean"], s["E_max"]["std"]]
        csv_rows.append(row)
    with open(comp_dir / "table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_headers)
        w.writerows(csv_rows)

    # -- table_t5.tex / table_emax.tex: paper format, ONE metric per table
    # (was a single combined table.tex with a T_5%/E_max sub-column pair per
    # model -- split in two since the stacked E_max cells needed for width
    # (see _fmt_emax_stacked below) don't make sense mixed into a table
    # that's mostly single-line T_5% cells). Both share the same row order/
    # \addlinespace block split. Custom-built (not _common.render_latex_table)
    # because that helper's header is a flat row of cells, not this
    # two-row multicolumn/cmidrule + sub-header (table_t5.tex doesn't even
    # need the two-row header -- one column per model is enough once E_max
    # isn't sharing the row).
    #
    # T_5% here is specifically T_max_5pct (first time the WORST-NODE error
    # crosses 5%), not T_mean_5pct (beam-mean) -- labelled $T_{5\%}^{\max}$
    # below so that's unambiguous from the header alone.
    #
    # T_5% cells drop the "(pct_reached%)" suffix (LaTeX tables only --
    # table_readable.txt below keeps it): every cell in this particular
    # dataset is (100%) (all 20 draws of every wave family cross the 5%
    # threshold for all 4 models), so the parenthetical never actually
    # carries information -- it's the same "100%" repeated ~88 times.
    def _fmt_t5_notrunc(median_reached, _pct_reached):
        return "--" if median_reached is None else f"{median_reached:.2f}"

    # E_max stacked mean/std (2 lines) instead of one "mean +/- std" line --
    # with 4 models, a single-line scientific-notation +/- pair per model
    # pushes the table past \textwidth. A nested tabular[c] (not \shortstack,
    # which only centers the 2 lines against EACH OTHER, not against this
    # row's own \textbf{Test wave}/Exc. cells) is the standard idiom for a
    # multi-line cell that's actually centered -- both horizontally (its own
    # single 'c' column) and vertically (the outer [c] placement option) --
    # against the rest of the row. No extra package needed.
    def _fmt_emax_stacked(mean, std):
        if mean is None:
            return "--"
        m_mant, m_exp = f"{mean:.2e}".split("e")
        s_mant, s_exp = f"{std:.2e}".split("e")
        return (f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}${m_mant} \\times 10^{{{int(m_exp)}}}$\\\\"
                 f"$\\pm\\, {s_mant} \\times 10^{{{int(s_exp)}}}$\\end{{tabular}}")

    def _row_cells_t5():
        rows_out = []
        for i, (family, excitation) in enumerate(ordered_rows):
            if i == block_split:
                rows_out.append(("addlinespace",))
                continue
            row_metrics = grid[(family, excitation)]
            exc_label = "R" if excitation == "R" else "R+L"
            cells = [LATEX_WAVE_LABELS[family], exc_label]
            for letter in letters:
                s = row_metrics.get(letter)
                t5 = s["T_max_5pct"] if s is not None else None
                cells.append(MISSING_ROW if t5 is None else _fmt_t5_notrunc(t5["median_reached"], t5["pct_reached"]))
            rows_out.append(tuple(cells))
        return rows_out

    def _row_cells_emax():
        rows_out = []
        for i, (family, excitation) in enumerate(ordered_rows):
            if i == block_split:
                rows_out.append(("addlinespace",))
                continue
            row_metrics = grid[(family, excitation)]
            exc_label = "R" if excitation == "R" else "R+L"
            cells = [LATEX_WAVE_LABELS[family], exc_label]
            for letter in letters:
                s = row_metrics.get(letter)
                emax = s["E_max"] if s is not None else None
                cells.append(MISSING_ROW if emax is None else _fmt_emax_stacked(emax["mean"], emax["std"]))
            rows_out.append(tuple(cells))
        return rows_out

    caption_common = (f"Rows are test conditions, identical for all four models; columns are the "
                       f"dataset the model was trained on. All rollouts start from rest. Excitation: "
                       f"R = right end only, R+L = both ends. Each cell is aggregated over "
                       f"$N={N_REPEATS}$ random parameter draws of that wave family (amplitude, "
                       f"frequency/pulse-width, ...), not a single fixed wave.")

    t5_lines = [
        "\\begin{table}[H]", "    \\centering", "    \\small",
        f"    \\caption{{$T_{{5\\%}}^{{\\max}}$ (first time the worst-node error crosses 5% of "
        f"reference amplitude) as a function of training dataset. {caption_common} Right-censored "
        f"(not every draw crosses the threshold) and reported as the median crossing time among "
        f"draws that did. \\todo{{fill in}}}}",
        "    \\label{tab:p6_dataset_comparison_t5}",
        "    \\begin{tabular}{@{}ll " + " ".join(["c"] * len(letters)) + "@{}}",
        "        \\toprule",
        "        \\textbf{Test wave} & \\textbf{Exc.} & " + " & ".join(f"\\textbf{{{l}}}" for l in letters) + " \\\\",
        "        \\midrule",
    ]
    for row in _row_cells_t5():
        t5_lines.append("        \\addlinespace" if row == ("addlinespace",) else "        " + " & ".join(row) + " \\\\")
    t5_lines += ["        \\bottomrule", "    \\end{tabular}", "\\end{table}"]
    (comp_dir / "table_t5.tex").write_text("\n".join(t5_lines) + "\n")

    emax_lines = [
        "\\begin{table}[H]", "    \\centering", "    \\small",
        f"    \\caption{{$E_{{\\max}}$ (peak max absolute rollout error, mean $\\pm$ std across the "
        f"{N_REPEATS} draws) as a function of training dataset. {caption_common} \\todo{{fill in}}}}",
        "    \\label{tab:p6_dataset_comparison_emax}",
        "    \\begin{tabular}{@{}ll " + " ".join(["c"] * len(letters)) + "@{}}",
        "        \\toprule",
        "        \\textbf{Test wave} & \\textbf{Exc.} & " + " & ".join(f"\\textbf{{{l}}}" for l in letters) + " \\\\",
        "        \\midrule",
    ]
    for row in _row_cells_emax():
        emax_lines.append("        \\addlinespace" if row == ("addlinespace",) else "        " + " & ".join(row) + " \\\\")
    emax_lines += ["        \\bottomrule", "    \\end{tabular}", "\\end{table}"]
    (comp_dir / "table_emax.tex").write_text("\n".join(emax_lines) + "\n")

    # -- table_readable.txt: plain-text sibling, same block order, with the
    # A/B/C/D -> dataset mapping spelled out since there's no caption here.
    readable_header = ["Test wave", "Exc."]
    for letter in letters:
        readable_header += [f"{letter} T_5%", f"{letter} E_max"]
    readable_rows = []
    for i, (family, excitation) in enumerate(ordered_rows):
        if i == block_split:
            readable_rows.append(("section", "-- R+L (both ends, anti-phase) --"))
        row_metrics = grid[(family, excitation)]
        exc_label = "R" if excitation == "R" else "R+L"
        row = [WAVE_LABELS[family], exc_label]
        for letter in letters:
            s = row_metrics.get(letter)
            if s is None:
                row += ["MISSING", "MISSING"]
            else:
                row += [fmt_time_censored_readable(s["T_max_5pct"]["median_reached"], s["T_max_5pct"]["pct_reached"]),
                         fmt_sci_mean_std_readable(s["E_max"]["mean"], s["E_max"]["std"])]
        readable_rows.append(row)
    readable_lines = [
        "Rollout performance as a function of training dataset (p6_diag_dataset_comparison)",
        "All rollouts start from rest. Exc.: R = right end only, R+L = both ends (anti-phase).",
        f"Each cell aggregates N={N_REPEATS} random parameter draws of that wave family "
        f"(amplitude, frequency/pulse-width, ...), not a single fixed wave.",
        "",
        "Dataset mapping (per src/beamsurrogate/data/generate.py PROFILES -- corrects "
        "thesis draft Table 11, which listed wrong wave families for B/C):",
    ] + [f"  {letter} = {DATASET_DESCRIPTIONS[letter]}" for letter in letters] + [
        "",
        "-- R (right end only) --",
        render_readable_table(header_cells=readable_header, rows=readable_rows),
        "",
    ] + readable_metric_notes({"T_max_5pct", "E_max"}) + [
        f"(E_max: mean +/- std; T_max_5%: median (%reached) -- across N={N_REPEATS} draws per cell)",
    ]
    (comp_dir / "table_readable.txt").write_text("\n".join(readable_lines) + "\n")

    summary = (
        f"{COMPARATIVE_ID} -- {len(available_letters)}/4 models available "
        f"({', '.join(sorted(available_letters)) or 'none'})\n"
        f"Test battery: {len(csv_rows)} (wave, excitation) rows.\n"
        f"training_curves_{{A,B,C,D}}.png: {n_curves_copied}/4 copied to {figures_dir}\n"
        f"table.csv / table.tex / table_readable.txt written to {comp_dir}\n"
    )
    (comp_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"Done -- outputs in {comp_dir}")


if __name__ == "__main__":
    main()
