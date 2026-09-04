#!/usr/bin/env python3
# The 94-rod triangular/rectangular lattice test (bottom row clamped, 3
# top-center nodes driven with a gaussian pulse, 10s rollout, FD reference
# vs a trained single-rod model reused as every rod's own physics), as a
# single parametrized module -- replaces the 12 hand-copied
# `<name>_test.py` scripts previously scattered across
# runs/p7/p7_assembly_of_rods/*/, each hardcoding its own NAME/SOURCE_RUN_ID
# and each broken the same way (see below). Physics/plotting logic here is
# ported as-is from the most complete of those copies
# (runs/p7/p7_assembly_of_rods/p7_medium_pinn_0.01/p7_medium_pinn_0.01_test.py)
# -- only the plumbing around it changed:
#
#   1. Path resolution fixed. The old copies read `runs/<id>/
#      config.resolved.yaml` and checked `runs/<id>/metrics.json` for the
#      "training finished" gate -- both paths from the pre-restructure flat
#      `runs/<run_id>/` layout, which no longer exists anywhere in this
#      repo (now `runs/<phase-or-group>/<run_id>/config.yaml` +
#      `results.yaml`). Every one of the 12 old copies would crash on this
#      alone if re-run today. resolve_run_dir() below globs for the run_id
#      instead of assuming a naming convention, so it works whether the
#      parent folder is `p7/p7_model_optimization_for_assembly`, a bare
#      `p7`, or anything else.
#   2. Parametrized: source_run_id is a function argument, not a constant
#      you hand-edit into a fresh copy of the file.
#   3. Divergence isn't always NaN. A confirmed real case
#      (p7_dataset_medium_bidir) "completed" while numerically exploding to
#      ratio NN/FD ~= 1.78e10 -- finite, so the old NaN-only assert missed
#      it entirely. compute_metrics() below flags this explicitly.
#   4. Frame cache keyed by source_run_id (content-derived), not a
#      hand-typed NAME that could silently go stale if reused for a
#      different model.
#   5. Returns a metrics dict (not just a summary.txt) so a sweep can
#      consume results directly without re-parsing text.
#
# Output layout: runs/p7/models/<run_id>/ holds BOTH the trained model
# (config.yaml/results.yaml/model.pth, from the training job) AND this
# module's assembly-test output (frames.npz/summary.txt/figures/) side by
# side in the same folder -- one folder per model, train and test together.
# Cross-model comparisons live separately under runs/p7/analysis/.
#
# Usage (single model, standalone):
#   python analysis/p7_structure_several_rods.py <source_run_id> [--cache-only]
from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from beamsurrogate.config import Config, load_config  # noqa: E402

ROLLOUT_T_END = 10.0
FULL_N = 9
INSET_N = FULL_N - 1
N_ROWS = 5
PULSE_SIGMA = 0.3
# Ratio NN/FD peak displacement outside this band is flagged as diverged,
# even if every value stayed finite (see item 3 above) -- 0.5-2.0 comfortably
# brackets every known-good result so far (best is 0.893) while catching
# the 1.78e10 blow-up case with enormous margin.
DIVERGED_RATIO_BAND = (0.5, 2.0)


class RunNotFinishedError(Exception):
    pass


def resolve_run_dir(run_id: str) -> Path:
    matches = sorted(REPO_ROOT.glob(f"runs/**/{run_id}/config.yaml"))
    if not matches:
        raise FileNotFoundError(f"No runs/**/{run_id}/config.yaml found under {REPO_ROOT / 'runs'}")
    if len(matches) > 1:
        raise FileNotFoundError(f"Ambiguous run_id {run_id!r} -- multiple config.yaml found: "
                                 f"{[str(m) for m in matches]}")
    return matches[0].parent


def load_source(run_id: str, allow_unfinished: bool = False) -> tuple[Config, Path]:
    run_dir = resolve_run_dir(run_id)
    if not (run_dir / "results.yaml").exists() and not allow_unfinished:
        raise RunNotFinishedError(
            f"{run_id} has no results.yaml yet -- its training job is most likely still "
            f"running, and model.pth is only the best-so-far checkpoint. To use the partial "
            f"checkpoint anyway, pass allow_unfinished=True.")
    cfg = load_config(run_dir / "config.yaml")
    return cfg, run_dir


# ---------------------------------------------------------------------------
# Topology: a triangular lattice occupying a rectangle. Rows of nodes are
# stacked vertically, alternating between a "full" row (FULL_N nodes, flush
# with both side edges) and an "inset" row (FULL_N-1 nodes, offset by half a
# spacing so it sits between the full row's nodes). Horizontal rods link
# neighbors within a row, diagonal rods link each inset-row node to the two
# full-row nodes it sits between. The row spacing is chosen so every
# triangle is equilateral, which makes every rod the same length L.
# ---------------------------------------------------------------------------
def _nid(row: int, i: int) -> str:
    return f"R{row}_{i}"


def build_lattice(cfg: Config, driven_bc: tuple | None = None):
    L = cfg.L
    DX = L
    DY = L * np.sqrt(3) / 2
    row_is_full = [row % 2 == 0 for row in range(N_ROWS)]

    pos = {}
    for row in range(N_ROWS):
        n = FULL_N if row_is_full[row] else INSET_N
        x0 = 0.0 if row_is_full[row] else DX / 2
        for i in range(n):
            pos[_nid(row, i)] = np.array([x0 + i * DX, row * DY])

    raw_edges = []
    # Horizontal rods within each row, except the top row (the driven
    # boundary never connects to itself -- no single rod ever has two
    # actively-driven ends, since the model was only ever trained with one
    # driven end per rod).
    for row in range(N_ROWS - 1):
        n = FULL_N if row_is_full[row] else INSET_N
        for i in range(n - 1):
            raw_edges.append((_nid(row, i), _nid(row, i + 1)))

    # Diagonal rods between every pair of consecutive rows.
    for row in range(N_ROWS - 1):
        if row_is_full[row]:
            full_row, inset_row = row, row + 1
        else:
            full_row, inset_row = row + 1, row
        for i in range(INSET_N):
            raw_edges.append((_nid(inset_row, i), _nid(full_row, i)))
            raw_edges.append((_nid(inset_row, i), _nid(full_row, i + 1)))

    FIXED_NODES = {_nid(0, i) for i in range(FULL_N)}
    DRIVEN_NODES = {_nid(N_ROWS - 1, i) for i in (3, 4, 5)}

    def role_rank(node_id: str) -> int:
        # The source model was only ever trained with the clamped end as a
        # rod's .start and the driven/pulse end as a rod's .end -- keep
        # every rod in that same orientation so the model sees inputs like
        # training.
        if node_id in FIXED_NODES:
            return 0
        if node_id in DRIVEN_NODES:
            return 2
        return 1

    edges = [(a, b) if role_rank(a) <= role_rank(b) else (b, a) for a, b in raw_edges]

    pulse_a = cfg.AMP_MAX
    default_bc = ("dirichlet", "gaussian", {"A": pulse_a, "sigma": PULSE_SIGMA})
    nodes = {}
    for node_id, node_pos in pos.items():
        if node_id in FIXED_NODES:
            nodes[node_id] = {"pos": node_pos, "driven": True, "bc": ("dirichlet", "rest", {})}
        elif node_id in DRIVEN_NODES:
            nodes[node_id] = {"pos": node_pos, "driven": True, "bc": driven_bc or default_bc}
        else:
            nodes[node_id] = {"pos": node_pos, "driven": False}

    return {"pos": pos, "edges": edges, "nodes": nodes, "fixed": FIXED_NODES, "driven": DRIVEN_NODES}


# ---------------------------------------------------------------------------
# Physics: the FD reference and the NN rollout. Both return an array of
# shape (n_frames, n_rods, Nx) plus the timestep index of each frame.
# ---------------------------------------------------------------------------
def load_model_bundle(cfg: Config, model_path: Path) -> dict:
    # Everything about a (source_run_id) that's independent of which
    # excitation drives the lattice -- split out from run_physics() so a
    # sweep replaying many excitations through the SAME model (e.g. one real
    # test trajectory per call) loads the model and recomputes norm_stats
    # ONCE, not once per call. The 200-trajectory HDF5 load + norm_stats
    # recompute dominates a single call's wall time (~50s+ of ~90s measured),
    # dwarfing the actual FD+NN rollout it's paying for -- see
    # analysis/p7_family_sweep.py's docstring for the measurement.
    import torch
    from beamsurrogate.registry import MODELS, DATASETS
    from beamsurrogate.data.split import load_hdf5_dataset, compute_norm_stats
    from beamsurrogate.data.norm import norm_stats_arrays
    from beamsurrogate.data.windows import make_feature_columns, make_output_columns
    from beamsurrogate.physics.solver import compute_rest_bias

    INPUT_FIELDS = list(cfg.features)
    INPUTS = make_feature_columns(INPUT_FIELDS, cfg)
    OUTPUTS = make_output_columns(cfg)

    model = MODELS[cfg.model](len(INPUTS), len(OUTPUTS), cfg)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    print(f"Recomputing norm_stats from a 200-trajectory subsample of {cfg.dataset}...")
    dataset_path = REPO_ROOT / "data" / DATASETS[cfg.dataset]
    (df, _FIELDS, INPUTS_check, OUTPUTS_check, *_rest) = load_hdf5_dataset(
        INPUT_FIELDS, cfg, dataset_path, max_trajectories=200)
    assert INPUTS_check == INPUTS and OUTPUTS_check == OUTPUTS
    norm_stats = compute_norm_stats(df, INPUTS, OUTPUTS, cfg)
    del df, _FIELDS, _rest
    gc.collect()
    mu_in, sd_in, mu_out, sd_out = norm_stats_arrays(norm_stats, INPUTS, OUTPUTS)
    rest_bias = compute_rest_bias(model, mu_in, sd_in, mu_out, sd_out, cfg)

    return {"model": model, "mu_in": mu_in, "sd_in": sd_in, "mu_out": mu_out, "sd_out": sd_out,
            "rest_bias": rest_bias, "INPUT_FIELDS": INPUT_FIELDS}


def run_physics(cfg: Config, bundle: dict, lattice: dict):
    import torch
    from beamsurrogate.data.windows import build_window
    from beamsurrogate.physics.waves import apply_boundary, bc_value

    nodes, edges = lattice["nodes"], lattice["edges"]
    rollout_nt = int(ROLLOUT_T_END / cfg.dt)

    model = bundle["model"]
    mu_in, sd_in, mu_out, sd_out = bundle["mu_in"], bundle["sd_in"], bundle["mu_out"], bundle["sd_out"]
    rest_bias = bundle["rest_bias"]
    INPUT_FIELDS = bundle["INPUT_FIELDS"]

    # ------------------------------ FD reference ---------------------------
    class FDRod:
        def __init__(self, start, end):
            self.start, self.end = start, end
            self.u = np.zeros(cfg.Nx)
            self.u_prev = np.zeros(cfg.Nx)

    fd_rods = [FDRod(s, e) for s, e in edges]
    node_rods_fd = {n: [] for n in nodes}
    for rod in fd_rods:
        node_rods_fd[rod.start].append((rod, "start"))
        node_rods_fd[rod.end].append((rod, "end"))

    fd_steps, fd_U = [], []

    print("Running FD reference...")
    _fd_t0 = time.perf_counter()
    for n in range(rollout_nt):
        t_new = (n + 1) * cfg.dt
        u_new_list = []
        for rod in fd_rods:
            u, u_prev = rod.u, rod.u_prev
            u_new = np.empty_like(u)
            u_new[1:-1] = (2 * u[1:-1] - u_prev[1:-1]
                            + cfg.CFL**2 * (u[:-2] - 2 * u[1:-1] + u[2:]))
            u_new_list.append(u_new)

        node_value = {}
        for node_id, node in nodes.items():
            if node["driven"]:
                node_value[node_id] = bc_value(node["bc"], t_new)
                continue
            touching = node_rods_fd[node_id]
            if len(touching) == 2:
                (rod1, side1), (rod2, side2) = touching
                b1, i1 = (0, 1) if side1 == "start" else (-1, -2)
                u_curr = rod1.u[b1]
                u_prev_node = rod1.u_prev[b1]
                node_value[node_id] = (2 * u_curr - u_prev_node
                                        + cfg.CFL**2 * (rod1.u[i1] - 2 * u_curr + rod2.u[b1 == 0 and 1 or -2]))
            else:
                inside_values = []
                for rod, u_new in zip(fd_rods, u_new_list):
                    if rod.start == node_id:
                        inside_values.append(u_new[1])
                    if rod.end == node_id:
                        inside_values.append(u_new[-2])
                node_value[node_id] = np.mean(inside_values)

        for rod, u_new in zip(fd_rods, u_new_list):
            u_new[0] = node_value[rod.start]
            u_new[-1] = node_value[rod.end]
            rod.u_prev = rod.u
            rod.u = u_new

        if (n + 1) % cfg.ndt == 0:
            fd_steps.append(n + 1)
            fd_U.append(np.stack([r.u for r in fd_rods]))

    fd_time_s = time.perf_counter() - _fd_t0
    print(f"FD reference done: {len(fd_steps)} frames in {fd_time_s:.3f}s (wall clock).")

    # ------------------------------ NN surrogate ---------------------------
    class NNRod:
        def __init__(self, start, end):
            self.start, self.end = start, end
            self.U = np.zeros((rollout_nt + 1, cfg.Ntot), dtype=np.float32)

    nn_rods = [NNRod(s, e) for s, e in edges]
    # Every node's touching rods, computed once instead of re-scanning all
    # nn_rods for every node on every timestep (that rescan -- previously
    # `[r for r in nn_rods if node_id in (r.start, r.end)]`, redone per node
    # per h per hop-block -- was the dominant cost of the NN rollout on this
    # lattice; this dict makes it an O(degree) lookup instead of O(n_rods)).
    node_rods_nn = {n: [] for n in nodes}
    for rod in nn_rods:
        node_rods_nn[rod.start].append(rod)
        node_rods_nn[rod.end].append(rod)
    history_needed = cfg.M_BACK * cfg.ndt
    hop = cfg.N_FWD * cfg.ndt
    nn_steps, nn_U = [], []

    print("Running NN surrogate...")
    _nn_t0 = time.perf_counter()
    for n in range(history_needed, rollout_nt - hop + 1, hop):
        m_list = [n - lag * cfg.ndt for lag in range(cfg.M_BACK)]
        X = np.concatenate(
            [build_window(m_list, lambda m, rod=rod: rod.U[m], INPUT_FIELDS, cfg) for rod in nn_rods],
            axis=0,
        )
        Xn = (X - mu_in) / sd_in
        with torch.no_grad():
            sortie = model(torch.tensor(Xn)).numpy()
        deltas = sortie * sd_out + mu_out - rest_bias
        deltas_per_rod = np.split(deltas, len(nn_rods))

        for h in range(1, cfg.N_FWD + 1):
            s = n + h * cfg.ndt
            t = s * cfg.dt

            for rod, d in zip(nn_rods, deltas_per_rod):
                rod.U[s, cfg.nodes] = rod.U[n, cfg.nodes] + d[:, h - 1]

            def rod_reading(rod, node_id, dist):
                if rod.start == node_id:
                    return rod.U[s, cfg.i_left + dist]
                return rod.U[s, cfg.i_right - 1 - dist]

            node_value = {}
            for node_id, node in nodes.items():
                if node["driven"]:
                    node_value[node_id] = bc_value(node["bc"], t)
                else:
                    # sum()/len() on this 2-6 element list instead of np.mean() --
                    # numpy's generic reduction (dtype checks, array creation, ufunc
                    # dispatch) has fixed overhead that dominates at this size; this
                    # call runs ~550k times over a rollout and was roughly half of
                    # the NN wall time (measured via cProfile on the isolated loop).
                    touching = node_rods_nn[node_id]
                    readings = [rod_reading(rod, node_id, 0) for rod in touching]
                    node_value[node_id] = float(sum(readings) / len(readings))

            for rod in nn_rods:
                for side, node_id in (("left", rod.start), ("right", rod.end)):
                    node = nodes[node_id]
                    if node["driven"]:
                        apply_boundary(rod.U[s], side, "dirichlet", node_value[node_id], cfg)
                        continue
                    others = [r for r in node_rods_nn[node_id] if r is not rod]
                    if not others:
                        apply_boundary(rod.U[s], side, "neumann", 0.0, cfg)
                        continue
                    if side == "left":
                        rod.U[s, cfg.i_left] = node_value[node_id]
                        for dist in range(1, cfg.SS + 1):
                            readings = [rod_reading(r, node_id, dist) for r in others]
                            rod.U[s, cfg.i_left - dist] = float(sum(readings) / len(readings))
                    else:
                        rod.U[s, cfg.i_right - 1] = node_value[node_id]
                        for dist in range(1, cfg.SS + 1):
                            readings = [rod_reading(r, node_id, dist) for r in others]
                            rod.U[s, cfg.i_right - 1 + dist] = float(sum(readings) / len(readings))

            if cfg.SMOOTH_ALPHA > 0:
                j0, j1 = cfg.i_left + 1, cfg.i_right
                for rod in nn_rods:
                    lap = rod.U[s, j0-1:j1-1] - 2*rod.U[s, j0:j1] + rod.U[s, j0+1:j1+1]
                    rod.U[s, j0:j1] += cfg.SMOOTH_ALPHA * lap

            nn_steps.append(s)
            nn_U.append(np.stack([rod.U[s, cfg.nodes] for rod in nn_rods]))

    nn_time_s = time.perf_counter() - _nn_t0
    print(f"NN rollout done: {len(nn_steps)} frames covering {ROLLOUT_T_END:.1f} s "
          f"in {nn_time_s:.3f}s (wall clock).")

    return (np.array(fd_steps), np.stack(fd_U).astype(np.float32),
            np.array(nn_steps), np.stack(nn_U).astype(np.float32),
            fd_time_s, nn_time_s)


def _get_frames(source_run_id: str, cfg: Config, bundle_factory, lattice: dict, output_dir: Path):
    # bundle_factory is a zero-arg callable, not a pre-built bundle -- so a
    # cache hit costs nothing (no model load, no norm_stats recompute), and a
    # caller sweeping many excitations through the same model can pass one
    # that just returns an already-loaded bundle instead of reloading it.
    cache_path = output_dir / f"{source_run_id}_frames.npz"
    if cache_path.exists():
        print(f"Reusing cached frames from {cache_path.name} (delete it to recompute).")
        cached = np.load(cache_path)
        fd_steps, fd_U = cached["fd_steps"], cached["fd_U"]
        nn_steps, nn_U = cached["nn_steps"], cached["nn_U"]
        fd_time_s = float(cached["fd_time_s"]) if "fd_time_s" in cached else None
        nn_time_s = float(cached["nn_time_s"]) if "nn_time_s" in cached else None
    else:
        fd_steps, fd_U, nn_steps, nn_U, fd_time_s, nn_time_s = run_physics(cfg, bundle_factory(), lattice)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, fd_steps=fd_steps, fd_U=fd_U, nn_steps=nn_steps, nn_U=nn_U,
                             fd_time_s=fd_time_s, nn_time_s=nn_time_s)
        print(f"Cached raw frames to {cache_path.name}")
    return fd_steps, fd_U, nn_steps, nn_U, fd_time_s, nn_time_s


def compute_metrics(source_run_id: str, fd_steps, fd_U, nn_steps, nn_U, fd_time_s, nn_time_s,
                     dt: float | None = None) -> dict:
    diverged_nan = bool(np.isnan(fd_U).any() or np.isnan(nn_U).any())

    fd_amp = np.abs(fd_U).max(axis=2)
    fd_peak = float(fd_amp.max())
    if diverged_nan:
        nn_peak = float("nan")
        ratio = float("nan")
    else:
        nn_amp = np.abs(nn_U).max(axis=2)
        nn_peak = float(nn_amp.max())
        ratio = nn_peak / fd_peak

    lo, hi = DIVERGED_RATIO_BAND
    diverged = diverged_nan or not (lo <= ratio <= hi)

    metrics = {
        "source_run_id": source_run_id,
        "n_rods": int(fd_U.shape[1]),
        "rollout_t_end": ROLLOUT_T_END,
        "fd_peak": fd_peak,
        "nn_peak": nn_peak,
        "ratio_nn_fd": ratio,
        "diverged": diverged,
        "diverged_nan": diverged_nan,
        "fd_time_s": fd_time_s,
        "nn_time_s": nn_time_s,
    }

    if diverged_nan:
        metrics.update(abs_max=float("nan"), abs_mean=float("nan"),
                        rel_max_pct=float("nan"), rel_mean_pct=float("nan"))
        return metrics

    # Overlap-based alignment, not a suffix assumption: an M_BACK=3 model's
    # nn_steps IS a contiguous suffix of fd_steps ending at fd's last step
    # (the old assert's assumption), but an M_BACK=2 model's nn_steps window
    # is shifted one step earlier -- same length, doesn't reach fd's final
    # step -- so the old exact-suffix assert failed there even though the
    # two arrays still overlap over almost their entire range. Intersecting
    # on step VALUES computes the error over whatever the two actually share,
    # correct for either case (and any other stencil timing offset).
    common_steps, fd_idx, nn_idx = np.intersect1d(fd_steps, nn_steps, assume_unique=True, return_indices=True)
    if len(common_steps) == 0:
        raise ValueError(f"{source_run_id}: FD and NN timesteps share no common steps at all")
    err_U = fd_U[fd_idx] - nn_U[nn_idx]
    abs_max = float(np.abs(err_U).max())
    abs_mean = float(np.abs(err_U).mean())
    metrics.update(abs_max=abs_max, abs_mean=abs_mean,
                    rel_max_pct=100 * abs_max / fd_peak, rel_mean_pct=100 * abs_mean / fd_peak)

    # Table-13-style scalars (t_div, time of max error, % time above
    # threshold) -- same definitions the single-rod evaluate/metrics.py uses
    # (t_div at 10% of peak amplitude, P_thr at 5%), applied here to the
    # per-step worst-lattice-node error instead of the worst-beam-node error.
    if dt is not None:
        err_max_curve = np.abs(err_U).max(axis=(1, 2))
        t_curve = common_steps * dt
        above_10 = err_max_curve > 0.10 * fd_peak
        t_div = float(t_curve[above_10][0]) if above_10.any() else None
        i_max = int(np.argmax(err_max_curve))
        above_5 = err_max_curve > 0.05 * fd_peak
        metrics.update(
            t_div=t_div,
            t_of_max_error=float(t_curve[i_max]),
            pct_time_above_threshold=100.0 * float(above_5.mean()),
        )
    return metrics


def first_crossing_times(err_U, err_t, fd_peak: float) -> tuple[float | None, float | None]:
    # T_5%^max / T_10%^max: first time the worst-lattice-point error crosses
    # 5% / 10% of the FD peak (T_10% is the same quantity as compute_metrics'
    # t_div, computed independently here since callers like
    # p7_export_latex_report.py/p7_analysis_assembly_metrics.py work from a
    # freshly rebuilt err_U rather than compute_metrics' fd_idx/nn_idx pair).
    err_max_curve = np.abs(err_U).max(axis=(1, 2))

    def _crossing(level_pct):
        level = level_pct / 100.0 * fd_peak
        above = err_max_curve > level
        return float(err_t[above][0]) if above.any() else None

    return _crossing(5), _crossing(10)


def write_summary_txt(metrics: dict, out_path: Path, tag: str | None = None) -> None:
    m = metrics
    excitation_desc = "gaussian pulse" if tag is None else f"held-out test trajectory ({tag})"
    lines = [
        f"p17 rectangular assembly -- {m['source_run_id']}",
        f"{m['n_rods']}-rod triangular lattice, {excitation_desc}, {m['rollout_t_end']:.1f}s rollout",
        "",
    ]
    if m["diverged_nan"]:
        lines += ["DIVERGED -- FD or NN frames contain NaN. No further metrics computed."]
    else:
        lines += [
            "PEAK DISPLACEMENT",
            f"  FD peak |u|            : {m['fd_peak']:.6f}",
            f"  NN peak |u|            : {m['nn_peak']:.6f}",
            f"  ratio NN/FD            : {m['ratio_nn_fd']:.3f}"
            + (f"  ** OUTSIDE {DIVERGED_RATIO_BAND} -- FLAGGED AS DIVERGED **" if m["diverged"] else ""),
            "",
            "FD vs NN DIFFERENCE (over every lattice point, all frames)",
            f"  absolute max |FD-NN|   : {m['abs_max']:.6f}",
            f"  absolute mean |FD-NN|  : {m['abs_mean']:.6f}",
            f"  relative max  (% of FD peak) : {m['rel_max_pct']:.2f}%",
            f"  relative mean (% of FD peak) : {m['rel_mean_pct']:.2f}%",
            "",
        ]
        if "t_div" in m:
            lines += [
                "ROLLOUT-TIME METRICS (worst-lattice-node error vs simulation time)",
                f"  t_div (first crossing of 10% of FD peak) : "
                + (f"{m['t_div']:.2f}s" if m["t_div"] is not None else "never reached"),
                f"  time of max error                        : {m['t_of_max_error']:.2f}s",
                f"  proportion of rollout time above 5% of FD peak : {m['pct_time_above_threshold']:.1f}%",
                "",
            ]
        lines += ["COMPUTATIONAL TIME -- measured on THIS lattice, not a single isolated rod"]
        if m["fd_time_s"] is None or m["nn_time_s"] is None:
            lines += ["  NOT AVAILABLE -- frames were loaded from a cache written before timing "
                      "was tracked. Delete the _frames.npz cache and re-run to measure it."]
        else:
            lines += [
                f"  FD wall time           : {m['fd_time_s']:.3f} s (full {m['rollout_t_end']:.1f}s rollout, "
                f"{m['n_rods']} rods)",
                f"  NN wall time           : {m['nn_time_s']:.3f} s",
                f"  NN / FD wall-time ratio: {m['nn_time_s'] / m['fd_time_s']:.2f}x "
                f"({'slower' if m['nn_time_s'] > m['fd_time_s'] else 'faster'})",
            ]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {out_path.name}")


def run_lattice_test(source_run_id: str, output_dir: Path, cache_only: bool = False,
                      allow_unfinished: bool = False, make_gifs: bool = True, make_figures: bool = True,
                      driven_bc: tuple | None = None, tag: str | None = None, bundle: dict | None = None) -> dict:
    cfg, run_dir = load_source(source_run_id, allow_unfinished=allow_unfinished)
    lattice = build_lattice(cfg, driven_bc=driven_bc)
    output_dir.mkdir(parents=True, exist_ok=True)

    # tag disambiguates cache/output filenames when the same model is replayed
    # against several different excitations (e.g. one real test trajectory per
    # call in a family sweep) -- tag=None reproduces today's exact filenames,
    # so the already-completed gaussian-only folders are untouched.
    #
    # bundle lets a caller sweeping many excitations through the same model
    # (see analysis/p7_family_sweep.py) pass an already-loaded
    # load_model_bundle() result instead of paying its ~50s+ reload
    # (model load + 200-trajectory norm_stats recompute) on every call --
    # left None here, a cache miss loads it fresh, same as before this option
    # existed.
    cache_key = f"{source_run_id}_{tag}" if tag else source_run_id
    bundle_factory = (lambda: bundle) if bundle is not None else (lambda: load_model_bundle(cfg, run_dir / "model.pth"))
    fd_steps, fd_U, nn_steps, nn_U, fd_time_s, nn_time_s = _get_frames(
        cache_key, cfg, bundle_factory, lattice, output_dir)

    metrics = compute_metrics(source_run_id, fd_steps, fd_U, nn_steps, nn_U, fd_time_s, nn_time_s, dt=cfg.dt)
    metrics["tag"] = tag
    write_summary_txt(metrics, output_dir / f"{cache_key}_summary.txt", tag=tag)

    if metrics["diverged_nan"]:
        return metrics   # nothing else to plot -- FD/NN frames contain NaN

    trace_path = (output_dir / f"{cache_key}_gaussian_trace.npz" if tag is None
                  else output_dir / f"{cache_key}_trace.npz")
    np.savez(trace_path,
              fd_t=fd_steps * cfg.dt, nn_t=nn_steps * cfg.dt,
              rod_names=np.array([f"{a}-{b}" for a, b in lattice["edges"]], dtype=object),
              fd_traces=np.abs(fd_U).max(axis=2).T, nn_traces=np.abs(nn_U).max(axis=2).T)

    if make_figures:
        _make_figures(cache_key, cfg, lattice, fd_steps, fd_U, nn_steps, nn_U,
                       output_dir, cache_only=cache_only, make_gifs=make_gifs, tag=tag)
    return metrics


# ---------------------------------------------------------------------------
# Figures: error-vs-time curves, share-exceeding-threshold curve, a static
# FD/NN snapshot grid, and (unless cache_only) the 3 top-down 2D gifs.
# Ported as-is from the reference script -- see its own comments (preserved
# below) for the reasoning behind node de-duplication, the shared colour
# scale, and the 3D layout constants (3D rendering itself stays disabled,
# same as the reference script -- 2D only).
# ---------------------------------------------------------------------------
def _make_figures(source_run_id, cfg, lattice, fd_steps, fd_U, nn_steps, nn_U, output_dir,
                    cache_only: bool, make_gifs: bool, tag: str | None = None):
    nodes, edges, pos, FIXED_NODES = lattice["nodes"], lattice["edges"], lattice["pos"], lattice["fixed"]
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    excitation_label = "gaussian pulse" if tag is None else f"test-trajectory replay ({tag})"

    # Same overlap-based alignment as compute_metrics() -- see its comment.
    err_steps, fd_idx, nn_idx = np.intersect1d(fd_steps, nn_steps, assume_unique=True, return_indices=True)
    err_U = fd_U[fd_idx] - nn_U[nn_idx]

    # Each rod stores cfg.Nx nodes, and its first/last sit ON a lattice node,
    # so a junction of degree K appears K times in err_U -- count every
    # lattice node exactly once (clamped nodes excluded: FD and NN both hold
    # them at ~0, pure floating-point noise, not real error).
    def _lattice_node_abs_error():
        cols = []
        for node_id in nodes:
            if node_id in FIXED_NODES:
                continue
            copies = []
            for r, (a, b) in enumerate(edges):
                if a == node_id:
                    copies.append(np.abs(err_U[:, r, 0]))
                if b == node_id:
                    copies.append(np.abs(err_U[:, r, -1]))
            cols.append(np.mean(copies, axis=0))
        return np.stack(cols, axis=1)

    abs_interior = np.abs(err_U[:, :, 1:-1]).reshape(len(err_steps), -1)
    abs_nodes = np.concatenate([abs_interior, _lattice_node_abs_error()], axis=1)
    err_t = err_steps * cfg.dt
    err_max_curve = abs_nodes.max(axis=1)
    err_mean_curve = abs_nodes.mean(axis=1)

    FD_PEAK = float(np.abs(fd_U).max())
    ERR_REF_LEVELS = ((5, "#4d4d4d"), (10, "#8a8a8a"))

    def save_error_curve(values, kind, out_path):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.semilogy(err_t, values, lw=1.4, color="#b2182b", label=f"{kind} |NN - FD|")
        for pct, color in ERR_REF_LEVELS:
            level = pct / 100 * FD_PEAK
            crossed = values > level
            note = f"reached at t={float(err_t[np.argmax(crossed)]):.2f}s" if crossed.any() else "never reached"
            if crossed.any():
                ax.axvline(float(err_t[np.argmax(crossed)]), ls="--", lw=1.0, color=color, alpha=0.8)
            ax.axhline(level, ls=":", lw=1.5, color=color, label=f"{pct}% of FD peak ({level:.2e}) -- {note}")
        ax.set_xlabel("simulation time  t  [s]")
        ax.set_ylabel(f"{kind} |NN - FD|  over the lattice")
        ax.set_title(f"{kind.capitalize()} error vs time -- {excitation_label}, {len(edges)}-rod "
                      f"triangular assembly ({source_run_id})", fontsize=11)
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.15)
        ax.set_xlim(err_t[0], err_t[-1])
        ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path.name}")

    save_error_curve(err_max_curve, "max", figures_dir / f"{source_run_id}_error_max.png")
    save_error_curve(err_mean_curve, "mean", figures_dir / f"{source_run_id}_error_mean.png")

    def save_percent_exceeding_curve(level_pct, out_path):
        level = level_pct / 100 * FD_PEAK
        frac = 100.0 * (abs_nodes > level).mean(axis=1)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(err_t, frac, lw=1.4, color="#b2182b")
        ax.set_xlabel("simulation time  t  [s]")
        ax.set_ylabel(f"% of points with |NN - FD| > {level_pct}% of FD peak")
        ax.set_title(f"Share of points past {level_pct}% error vs time ({source_run_id})", fontsize=11)
        ax.grid(True, alpha=0.35)
        ax.set_xlim(err_t[0], err_t[-1])
        ax.set_ylim(0, 100)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path.name}")

    save_percent_exceeding_curve(5, figures_dir / f"{source_run_id}_pct_exceeding_5pct.png")

    U_MAX = float(max(np.abs(fd_U).max(), np.abs(nn_U).max(), np.abs(err_U).max()))
    XS = np.concatenate([np.linspace(pos[a][0], pos[b][0], cfg.Nx) for a, b in edges])
    YS = np.concatenate([np.linspace(pos[a][1], pos[b][1], cfg.Nx) for a, b in edges])
    PAD = 0.4
    X_LO, X_HI = XS.min() - PAD, XS.max() + PAD
    Y_LO, Y_HI = YS.min() - PAD, YS.max() + PAD

    def save_snapshots_single(out_dir, target_t=(2.5, 5.0, 7.5, 10.0),
                                target_tag=("t25", "t50", "t75", "t100")):
        # One file per (source, timestep) -- |u| only (no sign), with a
        # colorbar showing its scale, no title, no caption -- named
        # panel_{fd,nn,err}_{tag}.png directly,
        # matching the LaTeX figure's \includegraphics{figures/panel_fd_t25.png}
        # paths exactly, so there is exactly one file per image rather than a
        # long-named original plus a separately-copied short-named duplicate.
        # 3 rows -- fd/nn/err -- so the thesis's Figure 24 (top: FD reference,
        # middle: surrogate, bottom: absolute error) has a real "err" panel,
        # not just fd/nn. err uses err_steps/err_U (the overlap-aligned FD-NN
        # difference computed above) rather than separately-nearest-matched
        # fd_idx/nn_idx, so the error shown is always the actual FD-NN
        # difference AT one real common step, not a difference of two
        # independently-nearest-picked (and potentially off-by-one) frames.
        out_dir.mkdir(exist_ok=True)
        fd_idx = [int(np.argmin(np.abs(fd_steps * cfg.dt - t))) for t in target_t]
        nn_idx = [int(np.argmin(np.abs(nn_steps * cfg.dt - t))) for t in target_t]
        e_idx = [int(np.argmin(np.abs(err_t - t))) for t in target_t]

        rows = (("FD reference", fd_U, fd_idx, "fd", 0.0, U_MAX, "Reds", "|u|"),
                ("NN surrogate", nn_U, nn_idx, "nn", 0.0, U_MAX, "Reds", "|u|"),
                ("|FD - NN| error", err_U, e_idx, "err", 0.0, U_MAX, "Reds", "|FD - NN|"))
        for row_label, U_frames, idx_list, kind, vmin, vmax, cmap, cbar_label in rows:
            for tag, idx in zip(target_tag, idx_list):
                u = np.abs(U_frames[idx].ravel())
                fig, ax = plt.subplots(figsize=(5.8, 5))
                scat = ax.scatter(XS, YS, c=u, cmap=cmap, vmin=vmin, vmax=vmax, s=6)
                ax.set_xlim(X_LO, X_HI); ax.set_ylim(Y_LO, Y_HI); ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                # The structure's aspect (wide/short lattice here) is nowhere near the
                # figure's own -- under aspect="equal" matplotlib shrinks ax's box to
                # match it, but only once the layout is actually drawn. A colorbar
                # attached via fraction=... BEFORE that draw sizes itself off the
                # pre-shrink (full-height) box, so it ends up much taller than the
                # structure it's labeling. Force the draw first, then size a colorbar
                # axes off ax's real (post-shrink) height so the two match.
                fig.canvas.draw()
                pos = ax.get_position()
                cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.03, pos.height])
                cbar = fig.colorbar(scat, cax=cax)
                cbar.set_label(cbar_label, fontsize=14)
                cbar.ax.tick_params(labelsize=12)
                out_path = out_dir / f"panel_{kind}_{tag}.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
        print(f"Saved {3 * len(target_t)} panels to {out_dir.name}/")

    save_snapshots_single(figures_dir / "snapshots")

    def frac_time_violating_at(level_pct):
        # Per lattice point, the FRACTION OF THE ROLLOUT during which
        # |FD-NN| exceeded level_pct% of the FD peak -- not just whether it
        # ever crossed once. A binary "ever exceeds" flag saturates towards
        # "every point violates" on a long rollout (every point crosses once,
        # eventually) and hides which points are the actual worst/most
        # recurrent offenders; this continuous version is the per-location
        # analogue of save_percent_exceeding_curve's aggregate time curve above.
        level = (level_pct / 100.0) * FD_PEAK
        return 100.0 * (np.abs(err_U) > level).mean(axis=0).ravel()

    def save_threshold_violation_map(out_path, level_pct=5):
        # Static spatial map (thesis Figure 25).
        frac_time_violating = frac_time_violating_at(level_pct)
        # Taller than the structure alone needs (unlike save_snapshots_single's
        # panels, this one's colorbar carries a full-sentence label that, at
        # this fontsize, is longer than the structure itself is tall -- extra
        # figure height gives the label room without it getting clipped by
        # bbox_inches="tight").
        fig, ax = plt.subplots(figsize=(10.8, 6.5))
        scat = ax.scatter(XS, YS, c=frac_time_violating, cmap="Oranges", vmin=0.0, vmax=100.0, s=15)
        ax.set_xlim(X_LO, X_HI); ax.set_ylim(Y_LO, Y_HI); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(scat, ax=ax, fraction=0.04, pad=0.03)
        cbar.set_label(f"% of rollout time with |FD-NN| > {level_pct}% of FD peak", fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path.name}")

    def save_worst_points_map(out_path, rank_level_pct=5, top_pct=10):
        # Binary spatial map: highlights the top_pct% of lattice points
        # ranked by frac_time_violating_at(rank_level_pct) -- i.e. the points
        # that violate the rank_level_pct% threshold most often over the
        # rollout, not points exceeding some fixed error magnitude.
        frac_time_violating = frac_time_violating_at(rank_level_pct)
        cutoff = np.percentile(frac_time_violating, 100.0 - top_pct)
        is_worst = frac_time_violating >= cutoff
        fig, ax = plt.subplots(figsize=(10.8, 4.5))
        ax.scatter(XS[~is_worst], YS[~is_worst], c="#d9d9d9", s=15)
        ax.scatter(XS[is_worst], YS[is_worst], c="#d62728", s=15,
                   label=f"worst {top_pct}% of points ({int(is_worst.sum())}/{is_worst.size})")
        ax.set_xlim(X_LO, X_HI); ax.set_ylim(Y_LO, Y_HI); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.set_title(f"Worst {top_pct}% of points by time spent violating the "
                     f"{rank_level_pct}%-of-FD-peak threshold", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path.name}")

    save_threshold_violation_map(figures_dir / "threshold_violation_map.png")

    violation_maps_dir = figures_dir / "violation_maps"
    violation_maps_dir.mkdir(exist_ok=True)
    shutil.copy2(figures_dir / "threshold_violation_map.png",
                 violation_maps_dir / "threshold_violation_map_5pct.png")
    save_threshold_violation_map(violation_maps_dir / "threshold_violation_map_10pct.png", level_pct=10)
    save_threshold_violation_map(violation_maps_dir / "threshold_violation_map_25pct.png", level_pct=25)
    save_worst_points_map(violation_maps_dir / "worst_10pct_points_map.png",
                          rank_level_pct=5, top_pct=10)

    if cache_only or not make_gifs:
        print("Skipping gif rendering (cache_only / make_gifs=False).")
        return

    def save_gif(U_frames, steps, label, out_path, signed: bool, cmap: str):
        # signed=False (fd/nn): plots |u| on a sequential "Reds" scale, 0..U_MAX --
        # matches the older per-run test scripts' displacement gifs.
        # signed=True (err): plots the raw FD-NN difference on a diverging
        # "coolwarm" scale, -U_MAX..U_MAX, so over/under-prediction show as
        # opposite colours -- unchanged from before.
        fig = plt.figure(figsize=(7.5, 4.2))
        ax = fig.add_axes([0.02, 0.04, 0.82, 0.86])
        ax.set_xlim(X_LO, X_HI); ax.set_ylim(Y_LO, Y_HI); ax.set_aspect("equal"); ax.set_axis_off()
        vmin, vmax = (-U_MAX, U_MAX) if signed else (0.0, U_MAX)
        scat = ax.scatter(XS, YS, c=np.zeros_like(XS), cmap=cmap, vmin=vmin, vmax=vmax, s=7)
        cax = fig.add_axes([0.875, 0.30, 0.018, 0.42])
        cbar = fig.colorbar(scat, cax=cax)
        cbar.set_label("displacement u" if signed else "|u|", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        title = fig.suptitle("", fontsize=10, y=0.955)

        def update(i):
            u = U_frames[i].ravel()
            scat.set_array(u if signed else np.abs(u))
            title.set_text(f"{label}  --  t = {steps[i] * cfg.dt:.2f} s  (top view)")
            return [scat, title]

        anim = animation.FuncAnimation(fig, update, frames=len(steps), interval=40)
        anim.save(out_path, writer="pillow", fps=25)
        plt.close(fig)
        print(f"Saved {out_path.name}")

    save_gif(fd_U, fd_steps, f"{excitation_label}, FD reference", figures_dir / f"{source_run_id}_fd_2d.gif",
              signed=False, cmap="Reds")
    save_gif(nn_U, nn_steps, f"{excitation_label}, NN surrogate ({source_run_id})",
              figures_dir / f"{source_run_id}_nn_2d.gif", signed=False, cmap="Reds")
    save_gif(err_U, err_steps, f"{excitation_label}, error FD - NN (red = NN under-predicts)",
              figures_dir / f"{source_run_id}_err_2d.gif", signed=True, cmap="coolwarm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run_id")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="default: runs/p7/models/<source_run_id>/ -- same folder as the "
                              "trained model itself (config.yaml/model.pth), so training and "
                              "assembly-test artifacts live together.")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--allow-unfinished", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or (REPO_ROOT / "runs" / "p7" / "models" / args.source_run_id)
    cache_only = args.cache_only or bool(os.environ.get("CACHE_ONLY"))
    metrics = run_lattice_test(args.source_run_id, output_dir, cache_only=cache_only,
                                 allow_unfinished=args.allow_unfinished)
    print(metrics)


if __name__ == "__main__":
    main()
