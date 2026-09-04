#!/usr/bin/env python3
# Computes report.txt for a finished p7_structure_several_rods.py run: every
# numeric \todo{} value the thesis's "Extension to a network of rods" LaTeX
# section needs (Table~\ref{tab:assembly_metrics} row, the threshold-map
# prose, worst-rod/junction stats). Written directly into that same model
# folder (runs/p7/models/<run_id>/report.txt) -- no separate export tree.
#
# The panel images themselves (figures/snapshots/panel_{fd,nn,err}_{t25,
# t50,t75,t100}.png, figures/threshold_violation_map.png) are already in
# their final LaTeX-ready names and location -- p7_structure_several_rods.py
# writes them directly, nothing to rename/copy here. This script just
# checks they exist and warns if not.
#
# Does NOT re-run any physics -- run p7_structure_several_rods.py <run_id>
# first (or p7_compare_assembly_models.py for all of them) if a run's cache
# or figures are missing.
#
# Usage: python analysis/p7_export_latex_report.py <run_id>
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))
import p7_structure_several_rods as m  # noqa: E402

MODELS_DIR = REPO_ROOT / "runs" / "p7" / "models"

TARGET_T = (2.5, 5.0, 7.5, 10.0)
TARGET_TAG = ("t25", "t50", "t75", "t100")
JUNCTION_MIN_DEGREE = 3
NEAR_JUNCTION_HOPS = 1
VIOLATION_PCT = 5


def node_degrees(edges, nodes):
    deg = {n: 0 for n in nodes}
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    return deg


def hops_to_nearest_junction(nodes, edges, junction_nodes):
    # BFS from every junction node at once (multi-source), so each node's
    # distance is to its NEAREST junction, not to any single one.
    adj = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    dist = {n: None for n in nodes}
    frontier = list(junction_nodes)
    for n in frontier:
        dist[n] = 0
    d = 0
    while frontier:
        nxt = []
        for n in frontier:
            for nb in adj[n]:
                if dist[nb] is None:
                    dist[nb] = d + 1
                    nxt.append(nb)
        frontier = nxt
        d += 1
    return dist


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run_id = sys.argv[1]

    run_dir = MODELS_DIR / run_id
    cache_path = run_dir / f"{run_id}_frames.npz"
    if not cache_path.exists():
        sys.exit(f"No cached frames at {cache_path} -- run p7_structure_several_rods.py {run_id} first.")

    cfg, _src_run_dir = m.load_source(run_id)
    lattice = m.build_lattice(cfg)
    nodes, edges, pos, FIXED_NODES = lattice["nodes"], lattice["edges"], lattice["pos"], lattice["fixed"]

    cached = np.load(cache_path)
    fd_steps, fd_U = cached["fd_steps"], cached["fd_U"]
    nn_steps, nn_U = cached["nn_steps"], cached["nn_U"]
    fd_time_s = float(cached["fd_time_s"]) if "fd_time_s" in cached else None
    nn_time_s = float(cached["nn_time_s"]) if "nn_time_s" in cached else None

    metrics = m.compute_metrics(run_id, fd_steps, fd_U, nn_steps, nn_U, fd_time_s, nn_time_s, dt=cfg.dt)
    if metrics["diverged_nan"] or metrics.get("diverged"):
        print(f"WARNING: {run_id} is flagged diverged (ratio={metrics.get('ratio_nn_fd')}). "
              f"Exporting anyway, but the numbers below are meaningless for a diverged run.")

    # --- same overlap-based FD/NN alignment as compute_metrics()/_make_figures() ---
    err_steps, fd_idx, nn_idx = np.intersect1d(fd_steps, nn_steps, assume_unique=True, return_indices=True)
    err_U = fd_U[fd_idx] - nn_U[nn_idx]
    err_t = err_steps * cfg.dt
    FD_PEAK = float(np.abs(fd_U).max())

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
        return np.stack(cols, axis=1), [n for n in nodes if n not in FIXED_NODES]

    node_err, interior_node_order = _lattice_node_abs_error()  # (n_frames, n_interior_nodes)

    # --- worst single ROD (not node) by peak error, and the mean-error curve's
    # rise/plateau shape (quartile samples of the lattice-mean error curve) ---
    peak_per_rod = np.abs(err_U).max(axis=(0, 2))
    rod_order = np.argsort(-peak_per_rod)
    top_rods = [(edges[r], float(peak_per_rod[r])) for r in rod_order[:5]]

    mean_curve_all = node_err.mean(axis=1)
    quartile_samples = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        i = int(frac * (len(err_t) - 1))
        quartile_samples.append((float(err_t[i]), float(mean_curve_all[i])))
    growth_ratio_last_half = (quartile_samples[3][1] / quartile_samples[1][1]
                               if quartile_samples[1][1] > 0 else float("nan"))
    curve_trend = ("still growing (no plateau) through the full rollout" if growth_ratio_last_half > 1.2
                   else "roughly plateaued over the second half of the rollout")

    # --- T_5% / T_10% first-crossing times (T_10% == compute_metrics' t_div) ---
    t5, t10 = m.first_crossing_times(err_U, err_t, FD_PEAK)

    # --- per-node "ever exceeds 5%" flag + junction-proximity breakdown ---
    ever_exceeded_node = node_err.max(axis=0) > (VIOLATION_PCT / 100.0) * FD_PEAK  # (n_interior_nodes,)
    deg = node_degrees(edges, nodes)
    junction_nodes = {n for n, d in deg.items() if d >= JUNCTION_MIN_DEGREE and n not in FIXED_NODES}
    dist_to_junction = hops_to_nearest_junction(nodes, edges, junction_nodes)

    near = np.array([dist_to_junction[n] is not None and dist_to_junction[n] <= NEAR_JUNCTION_HOPS
                      for n in interior_node_order])
    n_interior = len(interior_node_order)
    n_violating = int(ever_exceeded_node.sum())
    n_near = int(near.sum())
    n_violating_near = int((ever_exceeded_node & near).sum())

    pct_violating = 100.0 * n_violating / n_interior
    pct_violating_near_junction = 100.0 * n_violating_near / n_violating if n_violating else float("nan")
    pct_near_junction_uniform = 100.0 * n_near / n_interior  # expected share if violations were uniform

    # worst single junction node by peak error, and its degree
    junction_order = [n for n in interior_node_order if n in junction_nodes]
    junction_peak = node_err[:, [interior_node_order.index(n) for n in junction_order]].max(axis=0)
    if len(junction_order):
        worst_i = int(np.argmax(junction_peak))
        worst_junction_node = junction_order[worst_i]
        worst_junction_peak = float(junction_peak[worst_i])
        worst_junction_degree = deg[worst_junction_node]
    else:
        worst_junction_node, worst_junction_peak, worst_junction_degree = None, float("nan"), None

    # violation rate vs degree, since every rod in this lattice is
    # geometrically/physically identical (same L, same CFL/material) -- there
    # is no material impedance mismatch (Z2/Z1 == 1 everywhere) for this
    # uniform triangular lattice, so degree is the only real "junction
    # strength" variable here.
    degree_bins = {}
    for i, n in enumerate(interior_node_order):
        degree_bins.setdefault(deg[n], []).append(bool(ever_exceeded_node[i]))

    # --- confirm the panel snapshots + threshold map exist under their
    # LaTeX-standard names -- p7_structure_several_rods.py's own
    # save_snapshots_single()/save_threshold_violation_map() now write these
    # directly (figures/snapshots/panel_{fd,nn,err}_{tag}.png and
    # figures/threshold_violation_map.png), so there's nothing to copy here
    # any more; this just checks they're actually there.
    snap_dir = run_dir / "figures" / "snapshots"
    missing = []
    for kind in ("fd", "nn", "err"):
        for tag in TARGET_TAG:
            p = snap_dir / f"panel_{kind}_{tag}.png"
            if not p.exists():
                missing.append(p)
    thresh_path = run_dir / "figures" / "threshold_violation_map.png"
    if not thresh_path.exists():
        missing.append(thresh_path)
    if missing:
        print("WARNING: missing figures (run p7_structure_several_rods.py to generate them):")
        for p in missing:
            print(f"  {p}")

    # --- table + prose values ---
    lines = [
        f"LaTeX export for {run_id}",
        f"Figures at: {snap_dir} (panels) and {thresh_path}",
        "",
        "=== Table~\\ref{tab:assembly_metrics} -- Baseline row ===",
        f"  T_5%^max  (first crossing of 5% FD peak)  : {t5:.2f}s" if t5 is not None else "  T_5%^max  : never reached",
        f"  T_10%^max (first crossing of 10% FD peak) : {t10:.2f}s" if t10 is not None else "  T_10%^max : never reached",
        f"  E_max (abs max |FD-NN| over all points/times) : {metrics['abs_max']:.6f}",
        f"  Delta E_max (baseline)                        : 1.00",
        f"  t_{{E_max}} (time of max error)                : {metrics['t_of_max_error']:.2f}s",
        f"  P_thr (%% of rollout time with worst-node error > {VIOLATION_PCT}% of FD peak) : "
        f"{metrics['pct_time_above_threshold']:.1f}%",
        "",
        "=== Figure~\\ref{fig:threshold_violation_map} prose ===",
        f"  Violations occur at {pct_violating:.1f}% of the {n_interior} non-fixed lattice nodes "
        f"({n_violating} nodes) ever exceeding {VIOLATION_PCT}% of FD peak error.",
        f"  Of those violating nodes, {pct_violating_near_junction:.1f}% lie within "
        f"{NEAR_JUNCTION_HOPS} hop(s) of a junction (degree >= {JUNCTION_MIN_DEGREE}), "
        f"against {pct_near_junction_uniform:.1f}% expected if violations were spatially uniform.",
        f"  Worst single junction node: {worst_junction_node} (degree {worst_junction_degree}), "
        f"peak |FD-NN| = {worst_junction_peak:.6f}." if worst_junction_node else "  No junction nodes found.",
        "",
        "  CAVEAT: 'ever exceeds threshold' accumulates over the whole 10s rollout, so on a long "
        "rollout it tends to saturate towards 100% of nodes (every node eventually crosses the "
        "threshold once) -- it does NOT mean the error is spatially uniform at any given instant. "
        "Compare the panel_err_t*.png images directly for the actual spatial pattern at each instant.",
        "",
        f"  Top rods by peak |FD-NN| error (closest thing to 'junction between rods X and Y'):",
    ]
    for (a, b), peak in top_rods:
        lines.append(f"    {a}-{b}: peak |FD-NN| = {peak:.6f}")
    lines += [
        "",
        f"  Lattice-mean |FD-NN| error curve: "
        + ", ".join(f"t={t:.2f}s -> {v:.3e}" for t, v in quartile_samples)
        + f"  ==> {curve_trend}.",
        "",
        "  NOTE ON IMPEDANCE RATIO: this lattice is a UNIFORM triangular mesh -- every rod shares "
        "the same L, CFL and material, so Z2/Z1 = 1 at every junction (no material mismatch to "
        "report). The variable that actually correlates with junction error here is NODE DEGREE "
        "(how many rods meet at a node), not impedance. Violation rate by degree:",
    ]
    for d in sorted(degree_bins):
        vals = degree_bins[d]
        lines.append(f"    degree {d}: {100.0*sum(vals)/len(vals):.1f}% of {len(vals)} node(s) violate")

    lines += [
        "",
        "=== Peak displacement / global error (for the t=2.5s..10.0s prose paragraph) ===",
        f"  FD peak |u| : {metrics['fd_peak']:.6f}   NN peak |u| : {metrics['nn_peak']:.6f}   "
        f"ratio : {metrics['ratio_nn_fd']:.3f}",
        f"  relative mean error over whole rollout : {metrics['rel_mean_pct']:.2f}%",
        f"  relative max  error over whole rollout : {metrics['rel_max_pct']:.2f}%",
        "  NOTE: per-instant (t=2.5/5.0/7.5/10.0s) wavefront-agreement and error-map-uniformity "
        "values are NOT computed here -- they require reading the panel_*.png images directly "
        "(open figures/panel_err_t*.png and compare by eye, or tell me the specific instants and "
        "I can add per-frame peak/localisation stats to this script).",
    ]

    report_path = run_dir / "report.txt"
    report_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {report_path}")


if __name__ == "__main__":
    main()
