#!/usr/bin/env python3
# For the exact red points shown in p7_structure_several_rods.py's
# worst_10pct_points_map.png (top 10% of PHYSICAL POINTS ALONG THE RODS --
# every one of cfg.Nx points per rod, not just rod endpoints/junctions --
# ranked by fraction of rollout time spent violating the 5%-of-FD-peak
# threshold), computes what fraction of those points sit within N grid
# points of ANY junction (a rod endpoint where >=3 rods meet).
#
# Distance is measured in point-steps along the physical beam, matching
# what the user actually asked ("distance of 15 points"), not graph hops
# between rod endpoints. Because this lattice is a uniform mesh (every rod
# has the same cfg.Nx, same length -- see p7_export_latex_report.py's own
# "UNIFORM triangular mesh" note), a point at index i on rod (a,b) is
# exactly `i` points from end a and `Nx-1-i` points from end b, and each
# additional rod crossing costs exactly `Nx-1` more points -- so the
# graph-hop count from hops_to_nearest_junction() converts to an exact
# point-count via hop_count * (Nx - 1), no re-derivation of the BFS needed.
#
# Usage: python analysis/p7_worst10pct_junction_proximity.py <run_id> [max_dist_points]
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))
import p7_structure_several_rods as m  # noqa: E402
from p7_export_latex_report import node_degrees, hops_to_nearest_junction  # noqa: E402

MODELS_DIR = REPO_ROOT / "runs" / "p7" / "models"
JUNCTION_MIN_DEGREE = 3
RANK_LEVEL_PCT = 5   # matches save_worst_points_map(rank_level_pct=5, top_pct=10)
TOP_PCT = 10


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python analysis/p7_worst10pct_junction_proximity.py <run_id> [max_dist_points]")
    run_id = sys.argv[1]
    max_dist = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    run_dir = MODELS_DIR / run_id
    cache_path = run_dir / f"{run_id}_frames.npz"
    if not cache_path.exists():
        sys.exit(f"No cached frames at {cache_path} -- run p7_structure_several_rods.py {run_id} first.")

    cfg, _ = m.load_source(run_id)
    lattice = m.build_lattice(cfg)
    nodes, edges = lattice["nodes"], lattice["edges"]
    Nx = cfg.Nx

    cached = np.load(cache_path)
    fd_steps, fd_U = cached["fd_steps"], cached["fd_U"]
    nn_steps, nn_U = cached["nn_steps"], cached["nn_U"]

    err_steps, fd_idx, nn_idx = np.intersect1d(fd_steps, nn_steps, assume_unique=True, return_indices=True)
    err_U = fd_U[fd_idx] - nn_U[nn_idx]   # (n_frames, n_rods, Nx) -- same array save_worst_points_map uses
    FD_PEAK = float(np.abs(fd_U).max())

    # --- exact same ranking as save_worst_points_map, on the same raw points ---
    level = (RANK_LEVEL_PCT / 100.0) * FD_PEAK
    frac_time_violating = 100.0 * (np.abs(err_U) > level).mean(axis=0).ravel()  # (n_rods*Nx,)
    cutoff = np.percentile(frac_time_violating, 100.0 - TOP_PCT)
    is_worst = frac_time_violating >= cutoff

    # --- distance (in points along the beam) from every point to its nearest junction ---
    deg = node_degrees(edges, nodes)
    junction_nodes = {n for n, d in deg.items() if d >= JUNCTION_MIN_DEGREE}
    hop_count = hops_to_nearest_junction(nodes, edges, junction_nodes)  # rod-crossings to nearest junction

    dist_points = np.empty(len(edges) * Nx)
    for r, (a, b) in enumerate(edges):
        i = np.arange(Nx)
        via_a = i + hop_count[a] * (Nx - 1)
        via_b = (Nx - 1 - i) + hop_count[b] * (Nx - 1)
        dist_points[r * Nx:(r + 1) * Nx] = np.minimum(via_a, via_b)

    near = dist_points <= max_dist
    n_worst = int(is_worst.sum())
    n_worst_near = int((is_worst & near).sum())
    pct_worst_near = 100.0 * n_worst_near / n_worst if n_worst else float("nan")
    pct_all_near = 100.0 * near.mean()  # what you'd expect if the worst points were random

    print(f"{run_id} -- worst {TOP_PCT}% of points along the rods by time spent violating the "
          f"{RANK_LEVEL_PCT}%-of-FD-peak threshold ({n_worst}/{len(is_worst)} points, {Nx} points/rod)")
    print(f"  {pct_worst_near:.1f}% of those worst points ({n_worst_near}/{n_worst}) are within "
          f"{max_dist} points of a junction (degree >= {JUNCTION_MIN_DEGREE}).")
    print(f"  For comparison, {pct_all_near:.1f}% of ALL points on the lattice are within "
          f"{max_dist} points of a junction (baseline if the worst points were scattered randomly).")


if __name__ == "__main__":
    main()
