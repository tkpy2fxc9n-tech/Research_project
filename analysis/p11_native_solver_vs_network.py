#!/usr/bin/env python3
# 11a -- separates the two sources of error a coarse grid introduces:
# the network's own error, vs the error a plain FD solver already has at
# that SAME coarse resolution (no network at all). Both measured against
# the SAME truth: the true fine solution, restricted to the coarse grid's
# own points/times -- exactly what data/beam_dataset_simple_coarse_r{2,4}.h5
# stores (an EXACT subsample of the fine solution, not a re-simulation --
# see make_coarse_dataset.py), so there's no separate "denser fine truth"
# to fetch: it's already what runs/p11_coarse_r{2,4}/metrics.json's own
# err_max curve was computed against.
#
# So only ONE new thing is computed here: physics/solver.py's
# run_fd_simulation_general run directly at the coarse resolution
# (Nx=50/Nt=250 or Nx=25/Nt=125 -- an independent FD solve, NOT the
# training data), for the exact same boundary condition as the network's
# own rollout trajectory. Trajectory correspondence across the fine/coarse
# files is exact: split labels and file order are passed through unchanged
# by make_coarse_dataset.py, so idx_test[0] is the same physical scenario
# in every file (verified: both r2 and r4 give idx_test[0]=3).
#
# The native FD solve fills every raw timestep directly (no autoregressive
# hopping), so it has none of the tail-zero rollout artifact h3/h4/p11's
# own network curves can have -- and r2/r4's network curves don't hit it
# either (verified empirically: amp_max never hits exactly 0 at their tail,
# unlike r1/p3_pinn_0's last point -- see p11_coarse_grid_error.py).
#
# Usage: python analysis/p11_native_solver_vs_network.py
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from beamsurrogate.config import Config  # noqa: E402
from beamsurrogate.physics.solver import run_fd_simulation_general  # noqa: E402
from beamsurrogate.data.split import _reconstruct_full_state, BC_LABEL_TO_TYPE  # noqa: E402
from beamsurrogate.evaluate.rollout import RolloutResult  # noqa: E402
from beamsurrogate.evaluate.metrics import compute_error_curves  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"
RUN_ID = "p11_diag_native_solver"

SOURCES = [
    ("r = 2", "p11_coarse_r2", DATA_DIR / "beam_dataset_simple_coarse_r2.h5", "error_r2_vs_native_solver.png"),
    ("r = 4", "p11_coarse_r4", DATA_DIR / "beam_dataset_simple_coarse_r4.h5", "error_r4_vs_native_solver.png"),
]
COLOR_NETWORK = "#2a78d6"
COLOR_SOLVER = "#eb6834"


def _load_resolved_config(run_id: str) -> Config:
    resolved = yaml.safe_load((RUNS_DIR / run_id / "config.resolved.yaml").read_text())
    known = {f.name for f in dataclasses.fields(Config)}
    raw = {k: v for k, v in resolved.items() if k in known}
    for tuple_field in ("HIDDEN_SIZES", "CNN_CHANNELS"):
        if tuple_field in raw and isinstance(raw[tuple_field], list):
            raw[tuple_field] = tuple(raw[tuple_field])
    return Config(**raw)


def _decode_str_array(arr) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else v for v in arr]


def _find_idx_test0(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as f:
        split = _decode_str_array(f["split"][:])
    return next(i for i, s in enumerate(split) if s == "test")


def _load_one_trajectory(h5_path: Path, idx: int, cfg: Config):
    # Same per-trajectory reconstruction load_hdf5_dataset does internally
    # (data/split.py), for ONE index only -- no full-dataset windowing, so
    # this stays cheap enough to run interactively (no sbatch needed).
    with h5py.File(h5_path, "r") as f:
        u_traj = f["u"][idx]
        left_bc_value = f["left_bc_value"][idx]
        right_bc_value = f["right_bc_value"][idx]
        left_label = _decode_str_array([f["left_label"][idx]])[0]
        right_label = _decode_str_array([f["right_label"][idx]])[0]

    left_type = BC_LABEL_TO_TYPE[left_label]
    right_type = BC_LABEL_TO_TYPE[right_label]
    U_full = _reconstruct_full_state(u_traj, left_bc_value, right_bc_value, left_type, right_type, cfg)

    t_ctrl = (np.arange(cfg.Nt + 1) * cfg.dt).tolist()
    bc_left = (left_type, "table", {"t_ctrl": t_ctrl, "values": left_bc_value.tolist(), "source_label": left_label})
    bc_right = (right_type, "table",
                {"t_ctrl": t_ctrl, "values": right_bc_value.tolist(), "source_label": right_label})
    return U_full, bc_left, bc_right


def main():
    run_dir = RUNS_DIR / RUN_ID
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = [f"{RUN_ID} -- network error vs native coarse-solver error, both vs the true "
                      f"fine solution restricted to the coarse grid\n"]

    for label, run_id, h5_path, filename in SOURCES:
        network_metrics_path = RUNS_DIR / run_id / "metrics.json"
        if not network_metrics_path.exists():
            print(f"ERROR: {network_metrics_path} not found -- run {run_id} first.", file=sys.stderr)
            sys.exit(1)
        network_curves = json.load(open(network_metrics_path))["curves"]

        cfg = _load_resolved_config(run_id)
        idx = _find_idx_test0(h5_path)
        U_full, bc_left, bc_right = _load_one_trajectory(h5_path, idx, cfg)

        print(f"{label}: native FD solve at Nx={cfg.Nx}/Nt={cfg.Nt} (idx={idx}, CFL={cfg.CFL:.4f})...")
        U_native = run_fd_simulation_general(bc_left, bc_right, cfg)
        native_rollout = RolloutResult(U=U_native, U_reel=U_full, left_bc=bc_left, right_bc=bc_right)
        native_curves = compute_error_curves(native_rollout, cfg)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(network_curves["t"], network_curves["err_max"], "o-", color=COLOR_NETWORK, lw=2, ms=3,
                 label=f"reseau ({run_id}, entraine sur cette grille)")
        ax.plot(native_curves["t"], native_curves["err_max"], "s--", color=COLOR_SOLVER, lw=2, ms=3,
                 label="solveur FD natif a cette grille (pas de reseau)")
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_ylabel("max absolute error along the beam -- Linf (log)")
        ax.grid(True, which="both", alpha=0.4)
        ax.legend()
        ax.set_title(f"11a -- {label} : reseau vs solveur natif, tous deux vs verite fine")
        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

        summary_lines.append(
            f"{label} (network: {run_id}, idx={idx}): "
            f"final err_max network={network_curves['err_max'][-1]:.4e}  "
            f"final err_max native_solver={native_curves['err_max'][-1]:.4e}  "
            f"peak err_max network={max(network_curves['err_max']):.4e}  "
            f"peak err_max native_solver={max(native_curves['err_max']):.4e}")

    summary = "\n".join(summary_lines) + "\n"
    (run_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"Done -- outputs in {run_dir}")


if __name__ == "__main__":
    main()
