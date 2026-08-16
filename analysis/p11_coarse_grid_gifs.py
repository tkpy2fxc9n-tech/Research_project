#!/usr/bin/env python3
# 11a -- 4 rollout animations, visual companion to
# p11_coarse_grid_error.py / p11_native_solver_vs_network.py:
#
#   wave_r{2,4}_fine_vs_native_solver.gif : the TRUE fine-resolution wave
#     (100 points) vs physics/solver.py's run_fd_simulation_general run
#     DIRECTLY at the coarse resolution (Nx=50/Nt=250 or Nx=25/Nt=125 --
#     an independent FD solve, same boundary condition, NOT the training
#     data) -- isolates what plain grid coarsening costs a solver, no
#     network involved. NOT "fine truth subsampled to the coarse grid's own
#     points" (that's nearly the same signal as itself by construction --
#     an earlier version of this script did that by mistake).
#   wave_r{2,4}_native_solver_vs_network.gif : that SAME native coarse
#     solver run vs the coarse-trained network's own rollout prediction,
#     both already at that resolution -- the animated version of what
#     error_r{2,4}_vs_native_solver.png's two curves compare.
#
# Trajectory correspondence across fine/coarse files is exact (split labels
# and file order are passed through unchanged by make_coarse_dataset.py):
# idx_test[0] is the same physical scenario in every file.
#
# Usage: python analysis/p11_coarse_grid_gifs.py
from __future__ import annotations

import dataclasses
import gc
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from beamsurrogate.config import Config  # noqa: E402
from beamsurrogate.registry import MODELS  # noqa: E402
from beamsurrogate.data.split import load_hdf5_dataset, compute_norm_stats, _reconstruct_full_state, \
    BC_LABEL_TO_TYPE  # noqa: E402
from beamsurrogate.evaluate.rollout import run_rollout  # noqa: E402
from beamsurrogate.physics.solver import run_fd_simulation_general  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"
FIGURES_DIR = RUNS_DIR / "p11_diag_native_solver" / "figures"

FINE_H5 = DATA_DIR / "beam_dataset_simple.h5"
FINE_SOURCE_RUN_ID = "p3_pinn_0"   # gives the fine grid's own resolved Config

SOURCES = [("r = 2", "p11_coarse_r2", DATA_DIR / "beam_dataset_simple_coarse_r2.h5"),
           ("r = 4", "p11_coarse_r4", DATA_DIR / "beam_dataset_simple_coarse_r4.h5")]

COLOR_FINE = "#2a78d6"
COLOR_SOLVER = "#eb6834"
COLOR_NETWORK = "#1baf7a"


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


def _load_one_trajectory(h5_path: Path, idx: int, cfg: Config) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        u_traj = f["u"][idx]
        left_bc_value = f["left_bc_value"][idx]
        right_bc_value = f["right_bc_value"][idx]
        left_label = _decode_str_array([f["left_label"][idx]])[0]
        right_label = _decode_str_array([f["right_label"][idx]])[0]
    left_type = BC_LABEL_TO_TYPE[left_label]
    right_type = BC_LABEL_TO_TYPE[right_label]
    return _reconstruct_full_state(u_traj, left_bc_value, right_bc_value, left_type, right_type, cfg)


def _two_series_animation(x_a, U_a, nodes_a, dt_a, label_a, color_a,
                           x_b, U_b, nodes_b, dt_b, label_b, color_b,
                           frames, title, path: Path):
    # b's own step grid drives the animation (it's the coarser one -- a's
    # matching frame is picked via nearest dt). `frames` must be caller-
    # supplied, not "every raw step": autoregressive_rollout only ever
    # fills steps that are multiples of cfg.ndt (plus the initial
    # M_BACK*ndt-step copied history) -- every OTHER raw step stays at its
    # np.zeros() init forever. Animating every raw index against a network
    # rollout shows a network prediction frozen at 0 on 2 out of 3 frames.
    ymax = max(np.abs(U_a[:, nodes_a]).max(), np.abs(U_b[:, nodes_b]).max()) * 1.2
    fig, ax = plt.subplots(figsize=(9, 5))
    line_a, = ax.plot([], [], "-", color=color_a, lw=2, label=label_a)
    line_b, = ax.plot([], [], "o--", color=color_b, lw=1.5, ms=4, label=label_b)
    ax.set_xlim(0, 1); ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("x / L"); ax.set_ylabel("u"); ax.legend(loc="upper right"); ax.grid(True)
    title_obj = ax.set_title("")

    def update(m_b):
        t = m_b * dt_b
        m_a = min(int(round(t / dt_a)), U_a.shape[0] - 1)
        line_a.set_data(x_a, U_a[m_a, nodes_a])
        line_b.set_data(x_b, U_b[m_b, nodes_b])
        title_obj.set_text(f"{title} -- t = {t:.3f}")
        return line_a, line_b, title_obj

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    anim.save(path, writer="pillow", fps=20, dpi=110)
    plt.close(fig)


def main():
    cfg_fine = _load_resolved_config(FINE_SOURCE_RUN_ID)
    idx_fine = _find_idx_test0(FINE_H5)
    U_fine = _load_one_trajectory(FINE_H5, idx_fine, cfg_fine)
    x_fine = np.linspace(0.0, 1.0, len(cfg_fine.nodes))
    print(f"Fine truth loaded: idx={idx_fine}, Nx={cfg_fine.Nx}, Nt={cfg_fine.Nt}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for label, run_id, coarse_h5 in SOURCES:
        cfg = _load_resolved_config(run_id)
        r_tag = run_id.split("_")[-1]   # "r2" or "r4"
        x_coarse = np.linspace(0.0, 1.0, len(cfg.nodes))

        print(f"{label}: loading {coarse_h5} (full dataset, needed for norm_stats)...")
        (df, FIELDS, INPUTS, OUTPUTS, bc_pairs, idx_train, idx_val, idx_test,
         rollout_idx, fam) = load_hdf5_dataset(cfg.features, cfg, coarse_h5, max_trajectories=None)
        norm_stats = compute_norm_stats(df, INPUTS, OUTPUTS, cfg)
        del df
        gc.collect()

        bc_left, bc_right = bc_pairs[rollout_idx]
        print(f"{label}: native FD solve at Nx={cfg.Nx}/Nt={cfg.Nt} (idx={rollout_idx}, CFL={cfg.CFL:.4f})...")
        U_native_solver = run_fd_simulation_general(bc_left, bc_right, cfg)

        print(f"{label}: fine vs native-solver animation...")
        solver_frames = np.arange(0, cfg.Nt + 1)   # direct FD solve fills every raw step -- no gaps
        _two_series_animation(
            x_fine, U_fine, cfg_fine.nodes, cfg_fine.dt, "verite fine (100 pts)", COLOR_FINE,
            x_coarse, U_native_solver, cfg.nodes, cfg.dt, f"solveur FD natif (grille {r_tag})",
            COLOR_SOLVER, solver_frames, f"11a -- {label} : onde fine vs solveur FD natif grossier",
            FIGURES_DIR / f"wave_{r_tag}_fine_vs_native_solver.gif")

        print(f"{label}: loading model and running rollout...")
        model = MODELS[cfg.model](len(INPUTS), len(OUTPUTS), cfg)
        model.load_state_dict(torch.load(RUNS_DIR / run_id / "model.pth", weights_only=True))
        model.eval()
        rollout = run_rollout(model, FIELDS, bc_pairs, rollout_idx, cfg.features, norm_stats, INPUTS, OUTPUTS, cfg)

        print(f"{label}: native-solver vs network animation...")
        # Only steps autoregressive_rollout actually filled: the initial
        # copied history [0, history_needed], then every ndt-multiple hop
        # target beyond it (see _two_series_animation's own comment).
        history_needed = cfg.M_BACK * cfg.ndt
        network_frames = np.concatenate([
            np.arange(0, history_needed + 1),
            np.arange(history_needed + cfg.ndt, cfg.Nt + 1, cfg.ndt),
        ])
        _two_series_animation(
            x_coarse, U_native_solver, cfg.nodes, cfg.dt, f"solveur FD natif (grille {r_tag})", COLOR_SOLVER,
            x_coarse, rollout.U, cfg.nodes, cfg.dt, "prediction reseau", COLOR_NETWORK,
            network_frames, f"11a -- {label} : solveur FD natif vs prediction reseau",
            FIGURES_DIR / f"wave_{r_tag}_native_solver_vs_network.gif")

        print(f"{label}: fine vs network animation...")
        _two_series_animation(
            x_fine, U_fine, cfg_fine.nodes, cfg_fine.dt, "verite fine (100 pts)", COLOR_FINE,
            x_coarse, rollout.U, cfg.nodes, cfg.dt, "prediction reseau", COLOR_NETWORK,
            network_frames, f"11a -- {label} : onde fine vs prediction reseau",
            FIGURES_DIR / f"wave_{r_tag}_fine_vs_network.gif")

        del FIELDS, bc_pairs, rollout, model
        gc.collect()

    print(f"\nDone -- 4 gifs in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
