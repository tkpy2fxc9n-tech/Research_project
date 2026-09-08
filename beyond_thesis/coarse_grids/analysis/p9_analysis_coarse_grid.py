#!/usr/bin/env python3
# Phase 9 (coarse grid): three analyses of what grid coarsening costs,
# merged from p9_coarse_grid_error.py + p9_coarse_grid_gifs.py +
# p9_native_solver_vs_network.py (which had duplicated _load_config /
# _decode_str_array / _find_idx_test0 helpers -- now shared once below).
# Each analysis keeps its own original RUN_ID/output dir -- gifs and
# native_solver_vs_network already shared runs/p9_diag_native_solver/, only
# coarse_grid_error has its own runs/p9_diag_error_coarse/.
#
#   coarse_grid_error(): max absolute rollout error vs t, r1 (fine
#     grid, reference) vs r2 vs r4 -- pure post-processing over each
#     source's results.yaml, no heavy compute. r1 has no dedicated run of
#     its own: reuses p3_pinn_0 (the campaign's settled step-10 model).
#
#   coarse_grid_gifs(): HEAVY -- loads the trained model, runs FD solves,
#     6 rollout animations (2 resolutions x 3 comparisons: fine vs native
#     solver, native solver vs network, fine vs network). Visual companion
#     to the other two.
#
#   native_solver_vs_network(): separates the two sources of error a coarse
#     grid introduces -- the network's own error vs. the error a plain FD
#     solver already has at that SAME coarse resolution (no network at
#     all), both measured against the true fine solution restricted to the
#     coarse grid's own points (exactly what the *_coarse_r{2,4}.h5 files
#     store -- an exact subsample, not a re-simulation, so there's no
#     separate "denser fine truth" to fetch).
#
# Usage: python analysis/p9_analysis_coarse_grid.py
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

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "common"))

from beamsurrogate.config import Config  # noqa: E402
from beamsurrogate.registry import MODELS  # noqa: E402
from beamsurrogate.data.split import load_hdf5_dataset, compute_norm_stats, _reconstruct_full_state, \
    BC_LABEL_TO_TYPE  # noqa: E402
from beamsurrogate.evaluate.rollout import run_rollout, RolloutResult  # noqa: E402
from beamsurrogate.evaluate.metrics import compute_error_curves  # noqa: E402
from beamsurrogate.physics.solver import run_fd_simulation_general  # noqa: E402
from _common import run_dir as _run_dir  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "beyond_thesis" / "coarse_grids" / "runs"


# ---------------------------------------------------------------- shared --

def _load_config(run_id: str) -> Config:
    resolved = yaml.safe_load((_run_dir(REPO_ROOT, run_id) / "config.yaml").read_text())
    known = {f.name for f in dataclasses.fields(Config)} - {"run_id"}
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


# ------------------------------------------------------- coarse_grid_error --

def coarse_grid_error() -> bool:
    run_id = "p9_diag_error_coarse"
    sources = {
        "r1 = 1 (fine, reference)": "p3_pinn_0",
        "r = 2": "p9_coarse_r2",
        "r = 4": "p9_coarse_r4",
    }
    # Validated categorical palette (dataviz skill, palette.md slots 1-3:
    # the only 3-slot subset clearing all-pairs CVD separation) + line
    # style as the required secondary encoding.
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    styles = ["-", "--", ":"]
    # Tail-zero artifact -- FIXED AT THE SOURCE 2026-08-23 (evaluate/
    # metrics.py's _rollout_steps was sampling past the last step
    # autoregressive_rollout actually writes; see p3_analysis_pinn_
    # comparative.py's module docstring for the full diagnosis). r1's
    # source (p3_pinn_0) was one of the affected results.yaml, patched in
    # place; r2/r4 never hit it. No truncation needed here anymore.
    curves = {}
    for label, source_run_id in sources.items():
        source_metrics_path = _run_dir(REPO_ROOT, source_run_id) / "results.yaml"
        if not source_metrics_path.exists():
            print(f"WARNING: {source_metrics_path} not found -- run {source_run_id} first, skipping "
                  f"{run_id}.", file=sys.stderr)
            return False
        with open(source_metrics_path) as f:
            metrics = yaml.safe_load(f)
        if not metrics["curves"]["t"]:
            print(f"WARNING: {source_metrics_path} has no curves -- {source_run_id} may not have "
                  f"finished successfully, skipping {run_id}.", file=sys.stderr)
            return False
        curves[label] = metrics["curves"]

    run_dir = RUNS_DIR / run_id
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for (label, c), color, style in zip(curves.items(), colors, styles):
        ax.plot(c["t"], c["err_max"], style, color=color, lw=2, marker="o", ms=3, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("max absolute error along the beam (log)")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    ax.set_title("coarse grid: rollout error (max absolute) vs time")
    plt.tight_layout()
    plt.savefig(figures_dir / "error_coarse_grid.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary_lines = [f"{run_id} -- max rollout error vs grid coarsening\n"]
    for label, source_run_id in sources.items():
        e = curves[label]["err_max"]
        summary_lines.append(f"{label} (source: {source_run_id}): "
                              f"final err_max={e[-1]:.4e}  peak err_max={max(e):.4e}")
    summary = "\n".join(summary_lines) + "\n"
    (run_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"Done -- outputs in {run_dir}")
    return True


# -------------------------------------------------------- coarse_grid_gifs --

def _load_fine_trajectory_u_only(h5_path: Path, idx: int, cfg: Config) -> np.ndarray:
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


def coarse_grid_gifs() -> bool:
    figures_dir = RUNS_DIR / "p9_diag_native_solver" / "figures"
    fine_h5 = DATA_DIR / "beam_dataset_A.h5"
    fine_source_run_id = "p3_pinn_0"   # gives the fine grid's own resolved Config
    sources = [("r = 2", "p9_coarse_r2", DATA_DIR / "beam_dataset_simple_coarse_r2.h5"),
               ("r = 4", "p9_coarse_r4", DATA_DIR / "beam_dataset_simple_coarse_r4.h5")]
    color_fine, color_solver, color_network = "#2a78d6", "#eb6834", "#1baf7a"

    fine_cfg_path = _run_dir(REPO_ROOT, fine_source_run_id) / "config.yaml"
    if not fine_cfg_path.exists():
        print(f"WARNING: {fine_cfg_path} not found -- run {fine_source_run_id} first, skipping "
              f"coarse_grid_gifs.", file=sys.stderr)
        return False

    cfg_fine = _load_config(fine_source_run_id)
    idx_fine = _find_idx_test0(fine_h5)
    U_fine = _load_fine_trajectory_u_only(fine_h5, idx_fine, cfg_fine)
    x_fine = np.linspace(0.0, 1.0, len(cfg_fine.nodes))
    print(f"Fine truth loaded: idx={idx_fine}, Nx={cfg_fine.Nx}, Nt={cfg_fine.Nt}")

    figures_dir.mkdir(parents=True, exist_ok=True)

    for label, run_id, coarse_h5 in sources:
        model_path = _run_dir(REPO_ROOT, run_id) / "model.pth"
        if not model_path.exists():
            print(f"WARNING: {model_path} not found -- run {run_id} first, skipping it.", file=sys.stderr)
            continue

        cfg = _load_config(run_id)
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
            x_fine, U_fine, cfg_fine.nodes, cfg_fine.dt, "fine reference (100 pts)", color_fine,
            x_coarse, U_native_solver, cfg.nodes, cfg.dt, f"native FD solver (grid {r_tag})",
            color_solver, solver_frames, f"11a -- {label}: fine wave vs coarse native FD solver",
            figures_dir / f"wave_{r_tag}_fine_vs_native_solver.gif")

        print(f"{label}: loading model and running rollout...")
        model = MODELS[cfg.model](len(INPUTS), len(OUTPUTS), cfg)
        model.load_state_dict(torch.load(model_path, weights_only=True))
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
            x_coarse, U_native_solver, cfg.nodes, cfg.dt, f"native FD solver (grid {r_tag})", color_solver,
            x_coarse, rollout.U, cfg.nodes, cfg.dt, "network prediction", color_network,
            network_frames, f"11a -- {label}: native FD solver vs network prediction",
            figures_dir / f"wave_{r_tag}_native_solver_vs_network.gif")

        print(f"{label}: fine vs network animation...")
        _two_series_animation(
            x_fine, U_fine, cfg_fine.nodes, cfg_fine.dt, "fine reference (100 pts)", color_fine,
            x_coarse, rollout.U, cfg.nodes, cfg.dt, "network prediction", color_network,
            network_frames, f"11a -- {label}: fine wave vs network prediction",
            figures_dir / f"wave_{r_tag}_fine_vs_network.gif")

        del FIELDS, bc_pairs, rollout, model
        gc.collect()

    print(f"Done -- gifs in {figures_dir}")
    return True


# ------------------------------------------------ native_solver_vs_network --

def _load_coarse_trajectory_with_bc(h5_path: Path, idx: int, cfg: Config):
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


def native_solver_vs_network() -> bool:
    run_id = "p9_diag_native_solver"
    sources = [
        ("r = 2", "p9_coarse_r2", DATA_DIR / "beam_dataset_simple_coarse_r2.h5", "error_r2_vs_native_solver.png"),
        ("r = 4", "p9_coarse_r4", DATA_DIR / "beam_dataset_simple_coarse_r4.h5", "error_r4_vs_native_solver.png"),
    ]
    color_network, color_solver = "#2a78d6", "#eb6834"

    run_dir = RUNS_DIR / run_id
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = [f"{run_id} -- network error vs native coarse-solver error, both vs the true "
                      f"fine solution restricted to the coarse grid\n"]

    ran_any = False
    for label, source_run_id, h5_path, filename in sources:
        network_metrics_path = _run_dir(REPO_ROOT, source_run_id) / "results.yaml"
        if not network_metrics_path.exists():
            print(f"WARNING: {network_metrics_path} not found -- run {source_run_id} first, skipping it.",
                  file=sys.stderr)
            continue
        with open(network_metrics_path) as f:
            network_curves = yaml.safe_load(f)["curves"]

        cfg = _load_config(source_run_id)
        idx = _find_idx_test0(h5_path)
        U_full, bc_left, bc_right = _load_coarse_trajectory_with_bc(h5_path, idx, cfg)

        print(f"{label}: native FD solve at Nx={cfg.Nx}/Nt={cfg.Nt} (idx={idx}, CFL={cfg.CFL:.4f})...")
        U_native = run_fd_simulation_general(bc_left, bc_right, cfg)
        native_rollout = RolloutResult(U=U_native, U_reel=U_full, left_bc=bc_left, right_bc=bc_right)
        native_curves = compute_error_curves(native_rollout, cfg)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(network_curves["t"], network_curves["err_max"], "o-", color=color_network, lw=2, ms=3,
                 label=f"network ({source_run_id}, trained on this grid)")
        ax.plot(native_curves["t"], native_curves["err_max"], "s--", color=color_solver, lw=2, ms=3,
                 label="native FD solver at this grid (no network)")
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_ylabel("max absolute error along the beam (log)")
        ax.grid(True, which="both", alpha=0.4)
        ax.legend()
        ax.set_title(f"11a -- {label}: network vs native solver, both vs fine reference")
        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

        summary_lines.append(
            f"{label} (network: {source_run_id}, idx={idx}): "
            f"final err_max network={network_curves['err_max'][-1]:.4e}  "
            f"final err_max native_solver={native_curves['err_max'][-1]:.4e}  "
            f"peak err_max network={max(network_curves['err_max']):.4e}  "
            f"peak err_max native_solver={max(native_curves['err_max']):.4e}")
        ran_any = True

    if not ran_any:
        return False

    summary = "\n".join(summary_lines) + "\n"
    (run_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"Done -- outputs in {run_dir}")
    return True


def main():
    ran_any = False
    ran_any |= coarse_grid_error()
    ran_any |= coarse_grid_gifs()
    ran_any |= native_solver_vs_network()
    if not ran_any:
        sys.exit(1)


if __name__ == "__main__":
    main()
