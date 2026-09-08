"""
Minimal junction-isolation test: 3 rods in a row (N0-N1-N2-N3), bent at an
angle at each joint purely so the 3D render can tell them apart -- the
physics only ever sees arc-length position along each rod, so the angle
doesn't change anything about the 1D wave equation itself.

Why: the bridge-truss surrogate test (surrogate_rods_prototype.py) showed
the model's amplitude gets crushed by roughly an order of magnitude at every
real junction it crosses, while an isolated single rod (single_rod_isolation_test.py,
same pulse) tracks the FD reference almost perfectly. But every junction in
that bridge has degree >= 3 (a real branch, ghost band = AVERAGE of several
other rods). This test isolates the simplest possible junction instead --
degree 2, straight pass-through, only ONE other rod to borrow ghost data
from -- to see whether the amplitude crushing is a generic "the model has
never seen a non-flat ghost band" problem (should show up here too, even
though there's no averaging at all), or specific to averaging across a real
branch (should look fine here, and only break down at degree >= 3).

BCs match exactly what the model was trained on ("simple" dataset profile,
see data/generate.py): N0 = dirichlet rest (fixed), N3 = dirichlet gaussian
(pulse injected here) -- literally the single-rod isolation test's own BC
convention, just chained across 3 rods with 2 pure degree-2 junctions
(N1, N2) in between instead of 0.

Runs BOTH an FD reference (exact-continuity 3-point stencil at each
degree-2 junction, same as graph_rods_3d_prototype.py -- physically
equivalent to one continuous rod of length 3L) and the NN surrogate
(ghost bands borrowed from the neighboring rod, same as
surrogate_rods_prototype.py) side by side, over the same extended 14s
horizon, at the same pulse (A=AMP_MAX, sigma=0.3) used in both those
scripts, for direct comparability.

Run it with:
    python3 three_rod_series_test.py
Writes three_rod_series.gif, three_rod_series_amplitude_vs_time.png,
and three_rod_series_amplitude_trace.npz next to this script.
"""
from __future__ import annotations

from pathlib import Path
import sys
import dataclasses
import gc

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path("/home/aph25/Code_GH")
sys.path.insert(0, str(REPO_ROOT / "src"))

from beamsurrogate.config import Config  # noqa: E402
from beamsurrogate.registry import MODELS, DATASETS  # noqa: E402
from beamsurrogate.data.split import load_hdf5_dataset, compute_norm_stats  # noqa: E402
from beamsurrogate.data.norm import norm_stats_arrays  # noqa: E402
from beamsurrogate.data.windows import build_window, make_feature_columns, make_output_columns  # noqa: E402
from beamsurrogate.physics.waves import apply_boundary, bc_value  # noqa: E402
from beamsurrogate.physics.solver import compute_rest_bias  # noqa: E402

SOURCE_RUN_ID = "p1_feat_u"


def _load_resolved_config(run_id: str) -> Config:
    resolved = yaml.safe_load((REPO_ROOT / "runs" / run_id / "config.resolved.yaml").read_text())
    known = {f.name for f in dataclasses.fields(Config)}
    raw = {k: v for k, v in resolved.items() if k in known}
    for tuple_field in ("HIDDEN_SIZES", "CNN_CHANNELS"):
        if tuple_field in raw and isinstance(raw[tuple_field], list):
            raw[tuple_field] = tuple(raw[tuple_field])
    return Config(**raw)


cfg = _load_resolved_config(SOURCE_RUN_ID)
INPUT_FIELDS = list(cfg.features)
INPUTS = make_feature_columns(INPUT_FIELDS, cfg)
OUTPUTS = make_output_columns(cfg)

model = MODELS[cfg.model](len(INPUTS), len(OUTPUTS), cfg)
model.load_state_dict(torch.load(REPO_ROOT / "runs" / SOURCE_RUN_ID / "model.pth", weights_only=True))
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

# --- Topology: N0(rest, fixed) -- N1 -- N2 -- N3(driven, gaussian). Angled
# purely for the 3D render; physics only cares about arc-length along a rod.
PULSE_A = cfg.AMP_MAX
PULSE_SIGMA = 0.3
angles_deg = [0.0, 35.0, -25.0]   # rod0, rod1, rod2 directions in the xy-plane
positions = [np.array([0.0, 0.0])]
for ang in angles_deg:
    positions.append(positions[-1] + cfg.L * np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))]))
NODE_NAMES = ["N0", "N1", "N2", "N3"]
pos = dict(zip(NODE_NAMES, positions))
edges = [("N0", "N1"), ("N1", "N2"), ("N2", "N3")]

nodes = {
    "N0": {"pos": pos["N0"], "driven": True, "bc": ("dirichlet", "rest", {})},
    "N1": {"pos": pos["N1"], "driven": False},
    "N2": {"pos": pos["N2"], "driven": False},
    "N3": {"pos": pos["N3"], "driven": True, "bc": ("dirichlet", "gaussian", {"A": PULSE_A, "sigma": PULSE_SIGMA})},
}

ROLLOUT_T_END = 14.0
ROLLOUT_NT = int(ROLLOUT_T_END / cfg.dt)

# ============================== FD reference ==============================
# Same exact-continuity degree-2 stencil as graph_rods_3d_prototype.py --
# physically equivalent to one continuous rod of length 3*cfg.L. Plain
# Nx-length arrays (no ghost band needed, it's a direct interior stencil).


def gaussian_value(t, A, sigma):
    t0 = 4.0 * sigma
    return A * np.exp(-((t - t0) / sigma) ** 2)


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

fd_amp_trace = {(r.start, r.end): np.empty(ROLLOUT_NT + 1) for r in fd_rods}
for r in fd_rods:
    fd_amp_trace[(r.start, r.end)][0] = 0.0

print("Running FD reference...")
for n in range(ROLLOUT_NT):
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
        (rod1, side1), (rod2, side2) = touching   # always degree 2 here
        b1, i1 = (0, 1) if side1 == "start" else (-1, -2)
        b2, i2 = (0, 1) if side2 == "start" else (-1, -2)
        u_curr = rod1.u[b1]
        u_prev_node = rod1.u_prev[b1]
        node_value[node_id] = (2 * u_curr - u_prev_node
                                + cfg.CFL**2 * (rod1.u[i1] - 2 * u_curr + rod2.u[i2]))

    for rod, u_new in zip(fd_rods, u_new_list):
        u_new[0] = node_value[rod.start]
        u_new[-1] = node_value[rod.end]
        rod.u_prev = rod.u
        rod.u = u_new
        fd_amp_trace[(rod.start, rod.end)][n + 1] = np.abs(u_new).max()

# ============================== NN surrogate ================================
# Same ghost-band-borrowed-from-the-other-rod approach as
# surrogate_rods_prototype.py, just for this 3-rod chain.


class NNRod:
    def __init__(self, start, end):
        self.start, self.end = start, end
        self.U = np.zeros((ROLLOUT_NT + 1, cfg.Ntot), dtype=np.float32)


nn_rods = [NNRod(s, e) for s, e in edges]

history_needed = cfg.M_BACK * cfg.ndt
hop = cfg.N_FWD * cfg.ndt
nn_frames = []

print("Running NN surrogate...")
for n in range(history_needed, ROLLOUT_NT - hop + 1, hop):
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
                touching = [rod for rod in nn_rods if node_id in (rod.start, rod.end)]
                node_value[node_id] = float(np.mean([rod_reading(rod, node_id, 0) for rod in touching]))

        for rod in nn_rods:
            for side, node_id in (("left", rod.start), ("right", rod.end)):
                node = nodes[node_id]
                if node["driven"]:
                    apply_boundary(rod.U[s], side, "dirichlet", node_value[node_id], cfg)
                    continue
                others = [r for r in nn_rods if r is not rod and node_id in (r.start, r.end)]
                if side == "left":
                    rod.U[s, cfg.i_left] = node_value[node_id]
                    for dist in range(1, cfg.SS + 1):
                        rod.U[s, cfg.i_left - dist] = float(
                            np.mean([rod_reading(r, node_id, dist) for r in others]))
                else:
                    rod.U[s, cfg.i_right - 1] = node_value[node_id]
                    for dist in range(1, cfg.SS + 1):
                        rod.U[s, cfg.i_right - 1 + dist] = float(
                            np.mean([rod_reading(r, node_id, dist) for r in others]))

        if cfg.SMOOTH_ALPHA > 0:
            j0, j1 = cfg.i_left + 1, cfg.i_right
            for rod in nn_rods:
                lap = rod.U[s, j0-1:j1-1] - 2*rod.U[s, j0:j1] + rod.U[s, j0+1:j1+1]
                rod.U[s, j0:j1] += cfg.SMOOTH_ALPHA * lap

        nn_frames.append((s, {(rod.start, rod.end): rod.U[s, cfg.nodes].copy() for rod in nn_rods}))

print(f"NN rollout done: {len(nn_frames)} frames covering {ROLLOUT_T_END:.1f} s")

nn_t = np.array([s * cfg.dt for s, _ in nn_frames])
nn_amp_trace = {key: np.array([np.abs(snap[key]).max() for _, snap in nn_frames]) for key in fd_amp_trace}
fd_t = np.arange(ROLLOUT_NT + 1) * cfg.dt

print("\nPeak |u| per rod -- FD vs NN (distance = number of degree-2 junctions from the driven end N3):")
dist_from_source = {("N2", "N3"): 0, ("N1", "N2"): 1, ("N0", "N1"): 2}
for key in sorted(fd_amp_trace, key=lambda k: dist_from_source[k]):
    fd_peak = fd_amp_trace[key].max()
    nn_peak = nn_amp_trace[key].max()
    print(f"  dist {dist_from_source[key]}  {key[0]}-{key[1]:<3s} FD={fd_peak:.5f}  NN={nn_peak:.5f}  "
          f"ratio NN/FD={nn_peak / fd_peak:.3f}")

np.savez(SCRIPT_DIR / "three_rod_series_amplitude_trace.npz",
         fd_t=fd_t, nn_t=nn_t,
         rod_names=np.array([f"{k[0]}-{k[1]}" for k in fd_amp_trace], dtype=object),
         fd_traces=np.stack([fd_amp_trace[k] for k in fd_amp_trace]),
         nn_traces=np.stack([nn_amp_trace[k] for k in fd_amp_trace]))

# --- Dense amplitude-vs-time plot, FD vs NN, one panel per rod.
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for ax, key in zip(axes, sorted(fd_amp_trace, key=lambda k: dist_from_source[k])):
    ax.plot(fd_t, fd_amp_trace[key], label="FD", color="tab:blue")
    ax.plot(nn_t, nn_amp_trace[key], label="NN", color="tab:orange")
    ax.set_ylabel("max|u|")
    ax.set_title(f"{key[0]}-{key[1]} (dist {dist_from_source[key]} junctions from source)")
    ax.legend(fontsize=8)
axes[-1].set_xlabel("t (s)")
fig.suptitle("3 rods in series, 2 degree-2 junctions: FD vs NN amplitude vs time")
fig.tight_layout()
fig.savefig(SCRIPT_DIR / "three_rod_series_amplitude_vs_time.png", dpi=120)
print(f"Saved {SCRIPT_DIR / 'three_rod_series_amplitude_vs_time.png'}")

# --- 3D animation (NN surrogate only -- the amplitude plot above already
# carries the FD-vs-NN comparison), same rendering convention as the other
# scripts here.
A_MAX = cfg.AMP_MAX
Z_SCALE = 20

fig3d = plt.figure(figsize=(9, 6))
ax3d = fig3d.add_axes([0.02, 0.05, 0.76, 0.90], projection="3d")
cax = fig3d.add_axes([0.84, 0.20, 0.03, 0.6])
all_x = [p[0] for p in positions]
all_y = [p[1] for p in positions]
pad = 0.3
ax3d.set_xlim(min(all_x) - pad, max(all_x) + pad)
ax3d.set_ylim(min(all_y) - pad, max(all_y) + pad)
ax3d.set_zlim(-A_MAX * Z_SCALE, A_MAX * Z_SCALE)
ax3d.view_init(elev=35, azim=-60)
ax3d.set_box_aspect((max(all_x) - min(all_x) + 2 * pad, max(all_y) - min(all_y) + 2 * pad, 1.0), zoom=1.5)
ax3d.set_axis_off()

rod_lines = []
for rod in nn_rods:
    x0, y0 = pos[rod.start]
    x1, y1 = pos[rod.end]
    xs = np.linspace(x0, x1, cfg.Nx)
    ys = np.linspace(y0, y1, cfg.Nx)
    sc = ax3d.scatter(xs, ys, np.zeros(cfg.Nx), c=np.zeros(cfg.Nx), cmap="coolwarm",
                       vmin=-A_MAX, vmax=A_MAX, s=10)
    rod_lines.append((rod, xs, ys, sc))
fig3d.colorbar(rod_lines[0][3], cax=cax, label="displacement u")
title3d = ax3d.set_title("t = 0.00 s (NN surrogate, height x20 exaggerated)")


def update3d(i):
    s, snap = nn_frames[i]
    for rod, xs, ys, sc in rod_lines:
        u = snap[(rod.start, rod.end)]
        sc._offsets3d = (xs, ys, u * Z_SCALE)
        sc.set_array(u)
    title3d.set_text(f"t = {s * cfg.dt:.2f} s (NN surrogate, height x20 exaggerated)")
    return [sc for *_, sc in rod_lines] + [title3d]


anim = animation.FuncAnimation(fig3d, update3d, frames=len(nn_frames), interval=40)
anim.save(SCRIPT_DIR / "three_rod_series.gif", writer="pillow", fps=25)
print(f"Saved {SCRIPT_DIR / 'three_rod_series.gif'}")
