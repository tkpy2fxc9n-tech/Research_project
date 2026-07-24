"""
Same rod network as graph_rods_prototype.py, but every rod's interior
physics comes from the trained neural network (Surrogate_model/model.pth)
instead of finite differences.

This model was only ever trained with one end at rest (0) and the other
driven by a single Gaussian pulse, so every driven node here stays within
that family ("rest" or "gaussian") to avoid extrapolating outside training.

Reuses the training-time code (Config, Reseau, build_window, apply_boundary,
normalization, bias-at-rest correction, smoothing) from
Beam_surrogate_model/training/code/commun.py, so the model sees inputs built
exactly like during training.

Run it with:
    python3 surrogate_rods_prototype.py
It writes propagation_reseau_surrogate.gif next to this script.
"""

from pathlib import Path
import sys
from itertools import product

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "Surrogate_model"
COMMUN_CODE_DIR = Path("/home/aph25/Beam_surrogate_model/training/code")
sys.path.insert(0, str(COMMUN_CODE_DIR))
import commun as C  # noqa: E402

INPUT_FIELDS = ["U", "Ut", "Uxx"]

# Every field pinned explicitly (not left to Config's defaults): commun.py
# and config.py live in Beam_surrogate_model, a project under active,
# separate development -- its defaults have already changed once since this
# script was written (different HIDDEN_SIZES, different amplitude/frequency
# range, different INPUT_FIELDS). These values are what model.pth was
# actually trained with, read off Surrogate_model/training/plots/
# simulation_22072026/resume.txt, and must not drift with that project.
cfg = C.Config(
    E=1, rho=2, L=1,
    Nt=500, Nx=100, SS=10, t_end=5,
    ndt=3, M_BACK=2, N_FWD=2,
    N_GRID=10, AMP_MIN=0.005, AMP_MAX=0.1, OMEGA_MIN=3, OMEGA_MAX=10,
    HIDDEN_SIZES=(64, 32, 16),
    SMOOTH_ALPHA=0.20,
    SEED=0, SPLIT_SEED=42,
)
INPUTS = C.make_feature_columns(INPUT_FIELDS, cfg)
OUTPUTS = C.make_output_columns(cfg)
assert len(INPUTS) == 126 and len(OUTPUTS) == 2, "Config doesn't match the trained model"

modele = C.Reseau(n_inputs=len(INPUTS), n_outputs=len(OUTPUTS), hidden_sizes=cfg.HIDDEN_SIZES)
modele.load_state_dict(torch.load(MODEL_DIR / "model.pth", map_location="cpu"))
modele.eval()

# Normalization stats were never saved to disk, and the code that built the
# original train/val/test split no longer exists -- rebuilt here from the
# full grid of training simulations instead (validated against the model's
# own documented benchmark in resume.txt: gives a bounded, NaN-free rollout
# and R2 close to what training reported; a guessed 90/5/5 split did not).
NORM_CACHE = MODEL_DIR / "norm_stats_cache.npz"
if NORM_CACHE.exists():
    print(f"Loading cached normalization stats from {NORM_CACHE}")
    cache = np.load(NORM_CACHE, allow_pickle=True)
    norm_stats = pd.DataFrame({"mean": cache["mean"], "std": cache["std"]},
                               index=cache["cols"])
else:
    print("Rebuilding the training grid to recover normalization stats "
          "(not retraining -- just recomputing mean/std per column)...")
    grid = list(product(cfg.AMPLITUDES, cfg.PULSATIONS))
    bc_pairs = [(("dirichlet", "rest", {}), ("dirichlet", "gaussian", {"A": A, "omega": omega}))
                for A, omega in grid]
    df, _FIELDS, INPUTS_check, OUTPUTS_check = C.generate_dataset_general(INPUT_FIELDS, cfg, bc_pairs)
    assert INPUTS_check == INPUTS and OUTPUTS_check == OUTPUTS

    cols = INPUTS + OUTPUTS
    norm_stats = pd.DataFrame({"mean": df[cols].mean(), "std": df[cols].std()})
    norm_stats["std"] = norm_stats["std"].replace(0, 1)
    np.savez(NORM_CACHE, mean=norm_stats["mean"].values, std=norm_stats["std"].values,
             cols=norm_stats.index.values)
    print(f"Cached normalization stats to {NORM_CACHE}")

mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)
mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype(np.float32)
biais_repos = C._biais_repos(modele, mu_in, sd_in, mu_out, sd_out, cfg)


# Graph: same ethylvanillin-shaped topology as graph_rods_prototype.py.
ring_angles = {f"R{k+1}": np.pi/2 - k * (np.pi/3) for k in range(6)}
positions = {name: (cfg.L * np.cos(a), cfg.L * np.sin(a)) for name, a in ring_angles.items()}

positions["Cald"] = (0.0, 1.8 * cfg.L)
positions["O_ald"] = (-0.7 * cfg.L, 2.3 * cfg.L)
positions["H_ald"] = (0.7 * cfg.L, 2.3 * cfg.L)

positions["O_e"] = (1.7 * cfg.L, -0.9 * cfg.L)
positions["CH2"] = (2.5 * cfg.L, -0.5 * cfg.L)
positions["CH3"] = (3.3 * cfg.L, -0.9 * cfg.L)

positions["OH"] = (0.0, -1.8 * cfg.L)

LEAF_BC = {
    "O_ald": ("dirichlet", "rest", {}),
    "H_ald": ("dirichlet", "rest", {}),
    "OH":    ("dirichlet", "rest", {}),
    "CH3":   ("dirichlet", "gaussian", {"A": 0.05, "omega": 6.0}),  # pulse injected here
}

nodes = {}
for name, pos in positions.items():
    if name in LEAF_BC:
        nodes[name] = {"pos": pos, "driven": True, "bc": LEAF_BC[name]}
    else:
        nodes[name] = {"pos": pos, "driven": False}

edges = [
    ("R1", "R2"), ("R2", "R3"), ("R3", "R4"), ("R4", "R5"), ("R5", "R6"), ("R6", "R1"),
    ("R1", "Cald"), ("Cald", "O_ald"), ("Cald", "H_ald"),
    ("R3", "O_e"), ("O_e", "CH2"), ("CH2", "CH3"),
    ("R4", "OH"),
]

# Longer than the training Nt (500 steps = 5 s): the pulse has to cross ~7
# rods end to end. The model only ever looks at a short local window, so
# this doesn't require retraining or touching norm_stats.
ROLLOUT_T_END = 14.0
ROLLOUT_NT = int(ROLLOUT_T_END / cfg.dt)


class Rod:
    def __init__(self, start, end):
        self.start, self.end = start, end
        self.U = np.zeros((ROLLOUT_NT + 1, cfg.Ntot), dtype=np.float32)


rods = [Rod(start, end) for start, end in edges]

# Time stepping: same structure as commun.py's _autoregressive_rollout_general,
# generalized to many rods sharing nodes. Each "hop" advances N_FWD*ndt raw
# steps and gets both horizons from the model at once, from the same base
# state (h=2 is not chained through h=1, matching how the model was trained).
history_needed = cfg.M_BACK * cfg.ndt
hop = cfg.N_FWD * cfg.ndt
frames = []  # list of (s, {(start,end): Nx-length array})

for n in range(history_needed, ROLLOUT_NT - hop + 1, hop):
    m_list = [n - lag * cfg.ndt for lag in range(cfg.M_BACK)]

    X = np.concatenate(
        [C.build_window(m_list, lambda m, rod=rod: rod.U[m], INPUT_FIELDS, cfg) for rod in rods],
        axis=0,
    )
    Xn = (X - mu_in) / sd_in
    with torch.no_grad():
        sortie = modele(torch.tensor(Xn)).numpy()
    deltas = sortie * sd_out + mu_out - biais_repos
    deltas_per_rod = np.split(deltas, len(rods))

    for h in range(1, cfg.N_FWD + 1):
        s = n + h * cfg.ndt
        t = s * cfg.dt

        for rod, d in zip(rods, deltas_per_rod):
            rod.U[s, cfg.nodes] = rod.U[n, cfg.nodes] + d[:, h - 1]

        # Node values at time s: driven nodes from their imposed waveform;
        # free nodes from continuity + zero net force at a massless
        # junction. Near the junction, continuity keeps every rod's nearby
        # reading close to the shared value (differing only by a dx-order
        # slope term that cancels on average), so the plain average of the
        # value just inside every rod touching the node is correct to
        # O(dx^2).
        node_value = {}
        for node_id, node in nodes.items():
            if node["driven"]:
                node_value[node_id] = C.bc_value(node["bc"], t)
            else:
                inside = []
                for rod in rods:
                    if rod.start == node_id:
                        inside.append(rod.U[s, cfg.i_left + 1])
                    if rod.end == node_id:
                        inside.append(rod.U[s, cfg.i_right - 1])
                node_value[node_id] = float(np.mean(inside))

        for rod in rods:
            C.apply_boundary(rod.U[s], "left", "dirichlet", node_value[rod.start], cfg)
            C.apply_boundary(rod.U[s], "right", "dirichlet", node_value[rod.end], cfg)

        if cfg.SMOOTH_ALPHA > 0:
            j0, j1 = cfg.i_left + 1, cfg.i_right
            for rod in rods:
                lap = rod.U[s, j0-1:j1-1] - 2*rod.U[s, j0:j1] + rod.U[s, j0+1:j1+1]
                rod.U[s, j0:j1] += cfg.SMOOTH_ALPHA * lap

        frames.append((s, {(rod.start, rod.end): rod.U[s, cfg.nodes].copy() for rod in rods}))

print(f"Rollout done: {len(frames)} frames covering {ROLLOUT_T_END:.1f} s")


# Animate: every rod drawn as a line of points between its two nodes, colored
# by displacement.
A_MAX = 0.05
fig, ax = plt.subplots(figsize=(8, 7))
ax.set_xlim(-2.0, 4.0)
ax.set_ylim(-2.5, 3.0)
ax.set_aspect("equal")
ax.axis("off")

scatters = []
for rod in rods:
    x0, y0 = nodes[rod.start]["pos"]
    x1, y1 = nodes[rod.end]["pos"]
    xs = np.linspace(x0, x1, cfg.Nx)
    ys = np.linspace(y0, y1, cfg.Nx)
    sc = ax.scatter(xs, ys, c=np.zeros(cfg.Nx), cmap="coolwarm",
                     vmin=-A_MAX, vmax=A_MAX, s=8)
    scatters.append((rod, sc))

fig.colorbar(scatters[0][1], ax=ax, label="displacement u", shrink=0.7)
title = ax.set_title("t = 0.00 s")


def update(frame_idx):
    s, snapshot = frames[frame_idx]
    for rod, sc in scatters:
        sc.set_array(snapshot[(rod.start, rod.end)])
    title.set_text(f"t = {s * cfg.dt:.2f} s  (surrogate model)")
    return [sc for _, sc in scatters] + [title]


anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=40)
out_path = SCRIPT_DIR / "propagation_reseau_surrogate.gif"
anim.save(out_path, writer="pillow", fps=25)
print(f"Saved animation to {out_path}")
