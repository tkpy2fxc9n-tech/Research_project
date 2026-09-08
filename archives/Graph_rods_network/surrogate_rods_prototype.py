"""
Rod-network wave propagation on a bridge truss (Warren truss, every rod the
same length cfg.L -- no riverbanks/supports modeled, just the rod assembly
itself), with every rod's interior physics coming from the CURRENT
beamsurrogate pipeline's trained model -- run p1_feat_u (runs/p1_feat_u:
regime=pushforward, model=mlp, features=[U], M_BACK=2, N_FWD=3, SS=11) --
instead of finite differences.

This model was only ever trained with a "rest" (0) or "gaussian" bump
family (see Config.AMP_MIN/AMP_MAX/SIGMA_MIN/SIGMA_MAX in its
config.resolved.yaml), so the only driven node here (B0) uses "gaussian"
with A/sigma drawn from the middle of those trained ranges, to stay well
inside the training distribution.

Ghost points: the model needs an SS-wide ghost band beyond each rod's own
Nx points to build its input window even close to a boundary. A flat,
constant-valued ghost band (imposed via apply_boundary) is only correct for
a genuine free/driven end -- everywhere a rod is actually linked to others,
the ghost band is instead real data borrowed from those other rods: exactly
the other rod's own points when the junction is a straight 2-rod link (no
real branch, e.g. the far end of the bottom chord), or the average of every
other rod's own point at the matching distance when 3+ rods meet (a real
branch, e.g. every interior chord node) -- the natural generalization of
"copy the other rod" once there's more than one to copy from.

Reuses the actual pipeline code (Config, MODELS registry, build_window,
apply_boundary, compute_rest_bias) from src/beamsurrogate/, so the model
sees inputs built exactly like during training. Normalization stats aren't
saved to disk by that pipeline (recomputed fresh every run) -- rebuilt here
from a 200-trajectory subsample of the simple dataset (not the full 2000,
which would cost ~29G RSS just to get a mean/std -- fine for training,
overkill for this exploratory check; the resulting norm_stats converge fast
and are already stable at this sample size).

Run it with:
    python3 surrogate_rods_prototype.py
It writes propagation_reseau_surrogate.gif next to this script.
"""

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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the 3d projection

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
INPUT_FIELDS = list(cfg.features)   # ["U"] for p1_feat_u

INPUTS = make_feature_columns(INPUT_FIELDS, cfg)
OUTPUTS = make_output_columns(cfg)

modele = MODELS[cfg.model](len(INPUTS), len(OUTPUTS), cfg)
modele.load_state_dict(torch.load(REPO_ROOT / "runs" / SOURCE_RUN_ID / "model.pth", weights_only=True))
modele.eval()

print(f"Recomputing norm_stats from a 200-trajectory subsample of {cfg.dataset} "
      f"(matches {SOURCE_RUN_ID}'s own training normalization)...")
dataset_path = REPO_ROOT / "data" / DATASETS[cfg.dataset]
(df, _FIELDS, INPUTS_check, OUTPUTS_check, *_rest) = load_hdf5_dataset(
    INPUT_FIELDS, cfg, dataset_path, max_trajectories=200)
assert INPUTS_check == INPUTS and OUTPUTS_check == OUTPUTS
norm_stats = compute_norm_stats(df, INPUTS, OUTPUTS, cfg)
del df, _FIELDS, _rest
gc.collect()

mu_in, sd_in, mu_out, sd_out = norm_stats_arrays(norm_stats, INPUTS, OUTPUTS)
biais_repos = compute_rest_bias(modele, mu_in, sd_in, mu_out, sd_out, cfg)


# Graph: bridge truss (Warren truss, N_BAYS bays) -- a bottom chord and a
# top chord joined by zigzagging diagonals, all forming equilateral
# triangles so every rod has the same length cfg.L. A pulse is injected at
# one end of the bottom chord (B0); every other node, including the far end
# (B{N_BAYS}), is a free junction with no imposed support.
N_BAYS = 3
H = cfg.L * np.sqrt(3) / 2   # equilateral triangles -> every rod has length cfg.L

positions = {}
for i in range(N_BAYS + 1):
    positions[f"B{i}"] = (i * cfg.L, 0.0)
for i in range(N_BAYS):
    positions[f"T{i}"] = ((i + 0.5) * cfg.L, H)

# A at the top of its trained range (cfg.AMP_MAX); sigma NOT at the sharpest
# trained value (cfg.SIGMA_MIN=0.05) -- a leapfrog scheme at CFL<1 (here
# 0.70) is only dispersion-free at CFL=1, and a narrow pulse has broad
# frequency content, so it visibly "rings" as it travels. That ringing was
# confirmed directly (sigma=0.05 vs 0.275 vs 0.5, same cfg, FD only:
# increasingly clean as sigma grows) and was muddying the "does it actually
# propagate" question. sigma=0.3 (same value used in graph_rods_3d_prototype.py,
# for a fair FD-vs-NN comparison) keeps a strong, still fairly localized
# pulse without that artifact, and stays inside [SIGMA_MIN, SIGMA_MAX].
PULSE_A = cfg.AMP_MAX
PULSE_SIGMA = 0.3

LEAF_BC = {
    "B0": ("dirichlet", "gaussian", {"A": PULSE_A, "sigma": PULSE_SIGMA}),  # pulse injected here
}

nodes = {}
for name, pos in positions.items():
    if name in LEAF_BC:
        nodes[name] = {"pos": pos, "driven": True, "bc": LEAF_BC[name]}
    else:
        nodes[name] = {"pos": pos, "driven": False}

edges = (
    [(f"B{i}", f"B{i+1}") for i in range(N_BAYS)]              # bottom chord
    + [(f"T{i}", f"T{i+1}") for i in range(N_BAYS - 1)]         # top chord
    + [e for i in range(N_BAYS)                                # diagonals
       for e in ((f"B{i}", f"T{i}"), (f"T{i}", f"B{i+1}"))]
)

# Longer than the training Nt (500 steps = 5 s): the pulse has to cross
# several rods end to end. The model only ever looks at a short local
# window, so this doesn't require retraining or touching norm_stats.
ROLLOUT_T_END = 14.0
ROLLOUT_NT = int(ROLLOUT_T_END / cfg.dt)


class Rod:
    def __init__(self, start, end):
        self.start, self.end = start, end
        self.U = np.zeros((ROLLOUT_NT + 1, cfg.Ntot), dtype=np.float32)


rods = [Rod(start, end) for start, end in edges]

# Time stepping: same structure as beamsurrogate's physics/solver.py
# autoregressive_rollout, generalized to many rods sharing nodes. Each "hop"
# advances N_FWD*ndt raw steps and gets both horizons from the model at
# once, from the same base state (h=2 is not chained through h=1, matching
# how the model was trained).
history_needed = cfg.M_BACK * cfg.ndt
hop = cfg.N_FWD * cfg.ndt
frames = []  # list of (s, {(start,end): Nx-length array})

for n in range(history_needed, ROLLOUT_NT - hop + 1, hop):
    m_list = [n - lag * cfg.ndt for lag in range(cfg.M_BACK)]

    X = np.concatenate(
        [build_window(m_list, lambda m, rod=rod: rod.U[m], INPUT_FIELDS, cfg) for rod in rods],
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

        def rod_reading(rod, node_id, dist):
            # rod's own physical value at `dist` grid points from node_id
            # (dist=0 is the shared boundary point itself), moving INTO the
            # rod, away from the junction.
            if rod.start == node_id:
                return rod.U[s, cfg.i_left + dist]
            return rod.U[s, cfg.i_right - 1 - dist]

        # Boundary point itself: every rod touching a free node must agree
        # on its value there (continuity) -- the consensus average of every
        # rod's own independent prediction there, same role as before.
        node_value = {}
        for node_id, node in nodes.items():
            if node["driven"]:
                node_value[node_id] = bc_value(node["bc"], t)
            else:
                touching = [rod for rod in rods if node_id in (rod.start, rod.end)]
                node_value[node_id] = float(np.mean([rod_reading(rod, node_id, 0) for rod in touching]))

        # Ghost band beyond each rod's own domain: a flat imposed value only
        # for a genuine free/driven end (apply_boundary, unchanged). Where a
        # rod is actually linked to others, the ghost band is real data
        # borrowed from those other rods instead of a flat constant -- the
        # other rod's own points exactly when only one other rod is there (a
        # straight link, no real branch), or the average of every other
        # rod's own point at the matching distance when several meet (a real
        # branch: the natural generalization of "copy the other rod", using
        # the same continuity + zero net force reasoning as node_value above).
        for rod in rods:
            for side, node_id in (("left", rod.start), ("right", rod.end)):
                node = nodes[node_id]
                if node["driven"]:
                    apply_boundary(rod.U[s], side, "dirichlet", node_value[node_id], cfg)
                    continue
                others = [r for r in rods if r is not rod and node_id in (r.start, r.end)]
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
            for rod in rods:
                lap = rod.U[s, j0-1:j1-1] - 2*rod.U[s, j0:j1] + rod.U[s, j0+1:j1+1]
                rod.U[s, j0:j1] += cfg.SMOOTH_ALPHA * lap

        frames.append((s, {(rod.start, rod.end): rod.U[s, cfg.nodes].copy() for rod in rods}))

print(f"Rollout done: {len(frames)} frames covering {ROLLOUT_T_END:.1f} s")

# --- Dense amplitude-vs-time diagnostic, every rod, every recorded frame
# (every cfg.ndt=3 raw steps, i.e. every 0.03s -- NOT a handful of
# hand-picked animation frames eyeballed by hand, which is how the earlier
# "no propagation" read was reached and got the wave-arrival timing wrong).
node_adj = {node_id: [] for node_id in nodes}
for rod in rods:
    node_adj[rod.start].append(rod.end)
    node_adj[rod.end].append(rod.start)
_dist = {"B0": 0}
_frontier = ["B0"]
while _frontier:
    _next = []
    for node_id in _frontier:
        for other in node_adj[node_id]:
            if other not in _dist:
                _dist[other] = _dist[node_id] + 1
                _next.append(other)
    _frontier = _next
rod_dist = {(rod.start, rod.end): min(_dist[rod.start], _dist[rod.end]) for rod in rods}

t_frames = np.array([s * cfg.dt for s, _ in frames])
amp_trace = {key: np.array([np.abs(snap[key]).max() for _, snap in frames]) for key in rod_dist}

fig_amp, ax_amp = plt.subplots(figsize=(11, 5))
cmap = plt.get_cmap("viridis")
max_dist = max(rod_dist.values())
for key in sorted(amp_trace, key=lambda k: rod_dist[k]):
    d = rod_dist[key]
    ax_amp.plot(t_frames, amp_trace[key], color=cmap(d / max_dist),
                label=f"{key[0]}-{key[1]} (dist {d})")
ax_amp.set_xlabel("t (s)")
ax_amp.set_ylabel("max|u| along rod")
ax_amp.set_title("Surrogate (p1_feat_u): amplitude vs time, every rod, every recorded frame "
                  "(color = graph distance from B0)")
ax_amp.legend(fontsize=7, ncol=2, loc="upper right")
fig_amp.savefig(SCRIPT_DIR / "bridge_amplitude_vs_time_surrogate.png", dpi=120)
print(f"Saved {SCRIPT_DIR / 'bridge_amplitude_vs_time_surrogate.png'}")

np.savez(SCRIPT_DIR / "bridge_amplitude_trace_surrogate.npz", t=t_frames,
         rod_names=np.array([f"{k[0]}-{k[1]}" for k in amp_trace], dtype=object),
         rod_dists=np.array([rod_dist[k] for k in amp_trace]),
         traces=np.stack([amp_trace[k] for k in amp_trace]))
print(f"Saved {SCRIPT_DIR / 'bridge_amplitude_trace_surrogate.npz'}")
print("Peak |u| per rod (surrogate), sorted by distance from B0:")
for key in sorted(amp_trace, key=lambda k: rod_dist[k]):
    print(f"  dist {rod_dist[key]}  {key[0]}-{key[1]:<4s} peak={amp_trace[key].max():.5f}")


# Animate in 3D, same framing as graph_rods_3d_prototype.py: displacement u
# becomes an actual height (z), not just a color, viewed from an oblique 3/4
# angle. A_MAX matches THIS model's own trained amplitude range (cfg.AMP_MAX
# -- p1_feat_u was trained on 0.0005-0.005, ~20x smaller than the old
# archived model's 0.005-0.1) -- using the old hardcoded 0.05 here made
# every real displacement a tiny fraction of the color/height scale, so
# nothing visible ever showed up. Real amplitude would still be invisible
# next to the truss's spatial extent even at the right scale, so z is
# exaggerated by Z_SCALE for visibility -- NOT physically to scale,
# labelled as such on the z-axis.
A_MAX = cfg.AMP_MAX
Z_SCALE = 20

fig = plt.figure(figsize=(9, 6))
# ax gets an EXPLICIT, fixed rect, and the colorbar its own separate cax --
# fig.colorbar(ax=ax, ...) silently shrinks ax to make room for the
# colorbar, which is what made earlier centering attempts (tuned without
# accounting for that shrink) land off-center once the colorbar was added.
ax = fig.add_axes([0.02, 0.05, 0.76, 0.90], projection="3d")
cax = fig.add_axes([0.84, 0.20, 0.03, 0.6])
ax.set_xlim(-1.35, 3.05)   # tuned empirically against ax's fixed rect above
ax.set_ylim(-1.31, 0.96)   # so the truss lands centered in the full frame
ax.set_zlim(-A_MAX * Z_SCALE, A_MAX * Z_SCALE)
ax.view_init(elev=40, azim=-50)
ax.set_box_aspect((4.5, 2, 1.2), zoom=1.85)
ax.set_axis_off()

rod_lines = []
for rod in rods:
    x0, y0 = nodes[rod.start]["pos"]
    x1, y1 = nodes[rod.end]["pos"]
    xs = np.linspace(x0, x1, cfg.Nx)
    ys = np.linspace(y0, y1, cfg.Nx)
    sc = ax.scatter(xs, ys, np.zeros(cfg.Nx), c=np.zeros(cfg.Nx), cmap="coolwarm",
                     vmin=-A_MAX, vmax=A_MAX, s=8)
    rod_lines.append((rod, xs, ys, sc))

fig.colorbar(rod_lines[0][3], cax=cax, label="displacement u")
title = ax.set_title(f"t = 0.00 s   (surrogate model, displacement u shown as height, x{Z_SCALE} exaggerated)")


def update(frame_idx):
    s, snapshot = frames[frame_idx]
    for rod, xs, ys, sc in rod_lines:
        u = snapshot[(rod.start, rod.end)]
        sc._offsets3d = (xs, ys, u * Z_SCALE)
        sc.set_array(u)
    title.set_text(f"t = {s * cfg.dt:.2f} s   (surrogate model, displacement u shown as height, x{Z_SCALE} exaggerated)")
    return [sc for *_, sc in rod_lines] + [title]


anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=40)
out_path = SCRIPT_DIR / "propagation_reseau_surrogate.gif"
anim.save(out_path, writer="pillow", fps=25)
print(f"Saved animation to {out_path}")
