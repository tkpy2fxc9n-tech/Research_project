"""
Rod-network wave propagation on a bridge truss (Warren truss, every rod the
same length L -- no riverbanks/supports modeled, just the rod assembly
itself), rendered in 3D: displacement u becomes an actual out-of-plane
height (z), not just a color, viewed from an oblique 3/4 angle.

Junction handling: a node touched by exactly 2 rods (e.g. the far end of the
bottom chord) is really just one continuous rod split across two arrays, so
its shared point is updated with the same 3-point leapfrog stencil an
ordinary interior point would use (exact continuity, no reflection). A node
touched by 3+ rods (every other junction in this truss) is a real branch:
continuity + zero net force at a massless junction, approximated to O(dx^2)
by averaging the near-boundary reading of every rod meeting there.

Pulse amplitude (0.005) matches p1_feat_u's own trained AMP_MAX (see
surrogate_rods_prototype.py) so the two animations are comparable rather
than run at two different physical scales. u's real amplitude would be
nearly invisible next to the network's spatial extent, so z is exaggerated
by Z_SCALE for visibility -- NOT physically to scale, labelled as such on
the z-axis.

Pulse width sigma=0.3 (not the sharpest trained value, 0.05): a leapfrog
scheme at CFL<1 (here 0.70) is only dispersion-free at CFL=1, and a narrow
pulse has broad frequency content, so it visibly "rings"/ripples as it
travels -- confirmed directly by comparing this same run_fd_simulation-style
stencil at sigma=0.05 vs 0.275 vs 0.5 (increasingly clean as sigma grows).
sigma=0.3 keeps a strong, still-fairly-localized pulse without that
dispersion artifact muddying the "does it actually propagate" question.
t0=4*sigma matches the real pipeline's own gaussian_value() convention
(physics/waves.py) instead of a fixed t0 that only suited the old sigma.

Also saves bridge_amplitude_vs_time_fd.png: max|u| along EVERY rod (not a
handful of hand-picked animation frames) at every raw timestep, so wave
arrival at each rod can be read directly off a dense curve instead of eyeballed
off the animation.

Run it with:
    python3 graph_rods_3d_prototype.py
It writes propagation_reseau_3d.gif next to this script.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the 3d projection

OUTPUT_DIR = Path(__file__).resolve().parent

E, rho, L = 1.0, 2.0, 1.0
Nx = 100                       # points per rod, including its two end nodes
t_end, Nt = 15.0, 1500
dt = t_end / Nt
dx = L / (Nx - 1)
CFL = dt / dx * np.sqrt(E / rho)
assert CFL < 1, "unstable discretization, reduce dt or increase Nx"


def gaussian_pulse(t, sigma=0.3, A=0.005):
    t0 = 4.0 * sigma   # same convention as physics/waves.py's gaussian_value
    return A * np.exp(-((t - t0) / sigma) ** 2)


# Bridge truss (Warren truss, N_BAYS bays): a bottom chord and a top chord
# joined by zigzagging diagonals, all forming equilateral triangles so every
# rod -- chord or diagonal -- has exactly the same length L. A pulse is
# injected at one end of the bottom chord (B0); every other node, including
# the far end (B{N_BAYS}), is a free junction with no imposed support.
N_BAYS = 3
H = L * np.sqrt(3) / 2   # equilateral triangles -> every rod has length L

positions = {}
for i in range(N_BAYS + 1):
    positions[f"B{i}"] = (i * L, 0.0)
for i in range(N_BAYS):
    positions[f"T{i}"] = ((i + 0.5) * L, H)

LEAF_FUNCS = {
    "B0": gaussian_pulse,
}

nodes = {}
for name, pos in positions.items():
    if name in LEAF_FUNCS:
        nodes[name] = {"pos": pos, "driven": True, "func": LEAF_FUNCS[name]}
    else:
        nodes[name] = {"pos": pos, "driven": False}

edges = (
    [(f"B{i}", f"B{i+1}") for i in range(N_BAYS)]              # bottom chord
    + [(f"T{i}", f"T{i+1}") for i in range(N_BAYS - 1)]         # top chord
    + [e for i in range(N_BAYS)                                # diagonals
       for e in ((f"B{i}", f"T{i}"), (f"T{i}", f"B{i+1}"))]
)


class Rod:
    def __init__(self, start, end):
        self.start, self.end = start, end
        self.u = np.zeros(Nx)          # current displacement along the rod
        self.u_prev = np.zeros(Nx)     # displacement one step earlier


rods = [Rod(start, end) for start, end in edges]

# Precomputed once: which (rod, side) pairs touch each node. A node touched
# by exactly 2 rods isn't a real branch, just one physical rod split across
# two arrays -- handled separately below so it gets EXACT continuity instead
# of the branch case's O(dx^2)-only approximation.
node_rods = {node_id: [] for node_id in nodes}
for rod in rods:
    node_rods[rod.start].append((rod, "start"))
    node_rods[rod.end].append((rod, "end"))

# Graph distance (in rod hops) from the driven node B0, used only to order/
# color the amplitude-vs-time diagnostic below by "how far the wave has to
# travel to reach this rod".
_dist = {"B0": 0}
_frontier = ["B0"]
while _frontier:
    _next = []
    for node_id in _frontier:
        for rod, side in node_rods[node_id]:
            other = rod.end if side == "start" else rod.start
            if other not in _dist:
                _dist[other] = _dist[node_id] + 1
                _next.append(other)
    _frontier = _next
rod_dist = {(rod.start, rod.end): min(_dist[rod.start], _dist[rod.end]) for rod in rods}

history = []
SAVE_EVERY = 8

# Dense (every raw step, dt=0.01) max|u| per rod -- not a handful of
# hand-picked animation frames -- so wave arrival at each rod can be read
# directly off a curve instead of eyeballed off the gif.
amp_trace = {(rod.start, rod.end): np.empty(Nt + 1) for rod in rods}
for rod in rods:
    amp_trace[(rod.start, rod.end)][0] = 0.0

for n in range(Nt):
    t_new = (n + 1) * dt

    u_new_list = []
    for rod in rods:
        u, u_prev = rod.u, rod.u_prev
        u_new = np.empty_like(u)
        u_new[1:-1] = (2 * u[1:-1] - u_prev[1:-1]
                        + CFL**2 * (u[:-2] - 2 * u[1:-1] + u[2:]))
        u_new_list.append(u_new)

    node_value = {}
    for node_id, node in nodes.items():
        if node["driven"]:
            node_value[node_id] = node["func"](t_new)
            continue

        touching = node_rods[node_id]
        if len(touching) == 2:
            # Exactly 2 rods meeting end-to-end are really just one
            # continuous rod split across two arrays: compute the shared
            # point with the SAME 3-point leapfrog stencil an ordinary
            # interior point would use, borrowing one real neighbor from
            # each side (at time n, not the not-yet-finalized u_new) --
            # exact continuity, no discretization mismatch, no reflection.
            (rod1, side1), (rod2, side2) = touching
            b1, i1 = (0, 1) if side1 == "start" else (-1, -2)
            b2, i2 = (0, 1) if side2 == "start" else (-1, -2)
            u_curr = rod1.u[b1]        # == rod2.u[b2], time n
            u_prev_node = rod1.u_prev[b1]   # == rod2.u_prev[b2], time n-1
            node_value[node_id] = (2 * u_curr - u_prev_node
                                    + CFL**2 * (rod1.u[i1] - 2 * u_curr + rod2.u[i2]))
        else:
            # Real branch (3+ rods): continuity + zero net force at a
            # massless junction, approximated to O(dx^2) by averaging the
            # near-boundary reading of every rod meeting here.
            inside_values = []
            for rod, u_new in zip(rods, u_new_list):
                if rod.start == node_id:
                    inside_values.append(u_new[1])
                if rod.end == node_id:
                    inside_values.append(u_new[-2])
            node_value[node_id] = np.mean(inside_values)

    for rod, u_new in zip(rods, u_new_list):
        u_new[0] = node_value[rod.start]
        u_new[-1] = node_value[rod.end]
        rod.u_prev = rod.u
        rod.u = u_new
        amp_trace[(rod.start, rod.end)][n + 1] = np.abs(u_new).max()

    if n % SAVE_EVERY == 0:
        history.append({(rod.start, rod.end): rod.u.copy() for rod in rods})

# --- Dense amplitude-vs-time diagnostic, every rod, every raw step.
t_axis = np.arange(Nt + 1) * dt
fig_amp, ax_amp = plt.subplots(figsize=(11, 5))
cmap = plt.get_cmap("viridis")
max_dist = max(rod_dist.values())
for key in sorted(amp_trace, key=lambda k: rod_dist[k]):
    d = rod_dist[key]
    ax_amp.plot(t_axis, amp_trace[key], color=cmap(d / max_dist),
                label=f"{key[0]}-{key[1]} (dist {d})")
ax_amp.set_xlabel("t (s)")
ax_amp.set_ylabel("max|u| along rod")
ax_amp.set_title("FD: amplitude vs time, every rod, every raw step "
                  "(color = graph distance from B0)")
ax_amp.legend(fontsize=7, ncol=2, loc="upper right")
fig_amp.savefig(OUTPUT_DIR / "bridge_amplitude_vs_time_fd.png", dpi=120)
print(f"Saved {OUTPUT_DIR / 'bridge_amplitude_vs_time_fd.png'}")

np.savez(OUTPUT_DIR / "bridge_amplitude_trace_fd.npz", t=t_axis,
         rod_names=np.array([f"{k[0]}-{k[1]}" for k in amp_trace], dtype=object),
         rod_dists=np.array([rod_dist[k] for k in amp_trace]),
         traces=np.stack([amp_trace[k] for k in amp_trace]))
print(f"Saved {OUTPUT_DIR / 'bridge_amplitude_trace_fd.npz'}")
print("Peak |u| per rod (FD), sorted by distance from B0:")
for key in sorted(amp_trace, key=lambda k: rod_dist[k]):
    print(f"  dist {rod_dist[key]}  {key[0]}-{key[1]:<4s} peak={amp_trace[key].max():.5f}")


# --- 3D rendering: displacement u becomes an actual height (z), on top of
# each rod's flat resting (x, y) position, colored the same way as before.
A_MAX = 0.005   # matches the pulse amplitude above (and p1_feat_u's AMP_MAX)
Z_SCALE = 20   # z exaggeration factor -- see module docstring

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
ax.view_init(elev=40, azim=-50)     # steeper, more front-on -- opens up the
                                     # triangulation instead of foreshortening
                                     # it into an overlapping zigzag
ax.set_box_aspect((4.5, 2, 1.2), zoom=1.85)  # fewer bays -> less elongated
ax.set_axis_off()   # no panes/ticks/spines -- the network's own lines are
                     # enough spatial reference, a box around mostly-empty
                     # space read as clutter (and left stray edge lines when
                     # only panes/ticks were hidden individually)

rod_lines = []
for rod in rods:
    x0, y0 = nodes[rod.start]["pos"]
    x1, y1 = nodes[rod.end]["pos"]
    xs = np.linspace(x0, x1, Nx)
    ys = np.linspace(y0, y1, Nx)
    sc = ax.scatter(xs, ys, np.zeros(Nx), c=np.zeros(Nx), cmap="coolwarm",
                     vmin=-A_MAX, vmax=A_MAX, s=8)
    rod_lines.append((rod, xs, ys, sc))

fig.colorbar(rod_lines[0][3], cax=cax, label="displacement u")
title = ax.set_title(f"t = 0.00 s   (displacement u shown as height, x{Z_SCALE} exaggerated)")


def update(frame_idx):
    snapshot = history[frame_idx]
    for rod, xs, ys, sc in rod_lines:
        u = snapshot[(rod.start, rod.end)]
        sc._offsets3d = (xs, ys, u * Z_SCALE)
        sc.set_array(u)
    title.set_text(f"t = {frame_idx * SAVE_EVERY * dt:.2f} s   (displacement u shown as height, x{Z_SCALE} exaggerated)")
    return [sc for *_, sc in rod_lines] + [title]


anim = animation.FuncAnimation(fig, update, frames=len(history), interval=40)
out_path = OUTPUT_DIR / "propagation_reseau_3d.gif"
anim.save(out_path, writer="pillow", fps=25)
print(f"Saved animation to {out_path}")
