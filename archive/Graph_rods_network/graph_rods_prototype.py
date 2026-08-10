"""
Prototype: wave propagation on a network of rods (ground truth, finite differences).

Topology copied from the skeleton of an ethylvanillin diagram (ring + branches,
bond order ignored -- every line in the drawing becomes one rod): a 6-rod
hexagon ring, with three branches hanging off three of its nodes (an
aldehyde-like branch that itself splits in two, an ethoxy-like 3-rod chain,
and a single OH-like rod). No neural network is involved here: every rod is
advanced with the same explicit finite-difference leapfrog scheme used to
generate the original single-rod training data (Beam_surrogate_model). A
pulse is injected at the far end of the ethoxy chain, to see it travel
through the ring and out to the other branches.

Run it with:
    python3 graph_rods_prototype.py
It writes propagation_reseau.gif next to this script.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

OUTPUT_DIR = Path(__file__).resolve().parent

E, rho, L = 1.0, 2.0, 1.0
Nx = 100                       # points per rod, including its two end nodes
# pulse has to cross ~7 rods end to end, each taking L / sqrt(E/rho) seconds
t_end, Nt = 15.0, 1500
dt = t_end / Nt
dx = L / (Nx - 1)
CFL = dt / dx * np.sqrt(E / rho)
assert CFL < 1, "unstable discretization, reduce dt or increase Nx"


def gaussian_pulse(t, t0=0.4, sigma=0.08, A=0.05):
    return A * np.exp(-((t - t0) / sigma) ** 2)


def fixed_zero(t):
    return 0.0


# Graph: hexagon ring (R1..R6) plus three branches, copied from the
# ethylvanillin skeleton (connectivity only, not the chemistry). Each node
# is either a leaf (an imposed pulse or a fixed 0) or free (computed from
# whichever rods touch it).
ring_angles = {f"R{k+1}": np.pi/2 - k * (np.pi/3) for k in range(6)}   # R1 top, clockwise
positions = {name: (L * np.cos(a), L * np.sin(a)) for name, a in ring_angles.items()}

# aldehyde-like branch off R1: R1 -> Cald -> {O_ald, H_ald}
positions["Cald"]  = (0.0, 1.8 * L)
positions["O_ald"] = (-0.7 * L, 2.3 * L)
positions["H_ald"] = (0.7 * L, 2.3 * L)

# ethoxy-like branch off R3: R3 -> O_e -> CH2 -> CH3
positions["O_e"] = (1.7 * L, -0.9 * L)
positions["CH2"] = (2.5 * L, -0.5 * L)
positions["CH3"] = (3.3 * L, -0.9 * L)

# hydroxyl-like branch off R4: R4 -> OH
positions["OH"] = (0.0, -1.8 * L)

LEAF_FUNCS = {
    "O_ald": fixed_zero,
    "H_ald": fixed_zero,
    "OH":    fixed_zero,
    "CH3":   gaussian_pulse,     # pulse injected at the far end of the ethoxy chain
}

nodes = {}
for name, pos in positions.items():
    if name in LEAF_FUNCS:
        nodes[name] = {"pos": pos, "driven": True, "func": LEAF_FUNCS[name]}
    else:
        nodes[name] = {"pos": pos, "driven": False}

edges = [
    ("R1", "R2"), ("R2", "R3"), ("R3", "R4"), ("R4", "R5"), ("R5", "R6"), ("R6", "R1"),
    ("R1", "Cald"), ("Cald", "O_ald"), ("Cald", "H_ald"),
    ("R3", "O_e"), ("O_e", "CH2"), ("CH2", "CH3"),
    ("R4", "OH"),
]


class Rod:
    def __init__(self, start, end):
        self.start, self.end = start, end
        self.u = np.zeros(Nx)          # current displacement along the rod
        self.u_prev = np.zeros(Nx)     # displacement one step earlier


rods = [Rod(start, end) for start, end in edges]

history = []
SAVE_EVERY = 8

for n in range(Nt):
    t_new = (n + 1) * dt

    # 1) advance the interior of every rod: same leapfrog wave update as
    #    the single-rod code, applied independently per rod.
    u_new_list = []
    for rod in rods:
        u, u_prev = rod.u, rod.u_prev
        u_new = np.empty_like(u)
        u_new[1:-1] = (2 * u[1:-1] - u_prev[1:-1]
                        + CFL**2 * (u[:-2] - 2 * u[1:-1] + u[2:]))
        u_new_list.append(u_new)

    # 2) node values at the new time: driven from their function, free
    #    nodes from continuity + zero net force at a massless junction.
    #    Near the junction, continuity keeps every rod's nearby reading
    #    close to the shared value (differing only by a dx-order slope
    #    term that cancels on average), so the plain average of the value
    #    just inside every rod touching the node is correct to O(dx^2).
    node_value = {}
    for node_id, node in nodes.items():
        if node["driven"]:
            node_value[node_id] = node["func"](t_new)
        else:
            inside_values = []
            for rod, u_new in zip(rods, u_new_list):
                if rod.start == node_id:
                    inside_values.append(u_new[1])
                if rod.end == node_id:
                    inside_values.append(u_new[-2])
            node_value[node_id] = np.mean(inside_values)

    # 3) write the node values into each rod's two end points, advance state.
    for rod, u_new in zip(rods, u_new_list):
        u_new[0] = node_value[rod.start]
        u_new[-1] = node_value[rod.end]
        rod.u_prev = rod.u
        rod.u = u_new

    if n % SAVE_EVERY == 0:
        history.append({(rod.start, rod.end): rod.u.copy() for rod in rods})


# Animate: every rod drawn as a line of points between its two nodes,
# colored by displacement.
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
    xs = np.linspace(x0, x1, Nx)
    ys = np.linspace(y0, y1, Nx)
    sc = ax.scatter(xs, ys, c=np.zeros(Nx), cmap="coolwarm",
                     vmin=-A_MAX, vmax=A_MAX, s=8)
    scatters.append((rod, xs, sc))

fig.colorbar(scatters[0][2], ax=ax, label="displacement u", shrink=0.7)
title = ax.set_title("t = 0.00 s")


def update(frame_idx):
    snapshot = history[frame_idx]
    for rod, _xs, sc in scatters:
        sc.set_array(snapshot[(rod.start, rod.end)])
    title.set_text(f"t = {frame_idx * SAVE_EVERY * dt:.2f} s")
    return [sc for *_, sc in scatters] + [title]


anim = animation.FuncAnimation(fig, update, frames=len(history), interval=40)
out_path = OUTPUT_DIR / "propagation_reseau.gif"
anim.save(out_path, writer="pillow", fps=25)
print(f"Saved animation to {out_path}")
