# Wave propagation on a NETWORK of rods sharing junction nodes -- the phase-6
# generalization beyond a single beam. Ported (junction-averaging physics
# only, matplotlib/example topology stripped out) from
# Graph_rods_network/graph_rods_prototype.py (ground truth FD) and
# surrogate_rods_prototype.py (NN-driven interiors, same junction rule).
#
# A "junction" is a massless point shared by 2+ rods: continuity + zero-net-
# force at that point means the value just inside every touching rod reads
# the same shared value to O(dx), so the plain average of those "just
# inside" readings is correct to O(dx^2) -- this is the one piece of physics
# a single beam doesn't need.
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Rod:
    start: str
    end: str
    n_points: int
    u: np.ndarray = field(default=None)
    u_prev: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.u is None:
            self.u = np.zeros(self.n_points)
        if self.u_prev is None:
            self.u_prev = np.zeros(self.n_points)


def step_network(rods: list[Rod], node_specs: dict[str, dict], t_new: float, CFL: float) -> dict[str, float]:
    # node_specs[node_id] = {"driven": bool, "func": callable(t) -> float}
    # for driven (leaf) nodes; free (junction) nodes only need {"driven": False}.
    # Advances every rod one leapfrog step and applies the junction rule at
    # every node, in place. Returns the node values used at this step (handy
    # for logging/err_near_junction).
    u_new_list = []
    for rod in rods:
        u, u_prev = rod.u, rod.u_prev
        u_new = np.empty_like(u)
        u_new[1:-1] = (2 * u[1:-1] - u_prev[1:-1] + CFL ** 2 * (u[:-2] - 2 * u[1:-1] + u[2:]))
        u_new_list.append(u_new)

    node_value = {}
    for node_id, spec in node_specs.items():
        if spec["driven"]:
            node_value[node_id] = spec["func"](t_new)
        else:
            inside_values = []
            for rod, u_new in zip(rods, u_new_list):
                if rod.start == node_id:
                    inside_values.append(u_new[1])
                if rod.end == node_id:
                    inside_values.append(u_new[-2])
            node_value[node_id] = float(np.mean(inside_values)) if inside_values else 0.0

    for rod, u_new in zip(rods, u_new_list):
        u_new[0] = node_value[rod.start]
        u_new[-1] = node_value[rod.end]
        rod.u_prev = rod.u
        rod.u = u_new

    return node_value


def run_network_simulation(edges: list[tuple[str, str]], node_specs: dict[str, dict],
                            n_points: int, Nt: int, dt: float, CFL: float) -> list[Rod]:
    rods = [Rod(start, end, n_points) for start, end in edges]
    for n in range(Nt):
        step_network(rods, node_specs, (n + 1) * dt, CFL)
    return rods
