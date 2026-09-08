"""
Isolates a single, genuine branch (degree K >= 3, real averaging across
several rods) from the rest of the bridge, and sweeps K = 3, 4, 5 to see how
the NN surrogate's amplitude-crushing at a real junction (found in the
bridge test, surrogate_rods_prototype.py: ~10-30x too small vs FD) scales
with how many rods meet there.

Why this and not just the bridge: three_rod_series_test.py showed a
degree-2 "junction" (only ONE other rod to borrow ghost data from, so
np.mean([...]) of a single element = an exact copy) reproduces FD almost
perfectly -- no crushing at all. That means the crushing isn't about the
ghost band varying in space (it varies here too); it's specifically about
AVERAGING across several independent rods, which the model never saw
during training (always a single, physically coherent rod). This test
isolates exactly that: one star junction, nothing else, at increasing K.

Topology per K: one central node J, one driven arm (O0 -> J, dirichlet
gaussian pulse at O0) and K-1 "receiving" arms (J -> O_i, dirichlet rest,
fixed far ends) arranged at evenly spaced angles purely for visual
distinguishability. Both an FD reference (same O(dx^2) real-branch
averaging used for degree>=3 junctions in graph_rods_3d_prototype.py --
the physically correct approximation for continuity + zero net force at a
massless junction) and the NN surrogate (ghost bands averaged across the
other rods, same as surrogate_rods_prototype.py) are run for each K, so we
get, for every K, both (a) how much a real branch SHOULD split the
amplitude physically, and (b) how much the NN actually transmits -- letting
us isolate the model's EXCESS crushing beyond the genuine physical split.

Same pulse (A=AMP_MAX, sigma=0.3) and 14s horizon as the other tests here.

Run it with:
    python3 star_junction_test.py
Writes star_junction_amplitude_vs_time.png and star_junction_summary.png
next to this script.
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

PULSE_A = cfg.AMP_MAX
PULSE_SIGMA = 0.3
ROLLOUT_T_END = 14.0
ROLLOUT_NT = int(ROLLOUT_T_END / cfg.dt)


def gaussian_value(t, A, sigma):
    t0 = 4.0 * sigma
    return A * np.exp(-((t - t0) / sigma) ** 2)


def build_star(K):
    """J at the origin; arm 0 driven (O0->J, gaussian), arms 1..K-1 fixed
    rest (J->O_i), evenly spaced angles for visual distinguishability."""
    pos = {"J": np.array([0.0, 0.0])}
    edges = []
    node_bc = {}
    for i in range(K):
        ang = np.radians(360.0 * i / K)
        name = f"O{i}"
        pos[name] = cfg.L * np.array([np.cos(ang), np.sin(ang)])
        if i == 0:
            edges.append((name, "J"))   # driven end is the rod's "start"
            node_bc[name] = ("dirichlet", "gaussian", {"A": PULSE_A, "sigma": PULSE_SIGMA})
        else:
            edges.append(("J", name))   # J is the rod's "start" for the rest
            node_bc[name] = ("dirichlet", "rest", {})
    nodes = {name: {"pos": p, "driven": name != "J", "bc": node_bc.get(name)} for name, p in pos.items()}
    return nodes, edges


def run_fd(nodes, edges):
    class FDRod:
        def __init__(self, s, e):
            self.start, self.end = s, e
            self.u = np.zeros(cfg.Nx)
            self.u_prev = np.zeros(cfg.Nx)

    rods = [FDRod(s, e) for s, e in edges]
    node_rods = {n: [] for n in nodes}
    for rod in rods:
        node_rods[rod.start].append((rod, "start"))
        node_rods[rod.end].append((rod, "end"))

    amp = {(r.start, r.end): np.empty(ROLLOUT_NT + 1) for r in rods}
    for r in rods:
        amp[(r.start, r.end)][0] = 0.0

    for n in range(ROLLOUT_NT):
        t_new = (n + 1) * cfg.dt
        u_new_list = []
        for rod in rods:
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
            touching = node_rods[node_id]
            if len(touching) == 2:
                (rod1, side1), (rod2, side2) = touching
                b1, i1 = (0, 1) if side1 == "start" else (-1, -2)
                b2, i2 = (0, 1) if side2 == "start" else (-1, -2)
                u_curr = rod1.u[b1]
                u_prev_node = rod1.u_prev[b1]
                node_value[node_id] = (2 * u_curr - u_prev_node
                                        + cfg.CFL**2 * (rod1.u[i1] - 2 * u_curr + rod2.u[i2]))
            else:
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
            amp[(rod.start, rod.end)][n + 1] = np.abs(u_new).max()

    return amp


def run_nn(nodes, edges):
    class NNRod:
        def __init__(self, s, e):
            self.start, self.end = s, e
            self.U = np.zeros((ROLLOUT_NT + 1, cfg.Ntot), dtype=np.float32)

    rods = [NNRod(s, e) for s, e in edges]
    history_needed = cfg.M_BACK * cfg.ndt
    hop = cfg.N_FWD * cfg.ndt
    frames = []

    for n in range(history_needed, ROLLOUT_NT - hop + 1, hop):
        m_list = [n - lag * cfg.ndt for lag in range(cfg.M_BACK)]
        X = np.concatenate(
            [build_window(m_list, lambda m, rod=rod: rod.U[m], INPUT_FIELDS, cfg) for rod in rods],
            axis=0,
        )
        Xn = (X - mu_in) / sd_in
        with torch.no_grad():
            sortie = model(torch.tensor(Xn)).numpy()
        deltas = sortie * sd_out + mu_out - rest_bias
        deltas_per_rod = np.split(deltas, len(rods))

        for h in range(1, cfg.N_FWD + 1):
            s = n + h * cfg.ndt
            t = s * cfg.dt

            for rod, d in zip(rods, deltas_per_rod):
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
                    touching = [rod for rod in rods if node_id in (rod.start, rod.end)]
                    node_value[node_id] = float(np.mean([rod_reading(rod, node_id, 0) for rod in touching]))

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

    t = np.array([s * cfg.dt for s, _ in frames])
    amp = {key: np.array([np.abs(snap[key]).max() for _, snap in frames])
           for key in {(r.start, r.end) for r in rods}}
    return t, amp


Ks = [3, 4, 5]
results = {}
fd_t = np.arange(ROLLOUT_NT + 1) * cfg.dt

for K in Ks:
    print(f"\n=== K={K} (1 driven arm + {K-1} receiving arms) ===")
    nodes, edges = build_star(K)
    driven_key = edges[0]
    receiving_keys = edges[1:]

    print("Running FD reference...")
    fd_amp = run_fd(nodes, edges)
    print("Running NN surrogate...")
    nn_t, nn_amp = run_nn(nodes, edges)

    fd_driven_peak = fd_amp[driven_key].max()
    nn_driven_peak = nn_amp[driven_key].max()
    fd_recv_peaks = [fd_amp[k].max() for k in receiving_keys]
    nn_recv_peaks = [nn_amp[k].max() for k in receiving_keys]

    print(f"  driven arm   FD={fd_driven_peak:.5f}  NN={nn_driven_peak:.5f}")
    for k, fp, np_ in zip(receiving_keys, fd_recv_peaks, nn_recv_peaks):
        print(f"  recv {k[0]}-{k[1]:<3s} FD={fp:.5f}  NN={np_:.5f}  ratio NN/FD={np_/fp:.3f}")

    results[K] = dict(nodes=nodes, edges=edges, driven_key=driven_key, receiving_keys=receiving_keys,
                       fd_amp=fd_amp, nn_t=nn_t, nn_amp=nn_amp,
                       fd_driven_peak=fd_driven_peak, nn_driven_peak=nn_driven_peak,
                       fd_recv_mean=float(np.mean(fd_recv_peaks)), nn_recv_mean=float(np.mean(nn_recv_peaks)))

print("\n=== Summary: physical split (FD) vs NN transmission, receiving arms (mean) ===")
print(f"{'K':>3s} {'FD recv/driven':>16s} {'NN recv/driven':>16s} {'NN/FD excess crush':>20s}")
for K in Ks:
    r = results[K]
    fd_split = r["fd_recv_mean"] / r["fd_driven_peak"]
    nn_split = r["nn_recv_mean"] / r["nn_driven_peak"]
    excess = nn_split / fd_split
    print(f"{K:3d} {fd_split:16.4f} {nn_split:16.4f} {excess:20.4f}")

# --- Per-K amplitude-vs-time panel: driven arm + receiving arms, FD vs NN.
fig, axes = plt.subplots(len(Ks), 1, figsize=(10, 3.2 * len(Ks)), sharex=True)
for ax, K in zip(axes, Ks):
    r = results[K]
    ax.plot(fd_t, r["fd_amp"][r["driven_key"]], color="tab:blue", label="FD driven")
    ax.plot(r["nn_t"], r["nn_amp"][r["driven_key"]], color="tab:orange", label="NN driven")
    for i, k in enumerate(r["receiving_keys"]):
        ax.plot(fd_t, r["fd_amp"][k], color="tab:cyan", alpha=0.6,
                 label="FD receiving" if i == 0 else None)
        ax.plot(r["nn_t"], r["nn_amp"][k], color="tab:red", alpha=0.6,
                 label="NN receiving" if i == 0 else None)
    ax.set_ylabel("max|u|")
    ax.set_title(f"K={K} ({K-1} receiving arms)")
    ax.legend(fontsize=8, ncol=2)
axes[-1].set_xlabel("t (s)")
fig.suptitle("Single real branch, degree K: driven arm vs receiving arms, FD vs NN")
fig.tight_layout()
fig.savefig(SCRIPT_DIR / "star_junction_amplitude_vs_time.png", dpi=120)
print(f"\nSaved {SCRIPT_DIR / 'star_junction_amplitude_vs_time.png'}")

# --- Summary plot: recv/driven split ratio vs K, FD (physical) vs NN.
fig2, ax2 = plt.subplots(figsize=(7, 5))
fd_splits = [results[K]["fd_recv_mean"] / results[K]["fd_driven_peak"] for K in Ks]
nn_splits = [results[K]["nn_recv_mean"] / results[K]["nn_driven_peak"] for K in Ks]
ax2.plot(Ks, fd_splits, "o-", color="tab:blue", label="FD (physically correct split)")
ax2.plot(Ks, nn_splits, "o-", color="tab:orange", label="NN (actual transmission)")
ax2.set_xlabel("K (rods meeting at the junction)")
ax2.set_ylabel("receiving-arm peak / driven-arm peak")
ax2.set_xticks(Ks)
ax2.legend()
ax2.set_title("How much amplitude actually crosses a real branch, vs how much should")
fig2.savefig(SCRIPT_DIR / "star_junction_summary.png", dpi=120)
print(f"Saved {SCRIPT_DIR / 'star_junction_summary.png'}")
