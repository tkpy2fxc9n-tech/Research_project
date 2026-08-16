#!/usr/bin/env python3
# p14_check: verifies physics/scaling.py's Scaling class independently of
# any trained network. 3 real beams (steel / aluminium / polymer, different
# L, E, rho), the SAME nondimensional boundary-condition scenario mapped
# into each beam's own physical units via Scaling.from_nd(), simulated with
# the UNMODIFIED FD solver (physics/solver.py -- no code path here is
# specific to this script), then mapped back to nondimensional form via
# Scaling.to_nd(). If Scaling correctly cancels out L/E/rho, the 3
# resulting nondimensional rollouts are identical to machine epsilon --
# the wave equation is scale-invariant, so any real gap means a physical
# constant leaked into the "core" solver/generator somewhere it shouldn't
# have (a non-regression check, not a training run -- no model involved).
#
# Usage: python scripts/check_nondim_scaling.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beamsurrogate.config import Config
from beamsurrogate.physics.scaling import Scaling, physical_config
from beamsurrogate.physics.solver import run_fd_simulation_general
from beamsurrogate.physics.waves import bc_value

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "runs" / "p14_check" / "figures"

TOLERANCE = 1e-9
# Validated categorical palette (dataviz skill, references/palette.md slots
# 1-3: blue/orange/aqua -- the only 3-slot subset that clears all-pairs CVD
# separation, not just adjacent-pair). Line style is the secondary encoding
# the skill requires alongside the aqua slot's sub-3:1 surface contrast.
COLORS = {"acier": "#2a78d6", "aluminium": "#eb6834", "polymere": "#1baf7a"}
STYLES = {"acier": "-", "aluminium": "--", "polymere": ":"}

# Representative material properties (E in Pa, rho in kg/m^3) -- only L was
# specified by the p14 planning table; E/rho are illustrative real-world
# values for each material, not measured for a specific specimen.
BEAMS = [
    ("acier",     dict(L=1.0,  E=200e9, rho=7850.0)),
    ("aluminium", dict(L=2.0,  E=70e9,  rho=2700.0)),
    ("polymere",  dict(L=10.0, E=3e9,   rho=1180.0)),
]

# One canonical nondimensional scenario, shared by all 3 beams: a Gaussian
# push at the right end, left end at rest -- same topology as the "simple"
# dataset. C=0.5 (this check has no legacy baseline to match, unlike
# p14_nondim, so any stable Courant number works).
ND_CFG = Config(E=1.0, rho=1.0, L=1.0, Nx=100, Nt=500, t_end=250 * (0.5 / 99))
SIGMA_ND = ND_CFG.t_end / 12.0
A_ND = 1.0   # peak displacement = U0 by construction


def _plot_rollout_overlay(beams_data, nodes, path: Path):
    # u_nd(x_nd) at 4 instants spread over the nondimensional run, one line
    # per beam -- if Scaling is correct, the 3 lines coincide at every panel.
    x_nd = np.linspace(0.0, 1.0, len(nodes))
    frame_steps = np.linspace(0, ND_CFG.Nt, 4, dtype=int)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
    for ax, m in zip(axes, frame_steps):
        for name, d in beams_data.items():
            ax.plot(x_nd, d["U_nd"][m, nodes], STYLES[name], color=COLORS[name], lw=2,
                     alpha=0.8, label=name)
        ax.set_title(f"t̃ = {m * ND_CFG.dt:.2f}"); ax.set_xlabel("x̃"); ax.grid(True)
    axes[0].set_ylabel("ũ"); axes[0].legend()
    fig.suptitle("p14_check -- rollout adimensionnel superpose (acier / aluminium / polymere)")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_rollout_animation(beams_data, nodes, path: Path):
    x_nd = np.linspace(0.0, 1.0, len(nodes))
    frames = np.arange(0, ND_CFG.Nt + 1, max(ND_CFG.Nt // 100, 1))
    ymax = max(np.abs(d["U_nd"][:, nodes]).max() for d in beams_data.values()) * 1.2

    fig, ax = plt.subplots(figsize=(9, 5))
    lines = {name: ax.plot([], [], STYLES[name], color=COLORS[name], lw=2, label=name)[0]
              for name in beams_data}
    ax.set_xlim(0, 1); ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("x̃"); ax.set_ylabel("ũ"); ax.legend(loc="upper right"); ax.grid(True)
    title_obj = ax.set_title("")

    def update(m):
        for name, d in beams_data.items():
            lines[name].set_data(x_nd, d["U_nd"][m, nodes])
        title_obj.set_text(f"p14_check -- rollout adimensionnel -- t̃ = {m * ND_CFG.dt:.3f}")
        return list(lines.values()) + [title_obj]

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    anim.save(path, writer="pillow", fps=20, dpi=110)
    plt.close(fig)


def _make_physical_rollout_animation(beams_data, nodes, path: Path):
    # Same rollout as rollout_nondim.gif, but in REAL units: one x-axis per
    # beam (meters -- L differs 10x between acier and polymere, so they
    # can't share a normalized axis here) against a SHARED real-time clock.
    # Common time window = acier's own run (its own L/c is the shortest of
    # the 3, so every beam has valid data across that whole window) --
    # within it, acier/aluminium (c ~5000 m/s) visibly cross their beam
    # while polymere (c ~1600 m/s, AND a 10x longer beam) barely moves: the
    # point of this figure is exactly that visible speed gap, driven by
    # each material's own sqrt(E/rho).
    names = [n for n, _ in BEAMS]
    t_common = beams_data[names[0]]["t_phys"]   # acier: shortest own t_end
    frame_idx = np.arange(0, len(t_common), max(len(t_common) // 150, 1))

    fig, axes = plt.subplots(len(names), 1, figsize=(9, 8), sharex=False)
    lines, fills = {}, {}
    for ax, name in zip(axes, names):
        d = beams_data[name]
        ymax = np.abs(d["U_phys"]).max() * 1.2
        line, = ax.plot([], [], STYLES[name], color=COLORS[name], lw=2)
        lines[name] = line
        ax.set_xlim(0, d["L"]); ax.set_ylim(-ymax, ymax)
        ax.set_ylabel("u (m)"); ax.grid(True)
        ax.set_title(f"{name}  (c = {d['c']:.0f} m/s, L = {d['L']:.0f} m)", loc="left", fontsize=10)
    axes[-1].set_xlabel("x (m)")
    title_obj = fig.suptitle("")

    def update(k):
        t = t_common[k]
        artists = []
        for name in names:
            d = beams_data[name]
            n = min(int(round(t / d["dt"])), len(d["t_phys"]) - 1)
            lines[name].set_data(d["x_phys"], d["U_phys"][n, nodes])
            artists.append(lines[name])
        title_obj.set_text(f"p14_check -- meme temps reel ecoule, vitesse differente -- t = {t:.2e} s")
        return artists + [title_obj]

    anim = animation.FuncAnimation(fig, update, frames=frame_idx, interval=60, blit=False)
    plt.tight_layout()
    anim.save(path, writer="pillow", fps=18, dpi=110)
    plt.close(fig)


def _plot_bc_drive(beams_data, path: Path):
    fig, (ax_phys, ax_nd) = plt.subplots(1, 2, figsize=(14, 5))
    for name, d in beams_data.items():
        ax_phys.plot(d["t_phys"], d["drive_phys"], color=COLORS[name], lw=2, label=name)
        ax_nd.plot(d["t_nd"], d["drive_nd"], STYLES[name], color=COLORS[name], lw=2, alpha=0.8, label=name)
    ax_phys.set_xlabel("t (s)"); ax_phys.set_ylabel("u (m)")
    ax_phys.set_title("Signal moteur -- unites physiques (tres differentes)")
    ax_phys.legend(); ax_phys.grid(True)
    ax_nd.set_xlabel("t̃"); ax_nd.set_ylabel("ũ")
    ax_nd.set_title("Meme signal -- adimensionnel (confondu)")
    ax_nd.legend(); ax_nd.grid(True)
    fig.suptitle("p14_check -- signal moteur (bord droit) avant/apres adimensionnement")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    bc_left_nd = ("dirichlet", "rest", {})

    results = {}
    beams_data = {}
    for name, mat in BEAMS:
        scaling = Scaling(L=mat["L"], E=mat["E"], rho=mat["rho"], U0=1e-3)  # 1 mm reference push
        cfg_phys = physical_config(ND_CFG, scaling)

        bc_params_phys = {
            "A": float(scaling.amp_from_nd(A_ND)),
            "sigma": float(scaling.time_from_nd(SIGMA_ND)),
        }
        bc_right_phys = ("dirichlet", "gaussian", bc_params_phys)
        bc_left_phys = bc_left_nd  # "rest" has no numeric params to rescale

        U_phys = run_fd_simulation_general(bc_left_phys, bc_right_phys, cfg_phys)
        U_nd = scaling.amp_to_nd(U_phys)
        results[name] = U_nd

        t_phys = np.arange(cfg_phys.Nt + 1) * cfg_phys.dt
        drive_phys = np.array([bc_value(bc_right_phys, t) for t in t_phys])
        beams_data[name] = {
            "U_nd": U_nd, "U_phys": U_phys, "t_phys": t_phys, "drive_phys": drive_phys,
            "t_nd": scaling.time_to_nd(t_phys), "drive_nd": scaling.amp_to_nd(drive_phys),
            "L": mat["L"], "c": scaling.c, "dt": cfg_phys.dt,
            "x_phys": np.linspace(0.0, mat["L"], len(ND_CFG.nodes)),
        }
        print(f"{name:10s} L={mat['L']:5.1f}m E={mat['E']:.1e}Pa rho={mat['rho']:.0f}kg/m3  "
              f"c={scaling.c:.1f}m/s  A_phys={bc_params_phys['A']:.3e}m  sigma_phys={bc_params_phys['sigma']:.3e}s")

    ref_name = BEAMS[0][0]
    ref = results[ref_name]
    all_ok = True
    print()
    for name, U_nd in results.items():
        gap = float(np.abs(U_nd - ref).max())
        status = "OK" if gap < TOLERANCE else "FAILED"
        if gap >= TOLERANCE:
            all_ok = False
        print(f"[{status}] {name:10s} vs {ref_name}: max abs gap (nondimensional) = {gap:.3e} "
              f"(tolerance {TOLERANCE:.0e})")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    nodes = ND_CFG.nodes
    _plot_rollout_overlay(beams_data, nodes, FIGURES_DIR / "rollout_overlay.png")
    _make_rollout_animation(beams_data, nodes, FIGURES_DIR / "rollout_nondim.gif")
    _plot_bc_drive(beams_data, FIGURES_DIR / "bc_drive.png")
    _make_physical_rollout_animation(beams_data, nodes, FIGURES_DIR / "rollout_physical.gif")
    print(f"\nFigures written to {FIGURES_DIR}")

    if all_ok:
        print("\nAll beams OK -- Scaling.to_nd()/from_nd() correctly reproduce the same "
              "nondimensional rollout regardless of the beam's real L/E/rho.")
    else:
        print("\nFAILED -- at least one beam's nondimensional rollout diverges: "
              "check physics/scaling.py and every cfg.E/cfg.rho/cfg.L touchpoint.")
        sys.exit(1)


if __name__ == "__main__":
    main()
