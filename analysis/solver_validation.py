#!/usr/bin/env python3
# Validates the FD leapfrog solver (physics/solver.py) against two
# independent, textbook checks -- not a comparison against another piece of
# our own code, but against known physics:
#   1. Travelling wave: a Gaussian pulse driven at the left end should
#      propagate as u(x, t) = pulse(t - x/c), matching the exact analytical
#      solution of the continuous wave equation, before it reaches the far end.
#   2. Energy conservation: the discrete mechanical energy of a clamped rod
#      released from rest should stay constant over the full run, since the
#      leapfrog scheme has no numerical dissipation.
#
# Usage: python analysis/solver_validation.py
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from beamsurrogate.config import Config
from beamsurrogate.physics.solver import run_fd_simulation_general, run_fd_simulation_free
from beamsurrogate.physics.waves import gaussian_value

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "runs" / "solver_validation" / "figures"

BC_CLAMPED = ("dirichlet", "rest", {})  # displacement held at 0 at that end
PULSE_PARAMS = {"A": 1.0, "sigma": 0.2}
BC_PULSE = ("dirichlet", "gaussian", PULSE_PARAMS)  # a Gaussian pulse driven at that end


def _save(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def gaussian(x, x0, A, sigma):
    return A * np.exp(-((x - x0) / sigma) ** 2)


# ---------------------------------------------------------------------------
# Check 1: travelling wave vs. the exact analytical solution
# ---------------------------------------------------------------------------
def travelling_wave_error(Nx: int, C0: float = 0.70, L: float = 6.0, E: float = 1.0, rho: float = 2.0,
                           t_check: float = 1.0):
    # A pulse driven at x=0 travels as u(x, t) = pulse(t - x/c) until it
    # reaches the far end -- domain (L=6) is long enough that it never does
    # within t_check (wave needs L/c ~ 8.5s to cross; margin is enormous).
    c = np.sqrt(E / rho)
    dx = L / (Nx - 1)
    dt = C0 * dx / c
    Nt = int(round(t_check / dt))
    cfg = Config(E=E, rho=rho, L=L, Nx=Nx, Nt=Nt, t_end=Nt * dt, SS=11)
    t_actual = cfg.Nt * cfg.dt  # compare at the time actually reached, not the rounded target

    x = np.linspace(0.0, L, Nx)
    u_num = run_fd_simulation_general(BC_PULSE, BC_CLAMPED, cfg)[:, cfg.nodes]
    arrival_time = t_actual - x / c
    u_analytical = np.where(arrival_time > 0, gaussian_value(PULSE_PARAMS, arrival_time), 0.0)
    error = float(np.max(np.abs(u_num[-1] - u_analytical)))
    return error, cfg.dx, x, u_num[-1], u_analytical


def check_travelling_wave():
    error, dx, x, u_num, u_analytical = travelling_wave_error(Nx=600)
    print(f"Travelling-wave check: Nx=600, dx={dx:.4e}, L_inf error = {error:.3e}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, u_num, color="#2a78d6", label="FD solver")
    ax.plot(x, u_analytical, "--", color="#eb6834", label="Analytical (travelling wave)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Displacement")
    ax.legend()
    _save(fig, FIG_DIR / "solver_validation_dalembert.png")
    return error


def check_convergence_order():
    resolutions = [150, 300, 600, 1200]
    errors, dxs = [], []
    for Nx in resolutions:
        error, dx, *_ = travelling_wave_error(Nx=Nx)
        errors.append(error)
        dxs.append(dx)
        print(f"  Nx={Nx:4d}  dx={dx:.4e}  L_inf error={error:.3e}")

    orders = [np.log(errors[i] / errors[i + 1]) / np.log(dxs[i] / dxs[i + 1]) for i in range(len(errors) - 1)]
    order = float(np.mean(orders))
    print(f"Observed convergence order (avg over {len(orders)} refinements): {order:.2f}")
    return order


# ---------------------------------------------------------------------------
# Check 2: energy conservation
# ---------------------------------------------------------------------------
def check_energy_conservation():
    # Same physical parameters as the real dataset (config.py defaults).
    cfg = Config(E=1.0, rho=2.0, L=1.0, Nx=100, Nt=500, t_end=5.0, SS=11)
    x = np.linspace(0.0, cfg.L, cfg.Nx)
    u0 = gaussian(x, cfg.L / 2, A=1.0, sigma=0.08)

    # Sum over the TRUE clamped domain (i_left to i_right inclusive) -- the
    # ghost-band Dirichlet condition pins the wall one index past cfg.nodes,
    # so slicing to cfg.nodes alone would silently drop the last wall's
    # elastic-energy term.
    u_full = run_fd_simulation_free(BC_CLAMPED, BC_CLAMPED, u0, cfg)
    u = u_full[:, cfg.i_left:cfg.i_right + 1]  # shape (Nt+1, Nx+1)

    # Discrete energy, one line per term of the equation:
    #   E^n = (dx/2) * sum_i [ rho*((u_i^{n+1}-u_i^n)/dt)^2
    #                          + E*((u_{i+1}^{n+1}-u_i^{n+1})/dx)*((u_{i+1}^n-u_i^n)/dx) ]
    v = (u[1:, :-1] - u[:-1, :-1]) / cfg.dt
    du_next = (u[1:, 1:] - u[1:, :-1]) / cfg.dx
    du_curr = (u[:-1, 1:] - u[:-1, :-1]) / cfg.dx
    energy = 0.5 * cfg.dx * np.sum(cfg.rho * v ** 2 + cfg.E * du_next * du_curr, axis=1)

    drift = float((energy.max() - energy.min()) / energy.mean())
    print(f"Energy conservation: relative variation over the full run = {drift:.3e}")

    t = cfg.dt * np.arange(1, cfg.Nt + 1)
    rel = (energy - energy[0]) / energy[0]

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(t, rel, color="#2a78d6", lw=1.0)
    ax.axhline(0.0, color="k", lw=0.5, ls=":")
    ax.set_xlabel(r"Time  $t$  [s]")
    ax.set_ylabel(r"Relative energy drift  $(\mathcal{E}^n - \mathcal{E}^0)/\mathcal{E}^0$  [-]")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
    ax.margins(x=0)
    ax.text(0.5, 0.93, f"max $|$drift$|$ = {np.abs(rel).max():.1e}", transform=ax.transAxes,
            ha="center", va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    _save(fig, FIG_DIR / "solver_validation_energy.png")
    return drift


def main():
    print("=== Check 1: travelling wave vs. analytical solution ===")
    error = check_travelling_wave()
    print("\n=== Convergence order (grid refinement, fixed Courant number) ===")
    order = check_convergence_order()
    print("\n=== Check 2: energy conservation ===")
    drift = check_energy_conservation()

    error_pct = 100 * error / PULSE_PARAMS["A"]
    summary = (
        "Solver validation results\n"
        "==========================\n\n"
        f"Travelling-wave L_inf error (Nx=600): {error:.2e}  "
        f"({error_pct:.3f}% of the pulse's peak amplitude)\n"
        f"Observed convergence order: {order:.2f}  "
        "(dimensionless -- error shrinks as dx^order; 2 is the expected value)\n"
        f"Energy conservation, relative variation: {drift:.2e}  "
        "(dimensionless ratio -- essentially exact conservation)\n"
    )
    print("\n=== Summary (values to paste into the thesis) ===")
    print(summary)
    out_path = FIG_DIR / "solver_validation_results.txt"
    out_path.write_text(summary)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
