# "Free evolution / random initial state" (table: "Avoid dependence on
# forced-from-rest trajectories") -- the one family that isn't a boundary
# forcing function. Both ends stay at rest (no push at all); instead the bar
# *starts* bent into a random smooth shape and is simply let go.
#
# Nothing in commun.py supports a non-zero initial condition (run_fd_simulation
# / run_fd_simulation_general both hardcode u = u_1 = 0 at t=0), so this is
# the one piece of physics genuinely new to this project. It's a ~15-line
# variant of commun.run_fd_simulation_general's leapfrog loop, kept local
# here rather than added to the shared module. `C` (the imported commun
# module) is passed in explicitly rather than re-imported, since this module
# does no sys.path setup of its own.
import numpy as np


def sample_random_ic(rng, cfg) -> np.ndarray:
    # A handful of low-order sine modes on [0, L]: naturally zero at both
    # ends (consistent with the rest/homogeneous boundaries used for every
    # free-evolution run), smooth (low wavenumbers only, no sharp kinks).
    n_modes = int(rng.integers(2, 6))
    modes = rng.integers(1, 6, size=n_modes)
    amps = rng.uniform(-1.0, 1.0, size=n_modes)
    x = np.linspace(0.0, cfg.L, cfg.Nx)

    profile = np.zeros(cfg.Nx)
    for a, k in zip(amps, modes):
        profile += a * np.sin(k * np.pi * x / cfg.L)

    peak = np.abs(profile).max()
    A_total = float(rng.uniform(cfg.AMP_MIN, cfg.AMP_MAX) * 3.0)  # a bit larger: spread over the whole bar, not a local pulse
    if peak > 1e-12:
        profile *= A_total / peak
    return profile.astype(np.float64)


def run_fd_simulation_free(bc_left, bc_right, u0_profile: np.ndarray, cfg, C) -> np.ndarray:
    # Same leapfrog scheme as C.run_fd_simulation_general, except u/u_1 start
    # from u0_profile (zero initial velocity) instead of zero displacement.
    # bc_left/bc_right are expected to be the "rest" family on both ends
    # (zero forcing) -- the randomness here is entirely in the starting
    # shape, not in any boundary push.
    i_left, i_right, Ntot = cfg.i_left, cfg.i_right, cfg.Ntot
    u_storage = np.zeros((cfg.Nt + 1, Ntot))

    u = np.zeros(Ntot)
    u[cfg.nodes] = u0_profile
    C.apply_boundary_conditions(u, 0.0, bc_left, bc_right, cfg)
    u_1 = u.copy()
    u_storage[0] = u.copy()

    for n in range(cfg.Nt):
        t = n * cfg.dt
        u_new = np.zeros(Ntot)
        u_new[i_left:i_right + 1] = (
            2.0 * u[i_left:i_right + 1] - u_1[i_left:i_right + 1]
            + cfg.CFL ** 2 * (u[i_left - 1:i_right] - 2.0 * u[i_left:i_right + 1] + u[i_left + 1:i_right + 2])
        )
        C.apply_boundary_conditions(u_new, t + cfg.dt, bc_left, bc_right, cfg)
        u_1, u = u.copy(), u_new
        u_storage[n + 1] = u.copy()

    return u_storage
