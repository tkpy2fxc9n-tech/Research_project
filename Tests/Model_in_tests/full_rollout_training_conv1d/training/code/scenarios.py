# Weighted sampler over the table's 7 signal families, plus the
# free-evolution random-initial-condition sampler. A "scenario" is
# (bc_left, bc_right, u0_or_None): bc_left/bc_right are the usual
# (bc_type, family, params) BCSpec tuples from waves.py; u0_or_None is only
# set for "free_evolution" scenarios (a random initial displacement instead
# of a boundary push -- physics.run_fd_simulation_free handles that case).
import numpy as np

from waves import BC_TYPES, BC_WAVEFORMS, flip

FAMILY_SHARES = {
    "fourier": 0,
    "sinusoid": 0.5,
    "chirp": 0,
    "gaussian": 0.5,
    "shock": 0,
    "filtered_random": 0,
    "free_evolution": 0,
}
FORCED_FAMILIES = [f for f in FAMILY_SHARES if f != "free_evolution"]
_FORCED_WEIGHTS = np.array([FAMILY_SHARES[f] for f in FORCED_FAMILIES])
FORCED_WEIGHTS = _FORCED_WEIGHTS / _FORCED_WEIGHTS.sum()

# Fraction of forced (non-free-evolution) scenarios that use a single family
# correlated across both ends (in-phase or anti-phase/antisymmetric) instead
# of two independently-drawn ends -- covers the table's "correlated two-end
# histories" on top of the default independent-per-end sampling.
CORRELATED_PROB = 0.3


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


def _sample_one_end(rng, cfg):
    family = rng.choice(FORCED_FAMILIES, p=FORCED_WEIGHTS)
    bc_type = rng.choice(BC_TYPES)
    sampler, _ = BC_WAVEFORMS[family]
    return (bc_type, family, sampler(rng, cfg))


def sample_scenario(rng, cfg):
    family_choice = rng.choice(list(FAMILY_SHARES), p=list(FAMILY_SHARES.values()))

    if family_choice == "free_evolution":
        u0 = sample_random_ic(rng, cfg)
        rest = ("dirichlet", "rest", {"ic": "random"})
        return rest, rest, u0

    if rng.random() < CORRELATED_PROB:
        family = rng.choice(FORCED_FAMILIES, p=FORCED_WEIGHTS)
        bc_type = rng.choice(BC_TYPES)
        sampler, _ = BC_WAVEFORMS[family]
        params = sampler(rng, cfg)
        bc_left = (bc_type, family, params)
        if rng.random() < 0.5:
            bc_right = (bc_type, family, params)  # in-phase / symmetric
        else:
            bc_right = (bc_type, family, flip(family, params))  # anti-phase / antisymmetric
        return bc_left, bc_right, None

    bc_left = _sample_one_end(rng, cfg)
    bc_right = _sample_one_end(rng, cfg)
    return bc_left, bc_right, None


def sample_scenarios(cfg, n_samples, rng=None):
    rng = rng if rng is not None else np.random.default_rng(cfg.SEED)
    return [sample_scenario(rng, cfg) for _ in range(n_samples)]
