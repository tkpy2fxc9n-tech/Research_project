# Weighted sampler over the table's 7 signal families. A "scenario" is
# (bc_left, bc_right, u0_or_None): bc_left/bc_right are the usual
# (bc_type, family, params) BCSpec tuples consumed unchanged by commun.py's
# generalized physics; u0_or_None is only set for "free_evolution" scenarios
# (a random initial displacement instead of a boundary push -- see
# free_evolution.py). Downstream code (data_split.py, train.py,
# rollout_torch.py, all copied verbatim from full_rollout_training_general_bc)
# only ever needs the (bc_left, bc_right) pair, never u0 -- see main.py where
# the two are split apart right after dataset generation.
import numpy as np

import waveforms
import free_evolution

FAMILY_SHARES = {
    "fourier": 0.30,
    "sinusoid": 0.15,
    "chirp": 0.15,
    "gaussian": 0.15,
    "shock": 0.10,
    "filtered_random": 0.10,
    "free_evolution": 0.05,
}
FORCED_FAMILIES = [f for f in FAMILY_SHARES if f != "free_evolution"]
_FORCED_WEIGHTS = np.array([FAMILY_SHARES[f] for f in FORCED_FAMILIES])
FORCED_WEIGHTS = _FORCED_WEIGHTS / _FORCED_WEIGHTS.sum()

# Fraction of forced (non-free-evolution) scenarios that use a single family
# correlated across both ends (in-phase or anti-phase/antisymmetric) instead
# of two independently-drawn ends -- covers the table's "correlated two-end
# histories" on top of the default independent-per-end sampling.
CORRELATED_PROB = 0.3


def _sample_one_end(rng, cfg, C):
    family = rng.choice(FORCED_FAMILIES, p=FORCED_WEIGHTS)
    bc_type = rng.choice(C.BC_TYPES)
    sampler, _ = C.BC_WAVEFORMS[family]
    return (bc_type, family, sampler(rng, cfg))


def sample_scenario(rng, cfg, C):
    family_choice = rng.choice(list(FAMILY_SHARES), p=list(FAMILY_SHARES.values()))

    if family_choice == "free_evolution":
        u0 = free_evolution.sample_random_ic(rng, cfg)
        rest = ("dirichlet", "rest", {"ic": "random"})
        return rest, rest, u0

    if rng.random() < CORRELATED_PROB:
        family = rng.choice(FORCED_FAMILIES, p=FORCED_WEIGHTS)
        bc_type = rng.choice(C.BC_TYPES)
        sampler, _ = C.BC_WAVEFORMS[family]
        params = sampler(rng, cfg)
        bc_left = (bc_type, family, params)
        if rng.random() < 0.5:
            bc_right = (bc_type, family, params)  # in-phase / symmetric
        else:
            bc_right = (bc_type, family, waveforms.flip(family, params))  # anti-phase / antisymmetric
        return bc_left, bc_right, None

    bc_left = _sample_one_end(rng, cfg, C)
    bc_right = _sample_one_end(rng, cfg, C)
    return bc_left, bc_right, None


def sample_scenarios(cfg, n_samples, C, rng=None):
    rng = rng if rng is not None else np.random.default_rng(cfg.SEED)
    return [sample_scenario(rng, cfg, C) for _ in range(n_samples)]
