# Independent per-end sampling of a boundary-forcing family, restricted to
# "gaussian" (a single forcing pulse, amplitude/omega drawn from
# cfg.AMP_MIN/MAX and cfg.OMEGA_MIN/MAX) and "rest" (the end stays at 0).
# Both ends are drawn independently, imposed-displacement (dirichlet) only --
# same physics as this project's original single-family (left=rest,
# right=gaussian) setup, just no longer pinning which end gets which family.
# The dataset therefore naturally covers one-sided forcing (either end),
# two-sided forcing, and the fully-at-rest case.
import numpy as np

ALLOWED_FAMILIES = ["gaussian", "rest"]
BC_TYPE = "dirichlet"


def _sample_one_end(rng, cfg, C):
    family = rng.choice(ALLOWED_FAMILIES)
    sampler, _ = C.BC_WAVEFORMS[family]
    return (BC_TYPE, family, sampler(rng, cfg))


def sample_scenario(rng, cfg, C):
    return _sample_one_end(rng, cfg, C), _sample_one_end(rng, cfg, C)


def sample_scenarios(cfg, n_samples, C, rng=None):
    rng = rng if rng is not None else np.random.default_rng(cfg.SEED)
    return [sample_scenario(rng, cfg, C) for _ in range(n_samples)]
