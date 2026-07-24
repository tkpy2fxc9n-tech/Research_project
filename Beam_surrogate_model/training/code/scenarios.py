# Independent per-end sampling of a boundary-forcing family, restricted to
# non-sum, single-signal families already validated in commun.py (excludes
# "random_multitone", which sums several tones together -- explicitly out of
# scope for this project). Both ends may be pushed (or left quiet, via
# "rest"), drawn independently -- so the dataset naturally covers one-sided
# forcing (like the original gaussian_wave project), two-sided forcing, and
# every mix of families in between.
import numpy as np

ALLOWED_FAMILIES = ["gaussian", "sinusoid", "step", "ramp", "rest"]
BC_TYPE = "dirichlet"  # imposed displacement -- same physics as the original single-family project


def _sample_one_end(rng, cfg, C):
    family = rng.choice(ALLOWED_FAMILIES)
    sampler, _ = C.BC_WAVEFORMS[family]
    return (BC_TYPE, family, sampler(rng, cfg))


def sample_scenario(rng, cfg, C):
    return _sample_one_end(rng, cfg, C), _sample_one_end(rng, cfg, C)


def sample_scenarios(cfg, n_samples, C, rng=None):
    rng = rng if rng is not None else np.random.default_rng(cfg.SEED)
    return [sample_scenario(rng, cfg, C) for _ in range(n_samples)]
