# Phase 10: which rollout stabilization mechanism is active. Both mechanisms
# already exist as cfg-gated code (`if cfg.NOISE_STD > 0` in
# training/teacher_forcing.py and training/bptt.py; `if cfg.SMOOTH_ALPHA > 0`
# in physics/solver.py and training/losses.py) -- this registry just decides
# which of the two knobs stays nonzero for a given run, by returning a
# resolved copy of cfg with the other knob forced to 0. No new mechanism, no
# duplicated logic: downstream code keeps reading cfg.NOISE_STD/SMOOTH_ALPHA
# exactly as before. "both" (phase 10c) is the one case that forces nothing
# to 0 -- it leaves NOISE_STD/SMOOTH_ALPHA exactly as the config set them, so
# the two mechanisms run together instead of being mutually exclusive.
from __future__ import annotations

import dataclasses


def _none(cfg):
    return dataclasses.replace(cfg, NOISE_STD=0.0, SMOOTH_ALPHA=0.0)


def _noise(cfg):
    return dataclasses.replace(cfg, SMOOTH_ALPHA=0.0)


def _laplacian(cfg):
    return dataclasses.replace(cfg, NOISE_STD=0.0)


def _both(cfg):
    return cfg


STABILIZERS = {
    "none": _none,
    "noise": _noise,
    "laplacian": _laplacian,
    "both": _both,
}


def resolve_stabilizer(cfg):
    if cfg.stabilizer not in STABILIZERS:
        raise ValueError(f"Unknown stabilizer {cfg.stabilizer!r}, expected one of {sorted(STABILIZERS)}")
    return STABILIZERS[cfg.stabilizer](cfg)
