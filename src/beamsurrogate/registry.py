# Name -> callable dicts. A run YAML picks a string (`model:`, `regime:`,
# `stabilizer:`, `dataset:`); this module is the only place that turns that
# string into actual code. Adding a new variant of any of these means adding
# one entry here (and the implementation it points to) -- never copying a
# project directory.
from __future__ import annotations

from .models.mlp import build_mlp
from .models.cnn import build_cnn
from .training import teacher_forcing, pushforward, bptt
from .training.stabilizers import STABILIZERS, resolve_stabilizer

MODELS = {
    "mlp": build_mlp,
    "cnn": build_cnn,
}

REGIMES = {
    "teacher_forcing": teacher_forcing,
    "pushforward": pushforward,
    "bptt": bptt,
}

# dataset name -> filename under data/ (see scripts/make_dataset.py).
DATASETS = {
    "simple": "beam_dataset_simple.h5",
    "complex": "beam_dataset_complex.h5",
}

__all__ = ["MODELS", "REGIMES", "STABILIZERS", "DATASETS", "resolve_stabilizer"]
