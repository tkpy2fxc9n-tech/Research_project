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

# dataset name -> filename under data/ (see dataset/make_dataset.py).
# Keys A/B/C/D match the "dataset levels" convention of the thesis (Table 16):
# same 4 datasets previously named simple/medium/medium_bidir/complex,
# renamed here to match the report. The underlying generation *profile*
# names in beamsurrogate.data.generate.PROFILES are unchanged (still
# "simple"/"medium"/"medium_bidir"/"complex") -- only the registry key and
# the delivered filename were renamed, so regenerating one of these needs
# `dataset/make_dataset.py --profile <old-profile-name> --output data/beam_dataset_<X>.h5`.
DATASETS = {
    "A": "beam_dataset_A.h5",  # was "simple" -- Gaussian pulse only, right-driven, Dirichlet only
    "D": "beam_dataset_D.h5",  # was "complex" -- richest profile: mixed BC types/initial states, 6 wave families
    "B": "beam_dataset_B.h5",
    # was "medium": same topology as A (right-driven, Dirichlet only, never
    # pre-excited), but 5 waveform families instead of gaussian only.
    # Intermediate in difficulty between A and D.
    "C": "beam_dataset_C.h5",
    # was "medium_bidir": same recipe as B, except driving is no longer
    # one-sided -- half the trajectories drive both ends at once, a quarter
    # right only, a quarter left only.
    "simple_coarse_r2": "beam_dataset_simple_coarse_r2.h5",
    # A, subsampled 2x in space+time -- see dataset/make_coarse_dataset.py.
    # Not fresh FD runs at coarse resolution. Kept under its original name
    # since it's a derived variant, not one of the 4 report-level datasets.
    "simple_coarse_r4": "beam_dataset_simple_coarse_r4.h5",  # A, subsampled 4x
    "simple_nondim": "beam_dataset_simple_nondim.h5",
    # A regenerated with E=rho=L=1 and per-sample amplitude normalization --
    # see dataset/make_dataset_nondim.py. Also kept under its original name.
}

__all__ = ["MODELS", "REGIMES", "STABILIZERS", "DATASETS", "resolve_stabilizer"]
