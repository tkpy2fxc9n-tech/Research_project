# Shared result contract every regime (teacher_forcing/pushforward/bptt)
# returns, so evaluate/metrics.py never needs to special-case which regime
# trained the model.
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainResult:
    train_history: list
    val_history: list
    best_val: float
    train_time_s: float
    n_params: int
    extra_history: dict = field(default_factory=dict)   # regime-specific term curves, e.g. {"physics": [...]}
