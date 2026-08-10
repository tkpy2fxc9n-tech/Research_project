# Autoregressive rollout on one held-out trajectory + inference-speed
# benchmarking. Ported from Tests/Model_in_tests/full_rollout_training_conv1d/
# training/code/evaluation.py.
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from ..physics.solver import compute_rest_bias, autoregressive_rollout, run_fd_simulation_general
from ..physics.waves import BCSpec
from ..data.norm import norm_stats_arrays


@dataclass
class RolloutResult:
    U: np.ndarray
    U_reel: np.ndarray
    left_bc: "BCSpec"
    right_bc: "BCSpec"


def run_rollout(model, FIELDS: dict, bc_pairs: list[tuple], rollout_idx: int,
                 input_fields, norm_stats, INPUTS, OUTPUTS, cfg) -> RolloutResult:
    left_bc, right_bc = bc_pairs[rollout_idx]
    U_reel = FIELDS[rollout_idx]

    mu_in, sd_in, mu_out, sd_out = norm_stats_arrays(norm_stats, INPUTS, OUTPUTS)
    rest_bias = compute_rest_bias(model, mu_in, sd_in, mu_out, sd_out, cfg)
    U = autoregressive_rollout(model, U_reel, input_fields, mu_in, sd_in, mu_out, sd_out,
                                rest_bias, left_bc, right_bc, cfg)
    return RolloutResult(U=U, U_reel=U_reel, left_bc=left_bc, right_bc=right_bc)


def chrono(func, n_repeat=15, n_warmup=3):
    for _ in range(n_warmup):
        func()
    durations = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        func()
        durations.append(time.perf_counter() - t0)
    d = np.array(durations)
    return d.mean(), d.std(), float(np.median(d))


@dataclass
class BenchmarkResult:
    fd_time_med: float
    fd_time_std: float
    nn_time_med: float
    nn_time_std: float
    flops_per_call: float
    n_calls: int


def benchmark_inference(model, FIELDS, input_fields, norm_stats, INPUTS, OUTPUTS,
                         rollout: RolloutResult, cfg) -> BenchmarkResult:
    left_bc, right_bc, U_reel = rollout.left_bc, rollout.right_bc, rollout.U_reel
    mu_in, sd_in, mu_out, sd_out = norm_stats_arrays(norm_stats, INPUTS, OUTPUTS)
    rest_bias = compute_rest_bias(model, mu_in, sd_in, mu_out, sd_out, cfg)

    def fd_once():
        return run_fd_simulation_general(left_bc, right_bc, cfg)

    def rollout_once():
        return autoregressive_rollout(model, U_reel, input_fields, mu_in, sd_in, mu_out, sd_out,
                                       rest_bias, left_bc, right_bc, cfg)

    fd_mean, fd_std, fd_med = chrono(fd_once)
    nn_mean, nn_std, nn_med = chrono(rollout_once)

    n_calls = len(range(cfg.M_BACK * cfg.ndt, cfg.Nt - cfg.N_FWD * cfg.ndt + 1, cfg.N_FWD * cfg.ndt))
    n_features = cfg.M_BACK * (2 * cfg.SS + 1) * len(input_fields)

    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        model(torch.zeros((len(cfg.nodes), n_features)))
    flops_per_call = fc.get_total_flops() * n_calls

    return BenchmarkResult(
        fd_time_med=fd_med, fd_time_std=float(fd_std),
        nn_time_med=nn_med, nn_time_std=float(nn_std),
        flops_per_call=flops_per_call, n_calls=n_calls,
    )
