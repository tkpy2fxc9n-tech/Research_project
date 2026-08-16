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


# FLOPs per interior grid point per leapfrog step, counted directly off
# run_fd_simulation_general's update line (+, -, * = 1 FLOP each, same
# convention torch.utils.flop_counter.FlopCounterMode uses for the NN side
# below -- so the two counts are comparable): `2.0*u` appears twice in that
# expression (once standalone, once inside the Laplacian term) and numpy
# evaluates it twice as written, so this counts the actual work done, not a
# hand-optimized minimum.
#   2.0*u                         -> 1 mul
#   (2u) - u_1                    -> 1 sub
#   2.0*u (again, inside laplacian) -> 1 mul
#   u_left - (2u)                 -> 1 sub
#   (u_left-2u) + u_right         -> 1 add
#   CFL^2 * laplacian             -> 1 mul
#   term1 + CFL^2*laplacian       -> 1 add
FD_FLOPS_PER_POINT_STEP = 7


@dataclass
class BenchmarkResult:
    fd_time_med: float
    fd_time_std: float
    nn_time_med: float
    nn_time_std: float
    fd_flops: float
    nn_flops: float
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
    nn_flops = fc.get_total_flops() * n_calls

    # Analytical, not measured: numpy/BLAS don't expose a FLOP counter for
    # plain elementwise array ops the way torch's FlopCounterMode does for
    # the NN, so this is FD_FLOPS_PER_POINT_STEP x (interior points) x
    # (timesteps) -- one full run_fd_simulation_general call, matching what
    # fd_time_med/fd_time_std just benchmarked.
    n_interior_points = cfg.i_right - cfg.i_left + 1
    fd_flops = FD_FLOPS_PER_POINT_STEP * n_interior_points * cfg.Nt

    return BenchmarkResult(
        fd_time_med=fd_med, fd_time_std=float(fd_std),
        nn_time_med=nn_med, nn_time_std=float(nn_std),
        fd_flops=fd_flops, nn_flops=nn_flops, n_calls=n_calls,
    )
