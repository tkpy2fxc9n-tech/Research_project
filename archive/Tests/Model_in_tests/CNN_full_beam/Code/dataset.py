# Simulation generation over a list of scenarios (see scenarios.py) and the
# train/val/test split, done by whole SIMULATION rather than by row
# (essential since rollout evaluates entire trajectories, so a simulation
# can't be split between train and test).
#
# Much simpler than full_rollout_training_conv1d's dataset.py: the TBPTT
# training scheme (train.py) works directly on the complete simulations
# (FIELDS), so there's no big flat row-per-(timestep, node) DataFrame to
# build any more. This module only runs the FD simulations (in parallel),
# splits the simulation indices, and computes the normalization stats
# straight from the simulation arrays. extract_pairs() additionally builds
# (history -> true future deltas) pairs on demand -- used only for the
# one-step accuracy check on the test split after training, not for
# training itself.
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from physics import run_fd_simulation_general, build_field_history


def _n_workers_from_env() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return max(1, int(slurm_cpus))
    return os.cpu_count() or 1


def _simulate_one(args):
    idx, bc_left, bc_right, cfg = args
    return idx, run_fd_simulation_general(bc_left, bc_right, cfg)


def generate_simulations(cfg, bc_pairs, n_workers=None) -> dict:
    # Returns FIELDS: dict sim_idx -> u_storage (Nt+1, Ntot).
    tasks = [(idx, bc_left, bc_right, cfg) for idx, (bc_left, bc_right) in enumerate(bc_pairs)]
    n_workers = n_workers or min(len(tasks), _n_workers_from_env())

    FIELDS = {}
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for idx, u_storage in ex.map(_simulate_one, tasks):
                FIELDS[idx] = u_storage
    else:
        for task in tasks:
            idx, u_storage = _simulate_one(task)
            FIELDS[idx] = u_storage
    return FIELDS


def split_by_simulation(n_total: int, cfg):
    n_val = max(1, round(0.05 * n_total))
    n_test = max(1, round(0.05 * n_total))
    n_train = n_total - n_val - n_test

    rng = np.random.default_rng(cfg.SPLIT_SEED)
    order = rng.permutation(n_total)
    idx_train = order[:n_train].tolist()
    idx_val = order[n_train:n_train + n_val].tolist()
    idx_test = order[n_train + n_val:].tolist()

    # First test-split index is "the" rollout/visualization case.
    rollout_idx = idx_test[0]

    print("Split distribution (by simulation):")
    for s, idxs in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        n = len(idxs)
        print(f"  {s:5s} : {n:>3d} simulations ({100*n/n_total:.1f} %)")

    return idx_train, idx_val, idx_test, rollout_idx


def timestep_list(cfg) -> list[int]:
    # Timesteps with a full M_BACK history behind and N_FWD horizons ahead.
    return list(range(cfg.M_BACK * cfg.ndt, cfg.Nt - cfg.N_FWD * cfg.ndt + 1))


def extract_pairs(FIELDS: dict, sim_indices: list[int], cfg):
    # (history -> true future deltas) pairs for the listed simulations:
    # X (n_samples, M_BACK, Nx), Y (n_samples, N_FWD, Nx). One sample = the
    # WHOLE beam at one timestep of one simulation.
    n_list = timestep_list(cfg)
    nodes = cfg.nodes
    Nx = len(nodes)
    n_samples = len(sim_indices) * len(n_list)

    X = np.zeros((n_samples, cfg.M_BACK, Nx), dtype=np.float32)
    Y = np.zeros((n_samples, cfg.N_FWD, Nx), dtype=np.float32)
    row = 0
    for idx in sim_indices:
        u_storage = FIELDS[idx]
        for n in n_list:
            m_list = [n - lag * cfg.ndt for lag in range(cfg.M_BACK)]
            X[row] = build_field_history(m_list, lambda m: u_storage[m], cfg)
            for h in range(1, cfg.N_FWD + 1):
                Y[row, h - 1] = u_storage[n + h * cfg.ndt, nodes] - u_storage[n, nodes]
            row += 1
    return X, Y


def compute_norm_stats(FIELDS: dict, idx_train: list[int], INPUT_CHANNELS, OUTPUT_CHANNELS, cfg) -> pd.DataFrame:
    # One mean/std per input channel (past lag) and per output channel
    # (future horizon), pooled over ALL simulations, timesteps and beam
    # positions of the train split -- NEVER per absolute position, which
    # would break the conv weight-sharing premise (see the plan's point 5).
    X, Y = extract_pairs(FIELDS, idx_train, cfg)
    norm_stats = pd.DataFrame({
        "mean": np.concatenate([X.mean(axis=(0, 2)), Y.mean(axis=(0, 2))]),
        "std": np.concatenate([X.std(axis=(0, 2)), Y.std(axis=(0, 2))]),
    }, index=INPUT_CHANNELS + OUTPUT_CHANNELS)
    norm_stats["std"] = norm_stats["std"].replace(0, 1)
    return norm_stats
