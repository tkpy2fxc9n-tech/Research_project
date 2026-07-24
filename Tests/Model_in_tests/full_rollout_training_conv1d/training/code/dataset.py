# Dataset generation over a list of scenarios (see scenarios.py) -- each
# either a normal two-end forced BC pair (u0 is None -> run_fd_simulation_general)
# or a free-evolution scenario (u0 is a random initial profile ->
# run_fd_simulation_free) -- and the train/val/test split, done by whole
# SIMULATION rather than by row (essential since rollout evaluates entire
# trajectories, so a simulation can't be split between train and test).
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from physics import run_fd_simulation_general, run_fd_simulation_free, build_window, make_feature_columns, make_output_columns


def _n_workers_from_env() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return max(1, int(slurm_cpus))
    return os.cpu_count() or 1


def _simulate_one(args):
    idx, bc_left, bc_right, u0, input_fields, cfg, INPUTS, OUTPUTS = args
    if u0 is None:
        u_storage = run_fd_simulation_general(bc_left, bc_right, cfg)
    else:
        u_storage = run_fd_simulation_free(bc_left, bc_right, u0, cfg)

    nodes = cfg.nodes
    n_list = list(range(cfg.M_BACK * cfg.ndt, cfg.Nt - cfg.N_FWD * cfg.ndt + 1))

    X = np.zeros((len(n_list), len(nodes), len(INPUTS)), dtype=np.float32)
    Y = np.zeros((len(n_list), len(nodes), len(OUTPUTS)), dtype=np.float32)
    for i, n in enumerate(n_list):
        m_list = [n - lag * cfg.ndt for lag in range(cfg.M_BACK)]
        X[i] = build_window(m_list, lambda m: u_storage[m], input_fields, cfg)
        for h in range(1, cfg.N_FWD + 1):
            Y[i, :, h - 1] = u_storage[n + h * cfg.ndt, nodes] - u_storage[n, nodes]

    # n_step (and sim_idx, via broadcasting) must be repeated len(nodes)
    # times: X/Y have one row per (time step, node), but n_list only has one
    # entry per time step. Without the repeat, concat below misaligns by
    # row count (pandas unions the indices instead of raising), leaving
    # sim_idx/n_step as NaN for all but the first len(n_list) rows -- which
    # silently breaks the sim_idx-based train/val/test merge downstream.
    meta = pd.DataFrame({"sim_idx": idx, "n_step": np.repeat(n_list, len(nodes))})
    df_sim = pd.concat([
        meta.reset_index(drop=True),
        pd.DataFrame(X.reshape(-1, len(INPUTS)), columns=INPUTS),
        pd.DataFrame(Y.reshape(-1, len(OUTPUTS)), columns=OUTPUTS),
    ], axis=1)
    return idx, u_storage, df_sim


def generate_dataset_multisignal(input_fields, cfg, scenarios, n_workers=None):
    INPUTS = make_feature_columns(input_fields, cfg)
    OUTPUTS = make_output_columns(cfg)

    tasks = [(idx, bc_left, bc_right, u0, input_fields, cfg, INPUTS, OUTPUTS)
             for idx, (bc_left, bc_right, u0) in enumerate(scenarios)]
    n_workers = n_workers or min(len(tasks), _n_workers_from_env())

    FIELDS = {}
    dfs = []
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for idx, u_storage, df_sim in ex.map(_simulate_one, tasks):
                FIELDS[idx] = u_storage
                dfs.append(df_sim)
    else:
        for task in tasks:
            idx, u_storage, df_sim = _simulate_one(task)
            FIELDS[idx] = u_storage
            dfs.append(df_sim)

    df = pd.concat(dfs, ignore_index=True)
    return df, FIELDS, INPUTS, OUTPUTS


def split_by_simulation(bc_pairs: list, df: pd.DataFrame, cfg):
    n_total = len(bc_pairs)
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

    split_df = pd.DataFrame(
        [(i, "train") for i in idx_train]
        + [(i, "val") for i in idx_val]
        + [(i, "test") for i in idx_test],
        columns=["sim_idx", "split"],
    )
    df = df.merge(split_df, on="sim_idx", how="left")

    print("Split distribution (by simulation):")
    for s, idxs in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        n = len(idxs)
        print(f"  {s:5s} : {n:>3d} simulations ({100*n/n_total:.1f} %)")

    return df, idx_train, idx_val, idx_test, rollout_idx


def compute_norm_stats(df: pd.DataFrame, INPUTS, OUTPUTS, cfg) -> pd.DataFrame:
    train_mask = df["split"] == "train"
    cols = INPUTS + OUTPUTS
    norm_stats = pd.DataFrame({
        "mean": df.loc[train_mask, cols].mean(),
        "std": df.loc[train_mask, cols].std(),
    })
    norm_stats["std"] = norm_stats["std"].replace(0, 1)
    return norm_stats
