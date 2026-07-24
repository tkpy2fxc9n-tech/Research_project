# Dataset generation over a list of scenarios (see scenarios.py), each
# either a normal two-end forced BC pair (u0 is None -> dispatches to
# commun.run_fd_simulation_general, unchanged) or a free-evolution scenario
# (u0 is a random initial profile -> dispatches to
# free_evolution.run_fd_simulation_free, the one new physics function this
# project adds). Otherwise a direct copy of commun.py's own
# generate_dataset_general/_simulate_one_general pattern (same
# ProcessPoolExecutor parallelization, same windowing via the unchanged
# make_feature_columns/make_output_columns/build_window) -- kept local only
# because commun's version doesn't know about u0/free evolution.
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR
import free_evolution

sys.path.insert(0, str(COMMUN_DIR))
import commun as C


def _n_workers_from_env() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return max(1, int(slurm_cpus))
    return os.cpu_count() or 1


def _simulate_one_multisignal(args):
    idx, bc_left, bc_right, u0, input_fields, cfg, INPUTS, OUTPUTS = args
    if u0 is None:
        u_storage = C.run_fd_simulation_general(bc_left, bc_right, cfg)
    else:
        u_storage = free_evolution.run_fd_simulation_free(bc_left, bc_right, u0, cfg, C)

    nodes = cfg.nodes
    n_list = list(range(cfg.M_BACK * cfg.ndt, cfg.Nt - cfg.N_FWD * cfg.ndt + 1))

    X = np.zeros((len(n_list), len(nodes), len(INPUTS)), dtype=np.float32)
    Y = np.zeros((len(n_list), len(nodes), len(OUTPUTS)), dtype=np.float32)
    for i, n in enumerate(n_list):
        m_list = [n - lag * cfg.ndt for lag in range(cfg.M_BACK)]
        X[i] = C.build_window(m_list, lambda m: u_storage[m], input_fields, cfg)
        for h in range(1, cfg.N_FWD + 1):
            Y[i, :, h - 1] = u_storage[n + h * cfg.ndt, nodes] - u_storage[n, nodes]

    # n_step (and sim_idx, via broadcasting) must be repeated len(nodes)
    # times: X/Y have one row per (time step, node), but n_list only has one
    # entry per time step. Without the repeat, the concat below misaligns by
    # row count (pandas unions the indices instead of raising), leaving
    # sim_idx/n_step as NaN for all but the first len(n_list) rows -- which
    # silently breaks the sim_idx-based train/val/test merge downstream
    # (compute_norm_stats/make_dataloaders end up training on <1% of rows).
    meta = pd.DataFrame({"sim_idx": idx, "n_step": np.repeat(n_list, len(nodes))})
    df_sim = pd.concat([
        meta.reset_index(drop=True),
        pd.DataFrame(X.reshape(-1, len(INPUTS)), columns=INPUTS),
        pd.DataFrame(Y.reshape(-1, len(OUTPUTS)), columns=OUTPUTS),
    ], axis=1)
    return idx, u_storage, df_sim


def generate_dataset_multisignal(input_fields, cfg, scenarios, n_workers=None):
    INPUTS = C.make_feature_columns(input_fields, cfg)
    OUTPUTS = C.make_output_columns(cfg)

    tasks = [(idx, bc_left, bc_right, u0, input_fields, cfg, INPUTS, OUTPUTS)
             for idx, (bc_left, bc_right, u0) in enumerate(scenarios)]
    n_workers = n_workers or min(len(tasks), _n_workers_from_env())

    FIELDS = {}
    dfs = []
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for idx, u_storage, df_sim in ex.map(_simulate_one_multisignal, tasks):
                FIELDS[idx] = u_storage
                dfs.append(df_sim)
    else:
        for task in tasks:
            idx, u_storage, df_sim = _simulate_one_multisignal(task)
            FIELDS[idx] = u_storage
            dfs.append(df_sim)

    df = pd.concat(dfs, ignore_index=True)
    return df, FIELDS, INPUTS, OUTPUTS
