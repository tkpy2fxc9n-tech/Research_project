# Split by full SIMULATION (pair (A, omega)), not by row like
# C.split_and_normalize -- essential here since we roll out entire
# trajectories (a simulation cannot be split between train and test).
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR

sys.path.insert(0, str(COMMUN_DIR))
import commun as C


def split_by_simulation(df: pd.DataFrame, cfg: "C.Config", pin_rollout_pair: bool = True):
    grid = list(product(cfg.AMPLITUDES, cfg.PULSATIONS))
    n_total = len(grid)
    n_val = max(1, round(0.05 * n_total))
    n_test = max(1, round(0.05 * n_total))
    n_train = n_total - n_val - n_test

    rng = np.random.default_rng(cfg.SPLIT_SEED)
    order = rng.permutation(n_total)
    shuffled = [grid[i] for i in order]

    if pin_rollout_pair:
        rollout_pair = (cfg.AMPLITUDES[cfg.ROLLOUT_A_IDX], cfg.PULSATIONS[cfg.ROLLOUT_OMEGA_IDX])
        idx_rollout = shuffled.index(rollout_pair)
        idx_last = n_total - 1
        shuffled[idx_rollout], shuffled[idx_last] = shuffled[idx_last], shuffled[idx_rollout]

    pairs_train = shuffled[:n_train]
    pairs_val = shuffled[n_train:n_train + n_val]
    pairs_test = shuffled[n_train + n_val:]

    split_df = pd.DataFrame(
        [(A, omega, "train") for A, omega in pairs_train]
        + [(A, omega, "val") for A, omega in pairs_val]
        + [(A, omega, "test") for A, omega in pairs_test],
        columns=["A", "omega", "split"],
    )
    df = df.merge(split_df, on=["A", "omega"], how="left")

    print("Split distribution (by simulation):")
    for s, pairs in [("train", pairs_train), ("val", pairs_val), ("test", pairs_test)]:
        n = len(pairs)
        print(f"  {s:5s} : {n:>3d} simulations ({100*n/n_total:.1f} %)")

    return df, pairs_train, pairs_val, pairs_test


def compute_norm_stats(df: pd.DataFrame, INPUTS, OUTPUTS, cfg: "C.Config") -> pd.DataFrame:
    train_mask = df["split"] == "train"
    cols = INPUTS + OUTPUTS
    norm_stats = pd.DataFrame({
        "mean": df.loc[train_mask, cols].mean(),
        "std": df.loc[train_mask, cols].std(),
    })
    norm_stats["std"] = norm_stats["std"].replace(0, 1)
    return norm_stats
