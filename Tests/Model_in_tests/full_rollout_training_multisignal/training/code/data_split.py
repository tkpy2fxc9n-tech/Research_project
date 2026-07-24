# Split by full SIMULATION, not by row -- essential here since we roll out
# entire trajectories (a simulation cannot be split between train and test).
#
# Generalized from the other full_rollout_training projects: simulations
# there are indexed by a dense (A, omega) grid (product(AMPLITUDES,
# PULSATIONS)); here they're indexed by position in a randomly-sampled
# `bc_pairs` list (see commun.sample_bc_pairs), since a dense grid over
# {type, waveform family, params} x 2 sides isn't practical. The split
# mechanics (90/5/5 by simulation, one pinned rollout/visualization case)
# are otherwise unchanged.
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR

sys.path.insert(0, str(COMMUN_DIR))
import commun as C


def split_by_simulation(bc_pairs: list, df: pd.DataFrame, cfg: "C.Config"):
    n_total = len(bc_pairs)
    n_val = max(1, round(0.05 * n_total))
    n_test = max(1, round(0.05 * n_total))
    n_train = n_total - n_val - n_test

    rng = np.random.default_rng(cfg.SPLIT_SEED)
    order = rng.permutation(n_total)
    idx_train = order[:n_train].tolist()
    idx_val = order[n_train:n_train + n_val].tolist()
    idx_test = order[n_train + n_val:].tolist()

    # First test-split index is "the" rollout/visualization case (analogous
    # to pin_rollout_pair in the other projects' data_split.py, just without
    # a grid-center index to pin to).
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


def compute_norm_stats(df: pd.DataFrame, INPUTS, OUTPUTS, cfg: "C.Config") -> pd.DataFrame:
    train_mask = df["split"] == "train"
    cols = INPUTS + OUTPUTS
    norm_stats = pd.DataFrame({
        "mean": df.loc[train_mask, cols].mean(),
        "std": df.loc[train_mask, cols].std(),
    })
    norm_stats["std"] = norm_stats["std"].replace(0, 1)
    return norm_stats
