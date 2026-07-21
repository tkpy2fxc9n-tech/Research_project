# Split par SIMULATION complète (couple (A, omega)), pas par ligne comme
# C.split_and_normalize -- indispensable ici puisqu'on déroule des
# trajectoires entières (une simulation ne peut pas être coupée entre train
# et test).
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR
from wave_forcing import WAVE_TYPES

sys.path.insert(0, str(COMMUN_DIR))
import commun as C


def split_by_simulation(df: pd.DataFrame, cfg: "C.Config", pin_rollout_pair: bool = True):
    grid = list(product(WAVE_TYPES, cfg.AMPLITUDES, cfg.PULSATIONS))
    n_total = len(grid)
    n_val = max(1, round(0.05 * n_total))
    n_test = max(1, round(0.05 * n_total))
    n_train = n_total - n_val - n_test

    rng = np.random.default_rng(cfg.SPLIT_SEED)
    order = rng.permutation(n_total)
    shuffled = [grid[i] for i in order]

    if pin_rollout_pair:
        # Une paire (A, omega) de référence est figée dans le test set pour
        # CHAQUE wave_type, afin qu'un rollout de référence existe pour les
        # deux types après entraînement (cf. run_rollout_multi).
        rollout_A = cfg.AMPLITUDES[cfg.ROLLOUT_A_IDX]
        rollout_omega = cfg.PULSATIONS[cfg.ROLLOUT_OMEGA_IDX]
        for offset, wave_type in enumerate(WAVE_TYPES):
            rollout_triple = (wave_type, rollout_A, rollout_omega)
            idx_rollout = shuffled.index(rollout_triple)
            idx_target = n_total - 1 - offset
            shuffled[idx_rollout], shuffled[idx_target] = shuffled[idx_target], shuffled[idx_rollout]

    pairs_train = shuffled[:n_train]
    pairs_val = shuffled[n_train:n_train + n_val]
    pairs_test = shuffled[n_train + n_val:]

    split_df = pd.DataFrame(
        [(wt, A, omega, "train") for wt, A, omega in pairs_train]
        + [(wt, A, omega, "val") for wt, A, omega in pairs_val]
        + [(wt, A, omega, "test") for wt, A, omega in pairs_test],
        columns=["wave_type", "A", "omega", "split"],
    )
    df = df.merge(split_df, on=["wave_type", "A", "omega"], how="left")

    print("Distribution du split (par simulation) :")
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
