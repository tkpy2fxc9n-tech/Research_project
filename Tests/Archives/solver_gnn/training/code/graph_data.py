# Génération du dataset au format GNN : historique brut de u par nœud (pas
# le stencil spatial U/Ut/Uxx du MLP -- les voisins arrivent via les arêtes
# du graphe, pas via des colonnes). Réutilise C.run_fd_simulation (même
# simulation FD que Code_comparaison_des_inputs) et le split par simulation
# de full_rollout_training/data_split.py (indispensable : un même snapshot
# ne doit jamais être coupé entre train/val/test).
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR

sys.path.insert(0, str(COMMUN_DIR))
import commun as C


def make_feature_columns(cfg: "C.Config") -> list[str]:
    return [f"u_lag{lag}" for lag in range(cfg.M_BACK)] + ["pos_x", "pos_t", "A_norm", "omega_norm"]


def build_node_features(get_u, n: int, A: float, omega: float, cfg: "C.Config") -> np.ndarray:
    """Vecteur de features par nœud pour le pas n : historique brut de u au
    nœud lui-même (pas de fenêtre spatiale), position, temps, et les
    paramètres de forçage (A, omega) diffusés à tous les nœuds -- façon
    bc_left/bc_right de Brandstetter."""
    nodes = cfg.nodes
    n_features = cfg.M_BACK + 4
    X = np.zeros((len(nodes), n_features), dtype=np.float32)
    col = 0
    for lag in range(cfg.M_BACK):
        X[:, col] = get_u(n - lag * cfg.ndt)[nodes]
        col += 1
    X[:, col] = (nodes - cfg.i_left) / (cfg.Nx - 1)
    col += 1
    X[:, col] = (n * cfg.dt) / cfg.t_end
    col += 1
    X[:, col] = A / cfg.AMP_MAX
    col += 1
    X[:, col] = omega / cfg.OMEGA_MAX
    return X


def _simulate_one_gnn(args):
    A, omega, cfg, INPUTS, OUTPUTS = args
    nodes = cfg.nodes
    u_storage = C.run_fd_simulation(A, omega, cfg)

    n_list = list(range(cfg.M_BACK * cfg.ndt, cfg.Nt - cfg.N_FWD * cfg.ndt + 1))
    X = np.zeros((len(n_list), len(nodes), len(INPUTS)), dtype=np.float32)
    Y = np.zeros((len(n_list), len(nodes), len(OUTPUTS)), dtype=np.float32)
    for i, n in enumerate(n_list):
        X[i] = build_node_features(lambda m, U=u_storage: U[m], n, A, omega, cfg)
        for h in range(1, cfg.N_FWD + 1):
            Y[i, :, h - 1] = u_storage[n + h * cfg.ndt, nodes] - u_storage[n, nodes]

    meta = pd.DataFrame({"A": A, "omega": omega, "n_step": np.repeat(n_list, len(nodes))})
    df_sim = pd.concat([
        meta.reset_index(drop=True),
        pd.DataFrame(X.reshape(-1, len(INPUTS)), columns=INPUTS),
        pd.DataFrame(Y.reshape(-1, len(OUTPUTS)), columns=OUTPUTS),
    ], axis=1)
    return A, omega, u_storage, df_sim, n_list


def build_dataset(cfg: "C.Config", n_workers: int | None = None):
    """Équivalent GNN de C.generate_dataset. Retourne en plus `samples`, la
    liste (A, omega, n_step) dans le même ordre que les blocs de n_nodes
    lignes contiguës de `df` -- indispensable pour retrouver, pendant
    l'entraînement, quel bloc de lignes correspond à quel graphe (snapshot)
    sans avoir à re-parser le dataframe."""
    INPUTS = make_feature_columns(cfg)
    OUTPUTS = C.make_output_columns(cfg)

    grid = list(product(cfg.AMPLITUDES, cfg.PULSATIONS))
    tasks = [(A, omega, cfg, INPUTS, OUTPUTS) for A, omega in grid]
    n_workers = n_workers or min(len(grid), C._n_workers_from_env())

    FIELDS = {}
    frames = []
    samples = []
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for A, omega, u_storage, df_sim, n_list in ex.map(_simulate_one_gnn, tasks):
                FIELDS[(A, omega)] = u_storage
                frames.append(df_sim)
                samples.extend((A, omega, n) for n in n_list)
    else:
        for task in tasks:
            A, omega, u_storage, df_sim, n_list = _simulate_one_gnn(task)
            FIELDS[(A, omega)] = u_storage
            frames.append(df_sim)
            samples.extend((A, omega, n) for n in n_list)

    df = pd.concat(frames, ignore_index=True)
    n_nodes = len(cfg.nodes)
    assert len(df) == len(samples) * n_nodes, "incohérence entre df et samples -- ne pas trier/filtrer df ailleurs"
    return df, FIELDS, INPUTS, OUTPUTS, samples, n_nodes


def split_by_simulation(df: pd.DataFrame, cfg: "C.Config", pin_rollout_pair: bool = True):
    # Identique à full_rollout_training/data_split.py : split par simulation
    # complète (A, omega), pas par ligne -- une trajectoire ne peut pas être
    # coupée entre train/val/test.
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
