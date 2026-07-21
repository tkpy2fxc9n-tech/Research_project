# Port local des fonctions de commun.py qui dépendent de la forme du bord
# droit imposé (u_right_val) -- afin d'entraîner sur plusieurs types d'onde
# sans modifier commun.py, partagé par 7 dossiers methode_* et 3 autres
# variantes rollout. Même principe que rollout_torch.py/data_split.py :
# reporter localement uniquement ce qui doit changer, réutiliser le reste de
# commun.py tel quel.
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR

sys.path.insert(0, str(COMMUN_DIR))
import commun as C

WAVE_TYPES = ["gaussian_pulse", "sinusoidal"]


def u_right_val_multi(wave_type: str, A: float, omega: float, t: float) -> float:
    if wave_type == "gaussian_pulse":
        # Comportement inchangé, délégué tel quel à commun.py.
        return C.u_right_val(A, omega, t)
    if wave_type == "sinusoidal":
        return A * np.sin(omega * t)
    raise ValueError(f"Type d'onde inconnu : {wave_type!r}")


def run_fd_simulation_multi(wave_type: str, A: float, omega: float, cfg: "C.Config") -> np.ndarray:
    # Copie de C.run_fd_simulation (commun.py) -- seule différence : le
    # forçage au bord droit passe par u_right_val_multi.
    i_left, i_right, Ntot = cfg.i_left, cfg.i_right, cfg.Ntot
    u_storage = np.zeros((cfg.Nt + 1, Ntot))
    u = np.zeros(Ntot)
    u_1 = np.zeros(Ntot)
    for n in range(cfg.Nt):
        t = n * cfg.dt
        u_new = np.zeros(Ntot)
        u_new[i_left:i_right+1] = (
            2.0 * u[i_left:i_right+1] - u_1[i_left:i_right+1]
            + cfg.CFL**2 * (u[i_left-1:i_right] - 2.0*u[i_left:i_right+1] + u[i_left+1:i_right+2])
        )
        u_new[:i_left+1] = 0.0
        u_new[i_right:] = u_right_val_multi(wave_type, A, omega, t + cfg.dt)
        u_1, u = u.copy(), u_new
        u_storage[n+1] = u.copy()
    return u_storage


def autoregressive_rollout_multi(modele, U_reel, input_fields, mu_in, sd_in, mu_out, sd_out,
                                  biais_repos, wave_type: str, A: float, omega: float,
                                  cfg: "C.Config") -> np.ndarray:
    # Copie de C._autoregressive_rollout -- seule différence : le forçage au
    # bord droit passe par u_right_val_multi.
    history_needed = cfg.M_BACK * cfg.ndt
    U = np.zeros((cfg.Nt + 1, cfg.Ntot))
    for m in range(history_needed + 1):
        U[m] = U_reel[m]

    for n in range(history_needed, cfg.Nt - cfg.N_FWD*cfg.ndt + 1, cfg.N_FWD*cfg.ndt):
        m_list = [n - lag*cfg.ndt for lag in range(cfg.M_BACK)]
        X = (C.build_window(m_list, lambda m: U[m], input_fields, cfg) - mu_in) / sd_in
        with torch.no_grad():
            sortie = modele(torch.tensor(X)).numpy()
        deltas = sortie * sd_out + mu_out - biais_repos

        for h in range(1, cfg.N_FWD + 1):
            s = n + h*cfg.ndt
            U[s, cfg.nodes] = U[n, cfg.nodes] + deltas[:, h-1]
            U[s, :cfg.i_left+1] = 0.0
            U[s, cfg.i_right:] = u_right_val_multi(wave_type, A, omega, s*cfg.dt)

            if cfg.SMOOTH_ALPHA > 0:
                j0, j1 = cfg.i_left + 1, cfg.i_right
                lap = U[s, j0-1:j1-1] - 2*U[s, j0:j1] + U[s, j0+1:j1+1]
                U[s, j0:j1] += cfg.SMOOTH_ALPHA * lap

    return U


def _simulate_one_multi(args):
    # Copie de C._simulate_one, avec wave_type en plus.
    wave_type, A, omega, input_fields, cfg, INPUTS, OUTPUTS = args
    nodes = cfg.nodes
    u_storage = run_fd_simulation_multi(wave_type, A, omega, cfg)

    n_list = list(range(cfg.M_BACK*cfg.ndt, cfg.Nt - cfg.N_FWD*cfg.ndt + 1))

    X = np.zeros((len(n_list), len(nodes), len(INPUTS)), dtype=np.float32)
    Y = np.zeros((len(n_list), len(nodes), len(OUTPUTS)), dtype=np.float32)
    for i, n in enumerate(n_list):
        m_list = [n - lag*cfg.ndt for lag in range(cfg.M_BACK)]
        X[i] = C.build_window(m_list, lambda m: u_storage[m], input_fields, cfg)
        for h in range(1, cfg.N_FWD + 1):
            Y[i, :, h-1] = u_storage[n + h*cfg.ndt, nodes] - u_storage[n, nodes]

    meta = pd.DataFrame({
        "wave_type": wave_type, "A": A, "omega": omega,
        "n_step": np.repeat(n_list, len(nodes)),
    })
    df_sim = pd.concat([
        meta.reset_index(drop=True),
        pd.DataFrame(X.reshape(-1, len(INPUTS)), columns=INPUTS),
        pd.DataFrame(Y.reshape(-1, len(OUTPUTS)), columns=OUTPUTS),
    ], axis=1)
    return wave_type, A, omega, u_storage, df_sim


def generate_dataset_multi(input_fields: list[str], wave_types: list[str], cfg: "C.Config",
                            n_workers: int | None = None):
    # Copie de C.generate_dataset, mais la grille de simulations couvre
    # aussi wave_types (FIELDS est alors indexé par (wave_type, A, omega)).
    INPUTS = C.make_feature_columns(input_fields, cfg)
    OUTPUTS = C.make_output_columns(cfg)

    grid = list(product(wave_types, cfg.AMPLITUDES, cfg.PULSATIONS))
    tasks = [(wave_type, A, omega, input_fields, cfg, INPUTS, OUTPUTS) for wave_type, A, omega in grid]
    n_workers = n_workers or min(len(grid), C._n_workers_from_env())

    FIELDS = {}
    frames = []
    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for wave_type, A, omega, u_storage, df_sim in ex.map(_simulate_one_multi, tasks):
                FIELDS[(wave_type, A, omega)] = u_storage
                frames.append(df_sim)
    else:
        for task in tasks:
            wave_type, A, omega, u_storage, df_sim = _simulate_one_multi(task)
            FIELDS[(wave_type, A, omega)] = u_storage
            frames.append(df_sim)

    df = pd.concat(frames, ignore_index=True)
    return df, FIELDS, INPUTS, OUTPUTS


def run_rollout_multi(modele, FIELDS, input_fields, norm_stats, INPUTS, OUTPUTS,
                       wave_type: str, A: float, omega: float, cfg: "C.Config") -> "C.RolloutResult":
    # Copie de C.run_rollout, paramétrée par wave_type -- retourne un
    # C.RolloutResult standard, réutilisable tel quel par toutes les
    # fonctions de plot/animation de commun.py.
    U_reel = FIELDS[(wave_type, A, omega)]

    mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
    sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)
    mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
    sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype(np.float32)

    biais_repos = C._biais_repos(modele, mu_in, sd_in, mu_out, sd_out, cfg)
    U = autoregressive_rollout_multi(modele, U_reel, input_fields, mu_in, sd_in, mu_out, sd_out,
                                      biais_repos, wave_type, A, omega, cfg)
    return C.RolloutResult(U=U, U_reel=U_reel, A=A, omega=omega)
