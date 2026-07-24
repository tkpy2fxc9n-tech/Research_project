# Rollout autorégressif et benchmark pour le WaveGNN. Ne peut pas réutiliser
# C._autoregressive_rollout/C.benchmark_inference tels quels : ces fonctions
# appellent C.build_window en interne (le format stencil qu'on remplace).
# Retourne les mêmes C.RolloutResult/C.BenchmarkResult que commun.py, donc
# tous les plots (plot_rollout_error, plot_smape, make_rollout_animation...)
# restent réutilisables sans modification.
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR
from graph_data import build_node_features

sys.path.insert(0, str(COMMUN_DIR))
import commun as C

# Pas de correction "biais au repos" (C._biais_repos) en v1 : elle suppose
# que TOUTES les colonnes d'entrée sont des champs physiques nuls au repos,
# ce qui n'est plus vrai ici (pos_x/pos_t/A/omega ne sont pas des champs).
# On observe donc le rollout brut du GNN, sans cette correction post-hoc --
# à réintroduire en v2 si un biais systématique apparaît.


def _autoregressive_rollout_gnn(modele, U_reel, mu_in, sd_in, mu_out, sd_out, A, omega, cfg: "C.Config") -> np.ndarray:
    history_needed = cfg.M_BACK * cfg.ndt
    U = np.zeros((cfg.Nt + 1, cfg.Ntot))
    for m in range(history_needed + 1):
        U[m] = U_reel[m]

    for n in range(history_needed, cfg.Nt - cfg.N_FWD * cfg.ndt + 1, cfg.N_FWD * cfg.ndt):
        X = (build_node_features(lambda m, UU=U: UU[m], n, A, omega, cfg) - mu_in) / sd_in
        with torch.no_grad():
            sortie = modele(torch.tensor(X)).numpy()
        deltas = sortie * sd_out + mu_out

        for h in range(1, cfg.N_FWD + 1):
            s = n + h * cfg.ndt
            U[s, cfg.nodes] = U[n, cfg.nodes] + deltas[:, h - 1]
            U[s, :cfg.i_left + 1] = 0.0
            U[s, cfg.i_right:] = C.u_right_val(A, omega, s * cfg.dt)

    return U


def run_rollout_gnn(modele, FIELDS, norm_stats, INPUTS, OUTPUTS, cfg: "C.Config") -> "C.RolloutResult":
    A = cfg.AMPLITUDES[cfg.ROLLOUT_A_IDX]
    omega = cfg.PULSATIONS[cfg.ROLLOUT_OMEGA_IDX]
    U_reel = FIELDS[(A, omega)]

    mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
    sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)
    mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
    sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype(np.float32)

    U = _autoregressive_rollout_gnn(modele, U_reel, mu_in, sd_in, mu_out, sd_out, A, omega, cfg)
    return C.RolloutResult(U=U, U_reel=U_reel, A=A, omega=omega)


def benchmark_gnn(modele, FIELDS, norm_stats, INPUTS, OUTPUTS, rollout: "C.RolloutResult", cfg: "C.Config") -> "C.BenchmarkResult":
    A, omega, U_reel = rollout.A, rollout.omega, rollout.U_reel

    mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
    sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)
    mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
    sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype(np.float32)

    def fd_once():
        return C.run_fd_simulation(A, omega, cfg)

    def rollout_once():
        return _autoregressive_rollout_gnn(modele, U_reel, mu_in, sd_in, mu_out, sd_out, A, omega, cfg)

    fd_mean, fd_std, fd_med = C.chrono(fd_once)
    nn_mean, nn_std, nn_med = C.chrono(rollout_once)

    n_calls = len(range(cfg.M_BACK * cfg.ndt, cfg.Nt - cfg.N_FWD * cfg.ndt + 1, cfg.N_FWD * cfg.ndt))

    try:
        from torch.utils.flop_counter import FlopCounterMode
        n_features = cfg.M_BACK + 4
        with FlopCounterMode(display=False) as fc:
            modele(torch.zeros((len(cfg.nodes), n_features)))
        flops_per_call = fc.get_total_flops() * n_calls
    except Exception as e:
        print(f"FlopCounterMode indisponible pour le GNN ({e}) -- flops_per_call mis à NaN.")
        flops_per_call = float("nan")

    print(f"FD (réel)   : {fd_med*1e3:7.3f} ms")
    print(f"NN (rollout): {nn_med*1e3:7.3f} ms  (±{nn_std*1e3:.3f})")
    print(f"speedup FD/NN = {fd_med/nn_med:.2f}x   (>1 = le réseau est plus rapide)")

    return C.BenchmarkResult(
        fd_time_med=fd_med, fd_time_std=float(fd_std),
        nn_time_med=nn_med, nn_time_std=float(nn_std),
        flops_per_call=flops_per_call, n_calls=n_calls,
    )
