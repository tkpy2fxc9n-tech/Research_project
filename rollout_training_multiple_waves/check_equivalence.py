# Sanity check à lancer AVANT tout entraînement : vérifie que le portage
# torch (rollout_torch.py) reproduit fidèlement la physique numpy déjà
# validée dans commun.py, sur quelques hops et un modèle à poids aléatoires
# fixes. Ne teste PAS le gradient (juste les valeurs), le but est de
# détecter une erreur de transcription (ordre de colonnes, conditions aux
# limites, lissage) avant d'investir du temps de calcul dans un entraînement.
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR
from rollout_torch import build_window_torch, reconstruct_torch
from wave_forcing import WAVE_TYPES, run_fd_simulation_multi, u_right_val_multi

sys.path.insert(0, str(COMMUN_DIR))
import commun as C

N_HOPS_TEST = 3
TOLERANCE = 1e-4


def check_one_wave_type(wave_type: str, cfg, modele, input_fields, INPUTS, OUTPUTS,
                         mu_in, sd_in, mu_out, sd_out, biais_repos) -> float:
    # Compare, pour un wave_type donné, le rollout numpy de référence
    # (reconstruction manuelle via u_right_val_multi, même principe que
    # C.reconstruct mais paramétrée par wave_type) au portage torch
    # (rollout_torch.reconstruct_torch avec wave_type_list), sur N_HOPS_TEST
    # hops -- pour être sûr que le chemin utilisé par la boucle
    # d'entraînement différentiable est physiquement fidèle avant de lancer
    # un run coûteux.
    A, omega = cfg.AMPLITUDES[0], cfg.PULSATIONS[0]
    U_reel = run_fd_simulation_multi(wave_type, A, omega, cfg)

    history_needed = cfg.M_BACK * cfg.ndt
    n_stop = history_needed + N_HOPS_TEST * cfg.N_FWD * cfg.ndt

    # --- Référence : reconstruction manuelle des champs, tronquée à N_HOPS_TEST hops ---
    nodes = cfg.nodes
    U_ref = np.zeros((cfg.Nt + 1, cfg.Ntot))
    U_ref[:history_needed + 1] = U_reel[:history_needed + 1]
    for n in range(history_needed, n_stop, cfg.N_FWD * cfg.ndt):
        m_list = [n - lag * cfg.ndt for lag in range(cfg.M_BACK)]
        X = (C.build_window(m_list, lambda m: U_ref[m], input_fields, cfg) - mu_in) / sd_in
        with torch.no_grad():
            sortie = modele(torch.tensor(X)).numpy()
        deltas = sortie * sd_out + mu_out - biais_repos
        for h in range(1, cfg.N_FWD + 1):
            s = n + h * cfg.ndt
            u = np.zeros(cfg.Ntot)
            u[nodes] = U_ref[n, nodes] + deltas[:, h - 1]
            u[:cfg.i_left + 1] = 0.0
            u[cfg.i_right:] = u_right_val_multi(wave_type, A, omega, s * cfg.dt)
            if cfg.SMOOTH_ALPHA > 0:
                j0, j1 = cfg.i_left + 1, cfg.i_right
                lap = u[j0 - 1:j1 - 1] - 2 * u[j0:j1] + u[j0 + 1:j1 + 1]
                u[j0:j1] += cfg.SMOOTH_ALPHA * lap
            U_ref[s] = u

    # --- Version torch (rollout_torch.py), mêmes hops, groupe de taille 1 ---
    mu_in_t, sd_in_t = torch.tensor(mu_in), torch.tensor(sd_in)
    mu_out_t, sd_out_t = torch.tensor(mu_out), torch.tensor(sd_out)
    biais_repos_t = torch.tensor(biais_repos)

    history = [torch.tensor(U_reel[history_needed - lag * cfg.ndt][None, :], dtype=torch.float32)
               for lag in range(cfg.M_BACK, -1, -1)]

    with torch.no_grad():
        for n in range(history_needed, n_stop, cfg.N_FWD * cfg.ndt):
            X = (build_window_torch(history, input_fields, cfg) - mu_in_t) / sd_in_t
            pred_norm = modele(X)
            new_states, _ = reconstruct_torch(history[-1], pred_norm, [wave_type], [A], [omega], n,
                                               mu_out_t, sd_out_t, biais_repos_t, cfg)
            history = history[cfg.N_FWD:] + new_states

    U_torch_final = history[-1][0].numpy()
    U_ref_final = U_ref[n_stop]

    diff = np.abs(U_torch_final - U_ref_final)
    return float(diff.max())


def main():
    cfg = C.Config()
    C.set_seeds(cfg)

    input_fields = ["U", "Ut", "Uxx"]
    INPUTS = C.make_feature_columns(input_fields, cfg)
    OUTPUTS = C.make_output_columns(cfg)

    modele = C.Reseau(n_inputs=len(INPUTS), n_outputs=len(OUTPUTS), hidden_sizes=cfg.HIDDEN_SIZES)
    modele.eval()

    # Stats de normalisation factices (juste besoin de std != 0) -- ce script
    # ne teste que la fidélité de la reconstruction physique, pas les vraies
    # statistiques du dataset.
    mu_in = np.zeros(len(INPUTS), dtype=np.float32)
    sd_in = np.ones(len(INPUTS), dtype=np.float32)
    mu_out = np.zeros(len(OUTPUTS), dtype=np.float32)
    sd_out = np.ones(len(OUTPUTS), dtype=np.float32)
    biais_repos = C._biais_repos(modele, mu_in, sd_in, mu_out, sd_out, cfg)

    all_ok = True
    for wave_type in WAVE_TYPES:
        max_diff = check_one_wave_type(wave_type, cfg, modele, input_fields, INPUTS, OUTPUTS,
                                        mu_in, sd_in, mu_out, sd_out, biais_repos)
        ok = max_diff < TOLERANCE
        all_ok &= ok
        status = "OK" if ok else "ÉCHEC"
        print(f"[{wave_type:12s}] écart max absolu après {N_HOPS_TEST} hops : {max_diff:.3e} "
              f"(tolérance {TOLERANCE:.0e}) -- {status}")

    if all_ok:
        print("OK -- rollout_torch.py est équivalent au rollout numpy pour tous les wave_types.")
    else:
        print("ÉCHEC -- écart au-delà de la tolérance pour au moins un wave_type, "
              "vérifier wave_forcing.py/rollout_torch.py avant d'entraîner.")
        sys.exit(1)


if __name__ == "__main__":
    main()
