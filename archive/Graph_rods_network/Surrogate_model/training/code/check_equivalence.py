# Sanity check to run BEFORE any training: verifies that the torch port
# (rollout_torch.py) faithfully reproduces the numpy physics already
# validated in commun.py, over a few hops and a model with fixed random
# weights. Does NOT test the gradient (only the values), the goal is to
# catch a transcription error (column order, boundary conditions,
# smoothing) before investing compute time in a training run.
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR
from rollout_torch import build_window_torch, reconstruct_torch

sys.path.insert(0, str(COMMUN_DIR))
import commun as C

N_HOPS_TEST = 3
TOLERANCE = 1e-4


def main():
    cfg = C.Config()
    C.set_seeds(cfg)

    input_fields = ["U", "Ut", "Uxx"]
    INPUTS = C.make_feature_columns(input_fields, cfg)
    OUTPUTS = C.make_output_columns(cfg)

    A, omega = cfg.AMPLITUDES[0], cfg.PULSATIONS[0]
    U_reel = C.run_fd_simulation(A, omega, cfg)

    modele = C.Reseau(n_inputs=len(INPUTS), n_outputs=len(OUTPUTS), hidden_sizes=cfg.HIDDEN_SIZES)
    modele.eval()

    # Fake normalization stats (just need std != 0) -- this script only
    # tests the fidelity of the physical reconstruction, not the real
    # dataset statistics.
    mu_in = np.zeros(len(INPUTS), dtype=np.float32)
    sd_in = np.ones(len(INPUTS), dtype=np.float32)
    mu_out = np.zeros(len(OUTPUTS), dtype=np.float32)
    sd_out = np.ones(len(OUTPUTS), dtype=np.float32)
    biais_repos = C._biais_repos(modele, mu_in, sd_in, mu_out, sd_out, cfg)

    history_needed = cfg.M_BACK * cfg.ndt
    n_stop = history_needed + N_HOPS_TEST * cfg.N_FWD * cfg.ndt

    # --- Reference: existing numpy rollout (commun.py), truncated to N_HOPS_TEST hops ---
    U_ref = np.zeros((cfg.Nt + 1, cfg.Ntot))
    U_ref[:history_needed + 1] = U_reel[:history_needed + 1]
    for n in range(history_needed, n_stop, cfg.N_FWD * cfg.ndt):
        m_list = [n - lag * cfg.ndt for lag in range(cfg.M_BACK)]
        X = (C.build_window(m_list, lambda m: U_ref[m], input_fields, cfg) - mu_in) / sd_in
        with torch.no_grad():
            sortie = modele(torch.tensor(X)).numpy()
        champs = C.reconstruct(U_ref[n], n, sortie, A, omega, mu_out, sd_out, cfg, biais_repos=biais_repos)
        for s, u in champs.items():
            U_ref[s] = u

    # --- Torch version (rollout_torch.py), same hops, group of size 1 ---
    mu_in_t, sd_in_t = torch.tensor(mu_in), torch.tensor(sd_in)
    mu_out_t, sd_out_t = torch.tensor(mu_out), torch.tensor(sd_out)
    biais_repos_t = torch.tensor(biais_repos)

    history = [torch.tensor(U_reel[history_needed - lag * cfg.ndt][None, :], dtype=torch.float32)
               for lag in range(cfg.M_BACK, -1, -1)]

    with torch.no_grad():
        for n in range(history_needed, n_stop, cfg.N_FWD * cfg.ndt):
            X = (build_window_torch(history, input_fields, cfg) - mu_in_t) / sd_in_t
            pred_norm = modele(X)
            new_states, _ = reconstruct_torch(history[-1], pred_norm, [A], [omega], n,
                                               mu_out_t, sd_out_t, biais_repos_t, cfg)
            history = history[cfg.N_FWD:] + new_states

    U_torch_final = history[-1][0].numpy()
    U_ref_final = U_ref[n_stop]

    diff = np.abs(U_torch_final - U_ref_final)
    print(f"Max absolute gap after {N_HOPS_TEST} hops: {diff.max():.3e}  (tolerance {TOLERANCE:.0e})")

    if diff.max() < TOLERANCE:
        print("OK -- rollout_torch.py is equivalent to the numpy rollout in commun.py.")
    else:
        print("FAILED -- gap beyond tolerance, check rollout_torch.py before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
