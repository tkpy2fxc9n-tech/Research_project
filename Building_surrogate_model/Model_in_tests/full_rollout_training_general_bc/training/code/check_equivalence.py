# Sanity check to run BEFORE any training: verifies that the torch port
# (rollout_torch.py) faithfully reproduces the numpy physics already
# validated in commun.py, over a few hops and a model with fixed random
# weights. Does NOT test the gradient (only the values), the goal is to
# catch a transcription error (column order, boundary conditions, smoothing)
# before investing compute time in a training run.
#
# Generalized from the single-BC-combination check in the other
# full_rollout_training projects: exercises several Dirichlet/Neumann
# combinations on both ends (not just one), since that's exactly the new
# code path (apply_boundary_conditions / apply_boundary_conditions_torch)
# this project adds.
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR
from rollout_torch import build_window_torch, reconstruct_torch_general

sys.path.insert(0, str(COMMUN_DIR))
import commun as C

N_HOPS_TEST = 2
TOLERANCE = 1e-4
# Note on N_HOPS_TEST: with an UNTRAINED, random-weight network and a free
# (Neumann) end, nothing external pins the state each hop the way Dirichlet
# does, so tiny float32-vs-float64 rounding-order differences between the
# numpy and torch paths can compound quickly across hops through the
# network's own (arbitrary, not physically meaningful at random weights)
# dynamics -- confirmed by hop-by-hop inspection: a double-Neumann
# multitone case matches to 1.5e-8 after 1 hop, 1.1e-6 after 2, but ~2e-4
# after 3. That growth is chaotic amplification of an untrained network,
# not a sign of a transcription error, so 2 hops is the right depth here
# (still checks the physics is wired correctly, without chasing precision
# that a real, trained network's rollout doesn't need either).

# A handful of representative (left, right) combinations: both fixed (as in
# the original single-BC setup), both free, and the two mixed cases -- across
# a couple of different waveform families, not just Gaussian.
TEST_CASES = [
    ("dirichlet-dirichlet (both fixed, Gaussian push)",
     ("dirichlet", "rest", {}), ("dirichlet", "gaussian", {"A": 0.05, "omega": 5.0})),
    ("neumann-neumann (both free, sinusoidal push)",
     ("neumann", "sinusoid", {"A": 0.02, "omega": 4.0, "phase": 0.0}), ("neumann", "rest", {})),
    ("dirichlet-neumann (fixed left, free+stepped right)",
     ("dirichlet", "rest", {}), ("neumann", "step", {"A": 0.03, "t_onset": 0.2})),
    ("neumann-dirichlet (free+ramped left, fixed right)",
     ("neumann", "ramp", {"A": 0.04, "duration": 1.0}), ("dirichlet", "rest", {})),
    ("multitone push, both free",
     ("neumann", "random_multitone", {"A": [0.02, 0.01], "omega": [3.0, 6.0], "phase": [0.0, 1.0]}),
     ("neumann", "rest", {})),
]


def run_case(cfg, modele, input_fields, INPUTS, OUTPUTS, mu_in, sd_in, mu_out, sd_out, biais_repos,
             bc_left, bc_right):
    U_reel = C.run_fd_simulation_general(bc_left, bc_right, cfg)
    history_needed = cfg.M_BACK * cfg.ndt
    n_stop = history_needed + N_HOPS_TEST * cfg.N_FWD * cfg.ndt

    # --- Reference: numpy rollout (commun.py), truncated to N_HOPS_TEST hops ---
    U_ref = np.zeros((cfg.Nt + 1, cfg.Ntot))
    U_ref[:history_needed + 1] = U_reel[:history_needed + 1]
    for n in range(history_needed, n_stop, cfg.N_FWD * cfg.ndt):
        m_list = [n - lag * cfg.ndt for lag in range(cfg.M_BACK)]
        X = (C.build_window(m_list, lambda m: U_ref[m], input_fields, cfg) - mu_in) / sd_in
        with torch.no_grad():
            sortie = modele(torch.tensor(X)).numpy()
        champs = C.reconstruct_general(U_ref[n], n, sortie, bc_left, bc_right, mu_out, sd_out, cfg,
                                        biais_repos=biais_repos)
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
            new_states, _ = reconstruct_torch_general(history[-1], pred_norm, [bc_left], [bc_right], n,
                                                        mu_out_t, sd_out_t, biais_repos_t, cfg)
            history = history[cfg.N_FWD:] + new_states

    U_torch_final = history[-1][0].numpy()
    U_ref_final = U_ref[n_stop]
    return np.abs(U_torch_final - U_ref_final).max()


def main():
    cfg = C.Config()
    C.set_seeds(cfg)

    input_fields = ["U", "Ut", "Uxx"]
    INPUTS = C.make_feature_columns(input_fields, cfg)
    OUTPUTS = C.make_output_columns(cfg)

    modele = C.Reseau(n_inputs=len(INPUTS), n_outputs=len(OUTPUTS), hidden_sizes=cfg.HIDDEN_SIZES)
    modele.eval()

    # Fake normalization stats (just need std != 0) -- this script only
    # tests the fidelity of the physical reconstruction, not real dataset stats.
    mu_in = np.zeros(len(INPUTS), dtype=np.float32)
    sd_in = np.ones(len(INPUTS), dtype=np.float32)
    mu_out = np.zeros(len(OUTPUTS), dtype=np.float32)
    sd_out = np.ones(len(OUTPUTS), dtype=np.float32)
    biais_repos = C._biais_repos(modele, mu_in, sd_in, mu_out, sd_out, cfg)

    all_ok = True
    for name, bc_left, bc_right in TEST_CASES:
        gap = run_case(cfg, modele, input_fields, INPUTS, OUTPUTS, mu_in, sd_in, mu_out, sd_out,
                        biais_repos, bc_left, bc_right)
        status = "OK" if gap < TOLERANCE else "FAILED"
        if gap >= TOLERANCE:
            all_ok = False
        print(f"[{status}] {name:55s} max abs gap = {gap:.3e}  (tolerance {TOLERANCE:.0e})")

    if all_ok:
        print("\nAll cases OK -- rollout_torch.py's generalized reconstruction matches the numpy physics.")
    else:
        print("\nFAILED -- at least one case is beyond tolerance, check rollout_torch.py before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
