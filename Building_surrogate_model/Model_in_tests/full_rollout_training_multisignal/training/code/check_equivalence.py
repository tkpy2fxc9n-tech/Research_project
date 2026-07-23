# Sanity check to run BEFORE any training: verifies that the torch port
# (rollout_torch.py, copied verbatim from full_rollout_training_general_bc)
# faithfully reproduces the numpy physics already validated in commun.py,
# over a few hops and a model with fixed random weights, for every signal
# family this project trains on (including the new fourier/chirp/shock/
# filtered_random ones, and the free-evolution random-initial-state case).
# Does NOT test the gradient (only the values) -- the goal is to catch a
# transcription error before investing compute time in a training run.
#
# Uses the real CONFIG_OVERRIDES from main.py (M_BACK=4, N_FWD=4,
# HIDDEN_SIZES=(512,256,64)) rather than commun.Config()'s plain defaults,
# so this actually exercises the hyperparameters training will use.
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR
from rollout_torch import build_window_torch, reconstruct_torch_general
from main import CONFIG_OVERRIDES, INPUT_FIELDS
import waveforms
import free_evolution

sys.path.insert(0, str(COMMUN_DIR))
import commun as C

N_HOPS_TEST = 2
TOLERANCE = 1e-4
# Note on N_HOPS_TEST: with an UNTRAINED, random-weight network and a free
# (Neumann) end, nothing external pins the state each hop the way Dirichlet
# does, so tiny float32-vs-float64 rounding-order differences between the
# numpy and torch paths can compound quickly across hops through the
# network's own (arbitrary, not physically meaningful at random weights)
# dynamics -- 2 hops is enough to check the physics is wired correctly,
# without chasing precision a real, trained network's rollout doesn't need.


def build_test_cases(cfg):
    # Built from the actual samplers (fixed seed, for reproducibility)
    # instead of hand-typed params, so they always stay within valid ranges.
    rng = np.random.default_rng(7)
    fourier_p = waveforms.sample_fourier_params(rng, cfg)
    chirp_p = waveforms.sample_chirp_params(rng, cfg)
    shock_p = waveforms.sample_shock_params(rng, cfg)
    filtered_p = waveforms.sample_filtered_random_params(rng, cfg)
    sinusoid_p = C.sample_sinusoid_params(rng, cfg)
    gaussian_p = C.sample_gaussian_params(rng, cfg)

    return [
        ("dirichlet-dirichlet (both fixed, Gaussian push)",
         ("dirichlet", "rest", {}), ("dirichlet", "gaussian", gaussian_p)),
        ("neumann-neumann (both free, sinusoidal push)",
         ("neumann", "sinusoid", sinusoid_p), ("neumann", "rest", {})),
        ("fourier push, both free",
         ("neumann", "fourier", fourier_p), ("neumann", "rest", {})),
        ("chirp push (fixed left, swept right)",
         ("dirichlet", "rest", {}), ("dirichlet", "chirp", chirp_p)),
        ("smoothed shock (free left, fixed right)",
         ("neumann", "shock", shock_p), ("dirichlet", "rest", {})),
        ("filtered random history, both free",
         ("neumann", "filtered_random", filtered_p), ("neumann", "rest", {})),
    ]


def run_case(cfg, modele, input_fields, mu_in, sd_in, mu_out, sd_out, biais_repos,
             bc_left, bc_right, U_reel=None):
    if U_reel is None:
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
    cfg = C.Config(**CONFIG_OVERRIDES)
    C.set_seeds(cfg)
    waveforms.register(C)

    input_fields = INPUT_FIELDS
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
    for name, bc_left, bc_right in build_test_cases(cfg):
        gap = run_case(cfg, modele, input_fields, mu_in, sd_in, mu_out, sd_out, biais_repos, bc_left, bc_right)
        status = "OK" if gap < TOLERANCE else "FAILED"
        if gap >= TOLERANCE:
            all_ok = False
        print(f"[{status}] {name:55s} max abs gap = {gap:.3e}  (tolerance {TOLERANCE:.0e})")

    # Dedicated free-evolution case: random smooth initial state, both ends
    # at rest (no push) -- exercises free_evolution.run_fd_simulation_free,
    # the one genuinely new physics function this project adds.
    rng = np.random.default_rng(123)
    u0 = free_evolution.sample_random_ic(rng, cfg)
    rest = ("dirichlet", "rest", {"ic": "random"})
    U_reel_free = free_evolution.run_fd_simulation_free(rest, rest, u0, cfg, C)
    gap = run_case(cfg, modele, input_fields, mu_in, sd_in, mu_out, sd_out, biais_repos,
                    rest, rest, U_reel=U_reel_free)
    status = "OK" if gap < TOLERANCE else "FAILED"
    if gap >= TOLERANCE:
        all_ok = False
    name = "free evolution (random initial state, no push)"
    print(f"[{status}] {name:55s} max abs gap = {gap:.3e}  (tolerance {TOLERANCE:.0e})")

    if all_ok:
        print("\nAll cases OK -- rollout_torch.py's generalized reconstruction matches the numpy physics "
              "for all 7 signal families.")
    else:
        print("\nFAILED -- at least one case is beyond tolerance, check the new physics before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
