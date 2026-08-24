#!/usr/bin/env python3
# Sanity check to run BEFORE any training: verifies that the differentiable
# torch port (training/losses.py) faithfully reproduces the numpy physics
# (physics/solver.py, physics/waves.py) over a few rollout hops, for every
# signal family, with a CNN of fixed random weights. Does NOT test the
# gradient (only the values) -- the goal is to catch a transcription error
# before spending compute on a training run. Ported from
# Tests/Model_in_tests/full_rollout_training_conv1d/training/code/
# check_equivalence.py (the most complete of the three source copies).
#
# Usage: python checks/check_equivalence.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beamsurrogate.config import Config, set_seeds
from beamsurrogate.physics import waves
from beamsurrogate.physics.solver import (run_fd_simulation_general, run_fd_simulation_free,
                                           reconstruct_general, compute_rest_bias)
from beamsurrogate.data.windows import build_window, make_feature_columns, make_output_columns
from beamsurrogate.training.losses import build_window_torch, reconstruct_torch_general
from beamsurrogate.models.cnn import ConvNet

N_HOPS_TEST = 2
TOLERANCE = 1e-4
INPUT_FIELDS = ["U"]


def sample_random_ic(rng, cfg) -> np.ndarray:
    # A handful of low-order sine modes on [0, L] -- naturally zero at both
    # ends, smooth. Ported from Tests/.../full_rollout_training_conv1d/
    # training/code/scenarios.py's sample_random_ic (only user of it: the
    # free-evolution case below).
    n_modes = int(rng.integers(2, 6))
    modes = rng.integers(1, 6, size=n_modes)
    amps = rng.uniform(-1.0, 1.0, size=n_modes)
    x = np.linspace(0.0, cfg.L, cfg.Nx)
    profile = np.zeros(cfg.Nx)
    for a, k in zip(amps, modes):
        profile += a * np.sin(k * np.pi * x / cfg.L)
    peak = np.abs(profile).max()
    A_total = float(rng.uniform(cfg.AMP_MIN, cfg.AMP_MAX) * 3.0)
    if peak > 1e-12:
        profile *= A_total / peak
    return profile.astype(np.float64)


def build_test_cases(cfg):
    rng = np.random.default_rng(7)
    fourier_p = waves.sample_fourier_params(rng, cfg)
    chirp_p = waves.sample_chirp_params(rng, cfg)
    shock_p = waves.sample_shock_params(rng, cfg)
    filtered_p = waves.sample_filtered_random_params(rng, cfg)
    sinusoid_p = waves.sample_sinusoid_params(rng, cfg)
    gaussian_p = waves.sample_gaussian_params(rng, cfg)

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


def run_case(cfg, model, input_fields, mu_in, sd_in, mu_out, sd_out, rest_bias,
             bc_left, bc_right, U_reel=None):
    if U_reel is None:
        U_reel = run_fd_simulation_general(bc_left, bc_right, cfg)
    history_needed = cfg.M_BACK * cfg.ndt
    n_stop = history_needed + N_HOPS_TEST * cfg.N_FWD * cfg.ndt

    U_ref = np.zeros((cfg.Nt + 1, cfg.Ntot))
    U_ref[:history_needed + 1] = U_reel[:history_needed + 1]
    for n in range(history_needed, n_stop, cfg.N_FWD * cfg.ndt):
        m_list = [n - lag * cfg.ndt for lag in range(cfg.M_BACK)]
        X = (build_window(m_list, lambda m: U_ref[m], input_fields, cfg) - mu_in) / sd_in
        with torch.no_grad():
            pred_norm = model(torch.tensor(X)).numpy()
        states = reconstruct_general(U_ref[n], n, pred_norm, bc_left, bc_right, mu_out, sd_out, cfg,
                                      rest_bias=rest_bias)
        for s, u in states.items():
            U_ref[s] = u

    mu_in_t, sd_in_t = torch.tensor(mu_in), torch.tensor(sd_in)
    mu_out_t, sd_out_t = torch.tensor(mu_out), torch.tensor(sd_out)
    rest_bias_t = torch.tensor(rest_bias)

    history = [torch.tensor(U_reel[history_needed - lag * cfg.ndt][None, :], dtype=torch.float32)
               for lag in range(cfg.M_BACK, -1, -1)]

    with torch.no_grad():
        for n in range(history_needed, n_stop, cfg.N_FWD * cfg.ndt):
            X = (build_window_torch(history, input_fields, cfg) - mu_in_t) / sd_in_t
            pred_norm = model(X)
            new_states, _ = reconstruct_torch_general(history[-1], pred_norm, [bc_left], [bc_right], n,
                                                        mu_out_t, sd_out_t, rest_bias_t, cfg)
            history = history[cfg.N_FWD:] + new_states

    U_torch_final = history[-1][0].numpy()
    U_ref_final = U_ref[n_stop]
    return np.abs(U_torch_final - U_ref_final).max()


def main():
    cfg = Config()
    set_seeds(cfg)

    INPUTS = make_feature_columns(INPUT_FIELDS, cfg)
    OUTPUTS = make_output_columns(cfg)

    model = ConvNet(n_lags=cfg.M_BACK, n_points=2 * cfg.SS + 1,
                     n_fields=len(INPUT_FIELDS), n_outputs=len(OUTPUTS))
    model.eval()

    mu_in = np.zeros(len(INPUTS), dtype=np.float32)
    sd_in = np.ones(len(INPUTS), dtype=np.float32)
    mu_out = np.zeros(len(OUTPUTS), dtype=np.float32)
    sd_out = np.ones(len(OUTPUTS), dtype=np.float32)
    rest_bias = compute_rest_bias(model, mu_in, sd_in, mu_out, sd_out, cfg)

    all_ok = True
    for name, bc_left, bc_right in build_test_cases(cfg):
        gap = run_case(cfg, model, INPUT_FIELDS, mu_in, sd_in, mu_out, sd_out, rest_bias, bc_left, bc_right)
        status = "OK" if gap < TOLERANCE else "FAILED"
        if gap >= TOLERANCE:
            all_ok = False
        print(f"[{status}] {name:55s} max abs gap = {gap:.3e}  (tolerance {TOLERANCE:.0e})")

    rng = np.random.default_rng(123)
    u0 = sample_random_ic(rng, cfg)
    rest = ("dirichlet", "rest", {"ic": "random"})
    U_reel_free = run_fd_simulation_free(rest, rest, u0, cfg)
    gap = run_case(cfg, model, INPUT_FIELDS, mu_in, sd_in, mu_out, sd_out, rest_bias,
                    rest, rest, U_reel=U_reel_free)
    status = "OK" if gap < TOLERANCE else "FAILED"
    if gap >= TOLERANCE:
        all_ok = False
    print(f"[{status}] {'free evolution (random initial state, no push)':55s} "
          f"max abs gap = {gap:.3e}  (tolerance {TOLERANCE:.0e})")

    if all_ok:
        print("\nAll cases OK -- training/losses.py's torch reconstruction matches the numpy physics "
              "for all 7 signal families.")
    else:
        print("\nFAILED -- at least one case is beyond tolerance, check the physics before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
