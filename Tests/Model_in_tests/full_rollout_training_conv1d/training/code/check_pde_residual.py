# Sanity check to run BEFORE trusting the composite loss's physics term:
# verifies that rollout_torch.utt_uxx_torch (the differentiable torch
# version used during training, see train.py's rollout_group_tbptt) matches
# evaluation.compute_utt_uxx (the numpy version used for the
# plot_utt_uxx/resume.txt diagnostics) on the exact same (u_prev, u_curr,
# u_next) triples, for both step spacings actually used in this project
# (dt_eff=cfg.dt for consecutive raw steps, dt_eff=cfg.ndt*cfg.dt for the
# ndt-spaced steps a rollout hop produces). Does NOT test the gradient (only
# the values) -- same scope as check_equivalence.py, just for the physics
# residual that script doesn't cover.
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from config import Config, set_seeds
from physics import run_fd_simulation_general
from evaluation import compute_utt_uxx
from rollout_torch import utt_uxx_torch

TOLERANCE = 1e-4


def main():
    cfg = Config()
    set_seeds(cfg)

    left_bc = ("dirichlet", "rest", {})
    right_bc = ("neumann", "gaussian", {"A": 0.1, "sigma": 0.15})
    U_reel = run_fd_simulation_general(left_bc, right_bc, cfg)

    n_mid = cfg.Nt // 2  # far enough from both ends to have full history/future at either spacing
    cases = [
        ("consecutive raw steps (dt)", n_mid - 1, n_mid, n_mid + 1, cfg.dt),
        ("ndt-spaced steps (rollout hop spacing)", n_mid - cfg.ndt, n_mid, n_mid + cfg.ndt, cfg.ndt * cfg.dt),
    ]

    all_ok = True
    for name, n_prev, n_curr, n_next, dt_eff in cases:
        u_prev, u_curr, u_next = U_reel[n_prev], U_reel[n_curr], U_reel[n_next]

        utt_np, uxx_np = compute_utt_uxx(u_prev, u_curr, u_next, dt_eff, cfg)

        u_prev_t = torch.tensor(u_prev[None, :], dtype=torch.float32)
        u_curr_t = torch.tensor(u_curr[None, :], dtype=torch.float32)
        u_next_t = torch.tensor(u_next[None, :], dtype=torch.float32)
        with torch.no_grad():
            utt_t, uxx_t = utt_uxx_torch(u_prev_t, u_curr_t, u_next_t, dt_eff, cfg)

        gap_tt = np.abs(utt_t[0].numpy() - utt_np).max()
        gap_xx = np.abs(uxx_t[0].numpy() - uxx_np).max()
        gap = max(gap_tt, gap_xx)
        status = "OK" if gap < TOLERANCE else "FAILED"
        if gap >= TOLERANCE:
            all_ok = False
        print(f"[{status}] {name:40s} max abs gap: u_tt={gap_tt:.3e} u_xx={gap_xx:.3e}  (tolerance {TOLERANCE:.0e})")

    if all_ok:
        print("\nAll cases OK -- rollout_torch.utt_uxx_torch matches evaluation.compute_utt_uxx: "
              "the differentiable physics-residual term matches its numpy reference.")
    else:
        print("\nFAILED -- at least one case is beyond tolerance, check utt_uxx_torch before "
              "training with LAMBDA_PHYSICS > 0.")
        sys.exit(1)


if __name__ == "__main__":
    main()
