# Manual test: for a hand-picked Gaussian pulse (A, omega), NOT necessarily
# seen during training (nor even within the dataset's AMP/OMEGA range),
# simulates the ground truth (FD resolution) and the rollout predicted by
# the trained model, then displays both curves overlaid in the same
# gif (see commun.make_rollout_animation) + the associated error curves.
#
# Physical reminder (see u_right_val in commun.py): for a Gaussian, omega
# does NOT represent a frequency but controls the pulse width via
# sigma = interp(omega, [1, 10], [0.15, 0.07]) -- large omega = narrow
# pulse, small omega = wide pulse.
#
# Simplest usage: change A and OMEGA just below, then
#   python test_prediction.py
# (the --A/--omega flags remain available if you prefer passing them on
# the command line -- they then take priority over the values below).
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ============================================================
#  PARAMETERS TO MODIFY HERE TO CHANGE THE WAVE SHAPE
# ============================================================
A = 0.08          # amplitude of the Gaussian pulse
OMEGA = 4.5       # width: sigma = interp(omega, [1,10], [0.15,0.07])
                  # (large omega -> narrow pulse, small omega -> wide)
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR
from main import CONFIG_OVERRIDES, INPUT_FIELDS, PROJECT_DIR, PLOTS_DIR

sys.path.insert(0, str(COMMUN_DIR))
import commun as C


def parse_args():
    p = argparse.ArgumentParser(description="Tests the trained model on a Gaussian pulse "
                                             "(A, omega) -- modifiable directly at the top of the file, "
                                             "or via --A/--omega which take priority.")
    p.add_argument("--A", type=float, default=None,
                    help=f"Amplitude of the Gaussian pulse (default: A={A} set at the top of the file).")
    p.add_argument("--omega", type=float, default=None,
                    help=f"Width parameter, sigma = interp(omega, [1,10], [0.15,0.07]) -- "
                         f"the larger omega, the narrower the pulse "
                         f"(default: OMEGA={OMEGA} set at the top of the file).")
    p.add_argument("--model-path", type=Path, default=PROJECT_DIR / "model.pth")
    p.add_argument("--norm-stats", type=Path, default=PROJECT_DIR / "norm_stats.csv",
                    help="Normalization stats saved by main.py during training.")
    p.add_argument("--output-dir", type=Path, default=None,
                    help="Default: training/plots/test_A{A}_omega{omega}/.")
    args = p.parse_args()
    if args.A is None:
        args.A = A
    if args.omega is None:
        args.omega = OMEGA
    return args


def main():
    args = parse_args()

    if not args.model_path.exists():
        sys.exit(f"Model not found: {args.model_path} -- have you run training (main.py)?")
    if not args.norm_stats.exists():
        sys.exit(f"Normalization stats not found: {args.norm_stats} -- have you run "
                 f"training (main.py) after adding the norm_stats.csv save?")

    cfg = C.Config(**CONFIG_OVERRIDES)
    sigma = np.interp(args.omega, [1.0, 10.0], [0.15, 0.07])
    print(f"=== manual test — A={args.A}, omega={args.omega} (sigma≈{sigma:.4f}) ===")

    INPUTS = C.make_feature_columns(INPUT_FIELDS, cfg)
    OUTPUTS = C.make_output_columns(cfg)

    norm_stats = pd.read_csv(args.norm_stats, index_col=0)
    mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
    sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)
    mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
    sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype(np.float32)

    modele = C.Reseau(n_inputs=len(INPUTS), n_outputs=len(OUTPUTS), hidden_sizes=cfg.HIDDEN_SIZES)
    modele.load_state_dict(torch.load(args.model_path, weights_only=True))
    modele.eval()

    biais_repos = C._biais_repos(modele, mu_in, sd_in, mu_out, sd_out, cfg)

    print("Reference FD simulation (ground truth)...")
    U_reel = C.run_fd_simulation(args.A, args.omega, cfg)

    print("Autoregressive rollout of the model...")
    U_pred = C._autoregressive_rollout(modele, U_reel, INPUT_FIELDS, mu_in, sd_in, mu_out, sd_out,
                                        biais_repos, args.A, args.omega, cfg)

    rollout = C.RolloutResult(U=U_pred, U_reel=U_reel, A=args.A, omega=args.omega)

    output_dir = args.output_dir or PLOTS_DIR / f"test_A{args.A}_omega{args.omega}"
    output_dir.mkdir(parents=True, exist_ok=True)

    C.make_rollout_animation(rollout, cfg, output_dir)

    t_axis, l2_list, linf_list, smape_list = C.compute_errors(rollout, cfg)
    C.plot_rollout_error(t_axis, l2_list, linf_list, output_dir)
    C.plot_smape(t_axis, smape_list, output_dir)

    print(f"L2 relative   : final = {l2_list[-1]:.4e}  |  max = {max(l2_list):.4e}")
    print(f"Linf absolute : final = {linf_list[-1]:.4e}  |  max = {max(linf_list):.4e}")
    print(f"sMAPE (%)     : final = {smape_list[-1]:.2f}  |  max = {max(smape_list):.2f}")
    print(f"Done — gif and error curves in {output_dir}")


if __name__ == "__main__":
    main()
