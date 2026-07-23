# Quick, no-retraining check of an already-trained model: rebuilds the exact
# same Config (via CONFIG_OVERRIDES imported from main.py) the model was
# trained with, runs a chosen ground-truth simulation (FD) and the network's
# rollout side by side, and reports L2/Linf/sMAPE errors + the standard
# commun.make_rollout_animation gif.
#
# Simplest usage: change LEFT_BC and RIGHT_BC just below (a boundary
# condition is (bc_type, waveform_family, params) -- see commun.BC_WAVEFORMS
# for the available families: gaussian, sinusoid, step, ramp,
# random_multitone, rest), then
#   python test_prediction.py
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from main import CONFIG_OVERRIDES
from _commun_path import COMMUN_DIR

sys.path.insert(0, str(COMMUN_DIR))
import commun as C

INPUT_FIELDS = ["U", "Ut", "Uxx"]
TRAINING_DIR = SCRIPT_DIR.parent
PROJECT_DIR = TRAINING_DIR.parent
PLOTS_DIR = TRAINING_DIR / "plots"

# ============================================================
#  PARAMETERS TO MODIFY HERE TO CHANGE THE BOUNDARY CONDITIONS
# ============================================================
LEFT_BC = ("dirichlet", "rest", {})
RIGHT_BC = ("neumann", "gaussian", {"A": 0.08, "omega": 4.5})
# ============================================================


def parse_args():
    p = argparse.ArgumentParser(description="Checks an already-trained generalized-BC model against a "
                                             "chosen (left, right) boundary condition -- modifiable "
                                             "directly at the top of the file.")
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--norm-stats", type=Path, default=None,
                    help="Normalization stats saved by main.py during training.")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    model_path = args.model_path or PROJECT_DIR / "model.pth"
    norm_stats_path = args.norm_stats or PROJECT_DIR / "norm_stats.csv"

    if not model_path.exists():
        sys.exit(f"Model not found: {model_path} -- have you run training (training/code/main.py)?")
    if not norm_stats_path.exists():
        sys.exit(f"Normalization stats not found: {norm_stats_path} -- have you run "
                 f"training (training/code/main.py)?")

    cfg = C.Config(**CONFIG_OVERRIDES)
    print(f"=== test prediction -- left={C.bc_describe(LEFT_BC)}  right={C.bc_describe(RIGHT_BC)} ===")

    INPUTS = C.make_feature_columns(INPUT_FIELDS, cfg)
    OUTPUTS = C.make_output_columns(cfg)

    norm_stats = pd.read_csv(norm_stats_path, index_col=0)
    mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
    sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)
    mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
    sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype(np.float32)

    modele = C.Reseau(n_inputs=len(INPUTS), n_outputs=len(OUTPUTS), hidden_sizes=cfg.HIDDEN_SIZES)
    modele.load_state_dict(torch.load(model_path, weights_only=True))
    modele.eval()

    biais_repos = C._biais_repos(modele, mu_in, sd_in, mu_out, sd_out, cfg)

    print("Reference FD simulation (ground truth)...")
    U_reel = C.run_fd_simulation_general(LEFT_BC, RIGHT_BC, cfg)

    print("Autoregressive rollout of the model...")
    U_pred = C._autoregressive_rollout_general(modele, U_reel, INPUT_FIELDS, mu_in, sd_in, mu_out, sd_out,
                                                biais_repos, LEFT_BC, RIGHT_BC, cfg)

    rollout = C.RolloutResultGeneral(U=U_pred, U_reel=U_reel, left_bc=LEFT_BC, right_bc=RIGHT_BC)

    output_dir = args.output_dir or PLOTS_DIR / "test_prediction"
    output_dir.mkdir(parents=True, exist_ok=True)

    C.make_rollout_animation(rollout, cfg, output_dir)

    errors = C.compute_errors(rollout, cfg)
    t_axis, l2_list, linf_list, smape_list = errors
    C.plot_rollout_error(t_axis, l2_list, linf_list, output_dir)
    C.plot_smape(t_axis, smape_list, output_dir)

    print(f"L2 relative  : final = {l2_list[-1]:.4e}  |  max = {max(l2_list):.4e}")
    print(f"Linf absolute: final = {linf_list[-1]:.4e}  |  max = {max(linf_list):.4e}")
    print(f"sMAPE (%)    : final = {smape_list[-1]:.3f}  |  max = {max(smape_list):.3f}")
    print(f"Done — outputs in {output_dir}")


if __name__ == "__main__":
    main()
