# Generates a gif (real vs predicted rollout) for a hand-picked (left, right)
# boundary condition pair, using the model already trained by
# ../training/code/main.py.
#
# A boundary condition is (bc_type, waveform_family, params) -- bc_type in
# {"dirichlet","neumann"}, waveform_family one of the 7 signal families this
# project trains on: fourier, sinusoid, chirp, gaussian, shock,
# filtered_random, or "rest" (homogeneous/free end) -- see
# training/code/scenarios.py. Only the network construction (ReseauConv,
# training/code/model.py) is specific to this project -- everything else
# copied from full_rollout_training_multisignal's make_gif.py.
#
# Simplest usage: change LEFT_BC and RIGHT_BC just below, then
#   python make_gif.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
TRAINING_CODE_DIR = PROJECT_DIR / "training" / "code"
sys.path.insert(0, str(TRAINING_CODE_DIR))
# INPUT_FIELDS comes from main.py (not redefined here) so the rebuilt Config
# always matches exactly what the saved model was trained with.
from main import INPUT_FIELDS
from model import ReseauConv
from config import Config
from waves import bc_describe
from physics import (make_feature_columns, make_output_columns, run_fd_simulation_general,
                      autoregressive_rollout, biais_repos as compute_biais_repos)
from evaluation import RolloutResult

MODEL_PATH = PROJECT_DIR / "model.pth"
NORM_STATS_PATH = PROJECT_DIR / "norm_stats.csv"
OUTPUT_DIR = TEST_DIR / "outputs"

# ============================================================
#  PARAMETERS TO MODIFY HERE TO CHANGE THE BOUNDARY CONDITIONS
# ============================================================
LEFT_BC = ("dirichlet", "rest", {})
RIGHT_BC = ("neumann", "gaussian", {"A": 0.08, "omega": 4.5})
# ============================================================


def bc_filename_tag(bc):
    bc_type, family, _ = bc
    return f"{bc_type[:1]}-{family}"  # e.g. "d-gaussian", "n-fourier"


def make_relative_error_animation(rollout, cfg, gif_path):
    # Bottom panel shows error normalized by the pulse's own peak amplitude
    # (|predicted-real| / max(|real|)), not pointwise relative error --
    # pointwise relative error is unstable near the wave's zero-crossings
    # (dividing by ~0 blows up from noise alone, hiding genuinely large
    # errors elsewhere). One fixed y-axis for the whole gif, not rescaled
    # per frame.
    U, U_reel = rollout.U, rollout.U_reel
    nodes = cfg.nodes
    x = np.linspace(0, cfg.L, cfg.Nx)
    frames = np.arange(0, cfg.Nt + 1, cfg.ndt)

    fig_anim, (axA, axB) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ligne_reel, = axA.plot([], [], "r", lw=2, label="real")
    ligne_pred, = axA.plot([], [], "b--", lw=2, label="predicted")
    amp_ref = np.abs(U_reel[:, nodes]).max()
    ymax = amp_ref * 1.2
    axA.set_xlim(0, cfg.L); axA.set_ylim(-ymax, ymax)
    axA.set_ylabel("u"); axA.legend(loc="upper right"); axA.grid(True)

    ligne_err, = axB.plot([], [], "k", lw=1.5, label="error / peak amplitude")
    err_frames = [np.abs(U[m, nodes] - U_reel[m, nodes]) / amp_ref for m in frames]
    err_max = max(np.max([e.max() for e in err_frames]) * 1.2, 1e-9)
    axB.set_xlim(0, cfg.L); axB.set_ylim(0, err_max)
    axB.set_xlabel("x"); axB.set_ylabel("error / peak amplitude"); axB.legend(loc="upper right"); axB.grid(True)

    titre = fig_anim.suptitle("")

    def maj(m):
        ligne_reel.set_data(x, U_reel[m, nodes])
        ligne_pred.set_data(x, U[m, nodes])
        ligne_err.set_data(x, np.abs(U[m, nodes] - U_reel[m, nodes]) / amp_ref)
        titre.set_text(f"left={bc_describe(rollout.left_bc)}  right={bc_describe(rollout.right_bc)}\n"
                        f"t = {m*cfg.dt:.3f}  (step {m})")
        return ligne_reel, ligne_pred, ligne_err, titre

    anim = animation.FuncAnimation(fig_anim, maj, frames=frames, interval=50, blit=False)
    anim.save(gif_path, writer="pillow", fps=20, dpi=110)
    plt.close(fig_anim)


def main():
    if not MODEL_PATH.exists():
        sys.exit(f"Model not found: {MODEL_PATH} -- have you run training (training/code/main.py)?")
    if not NORM_STATS_PATH.exists():
        sys.exit(f"Normalization stats not found: {NORM_STATS_PATH} -- have you run "
                 f"training (training/code/main.py)?")

    cfg = Config()
    print(f"=== gif -- left={bc_describe(LEFT_BC)}  right={bc_describe(RIGHT_BC)} ===")

    INPUTS = make_feature_columns(INPUT_FIELDS, cfg)
    OUTPUTS = make_output_columns(cfg)

    norm_stats = pd.read_csv(NORM_STATS_PATH, index_col=0)
    mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
    sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)
    mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
    sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype(np.float32)

    modele = ReseauConv(n_lags=cfg.M_BACK, n_points=2 * cfg.SS + 1,
                        n_fields=len(INPUT_FIELDS), n_outputs=len(OUTPUTS))
    modele.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    modele.eval()

    biais_repos = compute_biais_repos(modele, mu_in, sd_in, mu_out, sd_out, cfg)

    print("Reference FD simulation (ground truth)...")
    U_reel = run_fd_simulation_general(LEFT_BC, RIGHT_BC, cfg)

    print("Autoregressive rollout of the model...")
    U_pred = autoregressive_rollout(modele, U_reel, INPUT_FIELDS, mu_in, sd_in, mu_out, sd_out,
                                                biais_repos, LEFT_BC, RIGHT_BC, cfg)

    rollout = RolloutResult(U=U_pred, U_reel=U_reel, left_bc=LEFT_BC, right_bc=RIGHT_BC)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gif_path = OUTPUT_DIR / f"propagation_onde_{bc_filename_tag(LEFT_BC)}_{bc_filename_tag(RIGHT_BC)}.gif"
    make_relative_error_animation(rollout, cfg, gif_path)

    print(f"Done — gif in {gif_path}")


if __name__ == "__main__":
    main()
