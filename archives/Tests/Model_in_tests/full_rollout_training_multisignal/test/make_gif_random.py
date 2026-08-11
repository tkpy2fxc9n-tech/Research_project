# Generates several rollout gifs (real vs predicted) for randomly-sampled
# scenarios, using the model already trained by ../training/code/main.py --
# showcases the network's behavior across the same 7-family signal mix it
# was trained on (see scenarios.FAMILY_SHARES: fourier, sinusoid, chirp,
# gaussian, shock, filtered_random, free_evolution), rather than a single
# hand-picked case (see make_gif.py for that).
#
# Usage:
#   python3 make_gif_random.py
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
from main import CONFIG_OVERRIDES
from _commun_path import COMMUN_DIR
import waveforms
import scenarios
import free_evolution

sys.path.insert(0, str(COMMUN_DIR))
import commun as C

INPUT_FIELDS = ["U"]
MODEL_PATH = PROJECT_DIR / "model.pth"
NORM_STATS_PATH = PROJECT_DIR / "norm_stats.csv"
OUTPUT_DIR = TEST_DIR / "outputs"

# ============================================================
#  PARAMETERS TO MODIFY HERE
# ============================================================
N_SAMPLES = 7      # number of random scenarios to render (7 -> good odds of covering most families)
SEED = None        # set an int for reproducible draws, None for true random
# ============================================================


def bc_filename_tag(bc):
    bc_type, family, _ = bc
    return f"{bc_type[:1]}-{family}"


def make_relative_error_animation(rollout_U, rollout_U_reel, left_bc, right_bc, cfg, gif_path):
    # Same rendering as make_gif.py's local animation function -- kept local
    # (not in commun.py) since commun.py is shared by several other projects
    # and its default plots shouldn't change for all of them because of this one.
    U, U_reel = rollout_U, rollout_U_reel
    nodes = cfg.nodes
    x = np.linspace(0, cfg.L, cfg.Nx)
    frames = np.arange(0, cfg.Nt + 1, cfg.ndt)

    fig_anim, (axA, axB) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ligne_reel, = axA.plot([], [], "r", lw=2, label="real")
    ligne_pred, = axA.plot([], [], "b--", lw=2, label="predicted")
    amp_ref = max(np.abs(U_reel[:, nodes]).max(), 1e-9)
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
        titre.set_text(f"left={C.bc_describe(left_bc)}  right={C.bc_describe(right_bc)}\n"
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

    cfg = C.Config(**CONFIG_OVERRIDES)
    waveforms.register(C)

    INPUTS = C.make_feature_columns(INPUT_FIELDS, cfg)
    OUTPUTS = C.make_output_columns(cfg)

    norm_stats = pd.read_csv(NORM_STATS_PATH, index_col=0)
    mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
    sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)
    mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
    sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype(np.float32)

    modele = C.Reseau(n_inputs=len(INPUTS), n_outputs=len(OUTPUTS), hidden_sizes=cfg.HIDDEN_SIZES)
    modele.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    modele.eval()

    biais_repos = C._biais_repos(modele, mu_in, sd_in, mu_out, sd_out, cfg)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    scenario_list = scenarios.sample_scenarios(cfg, N_SAMPLES, C, rng)

    for i, (left_bc, right_bc, u0) in enumerate(scenario_list):
        print(f"--- [{i+1}/{N_SAMPLES}] left={C.bc_describe(left_bc)}  right={C.bc_describe(right_bc)} ---")

        if u0 is None:
            U_reel = C.run_fd_simulation_general(left_bc, right_bc, cfg)
        else:
            U_reel = free_evolution.run_fd_simulation_free(left_bc, right_bc, u0, cfg, C)

        U_pred = C._autoregressive_rollout_general(modele, U_reel, INPUT_FIELDS, mu_in, sd_in, mu_out, sd_out,
                                                    biais_repos, left_bc, right_bc, cfg)

        gif_path = OUTPUT_DIR / f"propagation_onde_{i:02d}_{bc_filename_tag(left_bc)}_{bc_filename_tag(right_bc)}.gif"
        make_relative_error_animation(U_pred, U_reel, left_bc, right_bc, cfg, gif_path)
        print(f"Done -- gif in {gif_path}")


if __name__ == "__main__":
    main()
