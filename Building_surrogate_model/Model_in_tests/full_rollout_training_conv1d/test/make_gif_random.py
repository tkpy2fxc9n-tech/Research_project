# Generates 7 rollout gifs (real vs predicted) for random Gaussian pulses
# (A, omega), using the model already trained by ../training/code/main.py:
#   - 2 pairs with A AND omega both BELOW their training range (close to it)
#   - 2 pairs with A AND omega both ABOVE their training range (close to it)
#   - 3 pairs with A and omega both INSIDE their training range
#
# Adapted from The_surrogate_model/full_rollout_training_gaussian_wave/test/
# make_gif_random.py -- only the network construction changed (ReseauConv
# instead of commun.Reseau), since this project uses the Conv1d architecture
# (see training/code/model.py) but otherwise the exact same physics/dataset
# pipeline as the other full_rollout_training* projects.
#
# Physical reminder (see u_right_val in commun.py): omega does NOT represent
# a frequency, it controls the pulse width via sigma = interp(omega, [1,10],
# [0.15,0.07]). np.interp SATURATES outside [1, 10], so an out-of-range omega
# gives the exact same sigma (hence the same pulse shape) as omega=1 or
# omega=10 -- only A truly extrapolates outside its range. The out-of-range
# omega gifs are therefore mostly a test of the network on out-of-distribution
# *A*, combined with a boundary-clamped pulse width.
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
# CONFIG_OVERRIDES/INPUT_FIELDS come from main.py (not redefined here) so the
# rebuilt Config always matches exactly what the saved model was trained with.
from main import CONFIG_OVERRIDES, INPUT_FIELDS
from model import ReseauConv
from _commun_path import COMMUN_DIR

sys.path.insert(0, str(COMMUN_DIR))
import commun as C

MODEL_PATH = PROJECT_DIR / "model.pth"
NORM_STATS_PATH = PROJECT_DIR / "norm_stats.csv"
OUTPUT_DIR = TEST_DIR / "outputs"

# ============================================================
#  PARAMETERS TO MODIFY HERE
# ============================================================
N_UNDER = 2   # pairs with A and omega both below their range
N_OVER = 2    # pairs with A and omega both above their range
N_IN = 3      # pairs with A and omega both inside their range
SEED = None        # set an int for reproducible draws, None for true random

# A/omega are rounded to the same precision as the training grid itself
# (cfg.AMPLITUDES/cfg.PULSATIONS, see commun.py) -- 0.001 for A, 0.1 for
# omega -- so displayed/simulated values stay the same order of magnitude
# as what the model was actually trained on.
A_STEP = 0.001
OMEGA_STEP = 0.1
# "under"/"over" draws are pushed 1-MAX_MARGIN_STEPS steps past the bound
# (in units of A_STEP/OMEGA_STEP) -- keeps them close to the range while
# guaranteeing that rounding can never land them back on/inside the bound.
MAX_MARGIN_STEPS_A = 10
MAX_MARGIN_STEPS_OMEGA = 3
# ============================================================


def _max_under_steps(bound, step, max_margin_steps):
    # Caps how many steps a value can be pushed below `bound` so it can
    # never cross down to (or past) zero -- matters for AMP_MIN=0.005,
    # which is only 5 steps of 0.001 away from zero.
    room = int(round((bound - step) / step))
    return max(1, min(max_margin_steps, room))


def sample_pairs(cfg):
    rng = np.random.default_rng(SEED)
    amp_min, amp_max = cfg.AMP_MIN, cfg.AMP_MAX
    omega_min, omega_max = cfg.OMEGA_MIN, cfg.OMEGA_MAX

    max_steps_a_under = _max_under_steps(amp_min, A_STEP, MAX_MARGIN_STEPS_A)
    max_steps_omega_under = _max_under_steps(omega_min, OMEGA_STEP, MAX_MARGIN_STEPS_OMEGA)

    pairs = []
    for _ in range(N_UNDER):
        A = round(amp_min - rng.integers(1, max_steps_a_under + 1) * A_STEP, 3)
        omega = round(omega_min - rng.integers(1, max_steps_omega_under + 1) * OMEGA_STEP, 1)
        pairs.append(("under", A, omega))
    for _ in range(N_OVER):
        A = round(amp_max + rng.integers(1, MAX_MARGIN_STEPS_A + 1) * A_STEP, 3)
        omega = round(omega_max + rng.integers(1, MAX_MARGIN_STEPS_OMEGA + 1) * OMEGA_STEP, 1)
        pairs.append(("over", A, omega))
    for _ in range(N_IN):
        A = round(rng.uniform(amp_min, amp_max), 3)
        omega = round(rng.uniform(omega_min, omega_max), 1)
        pairs.append(("in", A, omega))
    return pairs


def range_tag(value, vmin, vmax):
    return "in" if vmin <= value <= vmax else "out"


def make_titled_rollout_animation(rollout, cfg, gif_path, a_tag, omega_tag):
    # Same rendering as commun.make_rollout_animation, plus A/omega (and
    # whether each is inside its training range) in the title (kept local:
    # commun.py is shared by several other projects and its title format
    # shouldn't change for all of them because of this one).
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

    # Error normalized by the pulse's own peak amplitude (not pointwise
    # relative error): |predicted-real| / max(|real|). Pointwise relative
    # error is unstable here -- away from the pulse "real" is not exactly 0
    # but a tiny numerical residual, so dividing by it blows up from noise
    # alone and swamps the y-axis, hiding the real mismatch at the peak.
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
        titre.set_text(f"A={rollout.A:.3f} ({a_tag}-range), "
                        f"omega={rollout.omega:.1f} ({omega_tag}-range) -- "
                        f"t = {m*cfg.dt:.3f} (step {m})")
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
    print(f"=== ranges: A in [{cfg.AMP_MIN}, {cfg.AMP_MAX}], omega in [{cfg.OMEGA_MIN}, {cfg.OMEGA_MAX}] ===")

    INPUTS = C.make_feature_columns(INPUT_FIELDS, cfg)
    OUTPUTS = C.make_output_columns(cfg)

    norm_stats = pd.read_csv(NORM_STATS_PATH, index_col=0)
    mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
    sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)
    mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
    sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype(np.float32)

    modele = ReseauConv(n_lags=cfg.M_BACK, n_points=2 * cfg.SS + 1,
                        n_fields=len(INPUT_FIELDS), n_outputs=len(OUTPUTS))
    modele.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    modele.eval()

    biais_repos = C._biais_repos(modele, mu_in, sd_in, mu_out, sd_out, cfg)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for label, A, omega in sample_pairs(cfg):
        sigma = np.interp(omega, [1.0, 10.0], [0.15, 0.07])
        print(f"--- [{label}-range] A={A:.3f}, omega={omega:.1f} (sigma~={sigma:.4f}) ---")

        U_reel = C.run_fd_simulation(A, omega, cfg)
        U_pred = C._autoregressive_rollout(modele, U_reel, INPUT_FIELDS, mu_in, sd_in, mu_out, sd_out,
                                            biais_repos, A, omega, cfg)
        rollout = C.RolloutResult(U=U_pred, U_reel=U_reel, A=A, omega=omega)

        a_tag = range_tag(A, cfg.AMP_MIN, cfg.AMP_MAX)
        omega_tag = range_tag(omega, cfg.OMEGA_MIN, cfg.OMEGA_MAX)
        gif_path = OUTPUT_DIR / (f"propagation_onde_A{A:.3f}-{a_tag}_"
                                  f"omega{omega:.1f}-{omega_tag}.gif")
        make_titled_rollout_animation(rollout, cfg, gif_path, a_tag, omega_tag)
        print(f"Done -- gif in {gif_path}")


if __name__ == "__main__":
    main()
