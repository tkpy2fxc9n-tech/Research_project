# Generates a gif (real vs predicted rollout) for a Gaussian pulse (A, omega)
# chosen by hand, using the model already trained by ../training/code/main.py.
#
# Physical reminder (see u_right_val in commun.py): for a Gaussian, omega
# does NOT represent a frequency but controls the pulse width via
# sigma = interp(omega, [1, 10], [0.15, 0.07]) -- large omega = narrow
# pulse, small omega = wide pulse.
#
# Simplest usage: change A and OMEGA just below, then
#   python make_gif.py
# (the --A/--omega flags remain available if you prefer passing them on
# the command line -- they then take priority over the values below).
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ============================================================
#  PARAMETERS TO MODIFY HERE TO CHANGE THE WAVE SHAPE
# ============================================================
A = 0.13          # amplitude of the Gaussian pulse
OMEGA = 3       # width: sigma = interp(omega, [1,10], [0.15,0.07])
# CONFIG_OVERRIDES = dict(
#     N_GRID=20,
#     AMP_MIN=0.005,
#     AMP_MAX=0.15,
#     OMEGA_MIN=1.0,
#     OMEGA_MAX=10.0,
                  # (large omega -> narrow pulse, small omega -> wide)
# ============================================================

TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
TRAINING_CODE_DIR = PROJECT_DIR / "training" / "code"
sys.path.insert(0, str(TRAINING_CODE_DIR))
# CONFIG_OVERRIDES/INPUT_FIELDS come from main.py (not redefined here) so the
# rebuilt Config always matches exactly what the saved model was trained with.
from main import CONFIG_OVERRIDES, INPUT_FIELDS
from _commun_path import COMMUN_DIR

sys.path.insert(0, str(COMMUN_DIR))
import commun as C


def make_relative_error_animation(rollout, cfg, gif_path):
    # Same rendering as commun.make_rollout_animation, but with the bottom
    # panel showing relative error instead of absolute error (kept local:
    # commun.py is shared by several other projects and its plots shouldn't
    # change for all of them because of this one).
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
        titre.set_text(f"Wave propagation — t = {m*cfg.dt:.3f}  (step {m})")
        return ligne_reel, ligne_pred, ligne_err, titre

    anim = animation.FuncAnimation(fig_anim, maj, frames=frames, interval=50, blit=False)
    anim.save(gif_path, writer="pillow", fps=20, dpi=110)
    plt.close(fig_anim)


def parse_args():
    p = argparse.ArgumentParser(description="Builds a rollout gif (real vs predicted) for a "
                                             "Gaussian pulse (A, omega) -- modifiable directly at "
                                             "the top of the file, or via --A/--omega which take priority.")
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
                    help="Default: test/outputs/ (gif named propagation_onde_A{A}_omega{omega}.gif).")
    args = p.parse_args()
    if args.A is None:
        args.A = A
    if args.omega is None:
        args.omega = OMEGA
    return args


def main():
    args = parse_args()

    if not args.model_path.exists():
        sys.exit(f"Model not found: {args.model_path} -- have you run training (training/code/main.py)?")
    if not args.norm_stats.exists():
        sys.exit(f"Normalization stats not found: {args.norm_stats} -- have you run "
                 f"training (training/code/main.py)?")

    cfg = C.Config(**CONFIG_OVERRIDES)
    sigma = np.interp(args.omega, [1.0, 10.0], [0.15, 0.07])
    print(f"=== gif A={args.A:.3f}, omega={args.omega:.1f} (sigma≈{sigma:.4f}) ===")

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

    output_dir = args.output_dir or TEST_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    gif_path = output_dir / f"propagation_onde_A{args.A:.3f}_omega{args.omega:.1f}.gif"
    make_relative_error_animation(rollout, cfg, gif_path)

    print(f"Done — gif in {gif_path}")


if __name__ == "__main__":
    main()
