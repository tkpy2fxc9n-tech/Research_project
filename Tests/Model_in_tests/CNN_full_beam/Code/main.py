# Entry point: differentiable full-rollout (TBPTT) training of the
# full-beam Conv1d model (model.ReseauConvFullBeam) -- unlike the reference
# project (full_rollout_training_conv1d), where the TBPTT machinery existed
# but plain teacher forcing did the actual training, HERE the TBPTT loop
# (train.train_full_rollout) IS the training: each group of simulations is
# rolled out end to end on the network's own predictions, with a weight
# correction every --tbptt-hops hops.
#
# Boundary conditions are the 3 requested ones -- Gaussian push (as a
# displacement or as a flux), free end, fixed-at-0 end -- composed
# independently on each end of the beam (up to 4x4=16 combinations, see
# scenarios.py). After training, one showcase gif is rendered per (left,
# right) combination, so every configuration the network can face is
# visualized at least once.
import argparse
import resource
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from config import Config, set_seeds
from waves import bc_describe
from physics import (make_input_channel_labels, make_output_channel_labels,
                      run_fd_simulation_general, autoregressive_rollout, biais_repos)
from model import ReseauConvFullBeam
from training import norm_values
import scenarios
import dataset
import train
import training
import evaluation

METHOD_NAME = "conv1d_full_beam_U_tbptt"

N_BC_SAMPLES = 200   # number of randomly-sampled (left, right) BC scenarios
GROUP_SIZE = 8       # simulations rolled out together per TBPTT step
TBPTT_HOPS = 10      # hops between two weight corrections

# Code/ holds everything: plots/ and logs/ are its direct children;
# model.pth/norm_stats.csv stay at the CNN_full_beam/ level (SCRIPT_DIR's
# parent), same convention as the other Model_in_tests projects.
PROJECT_DIR = SCRIPT_DIR.parent
PLOTS_DIR = SCRIPT_DIR / "plots"


def parse_args():
    p = argparse.ArgumentParser(description="Differentiable full-rollout (TBPTT) training of the full-beam "
                                             "Conv1d model, with Gaussian/free/fixed boundary conditions "
                                             "composed on both ends.")
    p.add_argument("--smoke-test", action="store_true",
                    help="Miniature run (few simulations, few epochs) to check that everything "
                         "runs without error before a full, expensive run.")
    p.add_argument("--epochs", type=int, default=None,
                    help="Number of epochs (default: 2 in --smoke-test, otherwise Config.N_EPOCHS). One "
                         "epoch = every training simulation rolled out end to end once.")
    p.add_argument("--n-samples", type=int, default=None,
                    help=f"Number of randomly-sampled BC scenarios to simulate "
                         f"(default: 16 in --smoke-test, {N_BC_SAMPLES} otherwise).")
    p.add_argument("--group-size", type=int, default=GROUP_SIZE,
                    help=f"Simulations rolled out together per TBPTT step (default: {GROUP_SIZE}).")
    p.add_argument("--tbptt-hops", type=int, default=TBPTT_HOPS,
                    help=f"Hops between two weight corrections during a rollout (default: {TBPTT_HOPS}).")
    p.add_argument("--early-stop-patience", type=int, default=None,
                    help="Stop training once this many consecutive epochs pass without a new best val "
                         "rollout error (default: Config.EARLY_STOP_PATIENCE; 0 disables early stopping).")
    return p.parse_args()


def build_config(n_epochs, early_stop_patience):
    # None -> Config's own default applies, unoverridden -- config.py stays
    # the single source of truth unless the CLI explicitly asks for a
    # different value.
    kwargs = {}
    if n_epochs is not None:
        kwargs["N_EPOCHS"] = n_epochs
    if early_stop_patience is not None:
        kwargs["EARLY_STOP_PATIENCE"] = early_stop_patience
    return Config(**kwargs)


# --- Per-combination showcase -------------------------------------------
# The single pinned rollout_idx test case (used for resume.txt's metrics/
# benchmark/PDE-consistency plots) only ever lands on ONE (left, right)
# combination. To see how the network behaves on EVERY configuration it can
# face, render one clean, deterministic scenario per (left, right)
# combination (16 gifs) straight into this run's own OUTPUT_DIR.
def make_combo_animation(U, U_reel, left_bc, right_bc, cfg, gif_path):
    # Error normalized by the case's own peak amplitude (not pointwise
    # relative error, which blows up near zero-crossings).
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
        titre.set_text(f"left={bc_describe(left_bc)}  right={bc_describe(right_bc)}\n"
                        f"t = {m*cfg.dt:.3f}  (step {m})")
        return ligne_reel, ligne_pred, ligne_err, titre

    anim = animation.FuncAnimation(fig_anim, maj, frames=frames, interval=50, blit=False)
    anim.save(gif_path, writer="pillow", fps=20, dpi=110)
    plt.close(fig_anim)


def run_combo_showcase(modele, cfg, norm_stats, INPUT_CHANNELS, OUTPUT_CHANNELS, output_dir: Path) -> dict:
    mu_in, sd_in = norm_values(norm_stats, INPUT_CHANNELS)
    mu_out, sd_out = norm_values(norm_stats, OUTPUT_CHANNELS)
    biais_repos_zero = biais_repos(modele, mu_in, sd_in, mu_out, sd_out, cfg)

    rng = np.random.default_rng(cfg.SEED + 1)  # independent of the training scenario draws
    combo_summary = {}
    for left_bc, right_bc in scenarios.all_combo_scenarios(cfg, rng):
        tag = f"{scenarios.end_tag(left_bc)}__{scenarios.end_tag(right_bc)}"
        U_reel = run_fd_simulation_general(left_bc, right_bc, cfg)
        U_pred = autoregressive_rollout(modele, U_reel, mu_in, sd_in, mu_out, sd_out,
                                         biais_repos_zero, left_bc, right_bc, cfg)

        gif_path = output_dir / f"propagation_onde_{tag}.gif"
        make_combo_animation(U_pred, U_reel, left_bc, right_bc, cfg, gif_path)

        fake_rollout = evaluation.RolloutResult(U=U_pred, U_reel=U_reel, left_bc=left_bc, right_bc=right_bc)
        t_axis, l2_list, linf_list, smape_list = evaluation.compute_errors(fake_rollout, cfg)
        summary = (f"L2 final={l2_list[-1]:.3e} max={max(l2_list):.3e} | "
                   f"sMAPE final={smape_list[-1]:.1f}% max={max(smape_list):.1f}%")
        combo_summary[tag] = summary
        print(f"  [{tag:32s}] {summary}  -> {gif_path.name}")

    return combo_summary


def main():
    args = parse_args()
    # Computed here (not at module level) so importing main.py just for its
    # constants doesn't create an empty plots/simulation_.../ folder on
    # every import.
    OUTPUT_DIR = PLOTS_DIR / f"simulation_{datetime.now():%d%m%Y_%H%M%S}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_epochs = args.epochs if args.epochs is not None else (2 if args.smoke_test else None)
    n_samples = args.n_samples if args.n_samples is not None else (16 if args.smoke_test else N_BC_SAMPLES)
    cfg = build_config(n_epochs, args.early_stop_patience)
    set_seeds(cfg)

    patience = cfg.EARLY_STOP_PATIENCE if cfg.EARLY_STOP_PATIENCE > 0 else None

    mode = "SMOKE TEST" if args.smoke_test else "run"
    print(f"=== CNN_full_beam [{mode}] — whole-beam input (Nx={cfg.Nx}), U only, "
          f"{n_samples} random BC scenarios (gaussian/free/fixed composed on both ends, "
          f"A:{cfg.AMP_MIN}-{cfg.AMP_MAX}, sigma:{cfg.OMEGA_MIN}-{cfg.OMEGA_MAX}), "
          f"{cfg.N_EPOCHS} epochs (early-stop patience={patience}), TBPTT full-rollout training, "
          f"group size {args.group_size}, correction every {args.tbptt_hops} hops ===")

    rng = np.random.default_rng(cfg.SEED)
    bc_pairs = scenarios.sample_scenarios(cfg, n_samples, rng)

    FIELDS = dataset.generate_simulations(cfg, bc_pairs)
    print(f"{len(FIELDS)} simulations of {cfg.Nt + 1} timesteps x {cfg.Ntot} grid points")

    idx_train, idx_val, idx_test, rollout_idx = dataset.split_by_simulation(len(bc_pairs), cfg)

    INPUT_CHANNELS = make_input_channel_labels(cfg)
    OUTPUT_CHANNELS = make_output_channel_labels(cfg)
    norm_stats = dataset.compute_norm_stats(FIELDS, idx_train, INPUT_CHANNELS, OUTPUT_CHANNELS, cfg)
    # Persisted next to model.pth (not in OUTPUT_DIR, which changes name
    # every run) so that test.py/test_random.py always find the
    # normalization stats of the last trained model.
    norm_stats.to_csv(PROJECT_DIR / "norm_stats.csv")

    modele = ReseauConvFullBeam(n_lags=cfg.M_BACK, n_outputs=len(OUTPUT_CHANNELS))
    print(modele)
    print(f"Parameters: {sum(p.numel() for p in modele.parameters()):,}")

    train_result = train.train_full_rollout(modele, FIELDS, bc_pairs, idx_train, idx_val,
                                             norm_stats, INPUT_CHANNELS, OUTPUT_CHANNELS, cfg,
                                             group_size=args.group_size, n_epochs=cfg.N_EPOCHS,
                                             model_path=PROJECT_DIR / "model.pth",
                                             tbptt_hops=args.tbptt_hops, patience=patience)
    train.plot_rollout_training_curve(train_result, OUTPUT_DIR)

    # One-step accuracy on the test simulations -- a simple control metric
    # alongside the rollout errors, NOT what trained the model.
    X_test, Y_test = dataset.extract_pairs(FIELDS, idx_test, cfg)
    tf_metrics = training.evaluate_teacher_forcing(modele, X_test, Y_test,
                                                    INPUT_CHANNELS, OUTPUT_CHANNELS, norm_stats, OUTPUT_DIR)

    rollout = evaluation.run_rollout(modele, FIELDS, bc_pairs, rollout_idx,
                                      norm_stats, INPUT_CHANNELS, OUTPUT_CHANNELS, cfg)
    evaluation.plot_utt_uxx(rollout, cfg, OUTPUT_DIR)
    evaluation.make_rollout_animation(rollout, cfg, OUTPUT_DIR)

    errors = evaluation.compute_errors(rollout, cfg)
    t_axis, l2_list, linf_list, smape_list = errors
    evaluation.plot_rollout_error(t_axis, l2_list, linf_list, OUTPUT_DIR)
    evaluation.plot_smape(t_axis, smape_list, OUTPUT_DIR)

    bench = evaluation.benchmark_inference(modele, norm_stats, INPUT_CHANNELS, OUTPUT_CHANNELS, rollout, cfg)

    print(f"Rendering one showcase rollout per (left, right) BC combination into {OUTPUT_DIR} ...")
    combo_summary = run_combo_showcase(modele, cfg, norm_stats, INPUT_CHANNELS, OUTPUT_CHANNELS, OUTPUT_DIR)

    extra_info = {
        "Architecture": " ".join(str(modele).split()),
        "Training scheme": f"differentiable full rollout (TBPTT) -- group size {args.group_size}, "
                           f"weight correction every {args.tbptt_hops} hops",
        "CLI args": f"epochs={cfg.N_EPOCHS}, n_samples={n_samples}, group_size={args.group_size}, "
                    f"tbptt_hops={args.tbptt_hops}, early_stop_patience={patience}",
    }
    for tag, summary in combo_summary.items():
        extra_info[f"Combo {tag}"] = summary
    n_sims_by_split = {"train": len(idx_train), "val": len(idx_val), "test": len(idx_test)}
    evaluation.export_resume(OUTPUT_DIR, cfg, METHOD_NAME, n_sims_by_split, INPUT_CHANNELS, OUTPUT_CHANNELS,
                              train_result, tf_metrics, rollout, bench, errors, extra_info)

    if args.smoke_test:
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"Peak memory (smoke test): {peak_rss_mb:.0f} MB")

    print(f"Done — outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
