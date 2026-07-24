# Entry point: PLAIN ONE-STEP (teacher-forcing) training of the Conv1d model
# (model.py) -- for each training row, predict the two delta_u horizons from
# a REAL (ground-truth) M_BACK history window, average their error, and
# update the weights immediately. No autoregressive rollout, no TBPTT: every
# batch is drawn straight from the flattened dataset (df/INPUTS/OUTPUTS),
# reusing commun.py's existing, already-validated train_model/make_dataloaders
# (same training loop the other, non-rollout full_rollout_training* projects
# use) instead of hand-rolling a new one.
#
# The autoregressive "full rollout" + TBPTT training this project used
# earlier is still implemented in train.py/rollout_torch.py (and validated by
# check_equivalence.py) but is no longer called from here -- kept in place,
# unused, in case it's wanted again later.
#
# Boundary conditions are still restricted to the 7 signal families from
# full_rollout_training_multisignal's design table (see scenarios.py for the
# shares) -- fourier, sinusoid, chirp, gaussian, shock, filtered_random,
# free_evolution -- instead of this project's original single fixed
# left=clamped / right=Gaussian-pulse forcing. Dataset generation
# (waveforms.py/scenarios.py/free_evolution.py/dataset_multisignal.py) and
# the post-training evaluation/plotting (commun.py's *_general functions,
# the per-family showcase below) are unchanged by this training-scheme swap
# -- only HOW the weights get updated changed, not what data is used or how
# the trained model gets evaluated.
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
from _commun_path import COMMUN_DIR
from data_split import split_by_simulation, compute_norm_stats
from model import ReseauConv
import waveforms
import scenarios
import dataset_multisignal
import free_evolution

sys.path.insert(0, str(COMMUN_DIR))
import commun as C

# U only: the network only ever sees past displacement at each node, no
# velocity (Ut) or curvature (Uxx) features -- same convention as
# full_rollout_training_multisignal.
INPUT_FIELDS = ["U"]
METHOD_NAME = "conv1d_U_multisignal_teacher_forcing"
# M_BACK=4 -- stencil x lag input is a 21x4 matrix (21 neighbours, 4 past
# moments t, t-ndt, t-2ndt, t-3ndt) with a single field (U). Sampling grid
# (AMP_MIN/MAX, OMEGA_MIN/MAX) left at Config's defaults: the 7 signal
# families now provide the input diversity, not a widened amplitude/frequency
# range.
# LAMBDA_PF=0 / NOISE_STD=0: this project's plain teacher-forcing scheme is
# meant to be exactly "real M_BACK window in -> predict -> average the two
# horizons' error -> update", with no pushforward augmentation and no input
# noise -- both are commun.train_model features, off by default. (Also,
# commun.make_pf_samples/pushforward_loss hardcode the (A, omega)-keyed
# FIELDS structure from the dense-grid projects, not the sim_idx-keyed
# general-BC one used here, so LAMBDA_PF>0 would need that generalized first.)
# Kept as an explicit constant (not just relying on Config's defaults) so
# test/make_gif*.py and check_equivalence.py can import it and always
# rebuild the exact Config the saved model was trained with, same convention
# as every other full_rollout_training* project.
CONFIG_OVERRIDES = {"M_BACK": 4, "LAMBDA_PF": 0.0, "NOISE_STD": 0.0}

N_BC_SAMPLES = 100  # number of randomly-sampled scenarios (matches this project's previous 10x10=100 grid)

# code/ is a subfolder of full_rollout_training_conv1d/training/ -- plots/
# and logs/ are its sibling folders; model.pth/norm_stats.csv stay at the
# full_rollout_training_conv1d/ level.
TRAINING_DIR = SCRIPT_DIR.parent
PROJECT_DIR = TRAINING_DIR.parent
PLOTS_DIR = TRAINING_DIR / "plots"


def parse_args():
    p = argparse.ArgumentParser(description="Plain one-step (teacher-forcing) training of the Conv1d model, "
                                             "with generalized boundary conditions restricted to 7 signal families.")
    p.add_argument("--smoke-test", action="store_true",
                    help="Miniature run (few samples, few epochs) to check that everything "
                         "runs without error before a full, expensive run.")
    p.add_argument("--epochs", type=int, default=None,
                    help="Number of epochs (default: 2 in --smoke-test, 5 otherwise). Plain teacher-forcing "
                         "is much cheaper per epoch than the old full-rollout scheme -- likely worth raising "
                         "well above 5 for a real production run.")
    p.add_argument("--n-samples", type=int, default=None,
                    help=f"Number of randomly-sampled scenarios to simulate "
                         f"(default: 16 in --smoke-test, {N_BC_SAMPLES} otherwise).")
    p.add_argument("--batch-size", type=int, default=None,
                    help="Rows per gradient update (default: Config.BATCH_SIZE, currently 512). Each batch's "
                         "loss is the average error over the whole batch AND both delta_u horizons.")
    p.add_argument("--early-stop-patience", type=int, default=15,
                    help="Stop training once this many consecutive epochs pass without a new best val loss "
                         "(0 disables early stopping, always running the full --epochs count).")
    return p.parse_args()


def build_config(n_epochs, batch_size):
    kwargs = {"N_EPOCHS": n_epochs, **CONFIG_OVERRIDES}
    if batch_size is not None:
        kwargs["BATCH_SIZE"] = batch_size
    return C.Config(**kwargs)


# --- Per-family showcase -----------------------------------------------
# The single pinned rollout_idx test case (used above for resume.txt's
# metrics/benchmark/PDE-consistency plots) only ever lands on ONE signal
# family -- whichever the random test-split draw happened to pick, often the
# same family run after run. To actually see how the network behaves on
# EVERY family it was trained on, build one clean, deterministic scenario per
# family in scenarios.FAMILY_SHARES (left=dirichlet/rest, right=neumann/
# <family> -- same convention as test/make_gif.py) and render a
# propagation_onde_<family>.gif for each, straight into this run's own
# OUTPUT_DIR (not just in the standalone test/ scripts).
FAMILY_LEFT_BC = ("dirichlet", "rest", {})


def build_family_scenario(family: str, cfg, rng):
    if family == "free_evolution":
        u0 = free_evolution.sample_random_ic(rng, cfg)
        rest = ("dirichlet", "rest", {"ic": "random"})
        return rest, rest, u0
    sampler, _ = C.BC_WAVEFORMS[family]  # waveforms.register(C) must already have run
    params = sampler(rng, cfg)
    return FAMILY_LEFT_BC, ("neumann", family, params), None


def make_family_animation(U, U_reel, left_bc, right_bc, cfg, gif_path):
    # Error normalized by the case's own peak amplitude (not pointwise
    # relative error, which blows up near zero-crossings) -- same rendering
    # as test/make_gif.py, kept local for the same reason: commun.py is
    # shared by several other projects and its plots shouldn't change for
    # all of them because of this one.
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


def run_family_showcase(modele, cfg, norm_stats, INPUTS, OUTPUTS, output_dir: Path) -> dict:
    mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
    sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)
    mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
    sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype(np.float32)
    biais_repos = C._biais_repos(modele, mu_in, sd_in, mu_out, sd_out, cfg)

    rng = np.random.default_rng(cfg.SEED + 1)  # independent of the training scenario draws
    family_summary = {}
    for family in scenarios.FAMILY_SHARES:
        left_bc, right_bc, u0 = build_family_scenario(family, cfg, rng)
        if u0 is None:
            U_reel = C.run_fd_simulation_general(left_bc, right_bc, cfg)
        else:
            U_reel = free_evolution.run_fd_simulation_free(left_bc, right_bc, u0, cfg, C)

        U_pred = C._autoregressive_rollout_general(modele, U_reel, INPUT_FIELDS, mu_in, sd_in, mu_out, sd_out,
                                                     biais_repos, left_bc, right_bc, cfg)

        gif_path = output_dir / f"propagation_onde_{family}.gif"
        make_family_animation(U_pred, U_reel, left_bc, right_bc, cfg, gif_path)

        fake_rollout = C.RolloutResultGeneral(U=U_pred, U_reel=U_reel, left_bc=left_bc, right_bc=right_bc)
        t_axis, l2_list, linf_list, smape_list = C.compute_errors(fake_rollout, cfg)
        summary = (f"L2 final={l2_list[-1]:.3e} max={max(l2_list):.3e} | "
                   f"sMAPE final={smape_list[-1]:.1f}% max={max(smape_list):.1f}%")
        family_summary[family] = summary
        print(f"  [{family:16s}] {summary}  -> {gif_path.name}")

    return family_summary


def main():
    args = parse_args()
    # Computed here (not at module level) so importing main.py just for its
    # constants (make_gif.py, check_equivalence.py...) doesn't create an
    # empty plots/simulation_.../ folder on every import.
    OUTPUT_DIR = PLOTS_DIR / f"simulation_{datetime.now():%d%m%Y_%H%M%S}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_epochs = args.epochs if args.epochs is not None else (2 if args.smoke_test else 5)
    n_samples = args.n_samples if args.n_samples is not None else (16 if args.smoke_test else N_BC_SAMPLES)
    cfg = build_config(n_epochs, args.batch_size)
    C.set_seeds(cfg)
    waveforms.register(C)  # adds fourier/chirp/shock/filtered_random to commun.BC_WAVEFORMS (process-local only)

    patience = args.early_stop_patience if args.early_stop_patience > 0 else None

    mode = "SMOKE TEST" if args.smoke_test else "run"
    print(f"=== full_rollout_training_conv1d [{mode}] — input fields: {INPUT_FIELDS} — "
          f"{n_samples} random scenarios across 7 signal families "
          f"(A:{cfg.AMP_MIN}-{cfg.AMP_MAX}, omega:{cfg.OMEGA_MIN}-{cfg.OMEGA_MAX}), "
          f"{n_epochs} epochs (early-stop patience={patience}), plain one-step training, "
          f"batch size {cfg.BATCH_SIZE} ===")

    rng = np.random.default_rng(cfg.SEED)
    scenario_list = scenarios.sample_scenarios(cfg, n_samples, C, rng)
    bc_pairs = [(bc_left, bc_right) for bc_left, bc_right, _ in scenario_list]

    df, FIELDS, INPUTS, OUTPUTS = dataset_multisignal.generate_dataset_multisignal(INPUT_FIELDS, cfg, scenario_list)
    print(f"{len(df):,} rows x {df.shape[1]} columns ({len(FIELDS)} simulations)")

    # idx_train/idx_val/idx_test (lists of simulation indices) aren't needed
    # by plain teacher-forcing training -- it batches over df's row-level
    # "split" column instead (via make_dataloaders below). Only rollout_idx
    # is still used: it pins the one simulation used for the diagnostic
    # rollout/benchmark/PDE-consistency plots further down.
    df, _, _, _, rollout_idx = split_by_simulation(bc_pairs, df, cfg)
    norm_stats = compute_norm_stats(df, INPUTS, OUTPUTS, cfg)
    # Persisted next to model.pth (not in OUTPUT_DIR, which changes name
    # every day) so that test/check_equivalence.py always find the
    # normalization stats of the last trained model, without regenerating
    # the dataset.
    norm_stats.to_csv(PROJECT_DIR / "norm_stats.csv")

    modele = ReseauConv(n_lags=cfg.M_BACK, n_points=2 * cfg.SS + 1,
                        n_fields=len(INPUT_FIELDS), n_outputs=len(OUTPUTS))
    print(modele)
    print(f"Parameters: {sum(p.numel() for p in modele.parameters()):,}")

    # Plain one-step training: each row of df is a REAL (ground-truth)
    # M_BACK window -> the two true delta_u horizons. train_model predicts,
    # takes nn.MSELoss()'s default mean reduction (== averaging the error
    # over BOTH horizons, and over the batch), then backward()+step()
    # immediately every batch -- no rollout, no gradient accumulation across
    # multiple predictions. PF_SAMPLES=[] since LAMBDA_PF=0 above disables
    # the pushforward term entirely (never touched).
    train_loader, X_val, y_val = C.make_dataloaders(df, INPUTS, OUTPUTS, norm_stats, cfg)
    mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
    sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)
    mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
    sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype(np.float32)
    train_result = C.train_model(modele, train_loader, X_val, y_val, FIELDS, [], INPUT_FIELDS,
                                  mu_in, sd_in, mu_out, sd_out, cfg, model_path=PROJECT_DIR / "model.pth",
                                  patience=patience)
    C.plot_training_curve(train_result, OUTPUT_DIR)

    df_test = df[df["split"] == "test"].reset_index(drop=True)
    tf_metrics = C.evaluate_teacher_forcing(modele, df_test, INPUTS, OUTPUTS, norm_stats, OUTPUT_DIR)

    rollout = C.run_rollout_general(modele, FIELDS, bc_pairs, rollout_idx, INPUT_FIELDS, norm_stats,
                                     INPUTS, OUTPUTS, cfg)
    C.plot_utt_uxx(rollout, cfg, OUTPUT_DIR)
    C.make_rollout_animation(rollout, cfg, OUTPUT_DIR)

    errors = C.compute_errors(rollout, cfg)
    t_axis, l2_list, linf_list, smape_list = errors
    C.plot_rollout_error(t_axis, l2_list, linf_list, OUTPUT_DIR)
    C.plot_smape(t_axis, smape_list, OUTPUT_DIR)

    bench = C.benchmark_inference_general(modele, FIELDS, INPUT_FIELDS, norm_stats, INPUTS, OUTPUTS, rollout, cfg)

    print(f"Rendering one showcase rollout per signal family into {OUTPUT_DIR} ...")
    family_summary = run_family_showcase(modele, cfg, norm_stats, INPUTS, OUTPUTS, OUTPUT_DIR)

    extra_info = {
        "Architecture": " ".join(str(modele).split()),
        "Training scheme": "plain one-step (teacher forcing) -- no rollout, no TBPTT",
        "CLI args": f"epochs={n_epochs}, n_samples={n_samples}, batch_size={cfg.BATCH_SIZE}, "
                    f"early_stop_patience={patience}",
    }
    for family, summary in family_summary.items():
        extra_info[f"Family {family}"] = summary
    C.export_resume_general(OUTPUT_DIR, cfg, METHOD_NAME, df, INPUTS, OUTPUTS, train_result, tf_metrics,
                             rollout, bench, errors, extra_info)

    if args.smoke_test:
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"Peak memory (smoke test): {peak_rss_mb:.0f} MB")

    print(f"Done — outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
