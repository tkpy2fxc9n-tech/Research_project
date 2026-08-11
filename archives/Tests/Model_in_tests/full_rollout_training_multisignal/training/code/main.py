# Entry point: PLAIN ONE-STEP (teacher-forcing) training with generalized
# (arbitrary Dirichlet/Neumann, both ends) boundary conditions, restricted to
# the 7 signal families from the project's design table (see scenarios.py
# for the shares, randomly sampled per scenario -- unchanged by this file)
# instead of full_rollout_training_general_bc's 6 built-in families -- full
# reuse (without modification) of the evaluation/plotting functions already
# present in commun.py, plus the *_general physics functions added there for
# that project, plus this project's own waveforms.py / scenarios.py /
# free_evolution.py / dataset_multisignal.py.
#
# For each training row, predicts the delta_u horizons from a REAL
# (ground-truth) M_BACK history window, averages their error, and updates the
# weights immediately -- no autoregressive rollout, no TBPTT. Every batch is
# drawn straight from the flattened dataset (df/INPUTS/OUTPUTS), reusing
# commun.py's existing train_model/make_dataloaders instead of a hand-rolled
# loop. The old autoregressive "full rollout" + TBPTT training is still
# implemented in train.py/rollout_torch.py (validated by check_equivalence.py)
# but is no longer called from here -- kept in place, unused, in case it's
# wanted again later.
import argparse
import resource
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR
from data_split import split_by_simulation, compute_norm_stats
import waveforms
import scenarios
import dataset_multisignal

sys.path.insert(0, str(COMMUN_DIR))
import commun as C

# U only: the network only ever sees past displacement at each node, no
# velocity (Ut) or curvature (Uxx) features.
INPUT_FIELDS = ["U"]
METHOD_NAME = "U_only_multisignal_teacher_forcing"

# code/ is a subfolder of full_rollout_training_multisignal/training/ --
# plots/ and logs/ are its sibling folders; model.pth and norm_stats.csv
# stay at the full_rollout_training_multisignal/ level.
TRAINING_DIR = SCRIPT_DIR.parent
PROJECT_DIR = TRAINING_DIR.parent
PLOTS_DIR = TRAINING_DIR / "plots"

# M_BACK=4/N_FWD=4 (was 2/2) and a much wider network (512,256,64 vs
# 64,32,16) -- see scenarios.FAMILY_SHARES for the 7-family training-signal
# mix these amplitude/omega ranges feed into. Centralized here (not
# redefined elsewhere) so that test_prediction.py/make_gif*.py rebuild a
# strictly identical Config.
# LAMBDA_PF=0 / NOISE_STD=0: plain teacher-forcing is meant to be exactly
# "real M_BACK window in -> predict -> average the horizons' error -> update",
# with no pushforward augmentation and no input noise -- both are
# commun.train_model features, off by default. (commun.make_pf_samples/
# pushforward_loss also hardcode the (A, omega)-keyed FIELDS structure from
# the dense-grid projects, not the sim_idx-keyed general-BC one used here, so
# LAMBDA_PF>0 would need that generalized first.)
CONFIG_OVERRIDES = dict(
    M_BACK=4,
    N_FWD=4,
    HIDDEN_SIZES=(512, 256, 64),
    AMP_MIN=0.005,
    AMP_MAX=0.15,
    OMEGA_MIN=1.0,
    OMEGA_MAX=10.0,
    LAMBDA_PF=0.0,
    NOISE_STD=0.0,
)

N_BC_SAMPLES = 400  # number of randomly-sampled scenarios (same default as full_rollout_training_general_bc)


def parse_args():
    p = argparse.ArgumentParser(description="Plain one-step (teacher-forcing) training with generalized "
                                             "boundary conditions, randomly distributed across 7 signal families.")
    p.add_argument("--smoke-test", action="store_true",
                    help="Miniature run (few samples, few epochs) to check that everything "
                         "runs without error before a full, expensive run.")
    p.add_argument("--epochs", type=int, default=None,
                    help="Number of epochs (default: 2 in --smoke-test, 300 otherwise). Plain teacher-forcing "
                         "is much cheaper per epoch than the old full-rollout scheme -- reconsider this default "
                         "once a first run shows how loss/time actually behaves.")
    p.add_argument("--n-samples", type=int, default=None,
                    help=f"Number of randomly-sampled scenarios to simulate "
                         f"(default: 16 in --smoke-test, {N_BC_SAMPLES} otherwise).")
    p.add_argument("--batch-size", type=int, default=None,
                    help="Rows per gradient update (default: Config.BATCH_SIZE, currently 512). Each batch's "
                         "loss is the average error over the whole batch AND all delta_u horizons.")
    return p.parse_args()


def build_config(n_epochs, batch_size):
    kwargs = {"N_EPOCHS": n_epochs, **CONFIG_OVERRIDES}
    if batch_size is not None:
        kwargs["BATCH_SIZE"] = batch_size
    return C.Config(**kwargs)


def main():
    args = parse_args()
    # Computed here (not at module level) so importing main.py just for its
    # constants (make_gif.py, test_prediction.py...) doesn't create an empty
    # plots/simulation_.../ folder on every import.
    OUTPUT_DIR = PLOTS_DIR / f"simulation_{datetime.now():%d%m%Y_%H%M%S}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_epochs = args.epochs if args.epochs is not None else (2 if args.smoke_test else 300)
    n_samples = args.n_samples if args.n_samples is not None else (16 if args.smoke_test else N_BC_SAMPLES)
    cfg = build_config(n_epochs, args.batch_size)
    C.set_seeds(cfg)
    waveforms.register(C)  # adds fourier/chirp/shock/filtered_random to commun.BC_WAVEFORMS (process-local only)

    mode = "SMOKE TEST" if args.smoke_test else "run"
    print(f"=== full_rollout_training_multisignal [{mode}] -- input fields: {INPUT_FIELDS} -- "
          f"{n_samples} random scenarios across 7 signal families "
          f"(A:{cfg.AMP_MIN}-{cfg.AMP_MAX}, omega:{cfg.OMEGA_MIN}-{cfg.OMEGA_MAX}), "
          f"{n_epochs} epochs, plain one-step training, batch size {cfg.BATCH_SIZE} ===")

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
    # every day) so that test scripts always find the normalization stats
    # of the last trained model, without regenerating the dataset.
    norm_stats.to_csv(PROJECT_DIR / "norm_stats.csv")

    modele = C.Reseau(n_inputs=len(INPUTS), n_outputs=len(OUTPUTS), hidden_sizes=cfg.HIDDEN_SIZES)
    print(modele)

    # Plain one-step training: each row of df is a REAL (ground-truth)
    # M_BACK window -> the true delta_u horizons. train_model predicts,
    # takes nn.MSELoss()'s default mean reduction (== averaging the error
    # over ALL horizons, and over the batch), then backward()+step()
    # immediately every batch -- no rollout, no gradient accumulation across
    # multiple predictions. PF_SAMPLES=[] since LAMBDA_PF=0 above disables
    # the pushforward term entirely (never touched).
    train_loader, X_val, y_val = C.make_dataloaders(df, INPUTS, OUTPUTS, norm_stats, cfg)
    mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
    sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)
    mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
    sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype(np.float32)
    train_result = C.train_model(modele, train_loader, X_val, y_val, FIELDS, [], INPUT_FIELDS,
                                  mu_in, sd_in, mu_out, sd_out, cfg, model_path=PROJECT_DIR / "model.pth")
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

    extra_info = {
        "Architecture": " ".join(str(modele).split()),
        "Training scheme": "plain one-step (teacher forcing) -- no rollout, no TBPTT",
        "CLI args": f"epochs={n_epochs}, n_samples={n_samples}, batch_size={cfg.BATCH_SIZE}",
    }
    C.export_resume_general(OUTPUT_DIR, cfg, METHOD_NAME, df, INPUTS, OUTPUTS, train_result, tf_metrics,
                             rollout, bench, errors, extra_info)

    if args.smoke_test:
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"Peak memory (smoke test): {peak_rss_mb:.0f} MB")

    print(f"Done — outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
