# Entry point: PLAIN ONE-STEP (teacher-forcing) training with generalized
# (both ends independently forced, see scenarios.py) boundary conditions,
# restricted to 5 non-sum signal families (gaussian, sinusoid, step, ramp,
# rest) -- full reuse (without modification) of commun.py's evaluation/
# plotting/training functions, including the *_general physics functions
# and the early-stopping `patience` argument already built into train_model.
#
# For each training row, predicts the delta_u horizons from a REAL
# (ground-truth) M_BACK history window, averages their error over the
# N_FWD horizons, and updates the weights immediately -- no autoregressive
# rollout, no TBPTT. Every batch is drawn straight from the flattened
# dataset (df/INPUTS/OUTPUTS) via commun.py's make_dataloaders.
#
# The old autoregressive "full rollout" + TBPTT training (single gaussian
# family, right-end-only forcing) is still in train.py/rollout_torch.py but
# is no longer called from here -- kept in place, unused, in case it's
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
import scenarios

sys.path.insert(0, str(COMMUN_DIR))
import commun as C

# U only: the network only ever sees past displacement at each node, no
# velocity (Ut) or curvature (Uxx) features.
INPUT_FIELDS = ["U"]
METHOD_NAME = "U_only_teacher_forcing_multiwave"

# code/ is a subfolder of Beam_surrogate_model/training/ -- plots/ and logs/
# are its sibling folders; model.pth and norm_stats.csv stay at the
# Beam_surrogate_model/ level (see PLOTS_DIR/PROJECT_DIR below).
TRAINING_DIR = SCRIPT_DIR.parent
PROJECT_DIR = TRAINING_DIR.parent
PLOTS_DIR = TRAINING_DIR / "plots"

# M_BACK=4/N_FWD=4 and a wider network (512,256,64 vs the old 64,32,16) --
# plain teacher-forcing has no rollout to help it, so more temporal context
# and capacity are given to make up for it. NOISE_STD=0: plain teacher-forcing
# is meant to be exactly "real M_BACK window in -> predict -> average the
# horizons' error -> update", with no input noise (a commun.train_model
# feature, off by default) and no pushforward augmentation (removed from this
# project's vendored commun.py entirely -- see commun.train_model).
CONFIG_OVERRIDES = dict(
    M_BACK=4,
    N_FWD=4,
    HIDDEN_SIZES=(512, 256, 64),
    AMP_MIN=0.005,
    AMP_MAX=0.15,
    OMEGA_MIN=1.0,
    OMEGA_MAX=10.0,
    NOISE_STD=0.0,
)

N_SCENARIOS = 400  # number of randomly-sampled (left, right) boundary scenarios
PATIENCE = 20       # epochs without a val improvement before stopping early


def parse_args():
    p = argparse.ArgumentParser(description="Plain one-step (teacher-forcing) training with both ends "
                                             "independently forced, randomly distributed across 5 signal "
                                             "families (gaussian, sinusoid, step, ramp, rest).")
    p.add_argument("--smoke-test", action="store_true",
                    help="Miniature run (few scenarios, few epochs) to check that everything "
                         "runs without error before a full, expensive run.")
    p.add_argument("--epochs", type=int, default=None,
                    help="Number of epochs (default: 2 in --smoke-test, 300 otherwise).")
    p.add_argument("--n-samples", type=int, default=None,
                    help=f"Number of randomly-sampled (left, right) scenarios to simulate "
                         f"(default: 16 in --smoke-test, {N_SCENARIOS} otherwise).")
    p.add_argument("--batch-size", type=int, default=None,
                    help="Rows per gradient update (default: Config.BATCH_SIZE). Each batch's loss is "
                         "the average error over the whole batch AND all delta_u horizons.")
    p.add_argument("--patience", type=int, default=PATIENCE,
                    help=f"Stop training after this many epochs without a val improvement "
                         f"(default: {PATIENCE}). Pass a value >= --epochs to disable early stopping.")
    return p.parse_args()


def build_config(n_epochs, batch_size):
    kwargs = {"N_EPOCHS": n_epochs, **CONFIG_OVERRIDES}
    if batch_size is not None:
        kwargs["BATCH_SIZE"] = batch_size
    return C.Config(**kwargs)


def main():
    args = parse_args()
    # Computed here (not at module level) so importing main.py just for its
    # constants (test.py, test_prediction.py...) doesn't create an empty
    # plots/simulation_.../ folder on every import.
    OUTPUT_DIR = PLOTS_DIR / f"simulation_{datetime.now():%d%m%Y_%H%M%S}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_epochs = args.epochs if args.epochs is not None else (2 if args.smoke_test else 300)
    n_samples = args.n_samples if args.n_samples is not None else (16 if args.smoke_test else N_SCENARIOS)
    cfg = build_config(n_epochs, args.batch_size)
    C.set_seeds(cfg)

    mode = "SMOKE TEST" if args.smoke_test else "run"
    print(f"=== teacher_forcing_multiwave [{mode}] -- input fields: {INPUT_FIELDS} -- "
          f"{n_samples} random (left, right) scenarios across {scenarios.ALLOWED_FAMILIES} "
          f"(A:{cfg.AMP_MIN}-{cfg.AMP_MAX}, omega:{cfg.OMEGA_MIN}-{cfg.OMEGA_MAX}), "
          f"{n_epochs} epochs, plain one-step training, batch size {cfg.BATCH_SIZE}, "
          f"patience {args.patience} ===")

    rng = np.random.default_rng(cfg.SEED)
    bc_pairs = scenarios.sample_scenarios(cfg, n_samples, C, rng)

    df, FIELDS, INPUTS, OUTPUTS = C.generate_dataset_general(INPUT_FIELDS, cfg, bc_pairs)
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
    # multiple predictions, no pushforward. patience=args.patience stops
    # training once val hasn't improved for that many epochs in a row (the
    # best checkpoint, already saved on every improvement, is what gets
    # reloaded at the end either way).
    train_loader, X_val, y_val = C.make_dataloaders(df, INPUTS, OUTPUTS, norm_stats, cfg)
    mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
    sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)
    mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
    sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype(np.float32)
    train_result = C.train_model(modele, train_loader, X_val, y_val, mu_in, sd_in, mu_out, sd_out, cfg,
                                  model_path=PROJECT_DIR / "model.pth", patience=args.patience)
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
        "Wave families": ", ".join(scenarios.ALLOWED_FAMILIES),
        "CLI args": f"epochs={n_epochs}, n_samples={n_samples}, batch_size={cfg.BATCH_SIZE}, "
                    f"patience={args.patience}",
    }
    C.export_resume_general(OUTPUT_DIR, cfg, METHOD_NAME, df, INPUTS, OUTPUTS, train_result, tf_metrics,
                             rollout, bench, errors, extra_info)

    if args.smoke_test:
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"Peak memory (smoke test): {peak_rss_mb:.0f} MB")

    print(f"Done — outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
