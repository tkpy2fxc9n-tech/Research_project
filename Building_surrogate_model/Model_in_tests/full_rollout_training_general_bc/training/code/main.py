# Entry point: differentiable "full rollout" training with generalized
# (arbitrary Dirichlet/Neumann, several waveform families, both ends)
# boundary conditions -- full reuse (without modification) of the
# evaluation/plotting functions already present in commun.py, plus the new
# *_general physics functions added there for this project.
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
from train import train_full_rollout, plot_rollout_training_curve

sys.path.insert(0, str(COMMUN_DIR))
import commun as C

INPUT_FIELDS = ["U", "Ut", "Uxx"]
METHOD_NAME = "full_rollout_general_bc"

# code/ is a subfolder of full_rollout_training_general_bc/training/ --
# plots/ and logs/ are its sibling folders; model.pth and norm_stats.csv
# stay at the full_rollout_training_general_bc/ level.
TRAINING_DIR = SCRIPT_DIR.parent
PROJECT_DIR = TRAINING_DIR.parent
PLOTS_DIR = TRAINING_DIR / "plots"

# Amplitude/frequency ranges shared by every waveform family in
# commun.BC_WAVEFORMS (same physically-useful range as the other
# full_rollout_training projects: beyond omega=10 or below 1, the Gaussian's
# interp clamps and adds no extra pulse width -- see u_right_val/gaussian_value
# in commun.py). Centralized here (not redefined elsewhere) so that
# test_prediction.py/make_gif*.py rebuild a strictly identical Config.
CONFIG_OVERRIDES = dict(
    AMP_MIN=0.005,
    AMP_MAX=0.15,
    OMEGA_MIN=1.0,
    OMEGA_MAX=10.0,
)

N_BC_SAMPLES = 400  # number of randomly-sampled (left_bc, right_bc) simulations


def parse_args():
    p = argparse.ArgumentParser(description="Differentiable full-rollout training with generalized "
                                             "(arbitrary Dirichlet/Neumann) boundary conditions.")
    p.add_argument("--smoke-test", action="store_true",
                    help="Miniature run (few samples, few epochs) to check that everything "
                         "runs without error before a full, expensive run.")
    p.add_argument("--epochs", type=int, default=None,
                    help="Number of epochs (default: 2 in --smoke-test, 5 otherwise).")
    p.add_argument("--n-samples", type=int, default=None,
                    help=f"Number of randomly-sampled boundary-condition pairs to simulate "
                         f"(default: 16 in --smoke-test, {N_BC_SAMPLES} otherwise).")
    p.add_argument("--group-size", type=int, default=8,
                    help="Number of simulations rolled out in parallel per weight update.")
    p.add_argument("--tbptt-hops", type=int, default=10,
                    help="Number of hops rolled out before each weight correction (cuts the "
                         "gradient thread without ever resetting the state to ground truth).")
    return p.parse_args()


def build_config(n_epochs):
    # N_EPOCHS is passed back to Config (even though train_full_rollout
    # receives n_epochs separately) only so that C.export_resume_general
    # shows the correct number of epochs in resume.txt.
    return C.Config(N_EPOCHS=n_epochs, **CONFIG_OVERRIDES)


def main():
    args = parse_args()
    # Computed here (not at module level) so importing main.py just for its
    # constants (make_gif.py, test_prediction.py...) doesn't create an empty
    # plots/simulation_.../ folder on every import.
    OUTPUT_DIR = PLOTS_DIR / f"simulation_{datetime.now():%d%m%Y_%H%M%S}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_epochs = args.epochs if args.epochs is not None else (2 if args.smoke_test else 5)
    n_samples = args.n_samples if args.n_samples is not None else (16 if args.smoke_test else N_BC_SAMPLES)
    cfg = build_config(n_epochs)
    C.set_seeds(cfg)

    mode = "SMOKE TEST" if args.smoke_test else "run"
    print(f"=== full_rollout_training_general_bc [{mode}] -- input fields: {INPUT_FIELDS} -- "
          f"{n_samples} random (left,right) boundary-condition pairs "
          f"(A:{cfg.AMP_MIN}-{cfg.AMP_MAX}, omega:{cfg.OMEGA_MIN}-{cfg.OMEGA_MAX}), "
          f"{n_epochs} epochs, groups of {args.group_size}, correction every {args.tbptt_hops} hops ===")

    rng = np.random.default_rng(cfg.SEED)
    bc_pairs = C.sample_bc_pairs(cfg, n_samples, rng)

    df, FIELDS, INPUTS, OUTPUTS = C.generate_dataset_general(INPUT_FIELDS, cfg, bc_pairs)
    print(f"{len(df):,} rows x {df.shape[1]} columns ({len(FIELDS)} simulations)")

    df, idx_train, idx_val, idx_test, rollout_idx = split_by_simulation(bc_pairs, df, cfg)
    norm_stats = compute_norm_stats(df, INPUTS, OUTPUTS, cfg)
    # Persisted next to model.pth (not in OUTPUT_DIR, which changes name
    # every day) so that test scripts always find the normalization stats
    # of the last trained model, without regenerating the dataset.
    norm_stats.to_csv(PROJECT_DIR / "norm_stats.csv")

    modele = C.Reseau(n_inputs=len(INPUTS), n_outputs=len(OUTPUTS), hidden_sizes=cfg.HIDDEN_SIZES)
    print(modele)

    train_result = train_full_rollout(modele, FIELDS, bc_pairs, idx_train, idx_val, INPUT_FIELDS,
                                       norm_stats, INPUTS, OUTPUTS, cfg, group_size=args.group_size,
                                       n_epochs=n_epochs, model_path=PROJECT_DIR / "model.pth",
                                       tbptt_hops=args.tbptt_hops)
    plot_rollout_training_curve(train_result, OUTPUT_DIR)

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

    C.export_resume_general(OUTPUT_DIR, cfg, METHOD_NAME, df, INPUTS, OUTPUTS, train_result, tf_metrics,
                             rollout, bench, errors)

    if args.smoke_test:
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"Peak memory (smoke test): {peak_rss_mb:.0f} MB")

    print(f"Done — outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
