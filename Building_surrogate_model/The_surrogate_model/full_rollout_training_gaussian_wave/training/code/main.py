# Entry point: differentiable "full rollout" training, then full reuse
# (without modification) of the evaluation and plotting functions already
# present in commun.py.
import argparse
import resource
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR
from data_split import split_by_simulation, compute_norm_stats
from train import train_full_rollout, plot_rollout_training_curve

sys.path.insert(0, str(COMMUN_DIR))
import commun as C

INPUT_FIELDS = ["U", "Ut", "Uxx"]
METHOD_NAME = "full_rollout_U_Ut_Uxx_gaussian_wide"

# code/ is a subfolder of full_rollout_training_gaussian_wave/training/ --
# plots/ and logs/ are its sibling folders; model.pth and norm_stats.csv
# stay at the full_rollout_training_gaussian_wave/ level (see Current_model).
TRAINING_DIR = SCRIPT_DIR.parent
PROJECT_DIR = TRAINING_DIR.parent
PLOTS_DIR = TRAINING_DIR / "plots"

# Widened grid (vs Current_model, N_GRID=10 / A:0.005-0.1 / omega:3-10) to
# cover many more Gaussian wave shapes: 20x20 = 400 simulations,
# amplitude 0.005-0.15, omega 1-10 (= the whole physically useful range, see
# u_right_val in commun.py: sigma = interp(omega, [1,10], [0.15,0.07]) --
# beyond omega=10 or below 1, the interp clamps and adds no extra pulse
# width). Centralized here (rather than duplicated) because
# test_prediction.py must rebuild a strictly identical Config so the
# reloaded model receives the right features.
CONFIG_OVERRIDES = dict(
    N_GRID=20,
    AMP_MIN=0.005,
    AMP_MAX=0.15,
    OMEGA_MIN=1.0,
    OMEGA_MAX=10.0,
)


def parse_args():
    p = argparse.ArgumentParser(description="Differentiable full-rollout training (no detach).")
    p.add_argument("--smoke-test", action="store_true",
                    help="Miniature run (reduced grid, few epochs) to check that everything "
                         "runs without error before a full, expensive run.")
    p.add_argument("--epochs", type=int, default=None,
                    help="Number of epochs (default: 2 in --smoke-test, 5 otherwise).")
    p.add_argument("--group-size", type=int, default=8,
                    help="Number of simulations rolled out in parallel per weight update.")
    p.add_argument("--tbptt-hops", type=int, default=10,
                    help="Number of hops rolled out before each weight correction (cuts the "
                         "gradient thread without ever resetting the state to ground truth).")
    return p.parse_args()


def build_config(args, n_epochs):
    # N_EPOCHS is passed back to Config (even though train_full_rollout
    # receives n_epochs separately) only so that C.export_resume shows the
    # correct number of epochs in resume.txt.
    kwargs = {"N_EPOCHS": n_epochs, **CONFIG_OVERRIDES}
    if args.smoke_test:
        kwargs["N_GRID"] = 4  # 16 simulations instead of 400 -- enough to check it runs
    return C.Config(**kwargs)


def main():
    args = parse_args()
    # Computed here (not at module level) so importing main.py just for its
    # constants (make_gif.py, test_prediction.py...) doesn't create an empty
    # plots/simulation_.../ folder on every import.
    OUTPUT_DIR = PLOTS_DIR / f"simulation_{datetime.now():%d%m%Y_%H%M%S}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_epochs = args.epochs if args.epochs is not None else (2 if args.smoke_test else 5)
    cfg = build_config(args, n_epochs)
    C.set_seeds(cfg)

    mode = "SMOKE TEST" if args.smoke_test else "run"
    print(f"=== full_rollout_training_gaussian_wave [{mode}] — input fields: {INPUT_FIELDS} — "
          f"grid {cfg.N_GRID}x{cfg.N_GRID} (A:{cfg.AMP_MIN}-{cfg.AMP_MAX}, omega:{cfg.OMEGA_MIN}-{cfg.OMEGA_MAX}), "
          f"{n_epochs} epochs, groups of {args.group_size}, correction every {args.tbptt_hops} hops ===")

    df, FIELDS, INPUTS, OUTPUTS = C.generate_dataset(INPUT_FIELDS, cfg)
    print(f"{len(df):,} rows x {df.shape[1]} columns ({len(FIELDS)} simulations)")

    df, pairs_train, pairs_val, pairs_test = split_by_simulation(df, cfg)
    norm_stats = compute_norm_stats(df, INPUTS, OUTPUTS, cfg)
    # Persisted next to model.pth (not in OUTPUT_DIR, which changes name
    # every day) so that test_prediction.py always finds the normalization
    # stats of the last trained model, without regenerating the dataset.
    norm_stats.to_csv(PROJECT_DIR / "norm_stats.csv")

    modele = C.Reseau(n_inputs=len(INPUTS), n_outputs=len(OUTPUTS), hidden_sizes=cfg.HIDDEN_SIZES)
    print(modele)

    train_result = train_full_rollout(modele, FIELDS, pairs_train, pairs_val, INPUT_FIELDS,
                                       norm_stats, INPUTS, OUTPUTS, cfg, group_size=args.group_size,
                                       n_epochs=n_epochs, model_path=PROJECT_DIR / "model.pth",
                                       tbptt_hops=args.tbptt_hops)
    plot_rollout_training_curve(train_result, OUTPUT_DIR)

    df_test = df[df["split"] == "test"].reset_index(drop=True)
    tf_metrics = C.evaluate_teacher_forcing(modele, df_test, INPUTS, OUTPUTS, norm_stats, OUTPUT_DIR)

    rollout = C.run_rollout(modele, FIELDS, INPUT_FIELDS, norm_stats, INPUTS, OUTPUTS, cfg)
    C.plot_utt_uxx(rollout, cfg, OUTPUT_DIR)
    C.make_rollout_animation(rollout, cfg, OUTPUT_DIR)

    errors = C.compute_errors(rollout, cfg)
    t_axis, l2_list, linf_list, smape_list = errors
    C.plot_rollout_error(t_axis, l2_list, linf_list, OUTPUT_DIR)
    C.plot_smape(t_axis, smape_list, OUTPUT_DIR)

    bench = C.benchmark_inference(modele, FIELDS, INPUT_FIELDS, norm_stats, INPUTS, OUTPUTS, rollout, cfg)

    C.export_resume(OUTPUT_DIR, cfg, METHOD_NAME, df, INPUTS, OUTPUTS, train_result, tf_metrics, rollout, bench, errors)

    if args.smoke_test:
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"Peak memory (smoke test): {peak_rss_mb:.0f} MB")

    print(f"Done — outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
