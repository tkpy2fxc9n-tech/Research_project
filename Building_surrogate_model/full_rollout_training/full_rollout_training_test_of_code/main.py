# Diagnostic variant of full_rollout_training: LEARNING_RATE=5e-3 (5x the
# default 1e-3), everything else identical (same fields, same
# hyperparameters, same split, SMOOTH_ALPHA=0.20 like the reference). Goal:
# see whether a more aggressive learning rate converges better/faster on
# the full rollout. Does not modify full_rollout_training/.
import argparse
import resource
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR
from data_split import split_by_simulation, compute_norm_stats
from train import train_full_rollout, plot_rollout_training_curve

sys.path.insert(0, str(COMMUN_DIR))
import commun as C

INPUT_FIELDS = ["U", "Ut", "Uxx"]
METHOD_NAME = "full_rollout_U_Ut_Uxx_lr5e-3"

OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


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
    # LEARNING_RATE=5e-3: only change vs full_rollout_training (1e-3 by default).
    kwargs = {"N_EPOCHS": n_epochs, "LEARNING_RATE": 5e-3}
    if args.smoke_test:
        kwargs["N_GRID"] = 4  # 16 simulations instead of 100 -- enough to check it runs
    return C.Config(**kwargs)


def main():
    args = parse_args()
    n_epochs = args.epochs if args.epochs is not None else (2 if args.smoke_test else 5)
    cfg = build_config(args, n_epochs)
    C.set_seeds(cfg)

    mode = "SMOKE TEST" if args.smoke_test else "run"
    print(f"=== full_rollout_training_upgrade [{mode}] — input fields: {INPUT_FIELDS} — "
          f"grid {cfg.N_GRID}x{cfg.N_GRID}, {n_epochs} epochs, groups of {args.group_size}, "
          f"correction every {args.tbptt_hops} hops — LEARNING_RATE={cfg.LEARNING_RATE} ===")

    df, FIELDS, INPUTS, OUTPUTS = C.generate_dataset(INPUT_FIELDS, cfg)
    print(f"{len(df):,} rows x {df.shape[1]} columns ({len(FIELDS)} simulations)")

    df, pairs_train, pairs_val, pairs_test = split_by_simulation(df, cfg)
    norm_stats = compute_norm_stats(df, INPUTS, OUTPUTS, cfg)

    modele = C.Reseau(n_inputs=len(INPUTS), n_outputs=len(OUTPUTS), hidden_sizes=cfg.HIDDEN_SIZES)
    print(modele)

    train_result = train_full_rollout(modele, FIELDS, pairs_train, pairs_val, INPUT_FIELDS,
                                       norm_stats, INPUTS, OUTPUTS, cfg, group_size=args.group_size,
                                       n_epochs=n_epochs, model_path=SCRIPT_DIR / "model.pth",
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
