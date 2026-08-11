# Entry point: differentiable "full rollout" training, then full reuse
# (without modification) of the evaluation and plotting functions already
# present in commun.py. Boundary conditions are generalized (dirichlet,
# either end independently) and restricted by scenarios.py to the
# "gaussian"/"rest" families -- see scenarios.ALLOWED_FAMILIES.
import argparse
import resource
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from config import Config, INPUT_FIELDS, METHOD_NAME, N_SCENARIOS
from data_split import split_by_simulation, compute_norm_stats
from train import train_full_rollout, plot_rollout_training_curve
import scenarios
import commun as C

# code/ is a subfolder of full_rollout_training/training/ -- plots/ and logs/
# are its sibling folders; model.pth stays at the full_rollout_training/ level.
TRAINING_DIR = SCRIPT_DIR.parent
PROJECT_DIR = TRAINING_DIR.parent
PLOTS_DIR = TRAINING_DIR / "plots"
# One subfolder per run (date + time, not just the date) -- older runs
# therefore all stay browsable under plots/, even several per day.
OUTPUT_DIR = PLOTS_DIR / f"simulation_{datetime.now():%d%m%Y_%H%M%S}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Differentiable full-rollout training (no detach), boundary "
                                             f"conditions independently gaussian/rest per end "
                                             f"({scenarios.ALLOWED_FAMILIES}).")
    p.add_argument("--smoke-test", action="store_true",
                    help="Miniature run (few scenarios, few epochs) to check that everything "
                         "runs without error before a full, expensive run.")
    p.add_argument("--epochs", type=int, default=None,
                    help=f"Number of epochs (default: 2 in --smoke-test, config.Config.N_EPOCHS="
                         f"{Config().N_EPOCHS} otherwise).")
    p.add_argument("--n-samples", type=int, default=None,
                    help=f"Number of randomly-sampled (left, right) BC scenarios to simulate "
                         f"(default: 16 in --smoke-test, config.N_SCENARIOS={N_SCENARIOS} otherwise).")
    p.add_argument("--group-size", type=int, default=None,
                    help=f"Number of simulations rolled out in parallel per weight update "
                         f"(default: config.Config.GROUP_SIZE={Config().GROUP_SIZE}).")
    p.add_argument("--tbptt-hops", type=int, default=None,
                    help=f"Number of hops rolled out before each weight correction -- cuts the "
                         f"gradient thread without ever resetting the state to ground truth "
                         f"(default: config.Config.TBPTT_HOPS={Config().TBPTT_HOPS}).")
    return p.parse_args()


def build_config(args, n_epochs):
    # Every default lives in config.Config -- this only forwards CLI
    # overrides (all default to None) on top of it, so config.py stays the
    # single place to edit for a lasting change. N_EPOCHS is passed back to
    # Config (even though train_full_rollout receives n_epochs separately)
    # only so that C.export_resume_general shows the correct number of
    # epochs in resume.txt.
    kwargs = {"N_EPOCHS": n_epochs}
    if args.group_size is not None:
        kwargs["GROUP_SIZE"] = args.group_size
    if args.tbptt_hops is not None:
        kwargs["TBPTT_HOPS"] = args.tbptt_hops
    return Config(**kwargs)


def main():
    args = parse_args()
    n_epochs = args.epochs if args.epochs is not None else (2 if args.smoke_test else Config().N_EPOCHS)
    n_samples = args.n_samples if args.n_samples is not None else (16 if args.smoke_test else N_SCENARIOS)
    cfg = build_config(args, n_epochs)
    C.set_seeds(cfg)

    mode = "SMOKE TEST" if args.smoke_test else "run"
    print(f"=== full_rollout_training [{mode}] — input fields: {INPUT_FIELDS} — "
          f"{n_samples} random (left, right) BC scenarios across {scenarios.ALLOWED_FAMILIES} "
          f"(A:{cfg.AMP_MIN}-{cfg.AMP_MAX}, omega:{cfg.OMEGA_MIN}-{cfg.OMEGA_MAX}), "
          f"{n_epochs} epochs, groups of {cfg.GROUP_SIZE}, correction every {cfg.TBPTT_HOPS} hops ===")

    rng = np.random.default_rng(cfg.SEED)
    bc_pairs = scenarios.sample_scenarios(cfg, n_samples, C, rng)

    df, FIELDS, INPUTS, OUTPUTS = C.generate_dataset_general(INPUT_FIELDS, cfg, bc_pairs)
    print(f"{len(df):,} rows x {df.shape[1]} columns ({len(FIELDS)} simulations)")

    df, idx_train, idx_val, idx_test, rollout_idx = split_by_simulation(bc_pairs, df, cfg)
    norm_stats = compute_norm_stats(df, INPUTS, OUTPUTS, cfg)

    modele = C.Reseau(n_inputs=len(INPUTS), n_outputs=len(OUTPUTS), hidden_sizes=cfg.HIDDEN_SIZES)
    print(modele)

    train_result = train_full_rollout(modele, FIELDS, bc_pairs, idx_train, idx_val, INPUT_FIELDS,
                                       norm_stats, INPUTS, OUTPUTS, cfg, group_size=cfg.GROUP_SIZE,
                                       n_epochs=n_epochs, model_path=PROJECT_DIR / "model.pth",
                                       tbptt_hops=cfg.TBPTT_HOPS)
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

    extra_info = {"Scenarios": f"{n_samples} random (left, right) pairs, families={scenarios.ALLOWED_FAMILIES}"}
    C.export_resume_general(OUTPUT_DIR, cfg, METHOD_NAME, df, INPUTS, OUTPUTS, train_result, tf_metrics,
                             rollout, bench, errors, extra_info)

    if args.smoke_test:
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"Peak memory (smoke test): {peak_rss_mb:.0f} MB")

    print(f"Done — outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
