#!/usr/bin/env python3
"""
p6_complex_zeroshot: evaluate an already-trained model (trained on the
SIMPLE dataset) against the COMPLEX dataset's own test trajectory, with NO
retraining -- quantifies the cost of the complex dataset's wider IC/parameter
distribution on a model that never saw it.

Not a training run, so it doesn't go through `python -m beamsurrogate`
(cli.py has no "load a checkpoint and just evaluate" mode). This script
rebuilds the same pieces cli.py's run() would for evaluation -- model
architecture + weights from the source run, and CRITICALLY the SAME
normalization stats the model was trained with, recomputed from the SIMPLE
dataset, NOT the complex one. Reusing the model's own calibration (not a
fresh one fitted to the complex data) is the entire point of "zero-shot" --
recomputing norm_stats from the complex dataset instead would silently turn
this into a different, uncontrolled experiment.

Memory cost: loads BOTH datasets' full windowed tables, one after the other
(the simple one is freed before the complex one is built, keeping peak RSS
close to a single normal training run's, not double). Still substantial --
run this via sbatch with real --mem (whatever the source run itself needed,
see its own env.json/logs), not interactively on a login node:
    sbatch --job-name=p6_complex_zeroshot --mem=<match source run> \\
        --cpus-per-task=8 --time=4:00:00 --partition=medium \\
        --wrap="cd /home/aph25/Code_GH && /home/aph25/Desktop/wave_env/bin/python scripts/eval_zeroshot.py"

Usage: python scripts/eval_zeroshot.py
"""
from __future__ import annotations

import dataclasses
import gc
import re
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from beamsurrogate.config import Config, set_seeds  # noqa: E402
from beamsurrogate.registry import MODELS, DATASETS  # noqa: E402
from beamsurrogate.data.split import load_hdf5_dataset, compute_norm_stats  # noqa: E402
from beamsurrogate.training.teacher_forcing import evaluate_one_step  # noqa: E402
from beamsurrogate.evaluate.rollout import run_rollout, benchmark_inference  # noqa: E402
from beamsurrogate.evaluate.metrics import build_metrics  # noqa: E402
from beamsurrogate.evaluate import plots  # noqa: E402
from beamsurrogate.cli import _git_info, _dataset_sha256  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"

RUN_ID = "p6_complex_zeroshot"
RUN_DIR = RUNS_DIR / "p6" / RUN_ID   # runs/<phase>/<run_id>/, same convention as every other run
# TBD -- set to whichever LAMBDA_PHYSICS weight wins the p3_pinn_* (phase 3) sweep,
# once it concludes. That run's model.pth (trained on the SIMPLE dataset) is
# what gets evaluated zero-shot against the complex dataset here.
SOURCE_RUN_ID = None
TARGET_DATASET = "complex"


def _find_run_dir(run_id: str) -> Path:
    # A run's own folder lives under runs/<phase>/<run_id>/ -- phase is the
    # pN prefix already in run_id, never looked up separately.
    phase = re.match(r"p\d+", run_id).group()
    return RUNS_DIR / phase / run_id


def _load_config(run_id: str) -> Config:
    raw = yaml.safe_load((_find_run_dir(run_id) / "config.yaml").read_text())
    known = {f.name for f in dataclasses.fields(Config)} - {"run_id"}
    raw = {k: v for k, v in raw.items() if k in known}
    for tuple_field in ("HIDDEN_SIZES", "CNN_CHANNELS"):
        if tuple_field in raw and isinstance(raw[tuple_field], list):
            raw[tuple_field] = tuple(raw[tuple_field])
    return Config(run_id=run_id, **raw)


def main():
    if SOURCE_RUN_ID is None:
        print("ERROR: SOURCE_RUN_ID is not set -- edit this script once the p3_pinn_* (phase 3) sweep "
              "has a winner, then rerun.", file=sys.stderr)
        sys.exit(1)
    source_model_path = _find_run_dir(SOURCE_RUN_ID) / "model.pth"
    if not source_model_path.exists():
        print(f"ERROR: {source_model_path} not found -- {SOURCE_RUN_ID} may not have finished.",
              file=sys.stderr)
        sys.exit(1)

    cfg_source = _load_config(SOURCE_RUN_ID)
    set_seeds(cfg_source)

    # 1) Simple dataset, ONLY to reproduce this model's own training-time
    # norm_stats exactly (deterministic given fixed SEED/SPLIT_SEED) -- not
    # used for anything else, freed immediately after.
    simple_path = DATA_DIR / DATASETS[cfg_source.dataset]
    print(f"Loading source (simple) dataset {simple_path} to recompute norm_stats "
          f"exactly as {SOURCE_RUN_ID} was trained with...")
    (df_simple, _FIELDS_simple, INPUTS, OUTPUTS, *_rest) = load_hdf5_dataset(
        cfg_source.features, cfg_source, simple_path, max_trajectories=None)
    norm_stats = compute_norm_stats(df_simple, INPUTS, OUTPUTS, cfg_source)
    del df_simple, _FIELDS_simple, _rest
    gc.collect()

    model = MODELS[cfg_source.model](len(INPUTS), len(OUTPUTS), cfg_source)
    model.load_state_dict(torch.load(source_model_path, weights_only=True))
    model.eval()
    print(f"Loaded {SOURCE_RUN_ID}'s model ({sum(p.numel() for p in model.parameters()):,} params).")

    # 2) Complex dataset: the actual zero-shot evaluation target. Same cfg
    # (architecture/windowing) except dataset + run_id, so INPUTS/OUTPUTS
    # column layout matches the model exactly.
    cfg_target = dataclasses.replace(cfg_source, dataset=TARGET_DATASET, run_id=RUN_ID)
    complex_path = DATA_DIR / DATASETS[cfg_target.dataset]
    print(f"Loading target (complex) dataset {complex_path}...")
    (df_target, FIELDS, INPUTS_t, OUTPUTS_t, bc_pairs, idx_train, idx_val, idx_test,
     rollout_idx, family_showcase_idx) = load_hdf5_dataset(
        cfg_target.features, cfg_target, complex_path, max_trajectories=None)
    assert INPUTS_t == INPUTS and OUTPUTS_t == OUTPUTS, \
        "complex dataset produced a different column layout than the simple one -- cfg mismatch"

    df_target = df_target[df_target["split"] == "test"].reset_index(drop=True)

    one_step_metrics, y_true_onestep, y_pred_onestep = evaluate_one_step(
        model, df_target, INPUTS, OUTPUTS, norm_stats)

    rollout = run_rollout(model, FIELDS, bc_pairs, rollout_idx, cfg_target.features,
                           norm_stats, INPUTS, OUTPUTS, cfg_target)
    bench = benchmark_inference(model, FIELDS, cfg_target.features, norm_stats, INPUTS, OUTPUTS,
                                 rollout, cfg_target)

    metrics = build_metrics(cfg_target, train_result=None, rollout=rollout, bench=bench,
                             one_step_metrics=one_step_metrics)

    run_dir = RUN_DIR
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # config.yaml (input) and results.yaml (output), same convention as every
    # other run -- never merged, no run_id key in either (the folder name,
    # runs/p6/p6_complex_zeroshot/, is the only identity).
    cfg_dict = {f.name: getattr(cfg_target, f.name) for f in dataclasses.fields(cfg_target) if f.name != "run_id"}
    for k in ("HIDDEN_SIZES", "CNN_CHANNELS"):
        cfg_dict[k] = list(cfg_dict[k])
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)

    results = {
        "env": {
            "git": _git_info(),
            "dataset_path": str(complex_path),
            "dataset_sha256": _dataset_sha256(complex_path),
            "source_run_id": SOURCE_RUN_ID,
        },
        "scalars": metrics["scalars"],
        "curves": metrics["curves"],
        "spectrum": metrics["spectrum"],
    }
    with open(run_dir / "results.yaml", "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)

    plots.plot_one_step_predictions(y_true_onestep, y_pred_onestep, OUTPUTS, one_step_metrics, figures_dir, cfg_target)
    plots.plot_rollout_error(metrics["curves"], figures_dir, cfg_target, t_div=metrics["scalars"]["t_div"])
    plots.plot_amplitude_and_energy(metrics["curves"], figures_dir, cfg_target)
    plots.plot_spectrum(metrics["spectrum"], figures_dir, cfg_target)
    plots.make_rollout_animation(rollout, cfg_target, figures_dir)

    summary = (
        f"{RUN_ID} -- zero-shot: {SOURCE_RUN_ID}'s model (trained on simple), "
        f"no retraining, evaluated on the complex dataset's own test trajectory.\n"
        f"norm_stats: recomputed from the SIMPLE dataset (matches training), NOT the complex one.\n"
        f"r2_onestep={metrics['scalars']['r2_onestep']}  t_div={metrics['scalars']['t_div']}  "
        f"E_short={metrics['scalars']['E_short']}\n"
    )
    (run_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"Done -- outputs in {run_dir}")


if __name__ == "__main__":
    main()
