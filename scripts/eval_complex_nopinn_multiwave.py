#!/usr/bin/env python3
"""
p6_dataset_complex was only ever visualized on ONE test trajectory
(rollout_idx = idx_test[0], see data/split.py) -- this reruns the SAME
already-trained model on several different test trajectories from the
complex dataset, to see how it behaves across a broader sample of waves
instead of a single cherry-picked case.

norm_stats recomputed from the FULL complex dataset (max_trajectories=None),
matching exactly how cli.py trains for real (see its main(): max_trajectories
is only capped at 16 under --smoke-test, otherwise None) -- same SEED/
SPLIT_SEED=42 as training, so this reproduces the model's own training-time
norm_stats deterministically. A first pass of this script used a
200-trajectory subsample as a shortcut (the same convention used
successfully throughout this session for the "simple" dataset) and got
wildly unstable results (amp_loss_pct in the billions of percent) that
didn't match the run's own recorded metrics.json (t_div=0.54s, amp_loss=
-4.1%) on the SAME trajectory -- root cause: "complex" has far more BC
families than "simple" (rest/gaussian only), so 200 trajectories don't
converge to the true population mean/std. This full-dataset version is
slower to load but gives trustworthy numbers.

Usage: python scripts/eval_complex_nopinn_multiwave.py
"""
from __future__ import annotations

import dataclasses
import gc
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from beamsurrogate.config import Config, set_seeds  # noqa: E402
from beamsurrogate.registry import MODELS, DATASETS  # noqa: E402
from beamsurrogate.data.split import load_hdf5_dataset, compute_norm_stats  # noqa: E402
from beamsurrogate.evaluate.rollout import run_rollout  # noqa: E402
from beamsurrogate.evaluate.metrics import build_metrics  # noqa: E402
from beamsurrogate.evaluate import plots  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"
SOURCE_RUN_ID = "p6_dataset_complex"
N_WAVES = 6


def _load_resolved_config(run_id: str) -> Config:
    resolved = yaml.safe_load((RUNS_DIR / run_id / "config.resolved.yaml").read_text())
    known = {f.name for f in dataclasses.fields(Config)}
    raw = {k: v for k, v in resolved.items() if k in known}
    for tuple_field in ("HIDDEN_SIZES", "CNN_CHANNELS"):
        if tuple_field in raw and isinstance(raw[tuple_field], list):
            raw[tuple_field] = tuple(raw[tuple_field])
    return Config(**raw)


def main():
    cfg = _load_resolved_config(SOURCE_RUN_ID)
    set_seeds(cfg)

    dataset_path = DATA_DIR / DATASETS[cfg.dataset]
    print(f"Loading {dataset_path} (FULL dataset, matching training's own norm_stats)...")
    (df, FIELDS, INPUTS, OUTPUTS, bc_pairs, idx_train, idx_val, idx_test,
     rollout_idx, family_showcase_idx) = load_hdf5_dataset(
        cfg.features, cfg, dataset_path, max_trajectories=None)
    norm_stats = compute_norm_stats(df, INPUTS, OUTPUTS, cfg)
    del df
    gc.collect()

    model = MODELS[cfg.model](len(INPUTS), len(OUTPUTS), cfg)
    model.load_state_dict(torch.load(RUNS_DIR / SOURCE_RUN_ID / "model.pth", weights_only=True))
    model.eval()
    print(f"Loaded {SOURCE_RUN_ID}'s model. {len(idx_test)} test trajectories available.")

    figures_dir = RUNS_DIR / SOURCE_RUN_ID / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    waves = idx_test[:N_WAVES]
    print(f"Rolling out on {len(waves)} test trajectories: {waves}")
    summary_lines = [f"{SOURCE_RUN_ID} -- multi-wave rollout check ({len(waves)} test trajectories)\n"]
    for i, idx in enumerate(waves):
        rollout = run_rollout(model, FIELDS, bc_pairs, idx, cfg.features, norm_stats, INPUTS, OUTPUTS, cfg)
        metrics = build_metrics(cfg, rollout=rollout)
        plots.make_rollout_animation(rollout, cfg, figures_dir, filename=f"rollout_wave{i}.gif")
        left_bc, right_bc = rollout.left_bc, rollout.right_bc
        line = (f"wave {i} (test idx {idx}): left={left_bc[1]} right={right_bc[1]}  "
                f"E_short={metrics['scalars']['E_short']:.4e}  t_div={metrics['scalars']['t_div']}  "
                f"amp_loss_pct={metrics['scalars']['amp_loss_pct']:.2f}")
        print(line)
        summary_lines.append(line)

    (figures_dir / "multiwave_summary.txt").write_text("\n".join(summary_lines) + "\n")
    print(f"\nDone -- {len(waves)} gifs + multiwave_summary.txt in {figures_dir}")


if __name__ == "__main__":
    main()
