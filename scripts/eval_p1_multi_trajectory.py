#!/usr/bin/env python3
# Multi-trajectory rollout eval for p1_pushforward / p1_bptt: replaces the
# single rollout_idx each run's own metrics.json/results.yaml was scored on
# with an evaluation over the FULL "simple" test split (99 trajectories),
# via evaluate_multi_rollout/aggregate_multi_rollout_metrics (evaluate/
# metrics.py). Both runs share the same dataset/physics config (see
# runs/p1_pushforward vs runs/p1_bptt config.yaml diff -- only regime and
# TBPTT-specific hyperparams differ), so the dataset+norm_stats load happens
# once here, not once per model.
#
# Must run on a compute node, not the login node -- see run_analysis.job's
# own comment: loading the full "simple" dataset to recompute norm_stats
# blows the login node's 32G cgroup cap (measured here: killed at ~31G RSS
# during compute_norm_stats's boolean-mask copy of the train block).
#
# Usage: sbatch scripts/run_analysis.job scripts/eval_p1_multi_trajectory.py
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from beamsurrogate.config import load_config, set_seeds  # noqa: E402
from beamsurrogate.registry import MODELS, DATASETS  # noqa: E402
from beamsurrogate.data.split import load_hdf5_dataset, compute_norm_stats  # noqa: E402
from beamsurrogate.evaluate.metrics import (  # noqa: E402
    evaluate_multi_rollout, aggregate_multi_rollout_metrics,
    CONTINUOUS_SCALAR_KEYS, CENSORED_TIME_KEYS,
)

RUNS_DIR = REPO_ROOT / "runs"
OUT_DIR = RUNS_DIR / "p1_analysis"
SOURCES = {"Pushforward": "p1_pushforward", "TBPTT": "p1_bptt"}


def main():
    # Both runs share dataset=simple + identical physics/windowing config,
    # so load once against p1_pushforward's config.yaml -- p1_bptt's model
    # is evaluated against the SAME FIELDS/bc_pairs/idx_test/norm_stats,
    # not a re-load (would be wasted work, and risks a second 32G+ spike).
    cfg = load_config(RUNS_DIR / "p1_pushforward" / "config.yaml")
    set_seeds(cfg)

    dataset_path = REPO_ROOT / "data" / DATASETS[cfg.dataset]
    print(f"Loading {dataset_path} (full dataset, needed to recompute norm_stats)...")
    (df, FIELDS, INPUTS, OUTPUTS, bc_pairs, idx_train, idx_val, idx_test,
     rollout_idx, family_showcase_idx) = load_hdf5_dataset(cfg.features, cfg, dataset_path, max_trajectories=None)
    norm_stats = compute_norm_stats(df, INPUTS, OUTPUTS, cfg)
    del df
    print(f"idx_test has {len(idx_test)} trajectories.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records = []
    summaries = {}

    for label, run_id in SOURCES.items():
        run_cfg = load_config(RUNS_DIR / run_id / "config.yaml")
        model = MODELS[run_cfg.model](len(INPUTS), len(OUTPUTS), run_cfg)
        model.load_state_dict(torch.load(RUNS_DIR / run_id / "model.pth", weights_only=True))
        model.eval()
        print(f"Evaluating {label} ({run_id}) on {len(idx_test)} test trajectories...")

        records = evaluate_multi_rollout(model, FIELDS, bc_pairs, idx_test, cfg.features,
                                          norm_stats, INPUTS, OUTPUTS, cfg)
        for r in records:
            r["regime"] = label
        all_records.extend(records)

        summary = aggregate_multi_rollout_metrics(records)
        summaries[label] = summary
        print(f"  {label}: E_short mean={summary['E_short']['mean']:.4e} "
              f"std={summary['E_short']['std']:.4e}  "
              f"T_max_10pct reached={summary['T_max_10pct']['pct_reached']:.1f}%")

    # Per-trajectory raw numbers -> CSV (Appendix-A material, and lets any
    # later statistic -- e.g. the paired Wilcoxon test -- be added without
    # re-running the rollouts).
    fieldnames = ["regime", "traj_idx"] + list(CONTINUOUS_SCALAR_KEYS) + list(CENSORED_TIME_KEYS)
    csv_path = OUT_DIR / "multi_trajectory_per_run.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)
    print(f"Saved {csv_path}")

    summary_path = OUT_DIR / "multi_trajectory_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
