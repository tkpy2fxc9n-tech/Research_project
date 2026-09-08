#!/usr/bin/env python3
# Multi-trajectory rollout eval for one dataset-load GROUP: every run_id
# passed in must share the exact same run_registry.signature_of(cfg) --
# verified below, never assumed -- so the expensive dataset+norm_stats load
# (see scripts/eval_p1_multi_trajectory.py's own comment on why this can't
# run on the login node: it blows the 32G cgroup cap there) happens once per
# group instead of once per run. Writes multi_trajectory_summary.json and
# multi_trajectory_per_run.csv into EACH run's own directory (unlike
# p1_analysis's shared folder -- these runs are scattered across many phase
# directories, each is its own place for downstream analysis/*.py scripts
# to find it).
#
# Usage: sbatch --mem=<X>G scripts/run_analysis.job scripts/eval_multi_trajectory.py <run_id> [<run_id> ...]
#   See scripts/plan_multi_trajectory_groups.py for which run_ids share a
#   signature and what --mem each group needs -- don't hand-assemble groups.
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from beamsurrogate.config import load_config, set_seeds  # noqa: E402
from beamsurrogate.registry import MODELS, DATASETS  # noqa: E402
from beamsurrogate.data.split import load_hdf5_dataset, compute_norm_stats  # noqa: E402
from beamsurrogate.evaluate.metrics import (  # noqa: E402
    evaluate_multi_rollout, aggregate_multi_rollout_metrics,
    CONTINUOUS_SCALAR_KEYS, CENSORED_TIME_KEYS,
)
from run_registry import find_run_dir, signature_of  # noqa: E402


def main():
    run_ids = sys.argv[1:]
    if not run_ids:
        print("Usage: eval_multi_trajectory.py <run_id> [<run_id> ...]", file=sys.stderr)
        sys.exit(1)

    run_dirs = {run_id: find_run_dir(REPO_ROOT, run_id) for run_id in run_ids}
    missing = [r for r, d in run_dirs.items() if d is None]
    if missing:
        print(f"ERROR: no trained run found for {missing} (check spelling, or model.pth doesn't "
              f"exist yet).", file=sys.stderr)
        sys.exit(1)

    cfgs = {run_id: load_config(run_dirs[run_id] / "config.yaml") for run_id in run_ids}
    sigs = {run_id: signature_of(cfgs[run_id]) for run_id in run_ids}
    if len(set(sigs.values())) > 1:
        print("ERROR: these run_ids don't share a dataset-load signature -- put them in separate "
              "sbatch calls (see plan_multi_trajectory_groups.py):", file=sys.stderr)
        for r, s in sigs.items():
            print(f"  {r}: {s}", file=sys.stderr)
        sys.exit(1)

    # Any one config is representative for the shared load -- dataset,
    # physics, and windowing are identical across the group by the check
    # above; only model-specific fields (HIDDEN_SIZES, ...) may differ,
    # and those aren't used until each run's own model is built below.
    load_cfg = cfgs[run_ids[0]]
    set_seeds(load_cfg)
    dataset_path = REPO_ROOT / "data" / DATASETS[load_cfg.dataset]
    print(f"Loading {dataset_path} for group {sigs[run_ids[0]]} ({len(run_ids)} run(s): {run_ids})...")
    (df, FIELDS, INPUTS, OUTPUTS, bc_pairs, idx_train, idx_val, idx_test,
     rollout_idx, family_showcase_idx) = load_hdf5_dataset(load_cfg.features, load_cfg, dataset_path,
                                                             max_trajectories=None)
    norm_stats = compute_norm_stats(df, INPUTS, OUTPUTS, load_cfg)
    del df
    print(f"idx_test has {len(idx_test)} trajectories.")

    for run_id in run_ids:
        run_cfg = cfgs[run_id]
        run_dir = run_dirs[run_id]
        model = MODELS[run_cfg.model](len(INPUTS), len(OUTPUTS), run_cfg)
        model.load_state_dict(torch.load(run_dir / "model.pth", weights_only=True))
        model.eval()
        print(f"Evaluating {run_id} on {len(idx_test)} test trajectories...")

        # run_cfg (not load_cfg) drives run_rollout/build_metrics: every
        # field they read (M_BACK/N_FWD/ndt/dt/nodes/dx/rho/E/Nt) is
        # guaranteed identical to load_cfg's by the signature check above,
        # so this is just "use each run's own Config object", not a second
        # source of truth.
        records = evaluate_multi_rollout(model, FIELDS, bc_pairs, idx_test, run_cfg.features,
                                          norm_stats, INPUTS, OUTPUTS, run_cfg)
        summary = aggregate_multi_rollout_metrics(records)
        print(f"  {run_id}: E_short mean={summary['E_short']['mean']:.4e}  "
              f"T_max_10pct reached={summary['T_max_10pct']['pct_reached']:.1f}%")

        fieldnames = ["traj_idx"] + list(CONTINUOUS_SCALAR_KEYS) + list(CENSORED_TIME_KEYS)
        csv_path = run_dir / "multi_trajectory_per_run.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

        summary_path = run_dir / "multi_trajectory_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"  Saved {csv_path.relative_to(REPO_ROOT)}, {summary_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
