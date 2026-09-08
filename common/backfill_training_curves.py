#!/usr/bin/env python3
# Regenerates training_curve.png + training_curve_train_val.png +
# training_curve_active_components.png for every already-finished run, using
# the exact same plots.plot_training_curve() / plot_training_curve_active_
# components() the live pipeline calls (see cli.py's run()) -- so backfilled
# figures and any future run's figures are produced by one code path, never
# two drifting copies. Needed because plot_training_curve() changed after
# these runs already finished: (1) it now also writes a train/val-only
# figure without the extra_history component curves, (2) for pushforward
# runs its "train" line is now the true weighted total (data + ramped
# pushforward + physics), not just the data component training/pushforward.
# py's train_history alone holds -- see reconstruct_train_total()'s own
# comment in plots.py. The active-components figure used to be a separate
# manual script (analysis/p1_diag_training_curves_active.py); it's included
# here too now that its logic lives in plots.py.
#
# Explicitly skips any run still training right now (see SKIP_RUN_DIRS
# below) rather than relying only on the missing-results.yaml check, since
# writing into a run's figures/ while its own Slurm job might do the same is
# exactly the kind of concurrent-write risk not worth taking for a backfill
# that can just be re-run later once those finish.
#
# Usage: python analysis/backfill_training_curves.py
from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from beamsurrogate.config import load_config  # noqa: E402
from beamsurrogate.training import TrainResult  # noqa: E402
from beamsurrogate.evaluate import plots  # noqa: E402

# Runs with an active Slurm job as of this backfill (2026-08-24, checked via
# squeue immediately before running) -- update/empty this list once they've
# finished, then re-run this script to cover them too. Purely pending jobs
# (not yet RUNNING) don't need an entry here: they can't have a results.yaml
# yet either way, so the existence check below already skips them.
#
# p0_baseline specifically: mid-retrain (job 149685) as of writing, but
# already has an OLD results.yaml on disk from its previous, pre-retrain run
# -- without this entry the existence check below would happily backfill
# against that stale file while the retrain concurrently overwrites it.
SKIP_RUN_DIRS = {
    REPO_ROOT / "baseline" / "runs" / "p0_baseline",
}


def main():
    config_paths = sorted(
        p for p in REPO_ROOT.glob("*/**/runs/*/config.yaml")
        if "Archives_runs" not in p.parts
    )

    done, skipped = [], []
    for config_path in config_paths:
        run_dir = config_path.parent
        if run_dir in SKIP_RUN_DIRS:
            skipped.append((run_dir.name, "active Slurm job -- rerun later"))
            continue
        results_path = run_dir / "results.yaml"
        if not results_path.exists():
            skipped.append((run_dir.name, "no results.yaml -- not finished"))
            continue

        results = yaml.safe_load(results_path.read_text())
        if "train_history" not in results:
            skipped.append((run_dir.name, "results.yaml predates train_history instrumentation"))
            continue

        cfg = load_config(config_path)
        train_result = TrainResult(
            train_history=results["train_history"],
            val_history=results["val_history"],
            best_val=0.0, train_time_s=0.0, n_params=0,
            extra_history=results["extra_history"],
        )
        figures_dir = run_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        plots.plot_training_curve(train_result, figures_dir, cfg)
        plots.plot_training_curve_active_components(train_result, figures_dir, cfg)
        done.append(run_dir.name)

    print(f"Backfilled {len(done)} run(s): {', '.join(done)}")
    if skipped:
        print(f"\nSkipped {len(skipped)} run(s):")
        for name, reason in skipped:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
