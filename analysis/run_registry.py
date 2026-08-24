# Shared by scripts/eval_multi_trajectory.py and
# scripts/plan_multi_trajectory_groups.py: which run_ids exist and have a
# trained model.pth, and the full (dataset, physics, windowing) signature
# that determines whether two runs can safely share ONE expensive dataset+
# norm_stats load. The signature must cover every Config field
# load_hdf5_dataset/compute_norm_stats/run_rollout/build_metrics actually
# use -- dataset+M_BACK+N_FWD+ndt+features alone is NOT enough, e.g. p9's
# coarse-grid runs change Nx/SS too. Two runs with an identical signature
# are, by construction, byte-for-byte interchangeable for FIELDS/bc_pairs/
# norm_stats/INPUTS/OUTPUTS -- only their model.pth and model-specific
# fields (HIDDEN_SIZES, CNN_CHANNELS, ...) differ.
from __future__ import annotations

import glob
from pathlib import Path

# Archived/backup copies are intentionally excluded -- they duplicate a
# run_id that also exists at its current, non-archived path (see
# runs/Archives_runs/20260820_p4_arch_pre_relaunch/ vs runs/p4/, and
# runs/_backups/p0_baseline_pre_retrain_20260823/ vs runs/p0/p0_baseline/).
# Evaluating both would silently double-count them in any run_id-keyed table.
EXCLUDE_PATH_SUBSTRINGS = ("Archives_runs", "_backups", "p1_analysis", "p2_analysis")


def discover_run_dirs(repo_root: Path) -> dict[str, Path]:
    # run_id -> run_dir, for every runs/**/config.yaml with a sibling
    # model.pth (i.e. actually trained), outside the excluded paths above.
    out = {}
    for f in glob.glob(str(repo_root / "runs" / "**" / "config.yaml"), recursive=True):
        if any(s in f for s in EXCLUDE_PATH_SUBSTRINGS):
            continue
        run_dir = Path(f).parent
        if not (run_dir / "model.pth").exists():
            continue
        out[run_dir.name] = run_dir
    return out


def find_run_dir(repo_root: Path, run_id: str) -> Path | None:
    return discover_run_dirs(repo_root).get(run_id)


def signature_of(cfg) -> tuple:
    return (cfg.dataset, cfg.Nt, cfg.Nx, cfg.SS, cfg.t_end, cfg.E, cfg.rho, cfg.L,
            cfg.M_BACK, cfg.N_FWD, cfg.ndt, tuple(cfg.features))
