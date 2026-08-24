#!/usr/bin/env python3
# Dry-run planner for eval_multi_trajectory.py: discovers every trained run
# (run_registry.discover_run_dirs), groups them by dataset-load signature
# (run_registry.signature_of), estimates each group's peak RSS from the
# same data_all sizing formula data/split.py's load_hdf5_dataset itself
# fills (rows_per_traj * (n_in+n_out) * 4 bytes), and prints one sbatch
# command per group with --mem sized to it. PRINTS ONLY -- never calls
# sbatch itself, so a plan can be reviewed before anything touches the
# cluster.
#
# Peak RSS estimate = data_all_bytes * 1.9 (compute_norm_stats's boolean-
# mask train-split copy roughly doubles peak vs data_all alone, train is
# ~91% of rows) + 3G base overhead, then +30% headroom, rounded up to the
# nearest 10G, floored at 96G (run_analysis.job's existing default, already
# validated against p1's measured ~38G actual peak). This is an ESTIMATE,
# calibrated against exactly one measured case (p1) -- if a submitted job
# still OOMs, bump that group's --mem by hand and resubmit just that group.
#
# Usage: python scripts/plan_multi_trajectory_groups.py
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import h5py
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from beamsurrogate.config import load_config  # noqa: E402
from beamsurrogate.registry import DATASETS  # noqa: E402
from run_registry import discover_run_dirs, signature_of  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
SS_INPUT_WIDTH = lambda ss: 2 * ss + 1  # noqa: E731 -- matches windows.py's stencil width


def _n_total_and_nx(dataset: str) -> tuple[int, int]:
    with h5py.File(DATA_DIR / DATASETS[dataset], "r") as f:
        return f["u"].shape[0], f["u"].shape[2]


def estimate_mem_gb(sig: tuple) -> int:
    dataset, Nt, Nx, SS, t_end, E, rho, L, M_BACK, N_FWD, ndt, features = sig
    n_total, nx_stored = _n_total_and_nx(dataset)
    n_list_len = len(range(M_BACK * ndt, Nt - N_FWD * ndt + 1))
    n_in = M_BACK * SS_INPUT_WIDTH(SS) * len(features)
    n_out = N_FWD
    total_rows = n_total * n_list_len * nx_stored
    data_all_gb = total_rows * (n_in + n_out) * 4 / 1e9
    est_peak_gb = data_all_gb * 1.9 + 3
    return max(96, int(est_peak_gb * 1.3 / 10 + 1) * 10)


def main():
    run_dirs = discover_run_dirs(REPO_ROOT)
    groups = defaultdict(list)
    for run_id, run_dir in sorted(run_dirs.items()):
        cfg = load_config(run_dir / "config.yaml")
        groups[signature_of(cfg)].append(run_id)

    print(f"{len(groups)} dataset-load groups, {sum(len(v) for v in groups.values())} trained runs\n")
    for sig, run_ids in sorted(groups.items()):
        mem = estimate_mem_gb(sig)
        print(f"# {sig}  ({len(run_ids)} run(s), est. --mem={mem}G)")
        print(f"sbatch --partition=short --mem={mem}G scripts/run_analysis.job "
              f"scripts/eval_multi_trajectory.py {' '.join(run_ids)}")
        print()


if __name__ == "__main__":
    main()
