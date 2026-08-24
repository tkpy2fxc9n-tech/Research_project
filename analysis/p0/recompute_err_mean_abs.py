#!/usr/bin/env python3
# One-off patch, p0_baseline only: its results.yaml predates err_mean_abs
# being added to compute_error_curves() (metrics.py), so the mean-absolute
# rollout-error curve p0_analysis_r2_vs_deltau.py's r2_vs_rollout() wants to
# plot isn't in there. Rather than retrain (~2.8h, see results.yaml's own
# train_time_s), this loads the already-trained model.pth (state_dict only,
# no optimizer) and reruns just the rollout/metrics pass, then merges
# err_mean_abs into the EXISTING results.yaml in place -- everything else
# (scalars, train_history, env) is left untouched.
#
# Needs the full dataset load load_hdf5_dataset()/compute_norm_stats() do,
# same as training -- too much for the login node's 32G cgroup cap (see
# scripts/run_analysis.job's header), so run this via:
#   sbatch --partition=short scripts/run_analysis.job analysis/p0/recompute_err_mean_abs.py
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import yaml

from beamsurrogate.config import load_config, set_seeds
from beamsurrogate.registry import MODELS
from beamsurrogate.data.split import load_hdf5_dataset, compute_norm_stats
from beamsurrogate.evaluate.rollout import run_rollout
from beamsurrogate.evaluate.metrics import compute_error_curves

run_dir = REPO_ROOT / "runs" / "p0" / "p0_baseline"
cfg = load_config(run_dir / "config.yaml")
set_seeds(cfg)

dataset_path = REPO_ROOT / "data" / "beam_dataset_A.h5"
(df, FIELDS, INPUTS, OUTPUTS, bc_pairs, idx_train, idx_val, idx_test,
 rollout_idx, family_showcase_idx) = load_hdf5_dataset(cfg.features, cfg, dataset_path, None)
norm_stats = compute_norm_stats(df, INPUTS, OUTPUTS, cfg)

model = MODELS[cfg.model](len(INPUTS), len(OUTPUTS), cfg)
model.load_state_dict(torch.load(run_dir / "model.pth", weights_only=True))
model.eval()

rollout = run_rollout(model, FIELDS, bc_pairs, rollout_idx, cfg.features, norm_stats, INPUTS, OUTPUTS, cfg)
new_curves = compute_error_curves(rollout, cfg)

results_path = run_dir / "results.yaml"
with open(results_path) as f:
    results = yaml.safe_load(f)

old_curves = results["curves"]
print("old t: n=%d first5=%r last5=%r" % (len(old_curves["t"]), old_curves["t"][:5], old_curves["t"][-5:]))
print("new t: n=%d first5=%r last5=%r" % (len(new_curves["t"]), new_curves["t"][:5], new_curves["t"][-5:]))
print("old err_max first10:", old_curves["err_max"][:10])
print("new err_max first10:", new_curves["err_max"][:10])

# Confirms the rollout above reproduced the SAME run (same rollout_idx, same
# steps) before trusting its err_mean_abs. Full-curve exact equality is NOT
# the right check here: this system's rollout error is chaotic (it's the
# whole point of the plot -- it blows up to ~1e31 by t=5), so even a tiny
# float difference from a different node/BLAS/thread-count diverges into
# unrelated late-time values. Only the early steps, before the divergence
# takes off, can actually distinguish "same run" from "wrong rollout_idx".
assert old_curves["t"] == new_curves["t"], "step grid mismatch -- rollout_idx or steps changed"
N_CHECK = 20
import math
for i in range(min(N_CHECK, len(old_curves["err_max"]))):
    old_v, new_v = old_curves["err_max"][i], new_curves["err_max"][i]
    assert math.isclose(old_v, new_v, rel_tol=1e-3, abs_tol=1e-9), (
        f"err_max mismatch at step {i} (t={old_curves['t'][i]}): "
        f"old={old_v!r} new={new_v!r} -- rollout is not reproducing the original run")

old_curves["err_mean_abs"] = new_curves["err_mean_abs"]

with open(results_path, "w") as f:
    yaml.safe_dump(results, f, sort_keys=False)

print("err_mean_abs added, length:", len(new_curves["err_mean_abs"]))
