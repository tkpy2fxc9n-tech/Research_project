#!/usr/bin/env python3
"""
Isolation test for the "no propagation" finding in the bridge-truss surrogate
test (surrogate_rods_prototype.py): does p1_feat_u sustain a propagating wave
on a SINGLE isolated rod, in a BC configuration matching exactly what it was
trained on (left end at rest, right end driven by a gaussian pulse -- the
"simple" dataset profile, see data/generate.py's PROFILES["simple"]), run
over the SAME extended 14s horizon (vs. the 5s it was trained on)?

No custom ghost-point/junction code here at all -- a single rod is exactly
what physics/solver.py's run_fd_simulation_general and autoregressive_rollout
were built for, so this reuses them directly instead of reimplementing
anything. If propagation dies out here too, the issue is with the model /
long-rollout generalization, not with the bridge-assembly ghost-point logic.
If it propagates fine here, the bridge code is the suspect instead.

Same pulse amplitude/sigma as the bridge test (A=AMP_MAX, sigma=SIGMA_MIN --
the strongest, sharpest pulse in training range) for direct comparability.

Run it with:
    python3 single_rod_isolation_test.py
It writes single_rod_isolation.gif next to this script.
"""
from __future__ import annotations

from pathlib import Path
import sys
import dataclasses
import gc

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path("/home/aph25/Code_GH")
sys.path.insert(0, str(REPO_ROOT / "src"))

from beamsurrogate.config import Config  # noqa: E402
from beamsurrogate.registry import MODELS, DATASETS  # noqa: E402
from beamsurrogate.data.split import load_hdf5_dataset, compute_norm_stats  # noqa: E402
from beamsurrogate.data.norm import norm_stats_arrays  # noqa: E402
from beamsurrogate.physics.solver import run_fd_simulation_general, autoregressive_rollout, compute_rest_bias  # noqa: E402

SOURCE_RUN_ID = "p1_feat_u"
ROLLOUT_T_END = 14.0   # same extended horizon as the bridge test (training was 5s)


def _load_resolved_config(run_id: str) -> Config:
    resolved = yaml.safe_load((REPO_ROOT / "runs" / run_id / "config.resolved.yaml").read_text())
    known = {f.name for f in dataclasses.fields(Config)}
    raw = {k: v for k, v in resolved.items() if k in known}
    for tuple_field in ("HIDDEN_SIZES", "CNN_CHANNELS"):
        if tuple_field in raw and isinstance(raw[tuple_field], list):
            raw[tuple_field] = tuple(raw[tuple_field])
    return Config(**raw)


cfg_train = _load_resolved_config(SOURCE_RUN_ID)
INPUT_FIELDS = list(cfg_train.features)
# Extended-horizon cfg: same dt as training (0.01), just a longer run --
# __post_init__ re-derives Ntot/i_left/i_right/nodes/dt/dx/CFL from these.
cfg = dataclasses.replace(cfg_train, Nt=int(ROLLOUT_T_END / cfg_train.dt), t_end=ROLLOUT_T_END)
assert abs(cfg.dt - cfg_train.dt) < 1e-12, "extended cfg drifted off the training dt"

from beamsurrogate.data.windows import make_feature_columns, make_output_columns  # noqa: E402
INPUTS = make_feature_columns(INPUT_FIELDS, cfg)
OUTPUTS = make_output_columns(cfg)

model = MODELS[cfg.model](len(INPUTS), len(OUTPUTS), cfg)
model.load_state_dict(torch.load(REPO_ROOT / "runs" / SOURCE_RUN_ID / "model.pth", weights_only=True))
model.eval()

print(f"Recomputing norm_stats from a 200-trajectory subsample of {cfg.dataset} "
      f"(matches {SOURCE_RUN_ID}'s own training normalization)...")
dataset_path = REPO_ROOT / "data" / DATASETS[cfg.dataset]
(df, _FIELDS, INPUTS_check, OUTPUTS_check, *_rest) = load_hdf5_dataset(
    INPUT_FIELDS, cfg_train, dataset_path, max_trajectories=200)
assert INPUTS_check == make_feature_columns(INPUT_FIELDS, cfg_train) and \
    OUTPUTS_check == make_output_columns(cfg_train)
norm_stats = compute_norm_stats(df, INPUTS_check, OUTPUTS_check, cfg_train)
del df, _FIELDS, _rest
gc.collect()
# INPUTS/OUTPUTS column labels depend on cfg.N_FWD only through the number
# of `delta_u@Nndt` columns; norm_stats computed at training Nt still gives
# correct per-column mean/std here since column identity is unchanged.
mu_in, sd_in, mu_out, sd_out = norm_stats_arrays(norm_stats, INPUTS, OUTPUTS)
rest_bias = compute_rest_bias(model, mu_in, sd_in, mu_out, sd_out, cfg)

# Same BC family the "simple" dataset (and so p1_feat_u) was actually
# trained on: left always at rest, right always a driven gaussian pulse
# (data/generate.py's PROFILES["simple"]). Same A/sigma as the bridge test.
bc_left = ("dirichlet", "rest", {})
bc_right = ("dirichlet", "gaussian", {"A": cfg.AMP_MAX, "sigma": 0.3})

print("Running FD ground truth...")
U_reel = run_fd_simulation_general(bc_left, bc_right, cfg)
print("Running NN autoregressive rollout...")
U_nn = autoregressive_rollout(model, U_reel, INPUT_FIELDS, mu_in, sd_in, mu_out, sd_out,
                               rest_bias, bc_left, bc_right, cfg)

print(f"FD max|u| over rollout: {np.abs(U_reel[:, cfg.nodes]).max():.5f}")
print(f"NN max|u| over rollout: {np.abs(U_nn[:, cfg.nodes]).max():.5f}")
print(f"FD last-frame max|u|: {np.abs(U_reel[-1, cfg.nodes]).max():.5f}")
print(f"NN last-frame max|u|: {np.abs(U_nn[-1, cfg.nodes]).max():.5f}")

# --- Amplitude vs time, EVERY raw step (not a hand-picked sample of
# animation frames) -- settles whether the NN genuinely collapses to 0 and
# STAYS there, or whether it's oscillating/jumping around 0 in a way that
# just happened to look flat at the frames inspected by eye.
t_axis = np.arange(cfg.Nt + 1) * cfg.dt
fd_amp = np.abs(U_reel[:, cfg.nodes]).max(axis=1)
nn_amp = np.abs(U_nn[:, cfg.nodes]).max(axis=1)
np.savez(SCRIPT_DIR / "single_rod_amplitude_trace.npz", t=t_axis, fd_amp=fd_amp, nn_amp=nn_amp)

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(t_axis, fd_amp, label="FD (ground truth)", color="tab:blue")
ax2.plot(t_axis, nn_amp, label="NN (p1_feat_u)", color="tab:orange")
ax2.set_xlabel("t (s)")
ax2.set_ylabel("max|u| along the rod")
ax2.legend()
ax2.set_title("Amplitude vs time -- every raw step, no sampling")
fig2.savefig(SCRIPT_DIR / "single_rod_amplitude_vs_time.png", dpi=120)
print(f"Saved {SCRIPT_DIR / 'single_rod_amplitude_vs_time.png'}")

# Does the NN ever come back up after first dropping near 0, anywhere in the
# rollout (not just at the final frame)? And exactly when does it first drop
# and stay below a small threshold for good?
thresh = 1e-4
below = nn_amp < thresh
first_below = np.argmax(below) if below.any() else None
stays_below_after = bool(below[first_below:].all()) if first_below is not None else False
print(f"NN first drops below {thresh} at t={t_axis[first_below]:.2f}s (step {first_below}); "
      f"stays below it for the rest of the rollout: {stays_below_after}")
print(f"NN amplitude min after that point: {nn_amp[first_below:].min():.6f}, "
      f"max after that point: {nn_amp[first_below:].max():.6f}")

# --- Animate: x = position along the rod, y = displacement, FD vs NN overlaid.
x = np.linspace(0, cfg.L, cfg.Nx)
A_MAX = cfg.AMP_MAX
SAVE_EVERY = 4
frame_idx = list(range(0, cfg.Nt + 1, SAVE_EVERY))

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(0, cfg.L)
ax.set_ylim(-A_MAX * 1.3, A_MAX * 1.3)
ax.set_xlabel("position along rod")
ax.set_ylabel("displacement u")
line_fd, = ax.plot([], [], label="FD (ground truth)", color="tab:blue")
line_nn, = ax.plot([], [], label="NN (p1_feat_u)", color="tab:orange")
ax.legend(loc="upper right")
title = ax.set_title("t = 0.00 s")


def update(i):
    s = frame_idx[i]
    line_fd.set_data(x, U_reel[s, cfg.nodes])
    line_nn.set_data(x, U_nn[s, cfg.nodes])
    title.set_text(f"t = {s * cfg.dt:.2f} s")
    return [line_fd, line_nn, title]


anim = animation.FuncAnimation(fig, update, frames=len(frame_idx), interval=40)
out_path = SCRIPT_DIR / "single_rod_isolation.gif"
anim.save(out_path, writer="pillow", fps=25)
print(f"Saved animation to {out_path}")
