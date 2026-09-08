#!/usr/bin/env python3
"""
13 -- p13_cost_precision: float64 (reference) vs float32, for BOTH
the network and the FD solver -- verify float32 doesn't move t_div before
claiming its throughput gain. Inference/rollout only, no retraining: the
existing trained checkpoint (float32, as trained) is the "Network f32" row
directly; "Network f64" casts that SAME model + arithmetic to float64.
"Solver f64" is run_fd_simulation_general's own existing (default numpy,
i.e. float64) behavior, unmodified; "Solver f32" is a local float32 variant
of the same leapfrog scheme (NOT a change to physics/solver.py itself --
that function is shared by every other run in this campaign and stays
float64-default; this script keeps its own float32 copy local).

Why a local dtype-parametrized copy of build_window/compute_rest_bias/
autoregressive_rollout is needed instead of just reusing the production
ones: data/windows.py's build_window and physics/solver.py's
compute_rest_bias both hardcode `dtype=np.float32` internally regardless of
their inputs' own dtype -- by design, since every other run in this
campaign trains and evaluates in float32 throughout. Testing float64 for
real (not silently getting float32 fed to a .double()'d model, which would
crash on the dtype mismatch, or silently no-op) requires copies with that
hardcoding replaced by a parameter.

Set SOURCE_RUN_ID below once known (same "etape 10" dependency as this
campaign's other new configs).

Usage: python scripts/cost_precision_benchmark.py
"""
from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "common"))

from beamsurrogate.config import Config  # noqa: E402
from beamsurrogate.registry import MODELS, DATASETS  # noqa: E402
from beamsurrogate.data.split import load_hdf5_dataset, compute_norm_stats  # noqa: E402
from beamsurrogate.data.norm import norm_stats_arrays  # noqa: E402
from beamsurrogate.data.windows import make_feature_columns, make_output_columns, field_value  # noqa: E402
from beamsurrogate.physics.waves import apply_boundary_conditions  # noqa: E402
from beamsurrogate.evaluate.metrics import compute_t_div, compute_e_short  # noqa: E402
from beamsurrogate.evaluate.rollout import RolloutResult, chrono  # noqa: E402
from run_registry import find_run_dir  # noqa: E402

def _run_path(run_id: str) -> Path:
    # Runs live under <phase>/runs/<run_id>/, the phase folder being named
    # after what it does (baseline/, pinn_loss/, ...). find_run_dir searches
    # for the id rather than rebuilding a path, so this keeps working
    # wherever a run sits in the tree.
    found = find_run_dir(REPO_ROOT, run_id)
    if found is None:
        raise SystemExit(f"no run folder found for {run_id!r}")
    return found


OUT_DIR = REPO_ROOT / "runtime_cost" / "runs" / "p13_cost_precision"
SOURCE_RUN_ID = "p3_pinn_0"   # "etape 10" final model: M_BACK=3, N_FWD=2, features=[U],
                                # HIDDEN_SIZES=[512,256,64], LAMBDA_PHYSICS=0 -- found under
                                # its phase-3 name, but its config matches this campaign's
                                # settled architecture exactly.


def _load_resolved_config(run_id: str) -> Config:
    resolved = yaml.safe_load((_run_path(run_id) / "config.resolved.yaml").read_text())
    known = {f.name for f in dataclasses.fields(Config)}
    raw = {k: v for k, v in resolved.items() if k in known}
    for tuple_field in ("HIDDEN_SIZES", "CNN_CHANNELS"):
        if tuple_field in raw and isinstance(raw[tuple_field], list):
            raw[tuple_field] = tuple(raw[tuple_field])
    return Config(**raw)


# --- Local, dtype-parametrized copies (see module docstring for why) -------
def build_window_dtype(m_list, get_u, input_fields, cfg, dtype):
    nodes = cfg.nodes
    n_features = cfg.M_BACK * (2 * cfg.SS + 1) * len(input_fields)
    X = np.zeros((len(nodes), n_features), dtype=dtype)
    col = 0
    for m in m_list:
        field_arrays = {f: field_value(f, get_u, m, cfg) for f in input_fields}
        for k in range(-cfg.SS, cfg.SS + 1):
            for f in input_fields:
                X[:, col] = field_arrays[f][nodes + k]
                col += 1
    return X


def compute_rest_bias_dtype(model, mu_in, sd_in, mu_out, sd_out, cfg, dtype):
    Xz = (np.zeros((len(cfg.nodes), len(mu_in)), dtype=dtype) - mu_in) / sd_in
    with torch.no_grad():
        return (model(torch.tensor(Xz)).numpy() * sd_out + mu_out)[0]


def autoregressive_rollout_dtype(model, U_reel, input_fields, mu_in, sd_in, mu_out, sd_out,
                                  rest_bias, bc_left, bc_right, cfg, dtype):
    history_needed = cfg.M_BACK * cfg.ndt
    U = np.zeros((cfg.Nt + 1, cfg.Ntot), dtype=dtype)
    for m in range(history_needed + 1):
        U[m] = U_reel[m]
    for n in range(history_needed, cfg.Nt - cfg.N_FWD * cfg.ndt + 1, cfg.N_FWD * cfg.ndt):
        m_list = [n - lag * cfg.ndt for lag in range(cfg.M_BACK)]
        X = (build_window_dtype(m_list, lambda m: U[m], input_fields, cfg, dtype) - mu_in) / sd_in
        with torch.no_grad():
            pred_norm = model(torch.tensor(X)).numpy()
        deltas = pred_norm * sd_out + mu_out - rest_bias
        for h in range(1, cfg.N_FWD + 1):
            s = n + h * cfg.ndt
            t = s * cfg.dt
            U[s, cfg.nodes] = U[n, cfg.nodes] + deltas[:, h - 1]
            apply_boundary_conditions(U[s], t, bc_left, bc_right, cfg)
            if cfg.SMOOTH_ALPHA > 0:
                j0, j1 = cfg.i_left + 1, cfg.i_right
                lap = U[s, j0 - 1:j1 - 1] - 2 * U[s, j0:j1] + U[s, j0 + 1:j1 + 1]
                U[s, j0:j1] += cfg.SMOOTH_ALPHA * lap
    return U


def run_fd_dtype(bc_left, bc_right, cfg, dtype):
    # Local float32-capable copy of physics/solver.py's run_fd_simulation_general
    # -- that function stays float64-default (every other caller relies on it).
    from beamsurrogate.physics.waves import apply_boundary_conditions as apply_bc
    i_left, i_right, Ntot = cfg.i_left, cfg.i_right, cfg.Ntot
    u_storage = np.zeros((cfg.Nt + 1, Ntot), dtype=dtype)
    u = np.zeros(Ntot, dtype=dtype)
    u_1 = np.zeros(Ntot, dtype=dtype)
    for n in range(cfg.Nt):
        t = n * cfg.dt
        u_new = np.zeros(Ntot, dtype=dtype)
        u_new[i_left:i_right + 1] = (
            2.0 * u[i_left:i_right + 1] - u_1[i_left:i_right + 1]
            + cfg.CFL ** 2 * (u[i_left - 1:i_right] - 2.0 * u[i_left:i_right + 1] + u[i_left + 1:i_right + 2])
        )
        apply_bc(u_new, t + cfg.dt, bc_left, bc_right, cfg)
        u_1, u = u.copy(), u_new
        u_storage[n + 1] = u.copy()
    return u_storage


def rollout_metrics(U, U_reel, cfg):
    class R:
        pass
    r = R()
    r.U, r.U_reel = U, U_reel
    return dict(t_div=compute_t_div(r, cfg), E_short=compute_e_short(r, cfg))


def main():
    if SOURCE_RUN_ID is None:
        print("ERROR: SOURCE_RUN_ID is not set -- edit this script once 'etape 10' is known.",
              file=sys.stderr)
        sys.exit(1)

    cfg = _load_resolved_config(SOURCE_RUN_ID)
    INPUTS = make_feature_columns(cfg.features, cfg)
    OUTPUTS = make_output_columns(cfg)

    model32 = MODELS[cfg.model](len(INPUTS), len(OUTPUTS), cfg)
    model32.load_state_dict(torch.load(_run_path(SOURCE_RUN_ID) / "model.pth", weights_only=True))
    model32.eval()
    model64 = MODELS[cfg.model](len(INPUTS), len(OUTPUTS), cfg)
    model64.load_state_dict(model32.state_dict())
    model64 = model64.double().eval()

    dataset_path = REPO_ROOT / "data" / DATASETS[cfg.dataset]
    (df, FIELDS, _INPUTS, _OUTPUTS, bc_pairs, idx_train, idx_val, idx_test,
     rollout_idx, fam) = load_hdf5_dataset(cfg.features, cfg, dataset_path, max_trajectories=None)
    norm_stats = compute_norm_stats(df, INPUTS, OUTPUTS, cfg)
    mu_in32, sd_in32, mu_out32, sd_out32 = norm_stats_arrays(norm_stats, INPUTS, OUTPUTS)
    left_bc, right_bc = bc_pairs[rollout_idx]
    U_reel32 = FIELDS[rollout_idx]

    results = {}

    # --- Network f32 (production dtype, as trained) ---
    rb32 = compute_rest_bias_dtype(model32, mu_in32, sd_in32, mu_out32, sd_out32, cfg, np.float32)
    t0 = time.perf_counter()
    U32 = autoregressive_rollout_dtype(model32, U_reel32, cfg.features, mu_in32, sd_in32, mu_out32, sd_out32,
                                        rb32, left_bc, right_bc, cfg, np.float32)
    results["Network f32"] = dict(**rollout_metrics(U32, U_reel32, cfg), time_s=time.perf_counter() - t0)

    # --- Network f64 (reference) ---
    mu_in64, sd_in64, mu_out64, sd_out64 = (a.astype(np.float64) for a in
                                             (mu_in32, sd_in32, mu_out32, sd_out32))
    U_reel64 = U_reel32.astype(np.float64)
    rb64 = compute_rest_bias_dtype(model64, mu_in64, sd_in64, mu_out64, sd_out64, cfg, np.float64)
    t0 = time.perf_counter()
    U64 = autoregressive_rollout_dtype(model64, U_reel64, cfg.features, mu_in64, sd_in64, mu_out64, sd_out64,
                                        rb64, left_bc, right_bc, cfg, np.float64)
    results["Network f64 : ref"] = dict(**rollout_metrics(U64, U_reel64, cfg), time_s=time.perf_counter() - t0)

    # --- Solver f64 (reference) / f32 ---
    t0 = time.perf_counter()
    Usolver64 = run_fd_dtype(left_bc, right_bc, cfg, np.float64)
    t_solver64 = time.perf_counter() - t0
    t0 = time.perf_counter()
    Usolver32 = run_fd_dtype(left_bc, right_bc, cfg, np.float32)
    t_solver32 = time.perf_counter() - t0
    results["Solver f64 : ref"] = dict(t_div=None, E_short=None, time_s=t_solver64)
    solver_err = float(np.max(np.abs(Usolver32.astype(np.float64) - Usolver64)))
    results["Solver f32"] = dict(t_div=None, E_short=None, time_s=t_solver32,
                                  max_abs_diff_vs_f64=solver_err)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(results).T
    df_out.to_csv(OUT_DIR / "cost_precision.csv")
    print(df_out.to_string())
    print(f"\nNetwork: does t_div move between f32 and f64? "
          f"f32={results['Network f32']['t_div']}  f64={results['Network f64 : ref']['t_div']}")
    print(f"Solver f32 max abs diff vs f64 over the whole rollout: {solver_err:.3e} "
          f"(large/blown-up = f32 unstable for this scheme)")
    print(f"\nSaved {OUT_DIR / 'cost_precision.csv'}")


if __name__ == "__main__":
    main()
