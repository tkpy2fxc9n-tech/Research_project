# The metrics.json contract every run writes -- same keys for every run,
# `null` when a metric doesn't apply (never omitted, so analysis/*.py can
# always index the same schema across runs/*/metrics.json). This module was
# almost entirely missing before this refactor: without it, none of the 27
# ablation runs would be comparable or even individually interpretable.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .rollout import RolloutResult, BenchmarkResult


def l2_rel(pred, true, eps=1e-12):
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + eps))


def smape(pred, true):
    m = true != 0
    return float(np.mean(2 * np.abs(true[m] - pred[m]) / (np.abs(true[m]) + np.abs(pred[m]))))


def _rollout_steps(cfg):
    # Same step grid every rollout error curve is sampled on: ndt-spaced,
    # starting once the network has actually produced a prediction.
    return np.arange(2 * cfg.ndt, cfg.Nt + 1, cfg.ndt)


def compute_error_curves(rollout: RolloutResult, cfg) -> dict:
    steps = _rollout_steps(cfg)
    nodes = cfg.nodes
    U, U_reel = rollout.U, rollout.U_reel

    err_rel_mean = [l2_rel(U[k, nodes], U_reel[k, nodes]) for k in steps]
    err_max = [float(np.max(np.abs(U[k, nodes] - U_reel[k, nodes]))) for k in steps]
    amp_max = [float(np.max(np.abs(U[k, nodes]))) for k in steps]
    energy = [_mechanical_energy(U, k, cfg) for k in steps]

    return {
        "t": (steps * cfg.dt).tolist(),
        "err_rel_mean": err_rel_mean,
        "err_max": err_max,
        "amp_max": amp_max,
        "energy": energy,
    }


def _mechanical_energy(U: np.ndarray, k: int, cfg) -> float:
    # Total mechanical energy at step k: (1/2) integral of rho*u_t^2 +
    # E*u_x^2 over the beam, trapezoidal in space (dx-weighted sum),
    # u_t/u_x from central differences (backward difference at k=0).
    nodes = cfg.nodes
    k_prev = max(k - cfg.ndt, 0)
    u = U[k, nodes]
    u_prev = U[k_prev, nodes]
    dt_eff = (k - k_prev) * cfg.dt if k != k_prev else cfg.dt
    u_t = (u - u_prev) / dt_eff if dt_eff > 0 else np.zeros_like(u)
    u_x = np.gradient(u, cfg.dx)
    density = 0.5 * cfg.rho * u_t ** 2 + 0.5 * cfg.E * u_x ** 2
    return float(np.sum(density) * cfg.dx)


def compute_e_short(rollout: RolloutResult, cfg, n_steps: int = 50) -> float:
    # Mean RMSE (nodes-averaged) over the first `n_steps` rollout steps.
    steps = _rollout_steps(cfg)[:n_steps]
    nodes = cfg.nodes
    U, U_reel = rollout.U, rollout.U_reel
    rmses = [float(np.sqrt(np.mean((U[k, nodes] - U_reel[k, nodes]) ** 2))) for k in steps]
    return float(np.mean(rmses)) if rmses else None


def compute_t_div(rollout: RolloutResult, cfg, threshold: float = 0.10) -> float | None:
    # First rollout time where the Linf error exceeds `threshold` times the
    # reference trajectory's own global peak amplitude. None if the rollout
    # never diverges by this criterion (or the reference is ~motionless).
    steps = _rollout_steps(cfg)
    nodes = cfg.nodes
    U, U_reel = rollout.U, rollout.U_reel
    amp_ref = float(np.abs(U_reel[:, nodes]).max())
    if amp_ref < 1e-12:
        return None
    for k in steps:
        err = float(np.max(np.abs(U[k, nodes] - U_reel[k, nodes])))
        if err > threshold * amp_ref:
            return float(k * cfg.dt)
    return None


def compute_amp_loss_pct(rollout: RolloutResult, cfg) -> float:
    # Positive = the surrogate under-shoots the true peak amplitude
    # (damping); negative = it over-shoots (amplifying/unstable).
    nodes = cfg.nodes
    amp_ref = float(np.abs(rollout.U_reel[:, nodes]).max())
    amp_pred = float(np.abs(rollout.U[:, nodes]).max())
    if amp_ref < 1e-12:
        return None
    return float(100.0 * (amp_ref - amp_pred) / amp_ref)


def compute_energy_drift_pct(curves: dict, warmup_idx: int = 10) -> float | None:
    # Drift of the SURROGATE's own energy curve relative to its value a few
    # steps in (skips t=0, where energy can be exactly 0 before a boundary
    # excitation has reached the interior) -- the equation is conservative,
    # so a well-behaved surrogate keeps this near 0 regardless of what the
    # reference does (H6/H8's decisive diagnostic).
    energy = curves["energy"]
    if len(energy) <= warmup_idx:
        warmup_idx = 0
    e0 = energy[warmup_idx]
    if abs(e0) < 1e-12:
        return None
    return float(100.0 * (energy[-1] - e0) / e0)


def compute_spectrum(rollout: RolloutResult, cfg) -> dict:
    # Wavenumber spectrum of predicted and reference final-step states
    # (H-analysis fig. 17): power = |rfft(u)|^2 along the spatial axis.
    nodes = cfg.nodes
    u_pred = rollout.U[-1, nodes]
    u_ref = rollout.U_reel[-1, nodes]
    k = (2 * np.pi * np.fft.rfftfreq(len(nodes), d=cfg.dx)).tolist()
    power_pred = (np.abs(np.fft.rfft(u_pred)) ** 2).tolist()
    power_ref = (np.abs(np.fft.rfft(u_ref)) ** 2).tolist()
    return {"k": k, "power_pred": power_pred, "power_ref": power_ref}


def compute_r2_onestep(one_step_metrics: dict) -> float | None:
    # Mean R² across every output horizon column (training/teacher_forcing.py
    # evaluate_one_step's per-column {"mse_norm", "r2"} dict).
    if not one_step_metrics:
        return None
    return float(np.mean([m["r2"] for m in one_step_metrics.values()]))


def build_metrics(cfg, *, train_result=None, rollout: RolloutResult | None = None,
                   bench: BenchmarkResult | None = None, one_step_metrics: dict | None = None,
                   err_near_junction: float | None = None) -> dict:
    curves = compute_error_curves(rollout, cfg) if rollout is not None else {
        "t": [], "err_rel_mean": [], "err_max": [], "amp_max": [], "energy": []}
    spectrum = compute_spectrum(rollout, cfg) if rollout is not None else {"k": [], "power_pred": [], "power_ref": []}

    scalars = {
        "r2_onestep": compute_r2_onestep(one_step_metrics) if one_step_metrics is not None else None,
        "E_short": compute_e_short(rollout, cfg) if rollout is not None else None,
        "t_div": compute_t_div(rollout, cfg) if rollout is not None else None,
        "amp_loss_pct": compute_amp_loss_pct(rollout, cfg) if rollout is not None else None,
        "energy_drift_pct": compute_energy_drift_pct(curves) if curves["energy"] else None,
        "n_params": int(train_result.n_params) if train_result is not None else None,
        "train_time_s": float(train_result.train_time_s) if train_result is not None else None,
        "net_evals_per_unit_time": float(bench.n_calls / cfg.t_end) if bench is not None else None,
        "err_near_junction": err_near_junction,
    }

    return {
        "run_id": cfg.run_id,
        "phase": cfg.phase,
        "hypothesis": cfg.hypothesis,
        "scalars": scalars,
        "curves": curves,
        "spectrum": spectrum,
    }


def write_metrics_json(path: str | Path, metrics: dict) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
