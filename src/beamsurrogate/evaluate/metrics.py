# The metrics.json contract every run writes -- same keys for every run,
# `null` when a metric doesn't apply (never omitted, so analysis/*.py can
# always index the same schema across runs/*/metrics.json). This module was
# almost entirely missing before this refactor: without it, none of the 27
# ablation runs would be comparable or even individually interpretable.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .rollout import RolloutResult, BenchmarkResult, run_rollout


def l2_rel(pred, true, eps=1e-12):
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + eps))


def smape(pred, true):
    m = true != 0
    return float(np.mean(2 * np.abs(true[m] - pred[m]) / (np.abs(true[m]) + np.abs(pred[m]))))


def _last_written_step(cfg):
    # autoregressive_rollout (physics/solver.py) fills U in blocks of
    # N_FWD*ndt starting at history_needed=M_BACK*ndt; when (Nt -
    # history_needed) isn't a multiple of that block size, its last block
    # stops short of Nt and the trailing rows of U are left at their
    # np.zeros(...) initialization -- never written by the model. Sampling
    # past this point measures |0 - U_reel|, i.e. the reference's own
    # amplitude, identical for every model and not a real error (first
    # diagnosed by hand as the "TAIL ARTIFACT" in
    # p3_analysis_pinn_comparative.py; this is the root-cause fix).
    history_needed = cfg.M_BACK * cfg.ndt
    block = cfg.N_FWD * cfg.ndt
    last_n = None
    for n in range(history_needed, cfg.Nt - block + 1, block):
        last_n = n
    return (last_n + block) if last_n is not None else history_needed


def _rollout_steps(cfg):
    # Same step grid every rollout error curve is sampled on: ndt-spaced,
    # starting once the network has actually produced a prediction, capped
    # at the last step autoregressive_rollout actually wrote (see
    # _last_written_step) rather than cfg.Nt.
    return np.arange(2 * cfg.ndt, _last_written_step(cfg) + 1, cfg.ndt)


def compute_error_curves(rollout: RolloutResult, cfg) -> dict:
    steps = _rollout_steps(cfg)
    nodes = cfg.nodes
    U, U_reel = rollout.U, rollout.U_reel

    err_rel_mean = [l2_rel(U[k, nodes], U_reel[k, nodes]) for k in steps]
    err_max = [float(np.max(np.abs(U[k, nodes] - U_reel[k, nodes]))) for k in steps]
    # Mean absolute error across the beam at this step, in physical units --
    # distinct from err_rel_mean (a ratio of L2 norms, normalized by the
    # INSTANTANEOUS reference amplitude at that same step, which is why it
    # is noisy/near-1 whenever the reference itself is near zero). This one
    # normalizes against the single, time-invariant amp_ref (see
    # compute_amp_ref) instead, so it is comparable on the same scale as
    # err_max and can share the T_*_pct/P_thr threshold machinery below.
    err_mean_abs = [float(np.mean(np.abs(U[k, nodes] - U_reel[k, nodes]))) for k in steps]
    amp_max = [float(np.max(np.abs(U[k, nodes]))) for k in steps]
    energy = [_mechanical_energy(U, k, cfg) for k in steps]
    # Same computation, on the FD reference -- plotted alongside `energy`
    # (the surrogate's own) so the two can be compared directly instead of
    # only judging the surrogate's energy against its own starting value
    # (see compute_energy_drift_pct, which still only looks at `energy`).
    energy_ref = [_mechanical_energy(U_reel, k, cfg) for k in steps]
    pearson_r = [_pearson_r(U[k, nodes], U_reel[k, nodes]) for k in steps]

    return {
        "t": (steps * cfg.dt).tolist(),
        "err_rel_mean": err_rel_mean,
        "err_max": err_max,
        "err_mean_abs": err_mean_abs,
        "amp_max": amp_max,
        "energy": energy,
        "energy_ref": energy_ref,
        "pearson_r": pearson_r,
    }


def _pearson_r(pred: np.ndarray, true: np.ndarray) -> float | None:
    # Shape agreement between predicted and true snapshots, independent of
    # amplitude -- a prediction with the right shape but wrong size still
    # scores high here, so this must always be read next to an
    # amplitude-sensitive curve (err_mean_abs/err_max), never alone.
    if np.std(pred) < 1e-12 or np.std(true) < 1e-12:
        return None
    return float(np.corrcoef(pred, true)[0, 1])


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
    # First rollout time where the max absolute error exceeds `threshold` times the
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


def compute_amp_ref(rollout: RolloutResult, cfg) -> float | None:
    # The single, time-invariant FD reference peak amplitude that every
    # threshold-based metric below (T_max_*, T_mean_*, P_thr) is anchored
    # to -- same quantity compute_t_div computes internally for itself,
    # exposed here as its own scalar so all of them share exactly one
    # value rather than each recomputing it (and so nomenclature.tex's
    # "X% of the FD peak amplitude" phrasing has one unambiguous referent).
    nodes = cfg.nodes
    amp_ref = float(np.abs(rollout.U_reel[:, nodes]).max())
    return amp_ref if amp_ref >= 1e-12 else None


def compute_t_mean_threshold(rollout: RolloutResult, cfg, amp_ref: float | None, threshold: float) -> float | None:
    # T_mean_5%/10%: first rollout time where the MEAN absolute error over
    # the beam (not the worst node, see compute_t_div) exceeds `threshold`
    # times amp_ref. None if never crossed.
    if amp_ref is None:
        return None
    steps = _rollout_steps(cfg)
    nodes = cfg.nodes
    U, U_reel = rollout.U, rollout.U_reel
    for k in steps:
        err = float(np.mean(np.abs(U[k, nodes] - U_reel[k, nodes])))
        if err > threshold * amp_ref:
            return float(k * cfg.dt)
    return None


def compute_p_thr(curves: dict, amp_ref: float | None, threshold: float = 0.05) -> float | None:
    # P_thr: percentage of the tracked rollout timesteps where the max
    # absolute error over the beam exceeds `threshold` * amp_ref -- "how
    # much of the simulation is currently bad", distinct from T_max_*%
    # ("when does it first go bad").
    if amp_ref is None or not curves["err_max"]:
        return None
    vals = curves["err_max"]
    n_above = sum(1 for v in vals if v > threshold * amp_ref)
    return float(100.0 * n_above / len(vals))


def compute_amp_loss_pct(rollout: RolloutResult, cfg) -> float:
    # Positive = the surrogate under-shoots the true peak amplitude
    # (damping); negative = it over-shoots (amplifying/unstable).
    nodes = cfg.nodes
    amp_ref = float(np.abs(rollout.U_reel[:, nodes]).max())
    amp_pred = float(np.abs(rollout.U[:, nodes]).max())
    if amp_ref < 1e-12:
        return None
    return float(100.0 * (amp_ref - amp_pred) / amp_ref)


def compute_error_correlation_time(rollout: RolloutResult, cfg) -> float | None:
    # How long the NN's error "remembers itself": autocorrelation of the
    # error signal at the beam's midpoint node, over the tracked rollout
    # steps. Correlation time = the lag at which the (mean-subtracted,
    # normalized) autocorrelation first drops to 1/e of its zero-lag value
    # -- short means the error behaves like fresh noise each step, long
    # means a persistent, structural bias. None if the error is ~0
    # throughout (nothing to correlate) or never drops below 1/e within the
    # tracked window.
    steps = _rollout_steps(cfg)
    node = cfg.nodes[len(cfg.nodes) // 2]
    U, U_reel = rollout.U, rollout.U_reel
    e = np.array([U[k, node] - U_reel[k, node] for k in steps])
    e = e - e.mean()
    var = float(np.dot(e, e))
    if var < 1e-24:
        return None
    acf = np.correlate(e, e, mode="full")[len(e) - 1:] / var
    below = np.where(acf < 1 / np.e)[0]
    if len(below) == 0:
        return None
    dt_eff = cfg.ndt * cfg.dt   # spacing between consecutive tracked rollout steps
    return float(below[0] * dt_eff)


def compute_energy_drift_pct(curves: dict, warmup_idx: int = 10) -> float | None:
    # Drift of the SURROGATE's own energy curve relative to its value a few
    # steps in (skips t=0, where energy can be exactly 0 before a boundary
    # excitation has reached the interior) -- the equation is conservative,
    # so a well-behaved surrogate keeps this near 0 regardless of what the
    # reference does (phase 3/phase 10's decisive diagnostic).
    energy = curves["energy"]
    if len(energy) <= warmup_idx:
        warmup_idx = 0
    e0 = energy[warmup_idx]
    if abs(e0) < 1e-12:
        return None
    return float(100.0 * (energy[-1] - e0) / e0)


def compute_spectrum(rollout: RolloutResult, cfg, n_snapshots: int = 5) -> dict:
    # Wavenumber spectrum of predicted and reference states, at n_snapshots
    # times spread evenly across the tracked rollout (first to last) --
    # shows whether energy shifts toward the jagged (short-wavelength) end
    # as the rollout progresses, not just a single final-step snapshot
    # (what this used to compute). A Hann window is applied to each
    # snapshot before the FFT: the beam's displacement isn't a periodic
    # signal, so a plain FFT leaks energy into spurious high-k components
    # from the sharp edges -- the window tapers those edges to ~0 first.
    steps = _rollout_steps(cfg)
    nodes = cfg.nodes
    idx = np.unique(np.linspace(0, len(steps) - 1, n_snapshots).astype(int))
    snapshot_steps = steps[idx]
    window = np.hanning(len(nodes))

    k = (2 * np.pi * np.fft.rfftfreq(len(nodes), d=cfg.dx)).tolist()
    power_pred, power_ref = [], []
    for m in snapshot_steps:
        power_pred.append((np.abs(np.fft.rfft(rollout.U[m, nodes] * window)) ** 2).tolist())
        power_ref.append((np.abs(np.fft.rfft(rollout.U_reel[m, nodes] * window)) ** 2).tolist())

    return {
        "k": k,
        "t_snapshots": (snapshot_steps * cfg.dt).tolist(),
        "power_pred": power_pred,
        "power_ref": power_ref,
    }


def compute_t_at_e_max(curves: dict) -> float | None:
    # Timestep at which E_max (the worst-node error over the whole rollout,
    # see build_metrics below) actually occurs -- E_max alone is just a
    # value, this says whether that worst case hit early (still settling)
    # or late (drifted there after a long stable stretch).
    if not curves["err_max"]:
        return None
    idx = int(np.argmax(curves["err_max"]))
    return float(curves["t"][idx])


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
        "t": [], "err_rel_mean": [], "err_max": [], "err_mean_abs": [], "amp_max": [],
        "energy": [], "energy_ref": [], "pearson_r": []}
    spectrum = compute_spectrum(rollout, cfg) if rollout is not None else {
        "k": [], "t_snapshots": [], "power_pred": [], "power_ref": []}
    amp_ref = compute_amp_ref(rollout, cfg) if rollout is not None else None
    t_div = compute_t_div(rollout, cfg) if rollout is not None else None   # threshold=0.10, kept for backward compat

    scalars = {
        "r2_onestep": compute_r2_onestep(one_step_metrics) if one_step_metrics is not None else None,
        "E_short": compute_e_short(rollout, cfg) if rollout is not None else None,
        "t_div": t_div,
        "amp_ref": amp_ref,
        "E_max": float(max(curves["err_max"])) if curves["err_max"] else None,
        "t_E_max": compute_t_at_e_max(curves),
        "P_thr_5pct": compute_p_thr(curves, amp_ref, 0.05) if rollout is not None else None,
        "T_max_5pct": compute_t_div(rollout, cfg, threshold=0.05) if rollout is not None else None,
        "T_max_10pct": t_div,
        "T_mean_5pct": compute_t_mean_threshold(rollout, cfg, amp_ref, 0.05) if rollout is not None else None,
        "T_mean_10pct": compute_t_mean_threshold(rollout, cfg, amp_ref, 0.10) if rollout is not None else None,
        "amp_loss_pct": compute_amp_loss_pct(rollout, cfg) if rollout is not None else None,
        "energy_drift_pct": compute_energy_drift_pct(curves) if curves["energy"] else None,
        "error_corr_time": compute_error_correlation_time(rollout, cfg) if rollout is not None else None,
        "n_params": int(train_result.n_params) if train_result is not None else None,
        "train_time_s": float(train_result.train_time_s) if train_result is not None else None,
        "net_evals_per_unit_time": float(bench.n_calls / cfg.t_end) if bench is not None else None,
        "net_evals_total": int(bench.n_calls) if bench is not None else None,
        # Wall-clock time (median/std over 15 repeats, 3 warmup -- see
        # evaluate/rollout.py's chrono()) and FLOPs for one full test-time
        # rollout, solver (fd_*) vs surrogate (nn_*) -- see BenchmarkResult
        # in evaluate/rollout.py for how each is computed, and the
        # FD_FLOPS_PER_POINT_STEP comment there for the FD FLOP convention
        # (analytical, not measured -- numpy has no FLOP counter for this).
        "fd_time_med_s": float(bench.fd_time_med) if bench is not None else None,
        "fd_time_std_s": float(bench.fd_time_std) if bench is not None else None,
        "nn_time_med_s": float(bench.nn_time_med) if bench is not None else None,
        "nn_time_std_s": float(bench.nn_time_std) if bench is not None else None,
        "fd_flops": float(bench.fd_flops) if bench is not None else None,
        "nn_flops": float(bench.nn_flops) if bench is not None else None,
        "err_near_junction": err_near_junction,
    }

    return {
        "scalars": scalars,
        "curves": curves,
        "spectrum": spectrum,
    }


def write_metrics_json(path: str | Path, metrics: dict) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


# Multi-trajectory rollout evaluation: build_metrics above scores exactly one
# rollout_idx, which is how every run.__call__ site (cli.py, the per-family
# eval scripts) has evaluated a model so far -- fine for the plots, but it
# means every scalar in a run's results.yaml/metrics.json is a single random
# draw from the test split. evaluate_multi_rollout/aggregate_multi_rollout_
# metrics loop the same run_rollout + build_metrics(rollout=...) pair over
# every requested test trajectory instead, so comparisons across configs
# aren't confounded by which one trajectory got picked.
#
# Split by censoring, same convention throughout: CONTINUOUS_SCALAR_KEYS are
# defined on every rollout regardless of outcome, so they're summarized as
# mean/std/median over the full set. CENSORED_TIME_KEYS (t_div and the
# T_*_pct family) are only defined on rollouts that actually crossed their
# threshold -- a rollout that never diverges is the BEST outcome, not a
# missing value, so these are summarized as (% of trajectories that reached
# the threshold, median time among those that did) rather than averaged
# naively over all trajectories.
CONTINUOUS_SCALAR_KEYS = ("E_short", "E_max", "t_E_max", "P_thr_5pct", "amp_loss_pct",
                           "energy_drift_pct", "error_corr_time")
CENSORED_TIME_KEYS = ("t_div", "T_max_5pct", "T_max_10pct", "T_mean_5pct", "T_mean_10pct")


def evaluate_multi_rollout(model, FIELDS, bc_pairs, idx_list, input_fields, norm_stats,
                            INPUTS, OUTPUTS, cfg) -> list[dict]:
    # One record per trajectory in idx_list, each the "scalars" half of
    # build_metrics's output plus which test index it came from. No curves/
    # spectrum kept per-trajectory -- those are only meaningful for a single
    # showcase rollout (that's still what cli.py's rollout.gif/error_vs_time
    # plots use), not something you average over 99 different trajectories.
    records = []
    for idx in idx_list:
        rollout = run_rollout(model, FIELDS, bc_pairs, idx, input_fields, norm_stats, INPUTS, OUTPUTS, cfg)
        scalars = build_metrics(cfg, rollout=rollout)["scalars"]
        records.append({"traj_idx": int(idx), **scalars})
    return records


def aggregate_multi_rollout_metrics(records: list[dict]) -> dict:
    n = len(records)
    summary = {"n_trajectories": n}
    for key in CONTINUOUS_SCALAR_KEYS:
        vals = [r[key] for r in records if r.get(key) is not None]
        if not vals:
            summary[key] = {"mean": None, "std": None, "median": None, "n": 0}
        else:
            arr = np.asarray(vals, dtype=float)
            summary[key] = {"mean": float(arr.mean()), "std": float(arr.std()),
                             "median": float(np.median(arr)), "n": len(vals)}
    for key in CENSORED_TIME_KEYS:
        vals = [r[key] for r in records if r.get(key) is not None]
        summary[key] = {
            "pct_reached": 100.0 * len(vals) / n if n else None,
            "median_reached": float(np.median(vals)) if vals else None,
            "n_reached": len(vals),
        }
    return summary
