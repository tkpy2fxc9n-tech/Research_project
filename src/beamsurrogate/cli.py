# Entry point: `python -m beamsurrogate --config configs/runs/<run_id>.yaml`.
# Replaces every project's own main.py -- which model/regime/stabilizer/
# dataset a run uses is entirely decided by the YAML (registry.py), never by
# which script you happen to invoke.
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from .config import Config, load_config, config_to_dict, set_seeds
from .registry import MODELS, REGIMES, DATASETS, resolve_stabilizer
from .data.split import load_hdf5_dataset, compute_norm_stats
from .data.norm import make_dataloaders
from .training.teacher_forcing import evaluate_one_step
from .evaluate.rollout import run_rollout, benchmark_inference
from .evaluate.metrics import build_metrics, write_metrics_json
from .evaluate import plots

REPO_ROOT = Path(__file__).resolve().parents[2]   # src/beamsurrogate/cli.py -> src -> repo root
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"


def _log_rss(label: str) -> None:
    # Diagnostic checkpoint for the full (non-smoke-test) dataset's memory
    # footprint -- current resident memory, not torch/pandas guesses, so a
    # Slurm run's .log shows exactly where usage climbs instead of relying
    # on extrapolation from a smaller local sample.
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    print(f"[mem] {label}: {int(line.split()[1]) / 1e6:.2f} Go", flush=True)
                    return
    except OSError:
        pass


def parse_args():
    p = argparse.ArgumentParser(description="Train and evaluate one beamsurrogate run from a config file.")
    p.add_argument("--config", type=Path, required=True, help="Path to a configs/runs/<run_id>.yaml file.")
    p.add_argument("--smoke-test", action="store_true",
                    help="Miniature run (few trajectories, few epochs) to check the pipeline end to end.")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="Override a config field, e.g. --set N_EPOCHS=5. Repeatable.")
    return p.parse_args()


def _parse_overrides(raw: list[str]) -> dict:
    overrides = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key] = yaml.safe_load(value)   # numbers/bools/strings parsed the same way YAML would
    return overrides


def _git_info() -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "diff", "--quiet"], cwd=REPO_ROOT).returncode != 0
        return {"sha": sha.stdout.strip() if sha.returncode == 0 else None, "dirty": dirty}
    except OSError:
        return {"sha": None, "dirty": None}


def _dataset_sha256(dataset_path: Path) -> str | None:
    # Reads the sidecar `<dataset>.sha256` scripts/make_dataset.py writes
    # after generation -- never hashes the (potentially multi-GB) file
    # itself on every run.
    sidecar = dataset_path.with_suffix(dataset_path.suffix + ".sha256")
    if sidecar.exists():
        return sidecar.read_text().split()[0].strip()
    return None


def write_env_json(path: Path, dataset_path: Path) -> None:
    env = {
        "git": _git_info(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _dataset_sha256(dataset_path),
        "hostname": socket.gethostname(),
        "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    with open(path, "w") as f:
        json.dump(env, f, indent=2)


def make_run_dir(run_id: str) -> Path:
    run_dir = RUNS_DIR / run_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir


def run(cfg: Config, run_dir: Path, dataset_path: Path, max_trajectories: int | None,
        make_animation: bool = True) -> dict:
    set_seeds(cfg)

    INPUT_FIELDS = cfg.features
    _log_rss("start")
    (df, FIELDS, INPUTS, OUTPUTS, bc_pairs, idx_train, idx_val, idx_test,
     rollout_idx, family_showcase_idx) = load_hdf5_dataset(INPUT_FIELDS, cfg, dataset_path, max_trajectories)
    _log_rss("after load_hdf5_dataset")
    norm_stats = compute_norm_stats(df, INPUTS, OUTPUTS, cfg)
    _log_rss("after compute_norm_stats")

    model = MODELS[cfg.model](len(INPUTS), len(OUTPUTS), cfg)
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader, X_val, y_val = make_dataloaders(df, INPUTS, OUTPUTS, norm_stats, cfg)
    _log_rss("after make_dataloaders")
    # df's train/val rows (>=95% of it) are never touched again after this
    # point -- only the test split is, much later, for evaluate_one_step.
    # Shrinking df now lets the rest of it be freed before training starts,
    # instead of sitting alongside train_loader's own already-extracted copy.
    df = df[df["split"] == "test"].reset_index(drop=True)
    _log_rss("after trimming df to test split")
    patience = cfg.EARLY_STOP_PATIENCE if cfg.EARLY_STOP_PATIENCE > 0 else None
    model_path = run_dir / "model.pth"

    regime = cfg.regime
    if regime == "teacher_forcing":
        train_result = REGIMES[regime].run(model, train_loader, X_val, y_val, cfg, model_path, patience)
    elif regime in ("bptt", "pushforward"):
        train_result = REGIMES[regime].run(model, FIELDS, bc_pairs, idx_train, idx_val, INPUT_FIELDS,
                                             norm_stats, INPUTS, OUTPUTS, cfg, train_loader, model_path, patience)
    else:
        raise ValueError(f"Unknown regime {regime!r}, expected one of {sorted(REGIMES)}")
    _log_rss("after training")

    one_step_metrics, y_true_onestep, y_pred_onestep = evaluate_one_step(model, df, INPUTS, OUTPUTS, norm_stats)
    _log_rss("after evaluate_one_step")

    rollout = run_rollout(model, FIELDS, bc_pairs, rollout_idx, INPUT_FIELDS, norm_stats, INPUTS, OUTPUTS, cfg)
    bench = benchmark_inference(model, FIELDS, INPUT_FIELDS, norm_stats, INPUTS, OUTPUTS, rollout, cfg)
    _log_rss("after rollout + benchmark")

    metrics = build_metrics(cfg, train_result=train_result, rollout=rollout, bench=bench,
                             one_step_metrics=one_step_metrics)

    figures_dir = run_dir / "figures"
    plots.plot_training_curve(train_result, figures_dir)
    plots.plot_one_step_predictions(y_true_onestep, y_pred_onestep, OUTPUTS, one_step_metrics, figures_dir)
    plots.plot_rollout_error(metrics["curves"], figures_dir)
    plots.plot_amplitude_and_energy(metrics["curves"], figures_dir)
    plots.plot_spectrum(metrics["spectrum"], figures_dir)
    if make_animation:
        plots.make_rollout_animation(rollout, cfg, figures_dir)

    return metrics


def main():
    args = parse_args()
    overrides = _parse_overrides(args.set)
    cfg = load_config(args.config, **overrides)

    if args.smoke_test:
        cfg = dataclasses.replace(cfg, N_EPOCHS=min(cfg.N_EPOCHS, 2))
        max_trajectories = 16
        make_animation = False
    else:
        max_trajectories = None
        make_animation = True

    cfg = resolve_stabilizer(cfg)

    dataset_path = DATA_DIR / DATASETS[cfg.dataset]
    if not dataset_path.exists():
        print(f"ERROR: dataset not found at {dataset_path}. Generate it first with "
              f"scripts/make_dataset.py --profile {cfg.dataset}.", file=sys.stderr)
        sys.exit(1)

    run_dir = make_run_dir(cfg.run_id)
    with open(run_dir / "config.resolved.yaml", "w") as f:
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False)
    write_env_json(run_dir / "env.json", dataset_path)

    mode = "SMOKE TEST" if args.smoke_test else "run"
    print(f"=== beamsurrogate [{mode}] run_id={cfg.run_id} model={cfg.model} regime={cfg.regime} "
          f"stabilizer={cfg.stabilizer} dataset={cfg.dataset} features={cfg.features} ===")

    metrics = run(cfg, run_dir, dataset_path, max_trajectories, make_animation)
    write_metrics_json(run_dir / "metrics.json", metrics)
    print(f"Done -- outputs in {run_dir}")


if __name__ == "__main__":
    main()
