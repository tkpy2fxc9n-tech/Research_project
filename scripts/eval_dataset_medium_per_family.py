#!/usr/bin/env python3
"""
p6_dataset_medium: one rollout figure per waveform family (gaussian,
sine_pulse, triangular, sawtooth, square -- the 5 families data/generate.py's
"medium" profile draws from), instead of the single arbitrary test
trajectory cli.py's own rollout.gif uses.

Reuses data/split.py's family_showcase_idx (one representative val/test
trajectory per family, added to SHOWCASE_FAMILIES for the 3 new families
here) rather than picking trajectories by hand.

norm_stats recomputed from the FULL medium dataset (max_trajectories=None,
matching cli.py's own real-training call) -- NOT a subsample: a 200-
trajectory subsample of the "complex" dataset gave badly wrong norm_stats
earlier this session (amp_loss_pct off by 10 orders of magnitude vs the
run's own recorded metrics.json), because "complex" mixes many BC families
with different scales. "medium" is narrower than "complex" but still spans
5 shapes, so this script doesn't take that risk again.

Usage: python scripts/eval_dataset_medium_per_family.py
"""
from __future__ import annotations

import dataclasses
import gc
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "common"))

from beamsurrogate.config import Config, set_seeds  # noqa: E402
from beamsurrogate.registry import MODELS, DATASETS  # noqa: E402
from beamsurrogate.data.split import load_hdf5_dataset, compute_norm_stats  # noqa: E402
from beamsurrogate.evaluate.rollout import run_rollout  # noqa: E402
from beamsurrogate.evaluate.metrics import build_metrics  # noqa: E402
from beamsurrogate.evaluate import plots  # noqa: E402
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

DATA_DIR = REPO_ROOT / "data"
SOURCE_RUN_ID = "p6_dataset_medium"
EXPECTED_FAMILIES = ["gaussian", "sine_pulse", "triangular", "sawtooth", "square"]


def _load_resolved_config(run_id: str) -> Config:
    resolved = yaml.safe_load((_run_path(run_id) / "config.resolved.yaml").read_text())
    known = {f.name for f in dataclasses.fields(Config)}
    raw = {k: v for k, v in resolved.items() if k in known}
    for tuple_field in ("HIDDEN_SIZES", "CNN_CHANNELS"):
        if tuple_field in raw and isinstance(raw[tuple_field], list):
            raw[tuple_field] = tuple(raw[tuple_field])
    return Config(**raw)


def main():
    cfg = _load_resolved_config(SOURCE_RUN_ID)
    set_seeds(cfg)

    dataset_path = DATA_DIR / DATASETS[cfg.dataset]
    print(f"Loading {dataset_path} (FULL dataset, matching training's own norm_stats)...")
    (df, FIELDS, INPUTS, OUTPUTS, bc_pairs, idx_train, idx_val, idx_test,
     rollout_idx, family_showcase_idx) = load_hdf5_dataset(
        cfg.features, cfg, dataset_path, max_trajectories=None)
    norm_stats = compute_norm_stats(df, INPUTS, OUTPUTS, cfg)
    del df
    gc.collect()

    missing = [fam for fam in EXPECTED_FAMILIES if fam not in family_showcase_idx]
    if missing:
        print(f"WARNING: no val/test/train trajectory found for families {missing} -- "
              f"showcase has {list(family_showcase_idx)}")

    model = MODELS[cfg.model](len(INPUTS), len(OUTPUTS), cfg)
    model.load_state_dict(torch.load(_run_path(SOURCE_RUN_ID) / "model.pth", weights_only=True))
    model.eval()
    print(f"Loaded {SOURCE_RUN_ID}'s model.")

    figures_dir = _run_path(SOURCE_RUN_ID) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = [f"{SOURCE_RUN_ID} -- one rollout per waveform family\n"]
    for fam in EXPECTED_FAMILIES:
        if fam not in family_showcase_idx:
            continue
        idx = family_showcase_idx[fam]
        rollout = run_rollout(model, FIELDS, bc_pairs, idx, cfg.features, norm_stats, INPUTS, OUTPUTS, cfg)
        metrics = build_metrics(cfg, rollout=rollout)
        plots.make_rollout_animation(rollout, cfg, figures_dir, filename=f"rollout_{fam}.gif")
        line = (f"{fam:12s} (test idx {idx}): E_short={metrics['scalars']['E_short']:.4e}  "
                f"t_div={metrics['scalars']['t_div']}  amp_loss_pct={metrics['scalars']['amp_loss_pct']:.2f}")
        print(line)
        summary_lines.append(line)

    (figures_dir / "per_family_summary.txt").write_text("\n".join(summary_lines) + "\n")
    print(f"\nDone -- outputs in {figures_dir}")


if __name__ == "__main__":
    main()
