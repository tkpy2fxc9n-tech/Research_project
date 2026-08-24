#!/usr/bin/env python3
# Family sweep: replays real held-out test trajectories from the "medium"
# dataset through the 4 shortlisted p7 assembly models (the non-diverged
# rows in runs/p7/analysis/comparison_summary.txt), using each
# trajectory's own recorded driving signal instead of the fixed synthetic
# gaussian pulse p7_structure_several_rods.py normally uses -- same
# "evaluate over the full held-out test split, report mean +/- std"
# methodology as scripts/eval_p1_multi_trajectory.py / runs/p1_analysis/,
# applied here per excitation family instead of pooled.
#
# Skips beamsurrogate.data.split.load_hdf5_dataset's full windowed load
# (builds a training-sized feature table for all 2000 trajectories -- the
# thing eval_p1_multi_trajectory.py's own comment warns blows the login
# node's 32G cap). Only bc_pairs/idx_test are needed here, both cheaply
# rebuildable from 5 small raw HDF5 arrays without ever touching `u`.
#
# Usage: python analysis/p7_family_sweep.py
from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "src"))
import p7_structure_several_rods as m  # noqa: E402
from beamsurrogate.config import load_config  # noqa: E402
from beamsurrogate.data.split import BC_LABEL_TO_TYPE  # noqa: E402
from beamsurrogate.registry import DATASETS  # noqa: E402

MODELS_DIR = REPO_ROOT / "runs" / "p7" / "models"
OUT_DIR = REPO_ROOT / "runs" / "p7" / "analysis"

# The non-diverged, shortlisted models from p7_compare_assembly_models.py.
# Every folder under runs/p7/models/ is named by its own run_id.
RUN_IDS = [
    "p7_dataset_medium_pinn_0.01",
    "p7_dataset_medium",
    "p7_dataset_medium_arch_256_128_32_pinn_0.01",
    "p7_dataset_medium_ut",
]


def _decode(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def load_driven_bc_pairs() -> dict[str, list[tuple[int, tuple]]]:
    # Rebuilds just the driven-side "table" BC per held-out test trajectory --
    # the one piece of split.py's load_hdf5_dataset()'s bc_pairs this script
    # needs -- straight from the HDF5 file's own arrays (all 4 shortlisted
    # models share dataset=medium, Nt=500, ndt=3, confirmed against their
    # config.yaml before writing this).
    cfg = load_config(MODELS_DIR / "p7_dataset_medium" / "config.yaml")
    dataset_path = REPO_ROOT / "data" / DATASETS[cfg.dataset]
    with h5py.File(dataset_path, "r") as f:
        n = f["u"].shape[0]
        right_bc_value = f["right_bc_value"][:n]
        right_label = _decode(f["right_label"][:n])
        right_family = _decode(f["right_family"][:n])
        left_driven = f["left_driven"][:n].astype(bool)
        right_driven = f["right_driven"][:n].astype(bool)
        split = _decode(f["split"][:n])

    assert np.all(right_driven) and not np.any(left_driven), (
        "expected the medium dataset's right end to always be the driven side -- "
        "re-check before trusting this sweep's per-family grouping")

    t_ctrl = (np.arange(cfg.Nt + 1) * cfg.dt).tolist()
    by_family: dict[str, list[tuple[int, tuple]]] = defaultdict(list)
    for idx in range(n):
        if split[idx] != "test":
            continue
        bc_type = BC_LABEL_TO_TYPE[right_label[idx]]
        driven_bc = (bc_type, "table",
                     {"t_ctrl": t_ctrl, "values": right_bc_value[idx].tolist(),
                      "source_label": right_label[idx]})
        by_family[right_family[idx]].append((int(idx), driven_bc))
    return by_family


def main():
    by_family = load_driven_bc_pairs()
    families = sorted(by_family)
    print("Held-out test trajectories per family:", {fam: len(v) for fam, v in by_family.items()})

    rows = []  # one dict per (model, family, traj_idx)
    t0 = time.perf_counter()
    n_total = sum(len(v) for v in by_family.values()) * len(RUN_IDS)
    n_done = 0

    for run_id in RUN_IDS:
        folder = run_id  # kept as a separate column for the CSV's existing schema
        output_dir = MODELS_DIR / run_id / "family_sweep"
        # Load the model + recompute norm_stats ONCE per model here, not once
        # per trajectory -- that reload is ~50s+ of dead weight per call (see
        # this script's docstring), and it's identical across every
        # trajectory/family driving the SAME model.
        cfg, run_dir = m.load_source(run_id)
        bundle = m.load_model_bundle(cfg, run_dir / "model.pth")
        for family in families:
            for traj_idx, driven_bc in by_family[family]:
                tag = f"{family}_idx{traj_idx}"
                metrics = m.run_lattice_test(run_id, output_dir, driven_bc=driven_bc, tag=tag,
                                              make_gifs=False, make_figures=False, bundle=bundle)
                n_done += 1
                elapsed = time.perf_counter() - t0
                rel_mean = metrics.get("rel_mean_pct")
                rel_mean_str = f"{rel_mean:.2f}%" if rel_mean is not None else "n/a"
                print(f"[{n_done}/{n_total}] {folder} {tag}: rel_mean={rel_mean_str}  "
                      f"diverged={metrics['diverged']}  ({elapsed:.0f}s elapsed)")
                rows.append({
                    "folder": folder, "run_id": run_id, "family": family, "traj_idx": traj_idx,
                    "rel_mean_pct": rel_mean, "rel_max_pct": metrics.get("rel_max_pct"),
                    "ratio_nn_fd": metrics.get("ratio_nn_fd"), "diverged": metrics["diverged"],
                })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "family_sweep_per_run.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}")

    # Aggregate: mean/std of rel_mean_pct per (folder, family), over non-diverged rows.
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    diverged_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        diverged_count[r["folder"]][r["family"]] += int(r["diverged"])
        if not r["diverged"] and r["rel_mean_pct"] is not None:
            agg[r["folder"]][r["family"]].append(r["rel_mean_pct"])

    lines = ["p17 rectangular assembly -- family sweep over held-out test trajectories",
             "mean +/- std of relative mean |FD-NN| error (%), per model per family "
             "(d = count that diverged, excluded from the mean/std)",
             ""]
    header = f"{'model':<45}" + "".join(f"{fam:>18}" for fam in families)
    lines.append(header)
    for folder in RUN_IDS:
        cells = []
        for fam in families:
            vals = agg[folder][fam]
            n_div = diverged_count[folder][fam]
            if vals:
                cell = f"{np.mean(vals):5.2f}+/-{np.std(vals):4.2f}"
                if n_div:
                    cell += f"({n_div}d)"
            else:
                cell = "ALL DIVERGED"
            cells.append(f"{cell:>18}")
        lines.append(f"{folder:<45}" + "".join(cells))
    report = "\n".join(lines) + "\n"
    (OUT_DIR / "family_sweep_summary.txt").write_text(report)
    print(report)

    # Plot: grouped bars, x=family, one bar per model, y=mean rel_mean_pct (+/- std).
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(families))
    width = 0.8 / len(RUN_IDS)
    colors = plt.cm.tab10.colors
    for i, folder in enumerate(RUN_IDS):
        means = [np.mean(agg[folder][fam]) if agg[folder][fam] else 0.0 for fam in families]
        stds = [np.std(agg[folder][fam]) if agg[folder][fam] else 0.0 for fam in families]
        ax.bar(x + (i - (len(RUN_IDS) - 1) / 2) * width, means, width, yerr=stds,
               capsize=3, label=folder, color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylabel("relative mean |FD-NN| error (%)  [mean +/- std over held-out test trajectories]")
    ax.set_title("p17 assembly -- model comparison across dataset excitation families")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "family_sweep.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'family_sweep.png'}")


if __name__ == "__main__":
    main()
