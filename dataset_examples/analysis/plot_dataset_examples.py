#!/usr/bin/env python3
# Thesis Figure 2: representative trajectories from the "complex" dataset,
# shown as space-time displacement maps, one panel per distinct
# (boundary-condition, driving-function-family) combination. Cheap, direct
# h5py read of a handful of trajectories -- NOT the full 2000-trajectory
# load_hdf5_dataset()/compute_norm_stats() pipeline training uses (that
# needs a compute node, see scripts/run_analysis.job's own header); this
# only ever touches a few rows of the (2000, 501, 100) `u` dataset.
#
# Usage: python analysis/plot_dataset_examples.py
from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "data" / "beam_dataset_D.h5"
OUT_PATH = REPO_ROOT / "dataset_examples" / "runs" / "figures" / "dataset_examples.png"

N_PANELS = 6


def _decode(x):
    return x.decode() if isinstance(x, bytes) else x


def main():
    with h5py.File(DATASET_PATH, "r") as f:
        left_family = np.array([_decode(x) for x in f["left_family"][:]])
        right_family = np.array([_decode(x) for x in f["right_family"][:]])
        left_driven = f["left_driven"][:]
        right_driven = f["right_driven"][:]
        t = f["t"][:]
        x = f["x"][:]

        # One trajectory per distinct (left_family, right_family, both-driven
        # vs one-side-at-rest) combo actually present, picking the first
        # matching index -- real dataset rows, not synthetic examples.
        seen, chosen = set(), []
        for i in range(len(left_family)):
            key = (left_family[i], right_family[i], bool(left_driven[i]), bool(right_driven[i]))
            if key in seen:
                continue
            seen.add(key)
            chosen.append(i)
            if len(chosen) >= N_PANELS:
                break

        n = len(chosen)
        ncols = 3
        nrows = -(-n // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
        vmax = 0.0
        u_panels = []
        for idx in chosen:
            u = f["u"][idx]
            u_panels.append(u)
            vmax = max(vmax, float(np.abs(u).max()))

        for k, idx in enumerate(chosen):
            ax = axes[k // ncols][k % ncols]
            u = u_panels[k]
            im = ax.imshow(u, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                            extent=[x[0], x[-1], t[0], t[-1]])
            l_lab = f"L:{left_family[idx]}" + ("" if left_driven[idx] else " (rest)")
            r_lab = f"R:{right_family[idx]}" + ("" if right_driven[idx] else " (rest)")
            ax.set_title(f"{l_lab}  /  {r_lab}", fontsize=9)
            ax.set_xlabel("x")
            ax.set_ylabel("t")
        for k in range(n, nrows * ncols):
            axes[k // ncols][k % ncols].axis("off")

        fig.colorbar(im, ax=axes, shrink=0.7, label="displacement u(x,t)")
        fig.suptitle("Representative trajectories from the complex dataset", fontsize=13)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {OUT_PATH} ({n} panels, indices={chosen})")


if __name__ == "__main__":
    main()
