#!/usr/bin/env python3
"""
11a -- grille grossiere: builds a coarse-grid training dataset by
SUBSAMPLING an existing fine-grid HDF5 dataset (default: beam_dataset_
simple.h5) in both space and time, rather than re-running FD at native
coarse resolution. This is deliberate, not a shortcut: r2/r4's whole point
is "targets = fine solution subsampled" (see the 11a planning table) --
the coarse-grid network should learn to match the TRUE fine solution
restricted to a coarser stencil, not inherit whatever extra truncation
error a fresh coarse FD run would introduce on top. The mandatory
comparison against a solver run natively AT the coarse grid (not this
subsampled data) happens separately, at eval time, via
run_fd_simulation_general with the coarse cfg -- this script only builds
the training-target dataset.

Time subsampling is exact (every r-th of the 501 stored steps; 500 is
divisible by both 2 and 4, so both endpoints land exactly on t=0/t=t_end).
Space subsampling can't be exact for r=2/r=4 (Nx-1=99 isn't divisible by
either), so it uses evenly-spaced-as-possible indices that always include
both boundary points (np.linspace(0, Nx-1, Nx//r).round()) -- dropping the
actual right boundary (the driven end, for the simple/medium profiles)
would silently break the boundary condition. Resulting dx ends up ~2.02x/
~4.125x the fine dx, not exactly 2x/4x -- see the printed ratios.

left_bc_value/right_bc_value are subsampled the same way as u's time axis
(they're the per-step driving values, same t grid) -- NOT re-evaluated from
the stored family/params, so what data/split.py's "table" BC reconstruction
sees is consistent with u by construction. left_label/right_label/
left_driven/right_driven/left_family/right_family/split/initial_state are
per-trajectory metadata, unaffected by grid resolution, copied through
unchanged (same split assignment as the fine dataset).

Usage:
    python scripts/make_coarse_dataset.py --r 2
    python scripts/make_coarse_dataset.py --r 4
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description="Subsample a fine HDF5 dataset into a coarse-grid one.")
    p.add_argument("--source", type=Path, default=DATA_DIR / "beam_dataset_simple.h5")
    p.add_argument("--r", type=int, required=True, choices=(2, 4))
    p.add_argument("--output", type=Path, default=None,
                    help="Default: data/beam_dataset_simple_coarse_r<r>.h5")
    return p.parse_args()


def main():
    args = parse_args()
    output_path = args.output or (DATA_DIR / f"beam_dataset_simple_coarse_r{args.r}.h5")
    r = args.r

    with h5py.File(args.source, "r") as fin:
        Nt_fine = fin.attrs["Nt"]
        Nx_fine = fin.attrs["Nx"]
        assert Nt_fine % r == 0, f"Nt={Nt_fine} not divisible by r={r}"
        Nt_coarse = Nt_fine // r
        Nx_coarse = Nx_fine // r

        t_idx = np.arange(0, Nt_fine + 1, r)   # exact: includes t=0 and t=t_end
        assert len(t_idx) == Nt_coarse + 1
        x_idx = np.round(np.linspace(0, Nx_fine - 1, Nx_coarse)).astype(int)
        assert len(x_idx) == len(set(x_idx)) == Nx_coarse, "space subsampling produced duplicate indices"
        assert x_idx[0] == 0 and x_idx[-1] == Nx_fine - 1, "space subsampling dropped a boundary point"

        dx_fine = fin.attrs["L"] / (Nx_fine - 1)
        dx_coarse = fin.attrs["L"] / (Nx_coarse - 1)
        dt_fine = fin.attrs["t_end"] / Nt_fine
        dt_coarse = fin.attrs["t_end"] / Nt_coarse
        print(f"r={r}: Nx {Nx_fine}->{Nx_coarse} (dx x{dx_coarse/dx_fine:.3f}), "
              f"Nt {Nt_fine}->{Nt_coarse} (dt x{dt_coarse/dt_fine:.3f})")

        n_total = fin["u"].shape[0]
        print(f"Subsampling {n_total} trajectories from {args.source} ...")
        u_coarse = fin["u"][:, t_idx, :][:, :, x_idx].astype(np.float32)
        left_bc_coarse = fin["left_bc_value"][:, t_idx].astype(np.float32)
        right_bc_coarse = fin["right_bc_value"][:, t_idx].astype(np.float32)

        passthrough = {k: fin[k][:] for k in
                       ("left_label", "right_label", "left_driven", "right_driven",
                        "left_family", "right_family", "split", "initial_state")}
        E, rho, L, t_end, seed = (fin.attrs[k] for k in ("E", "rho", "L", "t_end", "seed"))
        profile = fin.attrs["profile"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_path} ...")
    with h5py.File(output_path, "w") as fout:
        fout.attrs.update(E=E, rho=rho, L=L, Nx=Nx_coarse, Nt=Nt_coarse, t_end=t_end,
                           dt=dt_coarse, dx=dx_coarse, CFL=dt_coarse / dx_coarse * (E / rho) ** 0.5,
                           seed=seed, profile=f"{profile}_coarse_r{r}",
                           coarsened_from=str(args.source), coarsen_r=r,
                           created=datetime.now().isoformat(timespec="seconds"))
        fout.create_dataset("x", data=np.linspace(0, L, Nx_coarse, dtype=np.float32))
        fout.create_dataset("t", data=(np.arange(Nt_coarse + 1) * dt_coarse).astype(np.float32))
        fout.create_dataset("u", data=u_coarse)
        fout.create_dataset("left_bc_value", data=left_bc_coarse)
        fout.create_dataset("right_bc_value", data=right_bc_coarse)
        for k, v in passthrough.items():
            fout.create_dataset(k, data=v)

    print(f"Saved {n_total} trajectories to {output_path}")
    print(f"Hashing {output_path} ...")
    digest = sha256_file(output_path)
    sidecar = output_path.with_suffix(output_path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {output_path.name}\n")
    print(f"sha256 {digest} -> {sidecar}")


if __name__ == "__main__":
    main()
