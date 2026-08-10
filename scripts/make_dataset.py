#!/usr/bin/env python3
# Generates one of the two canonical HDF5 datasets (data/beam_dataset_
# complex.h5, data/beam_dataset_simple.h5) and writes its sha256 sidecar,
# which cli.py's env.json reads. NEVER called automatically by a training
# run -- you run this yourself, once, before launching any experiment.
#
# Usage:
#   python scripts/make_dataset.py --profile simple  --n-trajectories 3000
#   python scripts/make_dataset.py --profile complex --n-trajectories 3000
#
# For a quick local check before committing to the full 3000-trajectory
# generation, pass a small --n-trajectories (e.g. 20-100) and a --output
# pointing somewhere other than data/ -- see the smoke-test instructions in
# the README.
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beamsurrogate.config import Config
from beamsurrogate.data.generate import generate_dataset, PROFILES

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description="Generate one of the canonical beamsurrogate HDF5 datasets.")
    p.add_argument("--profile", choices=sorted(PROFILES), required=True)
    p.add_argument("--n-trajectories", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=None,
                    help=f"Default: {DATA_DIR}/beam_dataset_<profile>.h5")
    return p.parse_args()


def main():
    args = parse_args()
    output_path = args.output or (DATA_DIR / f"beam_dataset_{args.profile}.h5")

    cfg = Config()   # only the physics/sampling-range fields matter here
    generate_dataset(cfg, args.profile, args.n_trajectories, output_path, seed=args.seed)

    print(f"Hashing {output_path} ...")
    digest = sha256_file(output_path)
    sidecar = output_path.with_suffix(output_path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {output_path.name}\n")
    print(f"sha256 {digest} -> {sidecar}")


if __name__ == "__main__":
    main()
