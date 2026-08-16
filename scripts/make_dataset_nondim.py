#!/usr/bin/env python3
# Generates data/beam_dataset_simple_nondim.h5 -- the "simple" profile
# (same bc_options/rest_shares/family_shares/initial_state_shares, same
# SEED=42) regenerated with E=rho=L=1 and per-sample amplitude
# normalization, for configs/runs/p14_nondim.yaml.
#
# Deliberately a standalone script, NOT a --profile option on
# scripts/make_dataset.py: that script's `cfg = Config()` line is shared by
# every dataset regeneration, and this run needs Config fields (E, rho, L,
# t_end, SIGMA_MIN/MAX, OMEGA_MIN/MAX) that no other dataset should ever see
# touched.
#
# The physical -> nondimensional field values below come from applying
# t_nd = t*c/L (c = sqrt(E/rho)) to the CURRENT "simple" baseline's own
# Config defaults (E=1, rho=2, L=1, t_end=5, SIGMA_MIN/MAX=0.05/0.5,
# OMEGA_MIN/MAX=0.05/0.5) -- so this dataset represents the EXACT SAME
# discretized problem (same CFL, same Nx/Nt, same relative pulse
# widths/timing, same random draws given the same seed), just re-expressed
# in canonical nondimensional units. That's what makes "identical E_short/
# t_div vs the baseline" a meaningful bug detector for
# configs/runs/p14_nondim.yaml: any real gap means a dimensional
# constant leaked somewhere in the pipeline outside physics/scaling.py.
#
# Usage: python scripts/make_dataset_nondim.py
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beamsurrogate.config import Config
from beamsurrogate.data.generate import generate_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# c/L of the "simple" baseline (E=1, rho=2, L=1): sqrt(1/2) = 0.7071067811865476
_C_OVER_L = 0.7071067811865476


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    output_path = DATA_DIR / "beam_dataset_simple_nondim.h5"

    cfg = Config(
        E=1.0, rho=1.0, L=1.0,
        t_end=5.0 * _C_OVER_L,                 # 3.5355339059327378
        SIGMA_MIN=0.05 * _C_OVER_L,             # 0.03535533905932738
        SIGMA_MAX=0.5 * _C_OVER_L,              # 0.3535533905932738
        OMEGA_MIN=0.05 / _C_OVER_L,             # 0.07071067811865477
        OMEGA_MAX=0.5 / _C_OVER_L,              # 0.7071067811865477
        # AMP_MIN/AMP_MAX untouched (0.0005/0.005): normalize_per_sample
        # divides them out per trajectory regardless of the raw draw range.
    )
    print(f"CFL check: {cfg.CFL:.10f} (baseline's own CFL is 0.7000357134)")

    generate_dataset(cfg, "simple", n_trajectories=2000, output_path=output_path,
                      seed=cfg.SEED, normalize_per_sample=True)

    print(f"Hashing {output_path} ...")
    digest = sha256_file(output_path)
    sidecar = output_path.with_suffix(output_path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {output_path.name}\n")
    print(f"sha256 {digest} -> {sidecar}")


if __name__ == "__main__":
    main()
