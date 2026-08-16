#!/usr/bin/env python3
"""
13 -- p13_cost_scaling: "the same model applied to domains of
different sizes" -- N = Nx (spatial points along the SAME beam), sweeping
200/1000/5000/20000 (vs the campaign's own Nx=100). Nt/t_end/SS/M_BACK/
N_FWD/ndt/HIDDEN_SIZES are all held fixed -- the MLP is a per-point local
model (its input only depends on SS/M_BACK/features, never on Nx), so it
can be applied at any Nx without retraining.

No trained checkpoint needed: FLOPs and wall-clock cost depend only on
array SHAPES and the architecture, never on the actual weight VALUES, so
this uses a freshly-initialized (untrained) model of the settled
architecture (see the "etape 10" assumption noted throughout this
campaign's other configs) -- this script measures cost, not accuracy.
norm_stats are dummy (mean=0/std=1): they only shift numbers, never change
how much compute happens.

Reuses evaluate/rollout.py's own benchmark_inference (chrono() timing +
FD_FLOPS_PER_POINT_STEP-based FLOPs) unmodified -- this script just calls
it in a loop over Nx instead of once.

Expected shape (per the 11a/13 planning table): FLOP ratio (nn/fd) roughly
constant with N (both scale ~linearly with Nx); wall-clock ratio rises then
plateaus -- the gap between the two ratios is Python/dispatch overhead, not
an algorithmic effect.

Usage: python scripts/cost_scaling_benchmark.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from beamsurrogate.config import Config  # noqa: E402
from beamsurrogate.registry import MODELS  # noqa: E402
from beamsurrogate.data.windows import make_feature_columns, make_output_columns  # noqa: E402
from beamsurrogate.physics.solver import run_fd_simulation_general  # noqa: E402
from beamsurrogate.evaluate.rollout import RolloutResult, benchmark_inference  # noqa: E402

OUT_DIR = REPO_ROOT / "runs" / "p13_cost_scaling"
N_VALUES = [200, 1000, 5000, 20000]

# "etape 10" settled architecture/hyperparameters -- see
# p11_coarse_r2.yaml's note: adjust if it settled on something else.
BASE_KWARGS = dict(regime="pushforward", M_BACK=3, N_FWD=2, features=["U"],
                    LAMBDA_PHYSICS=0, HIDDEN_SIZES=(512, 256, 64))


def main():
    results = []
    for N in N_VALUES:
        # Nt scaled with Nx to keep dt/dx (hence CFL) at the baseline's own
        # ratio (0.99) -- fixing Nt=500 while Nx grew made dx shrink and CFL
        # blow past 1 (up to 141 at N=20000), so U_reel was NaN/Inf there.
        Nt = round(5.0 * (N - 1) / 0.99)
        cfg = Config(Nx=N, Nt=Nt, **BASE_KWARGS)
        INPUTS = make_feature_columns(cfg.features, cfg)
        OUTPUTS = make_output_columns(cfg)

        model = MODELS[cfg.model](len(INPUTS), len(OUTPUTS), cfg)
        model.eval()
        norm_stats = pd.DataFrame({"mean": 0.0, "std": 1.0}, index=INPUTS + OUTPUTS)

        bc_left = ("dirichlet", "rest", {})
        bc_right = ("dirichlet", "gaussian", {"A": cfg.AMP_MAX, "sigma": 0.3})
        U_reel = run_fd_simulation_general(bc_left, bc_right, cfg)
        rollout = RolloutResult(U=None, U_reel=U_reel, left_bc=bc_left, right_bc=bc_right)

        print(f"N={N}: CFL={cfg.CFL:.4f}, benchmarking...")
        bench = benchmark_inference(model, {}, cfg.features, norm_stats, INPUTS, OUTPUTS, rollout, cfg)
        flop_ratio = bench.nn_flops / bench.fd_flops
        time_ratio = bench.nn_time_med / bench.fd_time_med
        print(f"  fd_time={bench.fd_time_med:.4e}s  nn_time={bench.nn_time_med:.4e}s  "
              f"flop_ratio={flop_ratio:.2f}  time_ratio={time_ratio:.2f}")
        results.append(dict(N=N, fd_time_med=bench.fd_time_med, nn_time_med=bench.nn_time_med,
                             fd_flops=bench.fd_flops, nn_flops=bench.nn_flops,
                             flop_ratio=flop_ratio, time_ratio=time_ratio))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "cost_scaling.csv", index=False)
    print(df.to_string(index=False))

    # 6 separate figures (one series each) -- a combined plot mixed a
    # ~12800x-scale series with a ~100x-scale one, making the smaller series
    # unreadable. Same validated categorical slot (blue) throughout: a lone
    # series needs no legend, the title names it (dataviz skill).
    COLOR = "#2a78d6"

    def _single_plot(y_col, ylabel, title, filename, logy=True):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(df["N"], df[y_col], "o-", color=COLOR, lw=2, ms=6)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("N (Nx)"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(True, which="both", alpha=0.4)
        fig.savefig(OUT_DIR / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    _single_plot("flop_ratio", "FLOP ratio (NN / FD)",
                 "Cost scaling -- FLOP ratio (NN / FD)", "flop_ratio_vs_N.png", logy=False)
    _single_plot("time_ratio", "wall-clock ratio (NN / FD)",
                 "Cost scaling -- wall-clock ratio (NN / FD)", "time_ratio_vs_N.png", logy=False)
    _single_plot("nn_flops", "FLOPs (network)",
                 "Cost scaling -- network FLOPs", "nn_flops_vs_N.png")
    _single_plot("fd_flops", "FLOPs (solver)",
                 "Cost scaling -- solver FLOPs", "fd_flops_vs_N.png")
    _single_plot("nn_time_med", "wall-clock time (s, network)",
                 "Cost scaling -- network wall-clock time", "nn_time_vs_N.png")
    _single_plot("fd_time_med", "wall-clock time (s, solver)",
                 "Cost scaling -- solver wall-clock time", "fd_time_vs_N.png")
    print(f"\nSaved {OUT_DIR / 'cost_scaling.csv'} and 6 figures "
          f"(flop_ratio/time_ratio/nn_flops/fd_flops/nn_time/fd_time _vs_N.png)")


if __name__ == "__main__":
    main()
