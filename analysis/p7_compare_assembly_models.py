#!/usr/bin/env python3
# Ranks every trained model under runs/p7/models/<run_id>/ by how well it
# reproduces the FD reference on the 94-rod p17 rectangular assembly (see
# p7_structure_several_rods.py for the test itself). Reuses each model's
# cached _frames.npz rather than re-running the rollout -- run
# p7_structure_several_rods.py first (or just delete a stale cache) if a
# model's cache is missing or outdated.
#
# Usage: python analysis/p7_compare_assembly_models.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))
import p7_structure_several_rods as m  # noqa: E402

MODELS_DIR = REPO_ROOT / "runs" / "p7" / "models"
OUT_DIR = REPO_ROOT / "runs" / "p7" / "analysis"

# Every folder under runs/p7/models/ is named by its own run_id (training and
# assembly-test artifacts live together there -- see p7_structure_several_rods.py).
# The models below are the current, non-archived p7 shortlist; everything
# else (duplicate/superseded/diverged-worse models, and p7_simple's
# different 4-rod/5-pulse-shape test) has been moved to runs/Archives_runs/.
RUN_IDS = [
    "p7_dataset_medium_pinn_0.01",
    "p7_dataset_medium",
    "p7_dataset_medium_arch_256_128_32_pinn_0.01",
    "p7_dataset_medium_bidir",
    "p7_dataset_medium_ut",
]


def load_cached_metrics(run_id: str) -> dict:
    out_dir = MODELS_DIR / run_id
    cache_path = out_dir / f"{run_id}_frames.npz"
    if not cache_path.exists():
        return {"source_run_id": run_id, "error": f"no cache at {cache_path}"}
    cached = np.load(cache_path)
    fd_steps, fd_U = cached["fd_steps"], cached["fd_U"]
    nn_steps, nn_U = cached["nn_steps"], cached["nn_U"]
    fd_time_s = float(cached["fd_time_s"]) if "fd_time_s" in cached else None
    nn_time_s = float(cached["nn_time_s"]) if "nn_time_s" in cached else None
    cfg, _ = m.load_source(run_id)
    return m.compute_metrics(run_id, fd_steps, fd_U, nn_steps, nn_U, fd_time_s, nn_time_s, dt=cfg.dt)


def main():
    rows = [load_cached_metrics(run_id) for run_id in RUN_IDS]

    ok_rows = [r for r in rows if "error" not in r and not r.get("diverged_nan")]
    bad_rows = [r for r in rows if "error" in r or r.get("diverged_nan")]

    # Best-accuracy first: relative mean error over the whole lattice is the
    # single best proxy for "matches the FD reference" (max error is noisier,
    # dominated by whichever single point/frame happens to disagree most).
    ok_rows.sort(key=lambda r: r["rel_mean_pct"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "p17 rectangular assembly (94-rod lattice) -- model comparison",
        "Ranked by relative mean |FD-NN| error over the whole lattice (lower is better).",
        "",
        f"{'rank':<5}{'model':<45}{'rel_mean%':>10}{'rel_max%':>10}{'ratio NN/FD':>13}{'NN time (s)':>13}{'NN/FD speed':>13}",
    ]
    for i, r in enumerate(ok_rows, 1):
        speed_ratio = r["nn_time_s"] / r["fd_time_s"] if r["fd_time_s"] else float("nan")
        lines.append(
            f"{i:<5}{r['source_run_id']:<45}{r['rel_mean_pct']:>10.2f}{r['rel_max_pct']:>10.2f}"
            f"{r['ratio_nn_fd']:>13.3f}{r['nn_time_s']:>13.2f}{speed_ratio:>13.2f}x"
        )
    if bad_rows:
        lines += ["", "EXCLUDED (diverged or missing cache):"]
        for r in bad_rows:
            reason = r.get("error") or "diverged (NaN or outside sanity band)"
            lines.append(f"  {r['source_run_id']:<45} {reason}")

    if ok_rows:
        best = ok_rows[0]
        fastest = min(ok_rows, key=lambda r: r["nn_time_s"])
        lines += [
            "",
            f"Best accuracy : {best['source_run_id']} (rel_mean={best['rel_mean_pct']:.2f}%)",
            f"Fastest NN    : {fastest['source_run_id']} (nn_time={fastest['nn_time_s']:.2f}s)",
        ]

    report = "\n".join(lines) + "\n"
    (OUT_DIR / "comparison_summary.txt").write_text(report)
    print(report)
    print(f"Saved {OUT_DIR / 'comparison_summary.txt'}")


if __name__ == "__main__":
    main()
