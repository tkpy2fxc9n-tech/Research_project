#!/usr/bin/env python3
# H5 (phase 2): same 3 curves as plots.py's plot_training_curve
# (train/val/pushforward), but with the val rollout blow-up epochs dropped
# from the line so the y-axis isn't stretched to 1e18+ by a handful of
# unstable epochs, squashing the other two curves flat. Those blow-ups are
# real (autoregressive rollout diverging at specific epochs, see log), not a
# bug -- this is a *readability* view, not a claim that the original figure
# is wrong. Epoch curves aren't in metrics.json (only the final rollout's
# curves are), so this re-parses the run's own Slurm .log. Assumes the
# pushforward-regime log line format ("data: X | pushforward: Y | L2 rel
# error (val): Z") -- a bptt-regime log ("combined loss (train): X | L2 rel
# error (val): Y -- Zs") needs a different regex if reused there.
# Usage: python analysis/h5_training_curve_clean.py
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

RUN_ID = "p2_diag_training_curve_clean"
PHASE = 2
HYPOTHESIS = "H5"
SOURCE_RUN_ID = "p2_pushforward"
OUTLIER_FACTOR = 10  # a val point beyond OUTLIER_FACTOR x median(val) is dropped from the line

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s+--\s+data:\s+([\d.eE+-]+)\s+\|\s+pushforward:\s+([\d.eE+-]+)\s+\|\s+"
    r"L2 rel error \(val\):\s+([\d.eE+-]+)"
)


def main():
    env_path = REPO_ROOT / "runs" / SOURCE_RUN_ID / "env.json"
    if not env_path.exists():
        print(f"ERROR: {env_path} not found -- run {SOURCE_RUN_ID} first.", file=sys.stderr)
        sys.exit(1)
    slurm_job_id = json.loads(env_path.read_text())["slurm_job_id"]
    log_path = REPO_ROOT / "logs" / f"beamsurrogate_{slurm_job_id}.log"
    if not log_path.exists():
        print(f"ERROR: {log_path} not found (job {slurm_job_id} log missing/purged).", file=sys.stderr)
        sys.exit(1)

    epochs, data, pushforward, val = [], [], [], []
    for line in log_path.read_text().splitlines():
        m = EPOCH_RE.search(line)
        if m:
            epochs.append(int(m.group(1)))
            data.append(float(m.group(2)))
            pushforward.append(float(m.group(3)))
            val.append(float(m.group(4)))
    if not epochs:
        print(f"ERROR: no epoch lines matched in {log_path} -- log format may differ "
              f"(e.g. a bptt-regime run).", file=sys.stderr)
        sys.exit(1)

    val_median = statistics.median(val)
    threshold = OUTLIER_FACTOR * val_median
    val_clean_epochs = [e for e, v in zip(epochs, val) if v <= threshold]
    val_clean = [v for v in val if v <= threshold]
    n_dropped = len(val) - len(val_clean)

    run_dir = REPO_ROOT / "runs" / RUN_ID
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, data, label="train")
    ax.plot(val_clean_epochs, val_clean, label="val")
    ax.plot(epochs, pushforward, "--", label="pushforward (unweighted)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"Learning curve ({SOURCE_RUN_ID}) -- {n_dropped} val outlier epoch(s) removed")
    ax.set_yscale("log"); ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / "training_curve_clean.png", dpi=150, bbox_inches="tight")
    plt.close()

    dropped = [(e, v) for e, v in zip(epochs, val) if v > threshold]
    summary = (
        f"{RUN_ID} (phase {PHASE}, {HYPOTHESIS}) -- clean training curve (source: {SOURCE_RUN_ID})\n"
        f"val median={val_median:.4g}, outlier threshold={threshold:.4g} ({OUTLIER_FACTOR}x median)\n"
        f"Dropped {n_dropped} epoch(s) from the val line: "
        + ", ".join(f"epoch {e} (val={v:.4g})" for e, v in dropped) + "\n"
    )
    (run_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"Done -- outputs in {run_dir}")


if __name__ == "__main__":
    main()
