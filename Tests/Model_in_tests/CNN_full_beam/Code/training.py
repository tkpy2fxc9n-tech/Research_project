# Normalization helpers, the TrainResult record shared with train.py, and
# the one-step (teacher-forcing) accuracy check used as a post-training
# sanity metric on the test split. The actual training this project uses is
# the differentiable full-rollout TBPTT loop in train.py -- the plain
# per-batch teacher-forcing loop the reference project trained with is NOT
# ported here (train.py is the one real training path now).
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Threads fixed to 1: training/rollout times are compared across projects,
# a variable thread count would make them non-comparable.
torch.set_num_threads(1)


def norm_values(norm_stats: pd.DataFrame, cols) -> tuple[np.ndarray, np.ndarray]:
    mu = norm_stats.loc[cols, "mean"].values.astype(np.float32)
    sd = norm_stats.loc[cols, "std"].values.astype(np.float32)
    return mu, sd


def normalize_array(values: np.ndarray, cols, norm_stats: pd.DataFrame) -> np.ndarray:
    # values: (N, C, Nx) -- one scale per channel, shared across all beam
    # positions (broadcast over the batch and spatial axes).
    mu, sd = norm_values(norm_stats, cols)
    return (values.astype(np.float32) - mu[None, :, None]) / sd[None, :, None]


@dataclass
class TrainResult:
    historique_train: list
    historique_val: list
    meilleure_val: float
    train_time_s: float
    n_params: int


def evaluate_teacher_forcing(modele, X_test: np.ndarray, Y_test: np.ndarray,
                              INPUT_CHANNELS, OUTPUT_CHANNELS, norm_stats: pd.DataFrame,
                              output_dir: Path) -> dict:
    # One-step accuracy on the test split: each sample is a REAL
    # (ground-truth) whole-beam history -> the true future deltas, no
    # autoregressive chaining. Not what trains the model (see train.py) --
    # just a simple control metric alongside the rollout errors.
    X_norm = normalize_array(X_test, INPUT_CHANNELS, norm_stats)
    y_true_n = normalize_array(Y_test, OUTPUT_CHANNELS, norm_stats)

    modele.eval()
    with torch.no_grad():
        y_pred_n = modele(torch.tensor(X_norm)).numpy()  # (N, N_FWD, Nx)

    mu_out, sd_out = norm_values(norm_stats, OUTPUT_CHANNELS)
    y_pred = y_pred_n * sd_out[None, :, None] + mu_out[None, :, None]

    n_out = len(OUTPUT_CHANNELS)
    fig, axes = plt.subplots(1, n_out, figsize=(6*n_out, 6), squeeze=False)
    axes = axes.flatten()
    metrics = {}
    for i, (ax, col) in enumerate(zip(axes, OUTPUT_CHANNELS)):
        y_r, y_p = Y_test[:, i, :].ravel(), y_pred[:, i, :].ravel()
        ax.scatter(y_r, y_p, alpha=0.4, s=8)
        lim = max(abs(y_r).max(), abs(y_p).max())
        ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="perfect prediction")
        ax.set_xlabel(f"{col} real (physical)")
        ax.set_ylabel(f"{col} predicted (physical)")

        mse_norm = ((y_pred_n[:, i, :] - y_true_n[:, i, :]) ** 2).mean()
        r2 = 1 - mse_norm / y_true_n[:, i, :].var()

        ax.set_title(f"{col}\nMSE (norm)={mse_norm:.2e}  |  R²={r2:.3f}")
        ax.legend(); ax.grid(True)
        metrics[col] = {"mse_norm": float(mse_norm), "r2": float(r2)}

    fig.suptitle("One-step accuracy over the test simulations", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "test_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    return metrics
