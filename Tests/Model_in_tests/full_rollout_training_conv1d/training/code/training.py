# Normalization and dataloaders shared by the composite training loop
# (train.py) -- make_dataloaders' train_loader feeds the "data" term there.
# Also holds TrainResult (shared result type) and the post-training
# teacher-forcing diagnostic, both scheme-agnostic.
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Threads fixed to 1: training/rollout times are compared across
# the 4 methods, a variable thread count would make them non-comparable.
torch.set_num_threads(1)


def normalize_array(values: np.ndarray, cols, norm_stats: pd.DataFrame) -> np.ndarray:
    mu = norm_stats.loc[cols, "mean"].values.astype(np.float32)
    sd = norm_stats.loc[cols, "std"].values.astype(np.float32)
    return (values.astype(np.float32) - mu) / sd


def make_dataloaders(df: pd.DataFrame, INPUTS, OUTPUTS, norm_stats: pd.DataFrame, cfg):
    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"

    X_train = normalize_array(df.loc[train_mask, INPUTS].values, INPUTS, norm_stats)
    y_train = normalize_array(df.loc[train_mask, OUTPUTS].values, OUTPUTS, norm_stats)
    X_val = normalize_array(df.loc[val_mask, INPUTS].values, INPUTS, norm_stats)
    y_val = normalize_array(df.loc[val_mask, OUTPUTS].values, OUTPUTS, norm_stats)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=cfg.BATCH_SIZE, shuffle=True,
    )
    return train_loader, X_val, y_val


@dataclass
class TrainResult:
    train_history: list
    val_history: list
    best_val: float
    train_time_s: float
    n_params: int
    # Unweighted per-epoch averages of each composite-loss term -- lets
    # LAMBDA_PHYSICS/LAMBDA_DATA/LAMBDA_ROLLOUT be tuned by comparing
    # magnitudes directly (see train.plot_rollout_training_curve).
    data_loss_history: list = field(default_factory=list)
    physics_loss_history: list = field(default_factory=list)
    rollout_loss_history: list = field(default_factory=list)


def evaluate_teacher_forcing(model, df_test: pd.DataFrame, INPUTS, OUTPUTS, norm_stats: pd.DataFrame, output_dir: Path) -> dict:
    X_new = normalize_array(df_test[INPUTS].values, INPUTS, norm_stats)
    y_true_n = normalize_array(df_test[OUTPUTS].values, OUTPUTS, norm_stats)
    y_true = df_test[OUTPUTS].values

    model.eval()
    with torch.no_grad():
        y_pred_n = model(torch.tensor(X_new)).numpy()

    mu_out = norm_stats.loc[OUTPUTS, "mean"].values
    sd_out = norm_stats.loc[OUTPUTS, "std"].values
    y_pred = y_pred_n * sd_out + mu_out

    fig, axes = plt.subplots(1, len(OUTPUTS), figsize=(6*len(OUTPUTS), 6), squeeze=False)
    axes = axes.flatten()
    metrics = {}
    for i, (ax, col) in enumerate(zip(axes, OUTPUTS)):
        y_r, y_p = y_true[:, i], y_pred[:, i]
        ax.scatter(y_r, y_p, alpha=0.4, s=8)
        lim = max(abs(y_r).max(), abs(y_p).max())
        ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="perfect prediction")
        ax.set_xlabel(f"{col} real (physical)")
        ax.set_ylabel(f"{col} predicted (physical)")

        mse_norm = ((y_pred_n[:, i] - y_true_n[:, i]) ** 2).mean()
        r2 = 1 - mse_norm / y_true_n[:, i].var()

        ax.set_title(f"{col}\nMSE (norm)={mse_norm:.2e}  |  R²={r2:.3f}")
        ax.legend(); ax.grid(True)
        metrics[col] = {"mse_norm": float(mse_norm), "r2": float(r2)}

    fig.suptitle("Test over the full test split of the dataset", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "test_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    return metrics
