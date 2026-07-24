# Normalization, dataloaders, and the plain one-step (teacher forcing)
# training loop this project actually uses. No pushforward: every batch is
# real (ground-truth) M_BACK window -> two true delta_u horizons, weights
# updated immediately -- no rollout, no chained autoregressive hops during
# training.
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    historique_train: list
    historique_val: list
    meilleure_val: float
    train_time_s: float
    n_params: int


def train_model(modele, train_loader, X_val, y_val, cfg, model_path: Path,
                 patience: int | None = None) -> TrainResult:
    # patience=None (default) -> always runs the full cfg.N_EPOCHS, exactly
    # the old behavior (every existing caller). patience=N -> stops as soon
    # as N consecutive epochs pass without a new best val loss (the best
    # checkpoint, already saved to model_path on every improvement, is what
    # gets reloaded at the end either way).
    criterion = nn.MSELoss()
    optimiseur = torch.optim.Adam(modele.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiseur, mode="min", factor=0.5, patience=10)

    with torch.no_grad():
        modele(torch.zeros(1, X_val.shape[1]))

    historique_train, historique_val = [], []
    meilleure_val = float("inf")
    epochs_sans_amelioration = 0

    t0 = time.perf_counter()
    for epoch in range(1, cfg.N_EPOCHS + 1):
        modele.train()
        perte_train = 0.0

        for X_batch, y_batch in train_loader:
            optimiseur.zero_grad()

            if cfg.NOISE_STD > 0:
                X_in = X_batch + cfg.NOISE_STD * torch.randn_like(X_batch)
            else:
                X_in = X_batch

            prediction = modele(X_in)
            loss = criterion(prediction, y_batch)
            loss.backward()
            optimiseur.step()

            perte_train += loss.item()

        perte_train /= len(train_loader)

        modele.eval()
        with torch.no_grad():
            pred_val = modele(torch.tensor(X_val)).numpy()
        perte_val = ((pred_val - y_val) ** 2).mean()
        scheduler.step(perte_val)

        historique_train.append(perte_train)
        historique_val.append(perte_val)

        print(f"Epoch {epoch:4d}/{cfg.N_EPOCHS}  —  data: {perte_train:.4f}  |  val: {perte_val:.4f}")

        if perte_val < meilleure_val:
            meilleure_val = perte_val
            torch.save(modele.state_dict(), model_path)
            epochs_sans_amelioration = 0
        else:
            epochs_sans_amelioration += 1

        if patience is not None and epochs_sans_amelioration >= patience:
            print(f"Early stopping at epoch {epoch}: val loss hasn't improved for "
                  f"{patience} epochs (best={meilleure_val:.6f}).")
            break

    train_time_s = time.perf_counter() - t0

    modele.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Best model reloaded — minimum val: {meilleure_val:.6f}")

    n_params = sum(p.numel() for p in modele.parameters())
    return TrainResult(historique_train, historique_val, meilleure_val, train_time_s, n_params)


def plot_training_curve(result: TrainResult, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.historique_train, label="Data (train)")
    ax.plot(result.historique_val, label="Data (val)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Learning curve")
    ax.set_yscale("log"); ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "courbe_apprentissage.png", dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_teacher_forcing(modele, df_test: pd.DataFrame, INPUTS, OUTPUTS, norm_stats: pd.DataFrame, output_dir: Path) -> dict:
    X_new = normalize_array(df_test[INPUTS].values, INPUTS, norm_stats)
    y_true_n = normalize_array(df_test[OUTPUTS].values, OUTPUTS, norm_stats)
    y_true = df_test[OUTPUTS].values

    modele.eval()
    with torch.no_grad():
        y_pred_n = modele(torch.tensor(X_new)).numpy()

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
