# Plain one-step (teacher-forcing) training: predicts the N_FWD horizons
# from a real ground-truth M_BACK window, backpropagates the averaged error,
# steps -- no autoregressive rollout, no pushforward. Ported from
# Beam_surrogate_model/training/code/commun.py's `train_model`. Input-noise
# augmentation (cfg.NOISE_STD, phase 10) was already active here in the source
# project despite a stale comment claiming otherwise -- kept, not
# reintroduced.
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import TrainResult


def run(model, train_loader, X_val, y_val, cfg, model_path: Path, patience: int | None = None) -> TrainResult:
    # patience=None -> always runs the full cfg.N_EPOCHS. patience=N ->
    # stops as soon as N consecutive epochs pass without a new best val loss
    # (the best checkpoint, saved to model_path on every improvement, is
    # what gets reloaded at the end either way).
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    with torch.no_grad():
        model(torch.zeros(1, X_val.shape[1]))

    train_history, val_history = [], []
    best_val = float("inf")
    epochs_without_improvement = 0

    t0 = time.perf_counter()
    for epoch in range(1, cfg.N_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            if cfg.NOISE_STD > 0:
                X_in = X_batch + cfg.NOISE_STD * torch.randn_like(X_batch)
            else:
                X_in = X_batch

            prediction = model(X_in)
            data_loss = criterion(prediction, y_batch)
            data_loss.backward()
            optimizer.step()

            train_loss += data_loss.item()

        train_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            pred_val = model(torch.tensor(X_val)).numpy()
        val_loss = float(((pred_val - y_val) ** 2).mean())
        scheduler.step(val_loss)

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(f"Epoch {epoch:4d}/{cfg.N_EPOCHS}  --  data: {train_loss:.4f}  |  val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience is not None and epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}: val loss hasn't improved for "
                  f"{patience} epochs (best={best_val:.6f}).")
            break

    train_time_s = time.perf_counter() - t0

    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Best model reloaded -- minimum val: {best_val:.6f}")

    n_params = sum(p.numel() for p in model.parameters())
    return TrainResult(train_history, val_history, best_val, train_time_s, n_params)


def evaluate_one_step(model, df_test, INPUTS, OUTPUTS, norm_stats) -> tuple[dict, np.ndarray, np.ndarray]:
    # Returns the per-column {mse_norm, r2} dict alongside the physical
    # (denormalized) true/predicted arrays -- the latter two only used for
    # plots.plot_one_step_predictions, kept out of `metrics` (which gets
    # written verbatim to metrics.json).
    from ..data.norm import normalize_array

    X_new = normalize_array(df_test[INPUTS].values, INPUTS, norm_stats)
    y_true_n = normalize_array(df_test[OUTPUTS].values, OUTPUTS, norm_stats)
    y_true = df_test[OUTPUTS].values

    model.eval()
    with torch.no_grad():
        y_pred_n = model(torch.tensor(X_new)).numpy()

    mu_out = norm_stats.loc[OUTPUTS, "mean"].values
    sd_out = norm_stats.loc[OUTPUTS, "std"].values
    y_pred = y_pred_n * sd_out + mu_out

    metrics = {}
    for i, col in enumerate(OUTPUTS):
        mse_norm = ((y_pred_n[:, i] - y_true_n[:, i]) ** 2).mean()
        r2 = 1 - mse_norm / y_true_n[:, i].var()
        metrics[col] = {"mse_norm": float(mse_norm), "r2": float(r2)}
    return metrics, y_true, y_pred
