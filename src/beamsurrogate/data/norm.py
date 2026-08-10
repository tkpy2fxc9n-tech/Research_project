# Normalization helpers shared by every training regime.
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


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


def norm_stats_arrays(norm_stats: pd.DataFrame, INPUTS, OUTPUTS):
    mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
    sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)
    mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
    sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype(np.float32)
    return mu_in, sd_in, mu_out, sd_out
