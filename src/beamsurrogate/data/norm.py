# Normalization helpers shared by every training regime.
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def normalize_array(values: np.ndarray, cols, norm_stats: pd.DataFrame, *, inplace: bool = False) -> np.ndarray:
    mu = norm_stats.loc[cols, "mean"].values.astype(np.float32)
    sd = norm_stats.loc[cols, "std"].values.astype(np.float32)
    # copy=False: values is already float32 coming from load_hdf5_dataset's
    # combined block (data/split.py) -- astype's default copy=True would
    # otherwise duplicate it here for no reason.
    values = values.astype(np.float32, copy=False)
    if inplace:
        # Only safe when the caller guarantees `values` is a private,
        # disposable array (e.g. freshly boolean-filtered, not a view into
        # something still needed elsewhere) -- see make_dataloaders below.
        # Avoids allocating a second array on top of `values` just to hold
        # the normalized result, which matters at the full dataset's scale.
        values -= mu
        values /= sd
        return values
    return (values - mu) / sd


class FastTensorLoader:
    # Replaces DataLoader(TensorDataset(...)): PyTorch's default collate
    # builds each batch by looping over `batch_size` individual __getitem__
    # calls in plain Python -- single-threaded, no benefit from the
    # OMP_NUM_THREADS/torch intra-op parallelism that only speeds up the
    # matmuls -- a real per-batch tax at ~32k batches/epoch on the full
    # dataset. X/y are already whole in-memory tensors here, so one
    # fancy-index slice per batch replaces that Python loop entirely. Same
    # observable behavior as DataLoader(shuffle=True): reshuffles on every
    # fresh iter() (i.e. every epoch), last batch may be smaller.
    def __init__(self, X: torch.Tensor, y: torch.Tensor, batch_size: int):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.n = X.shape[0]

    def __iter__(self):
        perm = torch.randperm(self.n)
        for start in range(0, self.n, self.batch_size):
            idx = perm[start:start + self.batch_size]
            yield self.X[idx], self.y[idx]

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


def make_dataloaders(df: pd.DataFrame, INPUTS, OUTPUTS, norm_stats: pd.DataFrame, cfg):
    # Filters INPUTS and OUTPUTS separately (not into one shared block) and
    # normalizes in place: each filtered array is disposable and freed as
    # soon as the next one is built, instead of a shared filtered block
    # staying alive across both the X and y extractions (that overlap --
    # filtered block + normalized X + normalized y all resident together --
    # is what pushed the full dataset's loading past 64G).
    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()

    X_train = normalize_array(df[INPUTS].to_numpy(copy=False)[train_mask], INPUTS, norm_stats, inplace=True)
    y_train = normalize_array(df[OUTPUTS].to_numpy(copy=False)[train_mask], OUTPUTS, norm_stats, inplace=True)
    X_val = normalize_array(df[INPUTS].to_numpy(copy=False)[val_mask], INPUTS, norm_stats, inplace=True)
    y_val = normalize_array(df[OUTPUTS].to_numpy(copy=False)[val_mask], OUTPUTS, norm_stats, inplace=True)

    # from_numpy shares memory with X_train/y_train instead of copying them
    # again (torch.tensor always copies).
    train_loader = FastTensorLoader(torch.from_numpy(X_train), torch.from_numpy(y_train), cfg.BATCH_SIZE)
    return train_loader, X_val, y_val


def norm_stats_arrays(norm_stats: pd.DataFrame, INPUTS, OUTPUTS):
    mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
    sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)
    mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
    sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype(np.float32)
    return mu_in, sd_in, mu_out, sd_out
