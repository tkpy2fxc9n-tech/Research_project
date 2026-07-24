# Standalone correctness check for model.reshape_to_channels: builds a fake
# 126-value input where every number IS its own (lag, k, field) address, runs
# it through the exact reshape/permute code path used by ReseauConv.forward,
# and asserts every value lands at the channel/position matching its own
# address. Run this before trusting anything trained with model.py:
#   python test_reshape.py
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import reshape_to_channels

N_LAGS, N_POINTS, N_FIELDS = 4, 21, 1


def tag(lag: int, k_idx: int, field: int) -> float:
    # Unique, human-readable encoding: e.g. 1123 = lag 1, point index 12, field 3.
    return lag * 1000 + k_idx * 10 + field


def main():
    # Build X in the SAME nested order as make_feature_columns/build_window_torch:
    # for lag: for k_idx: for field: next column.
    B = 4  # a few rows, to also confirm the batch dimension isn't disturbed
    cols = []
    for lag in range(N_LAGS):
        for k_idx in range(N_POINTS):
            for field in range(N_FIELDS):
                cols.append(tag(lag, k_idx, field))
    X = torch.tensor([cols] * B, dtype=torch.float32)
    assert X.shape == (B, N_LAGS * N_POINTS * N_FIELDS)

    x = reshape_to_channels(X, N_LAGS, N_POINTS, N_FIELDS)
    assert x.shape == (B, N_LAGS * N_FIELDS, N_POINTS)

    errors = []
    for lag in range(N_LAGS):
        for field in range(N_FIELDS):
            channel = lag * N_FIELDS + field
            for k_idx in range(N_POINTS):
                expected = tag(lag, k_idx, field)
                got = x[0, channel, k_idx].item()
                if got != expected:
                    errors.append(
                        f"channel={channel} (lag={lag}, field={field}), point={k_idx}: "
                        f"expected {expected}, got {got}")

    if errors:
        print(f"FAILED -- {len(errors)} mismatches:")
        for e in errors[:20]:
            print(" ", e)
        sys.exit(1)

    print(f"OK -- all {N_LAGS * N_POINTS * N_FIELDS} values landed in the correct "
          f"(lag, field) channel and point position, across all {B} batch rows checked structurally.")


if __name__ == "__main__":
    main()
