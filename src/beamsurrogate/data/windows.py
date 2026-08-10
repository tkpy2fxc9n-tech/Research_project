# Feature windowing (stencil x lag -> flat feature vector). Shared by
# dataset generation, training, and rollout -- H4 (which of U/Ut/Uxx feed
# the network) is just the `input_fields` list passed through here, not a
# separate code path per feature set.
from __future__ import annotations

import numpy as np

FIELD_LABELS = {"U": "u", "Ut": "u_dot", "Uxx": "u_xx"}


def jlabel(k: int) -> str:
    return "j" if k == 0 else f"j{k:+d}"


def lag_label(lag: int) -> str:
    return "t" if lag == 0 else f"t-{lag}ndt"


def uxx_field(u: np.ndarray, cfg) -> np.ndarray:
    out = np.zeros(cfg.Ntot)
    i_left, i_right = cfg.i_left, cfg.i_right
    out[i_left:i_right + 1] = (u[i_left - 1:i_right] - 2 * u[i_left:i_right + 1] + u[i_left + 1:i_right + 2]) / cfg.dx ** 2
    return out


def field_value(field_name: str, get_u, m: int, cfg) -> np.ndarray:
    if field_name == "U":
        return get_u(m)
    if field_name == "Ut":
        return (get_u(m) - get_u(m - cfg.ndt)) / (cfg.ndt * cfg.dt)
    if field_name == "Uxx":
        return uxx_field(get_u(m), cfg)
    raise ValueError(f"Unknown input field: {field_name!r}")


def make_feature_columns(input_fields: list[str], cfg) -> list[str]:
    cols = []
    for lag in range(cfg.M_BACK):
        lab = lag_label(lag)
        for k in range(-cfg.SS, cfg.SS + 1):
            for f in input_fields:
                cols.append(f"{FIELD_LABELS[f]}({lab},{jlabel(k)})")
    return cols


def make_output_columns(cfg) -> list[str]:
    return [f"delta_u@{h}ndt" for h in range(1, cfg.N_FWD + 1)]


def build_window(m_list, get_u, input_fields: list[str], cfg) -> np.ndarray:
    # Column order must stay synchronized with make_feature_columns.
    nodes = cfg.nodes
    n_features = cfg.M_BACK * (2 * cfg.SS + 1) * len(input_fields)
    X = np.zeros((len(nodes), n_features), dtype=np.float32)
    col = 0
    for m in m_list:
        field_arrays = {f: field_value(f, get_u, m, cfg) for f in input_fields}
        for k in range(-cfg.SS, cfg.SS + 1):
            for f in input_fields:
                X[:, col] = field_arrays[f][nodes + k]
                col += 1
    return X
