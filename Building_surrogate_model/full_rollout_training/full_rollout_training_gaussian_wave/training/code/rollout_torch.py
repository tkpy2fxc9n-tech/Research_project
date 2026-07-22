# Portage torch (différentiable, sans jamais repasser par numpy) de la
# physique de reconstruction déjà présente et validée dans
# Code_comparaison_des_inputs/commun.py (uxx_field, field_value, build_window,
# reconstruct). Aucune opération in-place : chaque étape reconstruit un
# nouveau tenseur, pour ne jamais perturber le chemin de calcul du gradient.
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR

sys.path.insert(0, str(COMMUN_DIR))
import commun as C


def uxx_field_torch(u: torch.Tensor, cfg: "C.Config") -> torch.Tensor:
    # u : (G, Ntot). Port direct de C.uxx_field.
    i_left, i_right = cfg.i_left, cfg.i_right
    G = u.shape[0]
    interior = (u[:, i_left - 1:i_right] - 2 * u[:, i_left:i_right + 1]
                + u[:, i_left + 1:i_right + 2]) / cfg.dx ** 2
    left_pad = torch.zeros(G, i_left, dtype=u.dtype)
    right_pad = torch.zeros(G, cfg.Ntot - (i_right + 1), dtype=u.dtype)
    return torch.cat([left_pad, interior, right_pad], dim=1)


def build_window_torch(history: list[torch.Tensor], input_fields: list[str], cfg: "C.Config") -> torch.Tensor:
    # history : liste de M_BACK+1 tenseurs (G, Ntot), du plus ancien
    # (history[0]) au plus récent (history[-1] = état courant n).
    # Retourne X de forme (G*Nx, n_features), même ordre de colonnes que
    # C.make_feature_columns (lag, puis k, puis champ).
    nodes = cfg.nodes
    G = history[-1].shape[0]
    Nx = len(nodes)

    cols = []
    for lag in range(cfg.M_BACK):
        u_m = history[-(lag + 1)]
        field_arrays = {}
        for f in input_fields:
            if f == "U":
                field_arrays[f] = u_m
            elif f == "Ut":
                u_prev = history[-(lag + 2)]
                field_arrays[f] = (u_m - u_prev) / (cfg.ndt * cfg.dt)
            elif f == "Uxx":
                field_arrays[f] = uxx_field_torch(u_m, cfg)
            else:
                raise ValueError(f"Champ d'entrée inconnu : {f!r}")
        for k in range(-cfg.SS, cfg.SS + 1):
            idx = nodes + k
            for f in input_fields:
                cols.append(field_arrays[f][:, idx])  # (G, Nx)

    X = torch.stack(cols, dim=-1)  # (G, Nx, n_features)
    return X.reshape(G * Nx, -1)


def reconstruct_torch(baseline: torch.Tensor, pred_norm: torch.Tensor,
                       A_list: list[float], omega_list: list[float], n_curr: int,
                       mu_out_t: torch.Tensor, sd_out_t: torch.Tensor,
                       biais_repos_t: torch.Tensor | None, cfg: "C.Config") -> tuple[list[torch.Tensor], list[int]]:
    # baseline : (G, Ntot), état AVANT ce hop -- fixe pour tous les h (ne pas
    # chaîner h=1 dans h=2, comme C.reconstruct).
    # pred_norm : (G*Nx, N_FWD), sortie brute du réseau pour ce hop.
    # Retourne (liste de N_FWD tenseurs (G, Ntot), liste des indices temporels s correspondants).
    nodes = cfg.nodes
    i_left, i_right, Ntot = cfg.i_left, cfg.i_right, cfg.Ntot
    G = baseline.shape[0]
    Nx = len(nodes)

    pred_norm = pred_norm.reshape(G, Nx, cfg.N_FWD)
    deltas = pred_norm * sd_out_t + mu_out_t
    if biais_repos_t is not None:
        deltas = deltas - biais_repos_t

    new_states, s_list = [], []
    for h in range(1, cfg.N_FWD + 1):
        s = n_curr + h * cfg.ndt
        t = s * cfg.dt
        interior_nodes = baseline[:, nodes] + deltas[:, :, h - 1]  # (G, Nx), valeurs physiques aux noeuds

        right_vals = torch.tensor([C.u_right_val(A, omega, t) for A, omega in zip(A_list, omega_list)],
                                   dtype=baseline.dtype)
        right_block = right_vals.unsqueeze(1).expand(G, Ntot - i_right)

        # encastrement gauche (indices [0, i_left] inclus, écrase donc aussi
        # la valeur physique calculée au noeud i_left -- comme C.reconstruct)
        left_block = torch.zeros(G, i_left + 1, dtype=baseline.dtype)
        u_full = torch.cat([left_block, interior_nodes[:, 1:], right_block], dim=1)

        if cfg.SMOOTH_ALPHA > 0:
            j0, j1 = i_left + 1, i_right
            lap = u_full[:, j0 - 1:j1 - 1] - 2 * u_full[:, j0:j1] + u_full[:, j0 + 1:j1 + 1]
            smoothed_middle = u_full[:, j0:j1] + cfg.SMOOTH_ALPHA * lap
            u_full = torch.cat([u_full[:, :j0], smoothed_middle, u_full[:, j1:]], dim=1)

        new_states.append(u_full)
        s_list.append(s)

    return new_states, s_list
