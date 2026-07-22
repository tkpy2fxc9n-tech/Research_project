# Boucle d'entraînement "full rollout" (TBPTT) : pour chaque groupe de
# simulations, déroule la trajectoire complète (82 hops) sans jamais
# repartir de la vérité terrain, avec une correction des poids toutes les
# `tbptt_hops` hops (le fil du gradient est coupé à ce moment-là, mais pas
# l'état -- le rollout reste continu et autonome de bout en bout).
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR
from rollout_torch import build_window_torch, reconstruct_torch
from wave_forcing import autoregressive_rollout_multi

sys.path.insert(0, str(COMMUN_DIR))
import commun as C


def make_epoch_groups(pairs: list[tuple], group_size: int, rng: np.random.Generator) -> list[list[tuple]]:
    order = rng.permutation(len(pairs))
    shuffled = [pairs[i] for i in order]
    return [shuffled[i:i + group_size] for i in range(0, len(shuffled), group_size)]


def rollout_group_tbptt(modele, group_pairs, FIELDS, input_fields,
                         mu_in_t, sd_in_t, mu_out_t, sd_out_t, biais_repos_t,
                         criterion, optimiseur, cfg: "C.Config", tbptt_hops: int) -> tuple[float, int]:
    # Rollout complet (82 hops) pour un groupe de simulations, SANS jamais
    # repartir de la vérité terrain -- mais avec une correction des poids
    # toutes les `tbptt_hops` hops plutôt qu'une seule à la toute fin.
    # Entre deux corrections, l'écart entre état prédit et état vrai peut
    # devenir énorme (cf. diagnostic : cible normalisée qui explose dès le
    # 3e-4e hop), ce qui noie le signal utile si on attend les 82 hops pour
    # corriger. Ici, dès que `tbptt_hops` hops sont passés, on corrige puis
    # on détache l'état (le fil du gradient est coupé, mais le rollout
    # continue sur l'état PRÉDIT, jamais remis à la vérité terrain).
    nodes = cfg.nodes
    G, Nx = len(group_pairs), len(nodes)
    wave_type_list = [wt for wt, _, _ in group_pairs]
    A_list = [A for _, A, _ in group_pairs]
    omega_list = [omega for _, _, omega in group_pairs]
    history_needed = cfg.M_BACK * cfg.ndt

    history = []
    for lag in range(cfg.M_BACK, -1, -1):
        m = history_needed - lag * cfg.ndt
        arr = np.stack([FIELDS[pair][m] for pair in group_pairs], axis=0)
        history.append(torch.tensor(arr, dtype=torch.float32))

    hops = list(range(history_needed, cfg.Nt - cfg.N_FWD * cfg.ndt + 1, cfg.N_FWD * cfg.ndt))
    total_loss_log, n_updates = 0.0, 0
    segment_loss, segment_hops = torch.zeros(()), 0

    for i, n in enumerate(hops):
        X = (build_window_torch(history, input_fields, cfg) - mu_in_t) / sd_in_t
        pred_norm = modele(X)  # (G*Nx, N_FWD)

        baseline = history[-1]
        new_states, s_list = reconstruct_torch(baseline, pred_norm, wave_type_list, A_list, omega_list, n,
                                                 mu_out_t, sd_out_t, biais_repos_t, cfg)

        baseline_nodes = baseline[:, nodes]
        target_list = [
            torch.tensor(np.stack([FIELDS[pair][s][nodes] for pair in group_pairs], axis=0), dtype=torch.float32)
            - baseline_nodes
            for s in s_list
        ]
        target = torch.stack(target_list, dim=-1)  # (G, Nx, N_FWD)
        target_norm = ((target - mu_out_t) / sd_out_t).reshape(G * Nx, cfg.N_FWD)

        hop_loss = criterion(pred_norm, target_norm)
        segment_loss = segment_loss + hop_loss
        segment_hops += 1
        total_loss_log += hop_loss.item()

        history = history[cfg.N_FWD:] + new_states

        if segment_hops == tbptt_hops or i == len(hops) - 1:
            optimiseur.zero_grad()
            (segment_loss / segment_hops).backward()
            optimiseur.step()
            n_updates += 1
            history = [h.detach() for h in history]
            segment_loss, segment_hops = torch.zeros(()), 0

    return total_loss_log / len(hops), n_updates


def evaluate_val_rollout(modele, FIELDS, pairs_val, input_fields, norm_stats, INPUTS, OUTPUTS, cfg: "C.Config") -> float:
    # Réutilise le rollout d'évaluation numpy/no_grad déjà existant et
    # validé (C._autoregressive_rollout) -- pas besoin de réécrire une
    # deuxième version torch pour le monitoring, seule l'étape
    # d'entraînement doit rester différentiable.
    mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
    sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)
    mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
    sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype(np.float32)
    biais_repos = C._biais_repos(modele, mu_in, sd_in, mu_out, sd_out, cfg)

    modele.eval()
    errs = []
    with torch.no_grad():
        for wave_type, A, omega in pairs_val:
            U_reel = FIELDS[(wave_type, A, omega)]
            U_pred = autoregressive_rollout_multi(modele, U_reel, input_fields, mu_in, sd_in, mu_out, sd_out,
                                                    biais_repos, wave_type, A, omega, cfg)
            errs.append(C.l2_rel(U_pred[:, cfg.nodes], U_reel[:, cfg.nodes]))
    modele.train()
    return float(np.mean(errs))


def train_full_rollout(modele, FIELDS, pairs_train, pairs_val, input_fields,
                        norm_stats, INPUTS, OUTPUTS, cfg: "C.Config", group_size: int,
                        n_epochs: int, model_path: Path, tbptt_hops: int = 10) -> "C.TrainResult":
    criterion = nn.MSELoss()
    optimiseur = torch.optim.Adam(modele.parameters(), lr=cfg.LEARNING_RATE)

    mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
    sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)
    mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
    sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype(np.float32)
    mu_in_t, sd_in_t = torch.tensor(mu_in), torch.tensor(sd_in)
    mu_out_t, sd_out_t = torch.tensor(mu_out), torch.tensor(sd_out)

    rng = np.random.default_rng(cfg.SEED)
    historique_train, historique_val = [], []
    meilleure_val = float("inf")

    t0 = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        biais_repos_t = torch.tensor(C._biais_repos(modele, mu_in, sd_in, mu_out, sd_out, cfg))
        groups = make_epoch_groups(pairs_train, group_size, rng)

        modele.train()
        t_epoch0 = time.perf_counter()
        epoch_loss = 0.0
        for i, group_pairs in enumerate(groups):
            t_g0 = time.perf_counter()
            avg_loss, n_updates = rollout_group_tbptt(modele, group_pairs, FIELDS, input_fields,
                                                        mu_in_t, sd_in_t, mu_out_t, sd_out_t, biais_repos_t,
                                                        criterion, optimiseur, cfg, tbptt_hops)
            epoch_loss += avg_loss
            print(f"  epoch {epoch:3d}  groupe {i+1:3d}/{len(groups)} "
                  f"({len(group_pairs)} sims) -- perte moy/hop={avg_loss:.4f} -- "
                  f"{n_updates} corrections -- {time.perf_counter()-t_g0:.2f}s/groupe")
        epoch_loss /= len(groups)

        val_err = evaluate_val_rollout(modele, FIELDS, pairs_val, input_fields, norm_stats, INPUTS, OUTPUTS, cfg)
        historique_train.append(epoch_loss)
        historique_val.append(val_err)

        print(f"Epoch {epoch:4d}/{n_epochs} -- perte rollout (train): {epoch_loss:.4f}  |  "
              f"erreur L2 rel (val): {val_err:.4f} -- {time.perf_counter()-t_epoch0:.1f}s")

        if val_err < meilleure_val:
            meilleure_val = val_err
            torch.save(modele.state_dict(), model_path)

    train_time_s = time.perf_counter() - t0
    modele.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Meilleur modèle rechargé -- erreur L2 rel (val) minimale : {meilleure_val:.6f}")

    n_params = sum(p.numel() for p in modele.parameters())
    return C.TrainResult(historique_train, historique_val, [], meilleure_val, train_time_s, n_params)


def plot_rollout_training_curve(result: "C.TrainResult", output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.historique_train, label="Perte rollout (train)")
    ax.plot(result.historique_val, label="Erreur L2 relative (val)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Valeur")
    ax.set_title("Courbe d'apprentissage (full rollout différentiable)")
    ax.set_yscale("log"); ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "courbe_apprentissage.png", dpi=150, bbox_inches="tight")
    plt.close()
