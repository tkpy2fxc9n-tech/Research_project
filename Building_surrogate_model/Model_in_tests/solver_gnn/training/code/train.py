# Boucle d'entraînement du WaveGNN. Ne peut pas réutiliser C.train_model tel
# quel : celui-ci mélange les LIGNES individuelles (torch DataLoader shuffle)
# et fait un warm-up sur une seule ligne -- les deux cassent la notion de
# graphe (un batch doit toujours être un multiple entier de n_nodes lignes,
# chaque bloc de n_nodes lignes dans le bon ordre de nœuds). Ici on mélange
# des SNAPSHOTS entiers (des blocs de n_nodes lignes), pas des lignes.
# v2 : pushforward (cf. pushforward_loss_gnn), AdamW + MultiStepLR comme
# Brandstetter et al. (experiments/train.py) -- toujours pas de bruit
# d'entrée ni de lissage (cf. commun.train_model), à ajouter en v3 si besoin.
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR
from graph_data import build_node_features
from rollout import _autoregressive_rollout_gnn

sys.path.insert(0, str(COMMUN_DIR))
import commun as C


def _gather_blocks(blocks, X, Y, n_nodes):
    idx = np.concatenate([np.arange(k * n_nodes, (k + 1) * n_nodes) for k in blocks])
    return X[idx], Y[idx]


def evaluate_val_rollout_gnn(modele, FIELDS, pairs_val, mu_in, sd_in, mu_out, sd_out, cfg: "C.Config") -> float:
    # Équivalent GNN de full_rollout_training/train.py:evaluate_val_rollout --
    # sélectionner le "meilleur" modèle sur la MSE one-shot est trompeur : un
    # modèle peut prédire parfaitement un seul pas et pourtant diverger en
    # rollout autorégressif complet (observé sur le run précédent : MSE val
    # one-shot = 5e-5 mais L2 rollout final = 1.07, max = 81.7). On mesure
    # donc directement l'erreur qui compte : le rollout complet, no_grad.
    modele.eval()
    errs = []
    with torch.no_grad():
        for A, omega in pairs_val:
            U_reel = FIELDS[(A, omega)]
            U_pred = _autoregressive_rollout_gnn(modele, U_reel, mu_in, sd_in, mu_out, sd_out, A, omega, cfg)
            errs.append(C.l2_rel(U_pred[:, cfg.nodes], U_reel[:, cfg.nodes]))
    modele.train()
    return float(np.mean(errs))


def pushforward_loss_gnn(modele, FIELDS, PF_SAMPLES, mu_in, sd_in, mu_out, sd_out,
                          criterion, cfg: "C.Config", n_groups: int):
    # Même schéma que C.pushforward_loss (Brandstetter et al. 2022, pushforward
    # trick) : cfg.PF_HOPS sauts autorégressifs enchaînés, chaque saut prédit
    # à partir de SA PROPRE reconstruction du saut précédent (gradient détaché
    # sauf le dernier saut) -- pas des données réelles, comme en rollout
    # complet. Reconstruction physique (C.reconstruct) et échantillons
    # (C.make_pf_samples) réutilisés tels quels -- ils ne dépendent pas du
    # format d'entrée du modèle. Seule la construction de X change :
    # build_node_features (historique brut par nœud + position/temps/A/omega)
    # au lieu du stencil build_window du MLP.
    # Pas de correction "biais au repos" ici (cf. rollout.py) : les colonnes
    # pos_x/pos_t/A/omega ne sont pas des champs physiques nuls au repos.
    idxs = np.random.choice(len(PF_SAMPLES), n_groups, replace=False)
    groups = [PF_SAMPLES[i] for i in idxs]
    nN = len(cfg.nodes)

    field_at = [(lambda m, U=FIELDS[(A, omega)]: U[m]) for (A, omega, n) in groups]
    n_curr = [n for (A, omega, n) in groups]
    baseline = [FIELDS[(A, omega)][n] for (A, omega, n) in groups]

    pred = None
    for hop in range(cfg.PF_HOPS):
        last = hop == cfg.PF_HOPS - 1
        X = np.concatenate([
            (build_node_features(field_at[j], n_curr[j], A, omega, cfg) - mu_in) / sd_in
            for j, (A, omega, n) in enumerate(groups)
        ], axis=0)

        if last:
            pred = modele(torch.tensor(X))
            pred_np = pred.detach().numpy()
        else:
            with torch.no_grad():
                pred_np = modele(torch.tensor(X)).numpy()

        new_field_at, new_n_curr, new_baseline = [], [], []
        for j, (A, omega, n) in enumerate(groups):
            U = FIELDS[(A, omega)]
            Up = C.reconstruct(baseline[j], n_curr[j], pred_np[j * nN:(j + 1) * nN], A, omega, mu_out, sd_out, cfg)
            nprime = n_curr[j] + cfg.N_FWD * cfg.ndt
            new_field_at.append(lambda m, U=U, Up=Up: Up[m] if m in Up else U[m])
            new_n_curr.append(nprime)
            new_baseline.append(Up[nprime])
        field_at, n_curr, baseline = new_field_at, new_n_curr, new_baseline

    tgt_list = []
    for j, (A, omega, n) in enumerate(groups):
        U = FIELDS[(A, omega)]
        curr = baseline[j][cfg.nodes]
        tgt = np.stack([U[n_curr[j] + h * cfg.ndt][cfg.nodes] - curr for h in range(1, cfg.N_FWD + 1)], axis=1)
        tgt_list.append(((tgt - mu_out) / sd_out).astype(np.float32))

    return criterion(pred, torch.tensor(np.concatenate(tgt_list, axis=0)))


def train_gnn(modele, df, samples, pairs_train, pairs_val, INPUTS, OUTPUTS, norm_stats, FIELDS, PF_SAMPLES,
              cfg: "C.Config", model_path: Path, batch_graphs: int = 16) -> "C.TrainResult":
    n_nodes = modele.n_nodes
    criterion = nn.MSELoss()
    # AdamW + MultiStepLR : mêmes choix que Brandstetter et al.
    # (experiments/train.py) -- AdamW applique par défaut un weight decay de
    # 0.01 (Adam n'en applique aucun), et le scheduler réduit le lr de x0.4
    # aux epochs 1, 5, 10, 15 (leurs valeurs par défaut ; notre cfg.N_EPOCHS
    # vaut aussi 20 par défaut, donc ces paliers restent dans la même
    # proportion de l'entraînement).
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiseur, milestones=[1, 5, 10, 15], gamma=0.4)

    mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
    sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)
    mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
    sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype(np.float32)

    X_full = C.normalize_array(df[INPUTS].values, INPUTS, norm_stats)
    y_full = C.normalize_array(df[OUTPUTS].values, OUTPUTS, norm_stats)

    train_set = set(pairs_train)
    train_blocks = [k for k, (A, omega, n) in enumerate(samples) if (A, omega) in train_set]

    with torch.no_grad():
        modele(torch.zeros(n_nodes, X_full.shape[1]))  # warm-up sur UN graphe valide

    historique_train, historique_val, historique_pf = [], [], []
    meilleure_val = float("inf")

    t0 = time.perf_counter()
    for epoch in range(1, cfg.N_EPOCHS + 1):
        np.random.shuffle(train_blocks)
        # Montée progressive du poids pushforward (le modèle est nul au
        # début) -- identique à commun.train_model.
        lam_pf = cfg.LAMBDA_PF * min(1.0, epoch / cfg.PF_WARMUP)

        modele.train()
        perte_train = 0.0
        perte_pf_total = 0.0
        n_batches = 0
        for i in range(0, len(train_blocks), batch_graphs):
            batch_idx = train_blocks[i:i + batch_graphs]
            Xb, yb = _gather_blocks(batch_idx, X_full, y_full, n_nodes)

            optimiseur.zero_grad()
            prediction = modele(torch.tensor(Xb))
            data_loss = criterion(prediction, torch.tensor(yb))

            if lam_pf > 0:
                pf_loss = pushforward_loss_gnn(modele, FIELDS, PF_SAMPLES, mu_in, sd_in, mu_out, sd_out,
                                                criterion, cfg, cfg.N_PF_GROUPS)
            else:
                pf_loss = torch.tensor(0.0)

            total_loss = data_loss + lam_pf * pf_loss
            total_loss.backward()
            optimiseur.step()

            perte_train += data_loss.item()
            perte_pf_total += pf_loss.item()
            n_batches += 1
        perte_train /= n_batches
        perte_pf_total /= n_batches

        val_err = evaluate_val_rollout_gnn(modele, FIELDS, pairs_val, mu_in, sd_in, mu_out, sd_out, cfg)
        scheduler.step()

        historique_train.append(perte_train)
        historique_val.append(val_err)
        historique_pf.append(perte_pf_total)

        print(f"Epoch {epoch:4d}/{cfg.N_EPOCHS}  —  data: {perte_train:.4f}  |  "
              f"pushf: {perte_pf_total:.4f}  |  erreur L2 rel (val rollout): {val_err:.4f}")

        if val_err < meilleure_val:
            meilleure_val = val_err
            torch.save(modele.state_dict(), model_path)

    train_time_s = time.perf_counter() - t0

    modele.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Meilleur modèle rechargé — erreur L2 rel (val rollout) minimale : {meilleure_val:.6f}")

    n_params = sum(p.numel() for p in modele.parameters())
    return C.TrainResult(historique_train, historique_val, historique_pf, meilleure_val, train_time_s, n_params)
