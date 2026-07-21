# Boucle d'entraînement du WaveGNN. Ne peut pas réutiliser C.train_model tel
# quel : celui-ci mélange les LIGNES individuelles (torch DataLoader shuffle)
# et fait un warm-up sur une seule ligne -- les deux cassent la notion de
# graphe (un batch doit toujours être un multiple entier de n_nodes lignes,
# chaque bloc de n_nodes lignes dans le bon ordre de nœuds). Ici on mélange
# des SNAPSHOTS entiers (des blocs de n_nodes lignes), pas des lignes.
# v1 volontairement minimal : MSE + teacher forcing uniquement, pas de
# pushforward/bruit/lissage (voir le plan -- à ajouter en v2 si besoin).
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR

sys.path.insert(0, str(COMMUN_DIR))
import commun as C


def _gather_blocks(blocks, X, Y, n_nodes):
    idx = np.concatenate([np.arange(k * n_nodes, (k + 1) * n_nodes) for k in blocks])
    return X[idx], Y[idx]


def train_gnn(modele, df, samples, pairs_train, pairs_val, INPUTS, OUTPUTS, norm_stats,
              cfg: "C.Config", model_path: Path, batch_graphs: int = 16) -> "C.TrainResult":
    n_nodes = modele.n_nodes
    criterion = nn.MSELoss()
    optimiseur = torch.optim.Adam(modele.parameters(), lr=cfg.LEARNING_RATE)

    X_full = C.normalize_array(df[INPUTS].values, INPUTS, norm_stats)
    y_full = C.normalize_array(df[OUTPUTS].values, OUTPUTS, norm_stats)

    train_set = set(pairs_train)
    val_set = set(pairs_val)
    train_blocks = [k for k, (A, omega, n) in enumerate(samples) if (A, omega) in train_set]
    val_blocks = [k for k, (A, omega, n) in enumerate(samples) if (A, omega) in val_set]

    X_val, y_val = _gather_blocks(val_blocks, X_full, y_full, n_nodes)

    with torch.no_grad():
        modele(torch.zeros(n_nodes, X_full.shape[1]))  # warm-up sur UN graphe valide

    historique_train, historique_val, historique_pf = [], [], []
    meilleure_val = float("inf")

    t0 = time.perf_counter()
    for epoch in range(1, cfg.N_EPOCHS + 1):
        np.random.shuffle(train_blocks)

        modele.train()
        perte_train = 0.0
        n_batches = 0
        for i in range(0, len(train_blocks), batch_graphs):
            batch_idx = train_blocks[i:i + batch_graphs]
            Xb, yb = _gather_blocks(batch_idx, X_full, y_full, n_nodes)

            optimiseur.zero_grad()
            prediction = modele(torch.tensor(Xb))
            loss = criterion(prediction, torch.tensor(yb))
            loss.backward()
            optimiseur.step()

            perte_train += loss.item()
            n_batches += 1
        perte_train /= n_batches

        modele.eval()
        with torch.no_grad():
            pred_val = modele(torch.tensor(X_val)).numpy()
        perte_val = ((pred_val - y_val) ** 2).mean()

        historique_train.append(perte_train)
        historique_val.append(perte_val)
        historique_pf.append(0.0)  # pas de pushforward en v1 -- place-holder pour C.plot_training_curve

        print(f"Epoch {epoch:4d}/{cfg.N_EPOCHS}  —  data: {perte_train:.4f}  |  val: {perte_val:.4f}")

        if perte_val < meilleure_val:
            meilleure_val = perte_val
            torch.save(modele.state_dict(), model_path)

    train_time_s = time.perf_counter() - t0

    modele.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Meilleur modèle rechargé — val minimale : {meilleure_val:.6e}")

    n_params = sum(p.numel() for p in modele.parameters())
    return C.TrainResult(historique_train, historique_val, historique_pf, meilleure_val, train_time_s, n_params)
