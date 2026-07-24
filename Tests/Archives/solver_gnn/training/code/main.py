# Point d'entrée : GNN (message passing, façon Brandstetter et al.) pour
# prédire le déplacement de l'onde 1D forcée -- même physique que
# Code_comparaison_des_inputs, entrées différentes (historique brut de u par
# nœud + position + (A, omega) diffusés, au lieu du stencil U/Ut/Uxx aplati).
import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR
from graph_data import build_dataset, split_by_simulation, compute_norm_stats
from model import WaveGNN
from train import train_gnn
from rollout import run_rollout_gnn, benchmark_gnn

sys.path.insert(0, str(COMMUN_DIR))
import commun as C

METHOD_NAME = "gnn_mp_pde"
# code/ est un sous-dossier de solver_gnn/training/ -- plots/ et logs/ sont
# ses dossiers frères ; model.pth reste au niveau solver_gnn/ (partagé avec
# test/, qui le recharge pour évaluer sans ré-entraîner).
TRAINING_DIR = SCRIPT_DIR.parent
PROJECT_DIR = TRAINING_DIR.parent
PLOTS_DIR = TRAINING_DIR / "plots"
# Sous-dossier par run (date + heure, pas juste la date, pour ne pas écraser
# les résultats d'un run précédent lancé le même jour) -- les anciens runs
# restent donc tous consultables sous plots/.
OUTPUT_DIR = PLOTS_DIR / f"simulation_{datetime.now():%Y%m%d_%H%M%S}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="GNN (message passing) pour l'onde 1D forcée.")
    p.add_argument("--smoke-test", action="store_true",
                    help="Run miniature (grille réduite, peu d'epochs) pour vérifier que tout tourne sans erreur.")
    p.add_argument("--epochs", type=int, default=None,
                    help="Nombre d'epochs (défaut : 2 en --smoke-test, 20 sinon).")
    return p.parse_args()


def main():
    args = parse_args()
    n_epochs = args.epochs if args.epochs is not None else (2 if args.smoke_test else 20)
    kwargs = {"N_EPOCHS": n_epochs}
    if args.smoke_test:
        kwargs["N_GRID"] = 4  # 16 simulations au lieu de 100 -- suffisant pour vérifier que ça tourne
    cfg = C.Config(**kwargs)
    C.set_seeds(cfg)

    # commun.py fixe torch.set_num_threads(1) à l'import (calibré pour
    # comparer plusieurs méthodes MLP tournant en parallèle sur les mêmes
    # cpus) -- pas pertinent ici (un seul process), et les matmuls du GNN
    # profitent d'un BLAS multi-thread.
    torch.set_num_threads(8)

    mode = "SMOKE TEST" if args.smoke_test else "run"
    print(f"=== Code_gnn_pde_solver [{mode}] — grille {cfg.N_GRID}x{cfg.N_GRID}, {n_epochs} epochs ===")

    df, FIELDS, INPUTS, OUTPUTS, samples, n_nodes = build_dataset(cfg)
    print(f"{len(df):,} lignes x {df.shape[1]} colonnes ({len(FIELDS)} simulations, {len(samples)} snapshots)")

    df, pairs_train, pairs_val, pairs_test = split_by_simulation(df, cfg)
    norm_stats = compute_norm_stats(df, INPUTS, OUTPUTS, cfg)

    modele = WaveGNN(n_nodes=n_nodes, m_back=cfg.M_BACK, n_fwd=cfg.N_FWD)
    print(modele)
    print(f"Paramètres : {sum(p.numel() for p in modele.parameters()):,}")

    # Comme les méthodes MLP de Code_comparaison_des_inputs : échantillons de
    # départ pour le pushforward tirés de TOUTES les simulations (FIELDS),
    # pas seulement du split train (cf. methode_U_Ut_Uxx/main.py).
    PF_SAMPLES = C.make_pf_samples(FIELDS, cfg)

    train_result = train_gnn(modele, df, samples, pairs_train, pairs_val, INPUTS, OUTPUTS, norm_stats,
                              FIELDS, PF_SAMPLES, cfg, model_path=PROJECT_DIR / "model.pth")
    C.plot_training_curve(train_result, OUTPUT_DIR)

    df_test = df[df["split"] == "test"].reset_index(drop=True)
    tf_metrics = C.evaluate_teacher_forcing(modele, df_test, INPUTS, OUTPUTS, norm_stats, OUTPUT_DIR)

    rollout = run_rollout_gnn(modele, FIELDS, norm_stats, INPUTS, OUTPUTS, cfg)
    C.plot_utt_uxx(rollout, cfg, OUTPUT_DIR)
    C.make_rollout_animation(rollout, cfg, OUTPUT_DIR)

    errors = C.compute_errors(rollout, cfg)
    t_axis, l2_list, linf_list, smape_list = errors
    C.plot_rollout_error(t_axis, l2_list, linf_list, OUTPUT_DIR)
    C.plot_smape(t_axis, smape_list, OUTPUT_DIR)

    bench = benchmark_gnn(modele, FIELDS, norm_stats, INPUTS, OUTPUTS, rollout, cfg)

    C.export_resume(OUTPUT_DIR, cfg, METHOD_NAME, df, INPUTS, OUTPUTS, train_result, tf_metrics, rollout, bench, errors)

    print(f"Terminé — sorties dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
