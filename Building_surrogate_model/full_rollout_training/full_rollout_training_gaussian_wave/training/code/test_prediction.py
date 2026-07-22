# Test manuel : pour une impulsion gaussienne (A, omega) choisie à la main,
# PAS nécessairement vue à l'entraînement (ni même dans la plage AMP/OMEGA du
# dataset), simule la vérité terrain (résolution FD) et le rollout prédit par
# le modèle entraîné, puis affiche les deux courbes superposées dans un même
# gif (cf. commun.make_rollout_animation) + les courbes d'erreur associées.
#
# Rappel physique (cf. u_right_val dans commun.py) : pour une gaussienne,
# omega ne représente PAS une fréquence mais pilote la largeur de l'impulsion
# via sigma = interp(omega, [1, 10], [0.15, 0.07]) -- omega grand = impulsion
# étroite, omega petit = impulsion large.
#
# Usage le plus simple : modifiez A et OMEGA juste en-dessous, puis
#   python test_prediction.py
# (les flags --A/--omega restent disponibles si vous préférez les passer en
# ligne de commande -- ils ont alors priorité sur les valeurs ci-dessous).
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ============================================================
#  PARAMÈTRES À MODIFIER ICI POUR CHANGER LA FORME DE L'ONDE
# ============================================================
A = 0.08          # amplitude de l'impulsion gaussienne
OMEGA = 4.5       # largeur : sigma = interp(omega, [1,10], [0.15,0.07])
                  # (omega grand -> impulsion étroite, omega petit -> large)
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR
from main import CONFIG_OVERRIDES, INPUT_FIELDS, PROJECT_DIR, PLOTS_DIR

sys.path.insert(0, str(COMMUN_DIR))
import commun as C


def parse_args():
    p = argparse.ArgumentParser(description="Teste le modèle entraîné sur une impulsion gaussienne "
                                             "(A, omega) -- modifiables directement en haut du fichier, "
                                             "ou via --A/--omega qui ont priorité.")
    p.add_argument("--A", type=float, default=None,
                    help=f"Amplitude de l'impulsion gaussienne (défaut : A={A} défini en haut du fichier).")
    p.add_argument("--omega", type=float, default=None,
                    help=f"Paramètre de largeur, sigma = interp(omega, [1,10], [0.15,0.07]) -- "
                         f"plus omega est grand, plus l'impulsion est étroite "
                         f"(défaut : OMEGA={OMEGA} défini en haut du fichier).")
    p.add_argument("--model-path", type=Path, default=PROJECT_DIR / "model.pth")
    p.add_argument("--norm-stats", type=Path, default=PROJECT_DIR / "norm_stats.csv",
                    help="Stats de normalisation sauvegardées par main.py lors de l'entraînement.")
    p.add_argument("--output-dir", type=Path, default=None,
                    help="Défaut : training/plots/test_A{A}_omega{omega}/.")
    args = p.parse_args()
    if args.A is None:
        args.A = A
    if args.omega is None:
        args.omega = OMEGA
    return args


def main():
    args = parse_args()

    if not args.model_path.exists():
        sys.exit(f"Modèle introuvable : {args.model_path} -- avez-vous lancé l'entraînement (main.py) ?")
    if not args.norm_stats.exists():
        sys.exit(f"Stats de normalisation introuvables : {args.norm_stats} -- avez-vous lancé "
                 f"l'entraînement (main.py) après l'ajout de la sauvegarde norm_stats.csv ?")

    cfg = C.Config(**CONFIG_OVERRIDES)
    sigma = np.interp(args.omega, [1.0, 10.0], [0.15, 0.07])
    print(f"=== test manuel — A={args.A}, omega={args.omega} (sigma≈{sigma:.4f}) ===")

    INPUTS = C.make_feature_columns(INPUT_FIELDS, cfg)
    OUTPUTS = C.make_output_columns(cfg)

    norm_stats = pd.read_csv(args.norm_stats, index_col=0)
    mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
    sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)
    mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
    sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype(np.float32)

    modele = C.Reseau(n_inputs=len(INPUTS), n_outputs=len(OUTPUTS), hidden_sizes=cfg.HIDDEN_SIZES)
    modele.load_state_dict(torch.load(args.model_path, weights_only=True))
    modele.eval()

    biais_repos = C._biais_repos(modele, mu_in, sd_in, mu_out, sd_out, cfg)

    print("Simulation FD de référence (vérité terrain)...")
    U_reel = C.run_fd_simulation(args.A, args.omega, cfg)

    print("Rollout autorégressif du modèle...")
    U_pred = C._autoregressive_rollout(modele, U_reel, INPUT_FIELDS, mu_in, sd_in, mu_out, sd_out,
                                        biais_repos, args.A, args.omega, cfg)

    rollout = C.RolloutResult(U=U_pred, U_reel=U_reel, A=args.A, omega=args.omega)

    output_dir = args.output_dir or PLOTS_DIR / f"test_A{args.A}_omega{args.omega}"
    output_dir.mkdir(parents=True, exist_ok=True)

    C.make_rollout_animation(rollout, cfg, output_dir)

    t_axis, l2_list, linf_list, smape_list = C.compute_errors(rollout, cfg)
    C.plot_rollout_error(t_axis, l2_list, linf_list, output_dir)
    C.plot_smape(t_axis, smape_list, output_dir)

    print(f"L2 relative   : finale = {l2_list[-1]:.4e}  |  max = {max(l2_list):.4e}")
    print(f"Linf absolue  : finale = {linf_list[-1]:.4e}  |  max = {max(linf_list):.4e}")
    print(f"sMAPE (%)     : finale = {smape_list[-1]:.2f}  |  max = {max(smape_list):.2f}")
    print(f"Terminé — gif et courbes d'erreur dans {output_dir}")


if __name__ == "__main__":
    main()
