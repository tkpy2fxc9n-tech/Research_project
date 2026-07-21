# Script à lancer à la main pour tester le modèle entraîné (model.pth) sur
# une amplitude/pulsation/type d'onde choisis : calcule la "vraie" onde par
# différences finies et la prédiction autorégressive du réseau, puis génère
# un gif de comparaison (+ courbes d'erreur), en réutilisant tel quel les
# fonctions de commun.py et le portage wave_forcing.py du dossier parent.
#
# Exemples :
#   python test/generate_gif.py --amplitude 0.05 --pulsation 6.0 --wave-type sinusoidal
#   python test/generate_gif.py --amplitude 0.08 --pulsation 4.5 --wave-type gaussian_pulse
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TEST_DIR = Path(__file__).resolve().parent
ROOT_DIR = TEST_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
from _commun_path import COMMUN_DIR
from wave_forcing import WAVE_TYPES, run_fd_simulation_multi, autoregressive_rollout_multi

sys.path.insert(0, str(COMMUN_DIR))
import commun as C

# Doit rester synchronisé avec INPUT_FIELDS dans ../main.py.
INPUT_FIELDS = ["U", "Ut", "Uxx"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Génère un gif comparant la prédiction du réseau à la vraie onde (différences finies) "
                     "pour une amplitude/pulsation/type d'onde choisis.")
    p.add_argument("--amplitude", type=float, default=None,
                    help="Amplitude A du forçage au bord droit (demandée en console si omise).")
    p.add_argument("--pulsation", type=float, default=None,
                    help="Pulsation omega du forçage au bord droit (demandée en console si omise).")
    p.add_argument("--wave-type", type=str, default=None, choices=WAVE_TYPES,
                    help="Type d'onde du forçage au bord droit (demandé en console si omis).")
    p.add_argument("--model-path", type=Path, default=ROOT_DIR / "model.pth",
                    help="Chemin vers les poids du modèle entraîné (défaut : ../model.pth).")
    p.add_argument("--norm-stats", type=Path, default=ROOT_DIR / "outputs" / "norm_stats.csv",
                    help="Chemin vers les statistiques de normalisation sauvegardées par main.py "
                         "(défaut : ../outputs/norm_stats.csv).")
    p.add_argument("--output-dir", type=Path, default=None,
                    help="Dossier de sortie (défaut : test/outputs/<wave_type>_A<amplitude>_omega<pulsation>/).")
    return p.parse_args()


def ask_missing_args(args):
    # Permet de lancer le script sans aucun argument (python test/generate_gif.py) :
    # tout ce qui n'a pas été passé en ligne de commande est demandé en console.
    if args.amplitude is None:
        args.amplitude = float(input("Amplitude A (ex: 0.05) : ").strip())
    if args.pulsation is None:
        args.pulsation = float(input("Pulsation omega (ex: 6.0) : ").strip())
    if args.wave_type is None:
        reponse = input(f"Type d'onde {WAVE_TYPES} [gaussian_pulse] : ").strip()
        args.wave_type = reponse if reponse else "gaussian_pulse"
        if args.wave_type not in WAVE_TYPES:
            raise ValueError(f"Type d'onde inconnu : {args.wave_type!r} (attendu : {WAVE_TYPES})")
    return args


def main():
    args = ask_missing_args(parse_args())
    A, omega, wave_type = args.amplitude, args.pulsation, args.wave_type

    if not args.model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {args.model_path} -- avez-vous entraîné le réseau "
                                 f"(main.py) avant de lancer ce script ?")
    if not args.norm_stats.exists():
        raise FileNotFoundError(f"Statistiques de normalisation introuvables : {args.norm_stats} -- "
                                 f"générées par main.py à la fin de l'entraînement (outputs/norm_stats.csv).")

    output_dir = args.output_dir or TEST_DIR / "outputs" / f"{wave_type}_A{A}_omega{omega}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Même grille/architecture que l'entraînement (valeurs par défaut de
    # C.Config, comme dans check_equivalence.py -- N_EPOCHS n'affecte ni la
    # grille ni l'architecture donc n'a pas besoin d'être reproduit ici).
    cfg = C.Config()
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

    print(f"Type d'onde : {wave_type}  |  A = {A}  |  omega = {omega}")
    print("Calcul de la vraie onde (différences finies)...")
    U_reel = run_fd_simulation_multi(wave_type, A, omega, cfg)

    print("Inférence autorégressive du réseau...")
    U_pred = autoregressive_rollout_multi(modele, U_reel, INPUT_FIELDS, mu_in, sd_in, mu_out, sd_out,
                                           biais_repos, wave_type, A, omega, cfg)

    rollout = C.RolloutResult(U=U_pred, U_reel=U_reel, A=A, omega=omega)

    C.make_rollout_animation(rollout, cfg, output_dir)
    t_axis, l2_list, linf_list, smape_list = C.compute_errors(rollout, cfg)
    C.plot_rollout_error(t_axis, l2_list, linf_list, output_dir)
    C.plot_smape(t_axis, smape_list, output_dir)

    print(f"Erreur L2 relative -- finale : {l2_list[-1]:.4e}  |  max : {max(l2_list):.4e}")
    print(f"sMAPE (%)          -- finale : {smape_list[-1]:.3f}  |  max : {max(smape_list):.3f}")
    print(f"Sorties écrites dans {output_dir}")


if __name__ == "__main__":
    main()
