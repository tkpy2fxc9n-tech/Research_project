# Point d'entrée : entraînement "full rollout" différentiable, puis
# réutilisation intégrale (sans modification) des fonctions d'évaluation et
# de graphiques déjà existantes dans commun.py.
import argparse
import resource
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR
from data_split import split_by_simulation, compute_norm_stats
from train import train_full_rollout, plot_rollout_training_curve

sys.path.insert(0, str(COMMUN_DIR))
import commun as C

INPUT_FIELDS = ["U", "Ut", "Uxx"]
METHOD_NAME = "full_rollout_U_Ut_Uxx_gaussian_wide"

# code/ est un sous-dossier de full_rollout_training_gaussian_wave/training/ --
# plots/ et logs/ sont ses dossiers frères ; model.pth et norm_stats.csv
# restent au niveau full_rollout_training_gaussian_wave/ (cf. Current_model).
TRAINING_DIR = SCRIPT_DIR.parent
PROJECT_DIR = TRAINING_DIR.parent
PLOTS_DIR = TRAINING_DIR / "plots"
# Sous-dossier par run (date + heure, pas juste la date) -- les anciens runs
# restent donc tous consultables sous plots/, même plusieurs par jour.
OUTPUT_DIR = PLOTS_DIR / f"simulation_{datetime.now():%d%m%Y_%H%M%S}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Grille élargie (vs Current_model, N_GRID=10 / A:0.005-0.1 / omega:3-10) pour
# couvrir beaucoup plus de formes d'onde gaussiennes : 20x20 = 400 simulations,
# amplitude 0.005-0.15, omega 1-10 (= toute la plage physique utile, cf.
# u_right_val dans commun.py : sigma = interp(omega, [1,10], [0.15,0.07]) --
# au-delà de omega=10 ou en-deçà de 1, l'interp clampe et n'ajoute aucune
# largeur d'impulsion supplémentaire). Centralisé ici (plutôt que dupliqué)
# car test_prediction.py doit reconstruire un Config strictement identique
# pour que le modèle rechargé reçoive les bonnes features.
CONFIG_OVERRIDES = dict(
    N_GRID=20,
    AMP_MIN=0.005,
    AMP_MAX=0.15,
    OMEGA_MIN=1.0,
    OMEGA_MAX=10.0,
)


def parse_args():
    p = argparse.ArgumentParser(description="Entraînement full-rollout différentiable (sans detach).")
    p.add_argument("--smoke-test", action="store_true",
                    help="Run miniature (grille réduite, peu d'epochs) pour vérifier que tout "
                         "tourne sans erreur avant un run complet et coûteux.")
    p.add_argument("--epochs", type=int, default=None,
                    help="Nombre d'epochs (défaut : 2 en --smoke-test, 5 sinon).")
    p.add_argument("--group-size", type=int, default=8,
                    help="Nombre de simulations déroulées en parallèle par mise à jour des poids.")
    p.add_argument("--tbptt-hops", type=int, default=10,
                    help="Nombre de hops déroulés avant chaque correction des poids (coupe le fil du "
                         "gradient sans jamais remettre l'état à la vérité terrain).")
    return p.parse_args()


def build_config(args, n_epochs):
    # N_EPOCHS est repassé à Config (bien que train_full_rollout reçoive
    # n_epochs séparément) uniquement pour que C.export_resume affiche le
    # bon nombre d'epochs dans resume.txt.
    kwargs = {"N_EPOCHS": n_epochs, **CONFIG_OVERRIDES}
    if args.smoke_test:
        kwargs["N_GRID"] = 4  # 16 simulations au lieu de 400 -- suffisant pour vérifier que ça tourne
    return C.Config(**kwargs)


def main():
    args = parse_args()
    n_epochs = args.epochs if args.epochs is not None else (2 if args.smoke_test else 5)
    cfg = build_config(args, n_epochs)
    C.set_seeds(cfg)

    mode = "SMOKE TEST" if args.smoke_test else "run"
    print(f"=== full_rollout_training_gaussian_wave [{mode}] — champs d'entrée : {INPUT_FIELDS} — "
          f"grille {cfg.N_GRID}x{cfg.N_GRID} (A:{cfg.AMP_MIN}-{cfg.AMP_MAX}, omega:{cfg.OMEGA_MIN}-{cfg.OMEGA_MAX}), "
          f"{n_epochs} epochs, groupes de {args.group_size}, correction toutes les {args.tbptt_hops} hops ===")

    df, FIELDS, INPUTS, OUTPUTS = C.generate_dataset(INPUT_FIELDS, cfg)
    print(f"{len(df):,} lignes x {df.shape[1]} colonnes ({len(FIELDS)} simulations)")

    df, pairs_train, pairs_val, pairs_test = split_by_simulation(df, cfg)
    norm_stats = compute_norm_stats(df, INPUTS, OUTPUTS, cfg)
    # Persisté à côté de model.pth (pas dans OUTPUT_DIR, qui change de nom
    # chaque jour) pour que test_prediction.py retrouve toujours les stats de
    # normalisation du dernier modèle entraîné, sans régénérer le dataset.
    norm_stats.to_csv(PROJECT_DIR / "norm_stats.csv")

    modele = C.Reseau(n_inputs=len(INPUTS), n_outputs=len(OUTPUTS), hidden_sizes=cfg.HIDDEN_SIZES)
    print(modele)

    train_result = train_full_rollout(modele, FIELDS, pairs_train, pairs_val, INPUT_FIELDS,
                                       norm_stats, INPUTS, OUTPUTS, cfg, group_size=args.group_size,
                                       n_epochs=n_epochs, model_path=PROJECT_DIR / "model.pth",
                                       tbptt_hops=args.tbptt_hops)
    plot_rollout_training_curve(train_result, OUTPUT_DIR)

    df_test = df[df["split"] == "test"].reset_index(drop=True)
    tf_metrics = C.evaluate_teacher_forcing(modele, df_test, INPUTS, OUTPUTS, norm_stats, OUTPUT_DIR)

    rollout = C.run_rollout(modele, FIELDS, INPUT_FIELDS, norm_stats, INPUTS, OUTPUTS, cfg)
    C.plot_utt_uxx(rollout, cfg, OUTPUT_DIR)
    C.make_rollout_animation(rollout, cfg, OUTPUT_DIR)

    errors = C.compute_errors(rollout, cfg)
    t_axis, l2_list, linf_list, smape_list = errors
    C.plot_rollout_error(t_axis, l2_list, linf_list, OUTPUT_DIR)
    C.plot_smape(t_axis, smape_list, OUTPUT_DIR)

    bench = C.benchmark_inference(modele, FIELDS, INPUT_FIELDS, norm_stats, INPUTS, OUTPUTS, rollout, cfg)

    C.export_resume(OUTPUT_DIR, cfg, METHOD_NAME, df, INPUTS, OUTPUTS, train_result, tf_metrics, rollout, bench, errors)

    if args.smoke_test:
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"Pic mémoire (smoke test) : {peak_rss_mb:.0f} Mo")

    print(f"Terminé — sorties dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
