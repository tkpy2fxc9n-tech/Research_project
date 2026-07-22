# Point d'entrée : entraînement "full rollout" différentiable, puis
# réutilisation intégrale (sans modification) des fonctions d'évaluation et
# de graphiques déjà existantes dans commun.py.
import argparse
import resource
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _commun_path import COMMUN_DIR
from data_split import split_by_simulation, compute_norm_stats
from train import train_full_rollout, plot_rollout_training_curve
from wave_forcing import WAVE_TYPES, generate_dataset_multi, run_rollout_multi

sys.path.insert(0, str(COMMUN_DIR))
import commun as C

INPUT_FIELDS = ["U", "Ut", "Uxx"]
METHOD_NAME = "full_rollout_U_Ut_Uxx"

OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


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
    kwargs = {"N_EPOCHS": n_epochs}
    if args.smoke_test:
        kwargs["N_GRID"] = 4  # 16 simulations au lieu de 100 -- suffisant pour vérifier que ça tourne
    return C.Config(**kwargs)


def main():
    args = parse_args()
    n_epochs = args.epochs if args.epochs is not None else (2 if args.smoke_test else 5)
    cfg = build_config(args, n_epochs)
    C.set_seeds(cfg)

    mode = "SMOKE TEST" if args.smoke_test else "run"
    print(f"=== full_rollout_training [{mode}] — champs d'entrée : {INPUT_FIELDS} — "
          f"types d'onde : {WAVE_TYPES} — grille {cfg.N_GRID}x{cfg.N_GRID}, {n_epochs} epochs, "
          f"groupes de {args.group_size}, correction toutes les {args.tbptt_hops} hops ===")

    df, FIELDS, INPUTS, OUTPUTS = generate_dataset_multi(INPUT_FIELDS, WAVE_TYPES, cfg)
    print(f"{len(df):,} lignes x {df.shape[1]} colonnes ({len(FIELDS)} simulations)")

    df, pairs_train, pairs_val, pairs_test = split_by_simulation(df, cfg)
    norm_stats = compute_norm_stats(df, INPUTS, OUTPUTS, cfg)
    norm_stats.to_csv(OUTPUT_DIR / "norm_stats.csv")

    modele = C.Reseau(n_inputs=len(INPUTS), n_outputs=len(OUTPUTS), hidden_sizes=cfg.HIDDEN_SIZES)
    print(modele)

    train_result = train_full_rollout(modele, FIELDS, pairs_train, pairs_val, INPUT_FIELDS,
                                       norm_stats, INPUTS, OUTPUTS, cfg, group_size=args.group_size,
                                       n_epochs=n_epochs, model_path=SCRIPT_DIR / "model.pth",
                                       tbptt_hops=args.tbptt_hops)
    plot_rollout_training_curve(train_result, OUTPUT_DIR)

    df_test = df[df["split"] == "test"].reset_index(drop=True)
    tf_metrics = C.evaluate_teacher_forcing(modele, df_test, INPUTS, OUTPUTS, norm_stats, OUTPUT_DIR)

    # Un rollout/animation/resume par type d'onde, dans son propre
    # sous-dossier -- les fonctions de commun.py écrivent toutes sous un nom
    # de fichier fixe (propagation_onde.gif, resume.txt, ...), donc appeler
    # deux fois avec le même output_dir écraserait un type avec l'autre.
    A = cfg.AMPLITUDES[cfg.ROLLOUT_A_IDX]
    omega = cfg.PULSATIONS[cfg.ROLLOUT_OMEGA_IDX]
    for wave_type in WAVE_TYPES:
        wt_dir = OUTPUT_DIR / wave_type
        wt_dir.mkdir(exist_ok=True)

        rollout = run_rollout_multi(modele, FIELDS, INPUT_FIELDS, norm_stats, INPUTS, OUTPUTS,
                                     wave_type, A, omega, cfg)
        C.plot_utt_uxx(rollout, cfg, wt_dir)
        C.make_rollout_animation(rollout, cfg, wt_dir)

        errors = C.compute_errors(rollout, cfg)
        t_axis, l2_list, linf_list, smape_list = errors
        C.plot_rollout_error(t_axis, l2_list, linf_list, wt_dir)
        C.plot_smape(t_axis, smape_list, wt_dir)

        bench = C.benchmark_inference(modele, FIELDS, INPUT_FIELDS, norm_stats, INPUTS, OUTPUTS, rollout, cfg)

        C.export_resume(wt_dir, cfg, f"{METHOD_NAME}_{wave_type}", df, INPUTS, OUTPUTS,
                         train_result, tf_metrics, rollout, bench, errors)

    if args.smoke_test:
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"Pic mémoire (smoke test) : {peak_rss_mb:.0f} Mo")

    print(f"Terminé — sorties dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
