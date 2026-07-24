# Method U_Ut: U + Ut as input (pipeline shared in commun.py).
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
import commun as C

INPUT_FIELDS = ["U", "Ut"]
METHOD_NAME = "U_Ut"

OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    cfg = C.Config()
    C.set_seeds(cfg)

    print(f"=== Method {METHOD_NAME} — input fields: {INPUT_FIELDS} ===")
    df, FIELDS, INPUTS, OUTPUTS = C.generate_dataset(INPUT_FIELDS, cfg)
    print(f"{len(df):,} rows x {df.shape[1]} columns")

    df, norm_stats = C.split_and_normalize(df, INPUTS, OUTPUTS, cfg)
    train_loader, X_val, y_val = C.make_dataloaders(df, INPUTS, OUTPUTS, norm_stats, cfg)

    modele = C.Reseau(n_inputs=len(INPUTS), n_outputs=len(OUTPUTS), hidden_sizes=cfg.HIDDEN_SIZES)
    print(modele)

    mu_in = norm_stats.loc[INPUTS, "mean"].values.astype("float32")
    sd_in = norm_stats.loc[INPUTS, "std"].values.astype("float32")
    mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype("float32")
    sd_out = norm_stats.loc[OUTPUTS, "std"].values.astype("float32")
    PF_SAMPLES = C.make_pf_samples(FIELDS, cfg)

    train_result = C.train_model(modele, train_loader, X_val, y_val, FIELDS, PF_SAMPLES, INPUT_FIELDS,
                                  mu_in, sd_in, mu_out, sd_out, cfg, model_path=SCRIPT_DIR / "model.pth")
    C.plot_training_curve(train_result, OUTPUT_DIR)

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

    xlsx_path = SCRIPT_DIR.parent / "Comparaisons" / "comparative_table.xlsx"
    C.export_errors_to_xlsx(xlsx_path, METHOD_NAME, t_axis, l2_list)
    C.export_timings_to_xlsx(xlsx_path, METHOD_NAME, train_result.train_time_s,
                              bench.nn_time_med, bench.nn_time_std, bench.fd_time_med,
                              train_result.n_params, bench.flops_per_call)

    print(f"Done — outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
