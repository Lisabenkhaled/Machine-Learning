#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train.io import TARGET_COL, TIME_COL
from train.metrics import weighted_accuracy
from train.svr_model import SVRModel, fit_svr, predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraîne SVR avec CV temporelle large+fine pour spot_id_delta."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Dossier des données préparées",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
        help="Dossier des sorties",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=["rbf"],
        help="Noyau SVR retenu",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=3,
        help="Nombre de folds CV temporelle",
    )
    return parser.parse_args()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    wa = float(weighted_accuracy(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return wa, rmse, mae


def timeseries_cv_search(
    train_df: pd.DataFrame,
    c_values: list[float],
    epsilon_values: list[float],
    gamma_values: list[float | str],
    n_splits: int = 3,
    kernel: str = "rbf",
) -> tuple[pd.DataFrame, pd.DataFrame, SVRModel]:
    ordered = train_df.sort_values(TIME_COL).reset_index(drop=True).copy()
    tscv = TimeSeriesSplit(n_splits=n_splits)

    rows = []
    best_score = -np.inf
    best_rmse = np.inf
    best_model = None

    print(f"🔍 CV recherche : {len(c_values)}×{len(epsilon_values)}×{len(gamma_values)} = {len(c_values)*len(epsilon_values)*len(gamma_values)} combos × {n_splits} folds")

    for c in c_values:
        for epsilon in epsilon_values:
            for gamma in gamma_values:
                fold_scores = []
                for fold_id, (tr_idx, va_idx) in enumerate(tscv.split(ordered)):
                    fold_train = ordered.iloc[tr_idx].copy()
                    fold_valid = ordered.iloc[va_idx].copy()

                    model, _ = fit_svr(
                        train_df=fold_train,
                        valid_df=fold_valid,
                        c_values=[c],
                        epsilon_values=[epsilon],
                        gamma_values=[gamma],
                        scorer=weighted_accuracy,
                        kernel=kernel,
                    )

                    train_pred = predict(fold_train, model)
                    valid_pred = predict(fold_valid, model)

                    y_train = fold_train[TARGET_COL].to_numpy(dtype=float)
                    y_valid = fold_valid[TARGET_COL].to_numpy(dtype=float)
                    y_hat_train = train_pred[TARGET_COL].to_numpy(dtype=float)
                    y_hat_valid = valid_pred[TARGET_COL].to_numpy(dtype=float)

                    wa_train, rmse_train, mae_train = regression_metrics(y_train, y_hat_train)
                    wa_valid, rmse_valid, mae_valid = regression_metrics(y_valid, y_hat_valid)

                    fold_scores.append(wa_valid)
                    rows.append(
                        {
                            "C": c,
                            "epsilon": epsilon,
                            "gamma": gamma,
                            "fold": fold_id,
                            "train_size": len(fold_train),
                            "valid_size": len(fold_valid),
                            "train_wa": wa_train,
                            "train_rmse": rmse_train,
                            "train_mae": mae_train,
                            "valid_wa": wa_valid,
                            "valid_rmse": rmse_valid,
                            "valid_mae": mae_valid,
                            "wa_gap": wa_train - wa_valid,
                            "rmse_gap": rmse_valid - rmse_train,
                        }
                    )

                mean_valid_wa = np.mean(fold_scores)
                if mean_valid_wa > best_score:
                    best_score = mean_valid_wa
                    print(f"   Nouveau meilleur : C={c}, eps={epsilon}, gamma={gamma} → WA={mean_valid_wa:.4f}")

    cv_detail = pd.DataFrame(rows)
    cv_summary = (
        cv_detail.groupby(["C", "epsilon", "gamma"], as_index=False)
        .agg(
            cv_valid_wa_mean=("valid_wa", "mean"),
            cv_valid_wa_std=("valid_wa", "std"),
            cv_valid_rmse_mean=("valid_rmse", "mean"),
            cv_valid_rmse_std=("valid_rmse", "std"),
            cv_train_wa_mean=("train_wa", "mean"),
            cv_wa_gap_mean=("wa_gap", "mean"),
            cv_wa_gap_std=("wa_gap", "std"),
        )
        .sort_values(["cv_valid_wa_mean", "cv_valid_rmse_mean"], ascending=[False, True])
        .reset_index(drop=True)
    )

    best_params = cv_summary.iloc[0]
    best_c = best_params["C"]
    best_epsilon = best_params["epsilon"]
    best_gamma = best_params["gamma"]

    print(f"\n🏆 Meilleur CV : C={best_c}, eps={best_epsilon}, gamma={best_gamma}")
    print(
        f"WA CV mean/std : {cv_summary.iloc[0]['cv_valid_wa_mean']:.4f}±{cv_summary.iloc[0]['cv_valid_wa_std']:.4f} | "
        f"gap train-valid : {cv_summary.iloc[0]['cv_wa_gap_mean']:.4f}±{cv_summary.iloc[0]['cv_wa_gap_std']:.4f}"
    )

    # Final model sur tout train
    final_model, _ = fit_svr(
        train_df=train_df,
        valid_df=pd.DataFrame(),  # Ignoré
        c_values=[best_c],
        epsilon_values=[best_epsilon],
        gamma_values=[best_gamma],
        scorer=weighted_accuracy,
        kernel=kernel,
    )

    return cv_detail, cv_summary, final_model


def main() -> None:
    args = parse_args()
    np.random.seed(42)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.data_dir / "train_estimation.csv")
    valid_df = pd.read_csv(args.data_dir / "train_validation.csv")
    test_df = pd.read_csv(args.data_dir / "test_features.csv")

    print("🚀 SVR CV - Recherche large")
    # Grille large (plus riche que la première version)
    c_large = [0.1, 0.3, 1, 3, 10, 30]
    eps_large = [0.01, 0.05, 0.1, 0.2]
    gamma_large = ["scale", "auto", 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]

    cv_detail_large, cv_summary_large, best_large_model = timeseries_cv_search(
        train_df, c_large, eps_large, gamma_large, n_splits=args.cv_splits
    )

    # Grille fine
    print("\nSVR CV - Recherche fine")
    best_c = best_large_model.c
    best_eps = best_large_model.epsilon
    best_gamma = best_large_model.gamma

    c_fine = np.logspace(np.log10(max(best_c, 1e-3)) - 0.5, np.log10(max(best_c, 1e-3)) + 0.5, 5).tolist()
    eps_fine = np.linspace(max(best_eps * 0.5, 1e-3), best_eps * 2, 5).tolist()

    def gamma_neighbors(gamma_value):
        return [gamma_value / 3, gamma_value, gamma_value * 3]

    if isinstance(best_gamma, str):
        # approx gamma numérique équivalent sur données standardisées (~1 / n_features)
        n_feats = len(best_large_model.preprocessor.feature_cols)
        gamma_eff = 1.0 / max(n_feats, 1)
        gamma_fine = [best_gamma] + gamma_neighbors(gamma_eff)
    else:
        gamma_fine = gamma_neighbors(best_gamma)

    cv_detail_fine, cv_summary_fine, final_model = timeseries_cv_search(
        train_df, c_fine, eps_fine, gamma_fine, n_splits=args.cv_splits
    )

    # Exports riches
    cv_detail_large.assign(search_stage="large").to_csv(args.out_dir / "svr_cv_detail_large.csv", index=False)
    cv_summary_large.to_csv(args.out_dir / "svr_cv_summary_large.csv", index=False)
    cv_detail_fine.assign(search_stage="fine").to_csv(args.out_dir / "svr_cv_detail_fine.csv", index=False)
    cv_summary_fine.to_csv(args.out_dir / "svr_cv_summary_fine.csv", index=False)

    valid_pred = predict(valid_df, final_model)
    test_pred = predict(test_df, final_model)

    valid_pred.to_csv(args.out_dir / "svr_cv_validation_predictions.csv", index=False)
    test_pred.to_csv(args.out_dir / "svr_cv_test_submission.csv", index=False)

    # Rapport synthétique
    wa_val = float(weighted_accuracy(valid_df[TARGET_COL], valid_pred[TARGET_COL]))
    rmse_val = float(np.sqrt(np.mean((valid_df[TARGET_COL] - valid_pred[TARGET_COL]) ** 2)))
    mae_val = float(np.mean(np.abs(valid_df[TARGET_COL] - valid_pred[TARGET_COL])))

    best_row = cv_summary_fine.iloc[0]
    md = f"""# Modèle SVR

Métrique prioritaire : Weighted Accuracy sur validation chronologique.

## Meilleurs hyperparamètres
- kernel = {final_model.kernel}
- C = {final_model.c}
- epsilon = {final_model.epsilon}
- gamma = {final_model.gamma}

## Validation
- WA = {wa_val:.6f}
- RMSE = {rmse_val:.6f}
- MAE = {mae_val:.6f}

## Recherche (CV temporelle)
- WA CV mean/std = {best_row['cv_valid_wa_mean']:.6f} ± {best_row['cv_valid_wa_std']:.6f}
- Gap train-valid WA = {best_row['cv_wa_gap_mean']:.6f} ± {best_row['cv_wa_gap_std']:.6f}
- Grille large puis fine autour du meilleur triplet
- Préprocessing identique au modèle de référence (imputation/standardisation sur train uniquement)

"""
    (args.out_dir / "svr_metrics.md").write_text(md)

    print("\n SVR CV terminé !")
    print(f"WA validation finale : {wa_val:.4f}")
    print(f"Gap train-valid (moyen CV) : {best_row['cv_wa_gap_mean']:.4f} ± {best_row['cv_wa_gap_std']:.4f}")
    print("Fichiers : svr_cv_* + svr_cv_test_submission.csv")


if __name__ == "__main__":
    main()
