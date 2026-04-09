#!/usr/bin/env python3
from __future__ import annotations

"""Script d'entraînement/évaluation d'un KMeans non supervisé.

Entrées attendues:
- data/processed/train_estimation.csv
- data/processed/train_validation.csv
- data/processed/test_features.csv

Sorties:
- kmeans_k_grid.csv
- kmeans_valid_predictions.csv
- kmeans_test_submission.csv
- kmeans_metrics.md
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train.io import TARGET_COL, TIME_COL
from train.kmeans_model import fit_kmeans_regression, predict
from train.metrics import weighted_accuracy


def apply_calibration(y_pred: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "positive_clip":
        return np.where(y_pred <= 0, 0.1, y_pred)
    if strategy == "always_positive_1":
        return np.full_like(y_pred, 1.0)
    return y_pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraîne un KMeans avec mapping cluster -> moyenne de spot_id_delta.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Dossier des données préparées")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"), help="Dossier des sorties")
    parser.add_argument("--k-min", type=int, default=2, help="Nombre minimal de clusters testé")
    parser.add_argument("--k-max", type=int, default=20, help="Nombre maximal de clusters testé")
    parser.add_argument("--init", choices=["k-means++", "random"], default="k-means++", help="Initialisation KMeans")
    parser.add_argument("--n-init", type=int, default=20, help="Nombre de redémarrages KMeans")
    parser.add_argument("--random-state", type=int, default=42, help="Graine aléatoire")
    parser.add_argument(
        "--calibration",
        choices=["raw", "positive_clip", "always_positive_1"],
        default="raw",
        help="Calibration finale appliquée aux prédictions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.data_dir / "train_estimation.csv")
    valid_df = pd.read_csv(args.data_dir / "train_validation.csv")
    test_df = pd.read_csv(args.data_dir / "test_features.csv")

    ks = list(range(args.k_min, args.k_max + 1))
    model, diagnostics = fit_kmeans_regression(
        train_df,
        valid_df,
        ks=ks,
        scorer=weighted_accuracy,
        init=args.init,
        n_init=args.n_init,
        random_state=args.random_state,
    )

    valid_pred = predict(valid_df, model)
    test_pred = predict(test_df, model)

    y_valid = valid_df[TARGET_COL].to_numpy(dtype=float)
    y_hat_valid = valid_pred[TARGET_COL].to_numpy(dtype=float)
    y_hat_test = test_pred[TARGET_COL].to_numpy(dtype=float)

    y_hat_valid_cal = apply_calibration(y_hat_valid, args.calibration)
    y_hat_test_cal = apply_calibration(y_hat_test, args.calibration)

    valid_pred = valid_df[[TIME_COL]].copy()
    valid_pred[TARGET_COL] = y_hat_valid_cal
    test_pred[TARGET_COL] = y_hat_test_cal

    wa = float(weighted_accuracy(y_valid, y_hat_valid_cal))
    rmse = float(np.sqrt(np.mean((y_valid - y_hat_valid_cal) ** 2)))
    mae = float(np.mean(np.abs(y_valid - y_hat_valid_cal)))

    diagnostics = diagnostics.rename(
        columns={
            "weighted_accuracy": "raw_weighted_accuracy",
            "rmse": "raw_rmse",
            "mae": "raw_mae",
        }
    )

    diagnostics["init"] = args.init
    diagnostics["n_init"] = args.n_init
    diagnostics["random_state"] = args.random_state
    diagnostics["calibration"] = args.calibration
    diagnostics["selected_k"] = model.n_clusters
    diagnostics["selected_model_inertia"] = model.inertia_
    diagnostics["calibrated_weighted_accuracy"] = wa
    diagnostics["calibrated_rmse"] = rmse
    diagnostics["calibrated_mae"] = mae

    diagnostics.to_csv(args.out_dir / "kmeans_k_grid.csv", index=False)
    valid_pred.to_csv(args.out_dir / "kmeans_valid_predictions.csv", index=False)
    test_pred.to_csv(args.out_dir / "kmeans_test_submission.csv", index=False)

    diagnostics_md = diagnostics.to_csv(index=False)
    report = (
        "# Modèle non supervisé : KMeans avec mapping cluster -> moyenne de la cible\n\n"
        "Clustering KMeans sur les variables numériques standardisées, puis affectation à chaque cluster de la moyenne de `spot_id_delta` observée sur l'échantillon d'estimation.\n\n"
        f"k retenu: {model.n_clusters}\n\n"
        f"Initialisation: {args.init}\n\n"
        f"n_init: {args.n_init}\n\n"
        f"Inertie du modèle retenu: {model.inertia_:.6f}\n\n"
        f"Validation WA: {wa:.6f}\n\n"
        f"Validation RMSE: {rmse:.6f}\n\n"
        f"Validation MAE: {mae:.6f}\n\n"
        f"Calibration retenue: {args.calibration}\n\n"
        "## Grille testée\n\n"
        "```csv\n"
        f"{diagnostics_md}"
        "```\n"
    )
    (args.out_dir / "kmeans_metrics.md").write_text(report, encoding="utf-8")

    print("Sorties générées :")
    print(f"- {args.out_dir / 'kmeans_metrics.md'}")
    print(f"- {args.out_dir / 'kmeans_k_grid.csv'}")
    print(f"- {args.out_dir / 'kmeans_valid_predictions.csv'}")
    print(f"- {args.out_dir / 'kmeans_test_submission.csv'}")


if __name__ == "__main__":
    main()