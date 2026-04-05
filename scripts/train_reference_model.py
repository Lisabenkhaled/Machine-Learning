#!/usr/bin/env python3
from __future__ import annotations

"""Script d'entraînement/évaluation du modèle de référence (étape 4).

Entrées attendues (générées par `scripts/prepare_features.py`):
- data/processed/train_estimation.csv
- data/processed/train_validation.csv
- data/processed/test_features.csv

Sorties:
- reference_model_metrics.md (métriques + alpha retenu + grille)
- ridge_alpha_grid.csv
- ridge_valid_predictions.csv
- ridge_test_submission.csv
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
from train.metrics import weighted_accuracy
from train.reference_model import fit_ridge, predict


def apply_calibration(y_pred: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "positive_clip":
        return np.where(y_pred <= 0, 0.1, y_pred)
    if strategy == "always_positive_1":
        return np.full_like(y_pred, 1.0)
    return y_pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraîne un modèle de référence Ridge pour spot_id_delta.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Dossier des données préparées")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"), help="Dossier des sorties")
    parser.add_argument("--reference-alpha", type=float, default=1.0, help="Alpha fixe du modèle de référence")
    parser.add_argument(
        "--calibration",
        choices=["raw", "positive_clip", "always_positive_1"],
        default="positive_clip",
        help="Calibration fixe du modèle de référence",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.data_dir / "train_estimation.csv")
    valid_df = pd.read_csv(args.data_dir / "train_validation.csv")
    test_df = pd.read_csv(args.data_dir / "test_features.csv")

    model, diagnostics = fit_ridge(
        train_df,
        valid_df,
        alphas=[args.reference_alpha],
        scorer=weighted_accuracy,
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
    diagnostics["calibration"] = args.calibration
    diagnostics["calibrated_weighted_accuracy"] = wa
    diagnostics["calibrated_rmse"] = rmse
    diagnostics["calibrated_mae"] = mae

    diagnostics.to_csv(args.out_dir / "ridge_alpha_grid.csv", index=False)
    valid_pred.to_csv(args.out_dir / "ridge_valid_predictions.csv", index=False)
    test_pred.to_csv(args.out_dir / "ridge_test_submission.csv", index=False)

    diagnostics_md = diagnostics.to_csv(index=False)

    report = (
        "# Modèle de référence : Ridge Regression\n\n"
        "Métrique prioritaire: Weighted Accuracy (WA) sur l'échantillon de validation chronologique.\n\n"
        f"Alpha retenu: {model.alpha:.2f}\n\n"
        f"Validation WA: {wa:.6f}\n\n"
        f"Validation RMSE: {rmse:.6f}\n\n"
        f"Validation MAE: {mae:.6f}\n\n"
        f"Calibration WA retenue: {args.calibration}\n\n"
        "## Grille d'hyperparamètres testée\n\n"
        "```csv\n"
        f"{diagnostics_md}"
        "```\n"
    )
    (args.out_dir / "reference_model_metrics.md").write_text(report, encoding="utf-8")

    print("Sorties générées :")
    print(f"- {args.out_dir / 'reference_model_metrics.md'}")
    print(f"- {args.out_dir / 'ridge_alpha_grid.csv'}")
    print(f"- {args.out_dir / 'ridge_valid_predictions.csv'}")
    print(f"- {args.out_dir / 'ridge_test_submission.csv'}")


if __name__ == "__main__":
    main()
