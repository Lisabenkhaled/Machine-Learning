from __future__ import annotations

"""Modèle de référence régressif pour prédire `spot_id_delta`.

Ce module implémente une version compacte et explicable d'une Ridge regression:
- variables explicatives numériques uniquement;
- imputation des NaN par médiane (estimée sur train uniquement);
- standardisation (moyenne/écart-type du train uniquement);
- sélection d'alpha par grille selon la métrique imposée (WA), puis RMSE.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from train.io import TARGET_COL, TIME_COL


@dataclass
class Preprocessor:
    feature_cols: list[str]
    medians: np.ndarray
    means: np.ndarray
    stds: np.ndarray


@dataclass
class RidgeReferenceModel:
    alpha: float
    preprocessor: Preprocessor
    coef_: np.ndarray
    intercept_: float


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Conserve uniquement les colonnes numériques explicatives.

    On retire explicitement la cible `spot_id_delta`.
    `DELIVERY_START` n'est pas retenue car colonne datetime (non numérique).
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in numeric_cols if c != TARGET_COL]


def fit_preprocessor(df: pd.DataFrame, feature_cols: list[str]) -> Preprocessor:
    x = df[feature_cols].to_numpy(dtype=float)
    x[~np.isfinite(x)] = np.nan
    medians = np.nanmedian(x, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)

    x_imputed = np.where(np.isnan(x), medians, x)
    means = x_imputed.mean(axis=0)
    stds = x_imputed.std(axis=0)
    stds[stds == 0] = 1.0

    return Preprocessor(feature_cols=feature_cols, medians=medians, means=means, stds=stds)


def transform_features(df: pd.DataFrame, prep: Preprocessor) -> np.ndarray:
    x = df[prep.feature_cols].to_numpy(dtype=float)
    x[~np.isfinite(x)] = np.nan
    x = np.where(np.isnan(x), prep.medians, x)
    return (x - prep.means) / prep.stds


def fit_ridge(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    alphas: list[float],
    scorer,
) -> tuple[RidgeReferenceModel, pd.DataFrame]:
    """Ajuste plusieurs alpha de Ridge et choisit le meilleur selon la métrique imposée."""
    feature_cols = select_feature_columns(train_df)
    prep = fit_preprocessor(train_df, feature_cols)

    x_train = transform_features(train_df, prep)
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)

    x_valid = transform_features(valid_df, prep)
    y_valid = valid_df[TARGET_COL].to_numpy(dtype=float)

    n_features = x_train.shape[1]
    eye = np.eye(n_features, dtype=float)
    xtx = x_train.T @ x_train
    xty = x_train.T @ y_train

    rows: list[dict[str, float]] = []
    best: RidgeReferenceModel | None = None
    best_score = -np.inf
    best_rmse = np.inf

    for alpha in alphas:
        # Évite les instabilités numériques de OLS pur (alpha=0) sur jeux avec
        # colinéarité forte / colonnes quasi constantes.
        safe_alpha = max(float(alpha), 1e-8)
        beta = np.linalg.solve(xtx + safe_alpha * eye, xty)
        intercept = float(y_train.mean())

        y_pred = x_valid @ beta + intercept
        score = float(scorer(y_valid, y_pred))
        rmse = float(np.sqrt(np.mean((y_valid - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_valid - y_pred)))

        rows.append({"alpha": float(alpha), "weighted_accuracy": score, "rmse": rmse, "mae": mae})

        if (score > best_score) or (np.isclose(score, best_score) and rmse < best_rmse):
            best_score = score
            best_rmse = rmse
            best = RidgeReferenceModel(alpha=float(alpha), preprocessor=prep, coef_=beta, intercept_=intercept)

    if best is None:
        raise RuntimeError("Aucun modèle n'a pu être ajusté.")

    diagnostics = pd.DataFrame(rows).sort_values(["weighted_accuracy", "rmse"], ascending=[False, True])
    return best, diagnostics


def predict(df: pd.DataFrame, model: RidgeReferenceModel) -> pd.DataFrame:
    x = transform_features(df, model.preprocessor)
    centered_pred = x @ model.coef_
    y_pred = centered_pred + model.intercept_
    return pd.DataFrame({TIME_COL: df[TIME_COL], TARGET_COL: y_pred})
