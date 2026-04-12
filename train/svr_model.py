from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.svm import SVR

from train.io import TARGET_COL, TIME_COL
from train.reference_model import Preprocessor, fit_preprocessor, select_feature_columns


def safe_transform_features(df: pd.DataFrame, prep: Preprocessor) -> np.ndarray:
    """Transforme avec gestion des colonnes manquantes tout en conservant l'ordre attendu.

    On reconstruit explicitement la matrice de features dans l'ordre `prep.feature_cols`,
    en imputant par la médiane d'entraînement toute colonne absente dans `df` afin de
    garder la même dimension que celle utilisée pour l'apprentissage du modèle.
    """

    missing = [c for c in prep.feature_cols if c not in df.columns]
    if missing:
        print(f"⚠️ Colonnes manquantes dans df : {missing}")

    n_rows = len(df)
    n_features = len(prep.feature_cols)
    x = np.empty((n_rows, n_features), dtype=float)

    for idx, col in enumerate(prep.feature_cols):
        if col in df.columns:
            col_values = df[col].to_numpy(dtype=float)
            x[:, idx] = col_values
        else:
            x[:, idx] = np.nan

    x = np.where(np.isnan(x), prep.medians, x)
    x = (x - prep.means) / prep.stds
    return x


@dataclass
class SVRModel:
    c: float
    epsilon: float
    gamma: float | str
    kernel: str
    preprocessor: Preprocessor
    model: SVR


def fit_svr(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    c_values: list[float],
    epsilon_values: list[float],
    gamma_values: list[float | str],
    scorer,
    kernel: str = "rbf",
) -> tuple[SVRModel, pd.DataFrame]:
    feature_cols = select_feature_columns(train_df)
    prep = fit_preprocessor(train_df, feature_cols)

    x_train = safe_transform_features(train_df, prep)
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)

    # Cas d'entraînement final sans validation: une seule combinaison attendue.
    skip_validation = valid_df is None or valid_df.empty
    if skip_validation:
        if len(c_values) != 1 or len(epsilon_values) != 1 or len(gamma_values) != 1:
            raise ValueError("valid_df est vide mais plusieurs combinaisons d'hyperparamètres sont fournies")

        c = float(c_values[0])
        epsilon = float(epsilon_values[0])
        gamma = gamma_values[0]

        model = SVR(kernel=kernel, C=c, epsilon=epsilon, gamma=gamma)
        model.fit(x_train, y_train)

        # Diagnostics simples sur le train pour suivi.
        y_pred_train = model.predict(x_train)
        train_score = float(scorer(y_train, y_pred_train))
        train_rmse = float(np.sqrt(np.mean((y_train - y_pred_train) ** 2)))
        train_mae = float(np.mean(np.abs(y_train - y_pred_train)))

        diagnostics = pd.DataFrame(
            [
                {
                    "kernel": kernel,
                    "C": c,
                    "epsilon": epsilon,
                    "gamma": gamma,
                    "weighted_accuracy": train_score,
                    "rmse": train_rmse,
                    "mae": train_mae,
                    "dataset": "train",
                }
            ]
        )

        best_model = SVRModel(
            c=c,
            epsilon=epsilon,
            gamma=gamma,
            kernel=kernel,
            preprocessor=prep,
            model=model,
        )
        return best_model, diagnostics

    x_valid = safe_transform_features(valid_df, prep)
    y_valid = valid_df[TARGET_COL].to_numpy(dtype=float)

    rows: list[dict[str, float | str]] = []
    best_model: SVRModel | None = None
    best_score = -np.inf
    best_rmse = np.inf

    for c in c_values:
        for epsilon in epsilon_values:
            for gamma in gamma_values:
                model = SVR(
                    kernel=kernel,
                    C=float(c),
                    epsilon=float(epsilon),
                    gamma=gamma,
                )
                model.fit(x_train, y_train)
                y_pred = model.predict(x_valid)

                score = float(scorer(y_valid, y_pred))
                rmse = float(np.sqrt(np.mean((y_valid - y_pred) ** 2)))
                mae = float(np.mean(np.abs(y_valid - y_pred)))

                rows.append(
                    {
                        "kernel": kernel,
                        "C": float(c),
                        "epsilon": float(epsilon),
                        "gamma": gamma,
                        "weighted_accuracy": score,
                        "rmse": rmse,
                        "mae": mae,
                    }
                )

                if score > best_score or (np.isclose(score, best_score) and rmse < best_rmse):
                    best_score = score
                    best_rmse = rmse
                    best_model = SVRModel(
                        c=float(c),
                        epsilon=float(epsilon),
                        gamma=gamma,
                        kernel=kernel,
                        preprocessor=prep,
                        model=model,
                    )

    if best_model is None:
        raise RuntimeError("Aucun modèle SVR n'a pu être ajusté.")

    diagnostics = pd.DataFrame(rows).sort_values(
        ["weighted_accuracy", "rmse"],
        ascending=[False, True],
    )
    return best_model, diagnostics


def predict(df: pd.DataFrame, model: SVRModel) -> pd.DataFrame:
    x = safe_transform_features(df, model.preprocessor)
    y_pred = model.model.predict(x)
    return pd.DataFrame(
        {
            TIME_COL: df[TIME_COL],
            TARGET_COL: y_pred,
        }
    )
