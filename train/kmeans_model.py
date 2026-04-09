from __future__ import annotations

"""Modèle KMeans non supervisé avec mapping cluster -> cible moyenne.

Principe:
- variables explicatives numériques uniquement ;
- imputation médiane + standardisation estimées sur train uniquement ;
- KMeans (distance euclidienne) sur train ;
- pour chaque cluster, on associe la moyenne de `spot_id_delta` observée sur train ;
- sélection de k selon WA sur validation, puis RMSE.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from train.io import TARGET_COL, TIME_COL
from train.reference_model import Preprocessor, fit_preprocessor, select_feature_columns, transform_features


@dataclass
class KMeansClusterModel:
    n_clusters: int
    init: str
    n_init: int
    random_state: int
    preprocessor: Preprocessor
    kmeans: KMeans
    cluster_values_: np.ndarray
    global_mean_: float
    inertia_: float


def _compute_cluster_values(labels: np.ndarray, y: np.ndarray, n_clusters: int) -> tuple[np.ndarray, float]:
    global_mean = float(np.mean(y))
    cluster_values = np.full(n_clusters, global_mean, dtype=float)
    for c in range(n_clusters):
        mask = labels == c
        if np.any(mask):
            cluster_values[c] = float(np.mean(y[mask]))
    return cluster_values, global_mean


def fit_kmeans_regression(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    ks: list[int],
    scorer,
    init: str = "k-means++",
    n_init: int = 20,
    random_state: int = 42,
) -> tuple[KMeansClusterModel, pd.DataFrame]:
    feature_cols = select_feature_columns(train_df)
    prep = fit_preprocessor(train_df, feature_cols)

    x_train = transform_features(train_df, prep)
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)

    x_valid = transform_features(valid_df, prep)
    y_valid = valid_df[TARGET_COL].to_numpy(dtype=float)

    rows: list[dict[str, float]] = []
    best: KMeansClusterModel | None = None
    best_score = -np.inf
    best_rmse = np.inf

    for k in ks:
        km = KMeans(
            n_clusters=int(k),
            init=init,
            n_init=int(n_init),
            random_state=int(random_state),
            algorithm="lloyd",
        )
        train_labels = km.fit_predict(x_train)
        cluster_values, global_mean = _compute_cluster_values(train_labels, y_train, int(k))

        valid_labels = km.predict(x_valid)
        y_pred = cluster_values[valid_labels]

        score = float(scorer(y_valid, y_pred))
        rmse = float(np.sqrt(np.mean((y_valid - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_valid - y_pred)))
        rows.append(
            {
                "k": int(k),
                "weighted_accuracy": score,
                "rmse": rmse,
                "mae": mae,
                "inertia": float(km.inertia_),
            }
        )

        if (score > best_score) or (np.isclose(score, best_score) and rmse < best_rmse):
            best_score = score
            best_rmse = rmse
            best = KMeansClusterModel(
                n_clusters=int(k),
                init=init,
                n_init=int(n_init),
                random_state=int(random_state),
                preprocessor=prep,
                kmeans=km,
                cluster_values_=cluster_values,
                global_mean_=global_mean,
                inertia_=float(km.inertia_),
            )

    if best is None:
        raise RuntimeError("Aucun modèle KMeans n'a pu être ajusté.")

    diagnostics = pd.DataFrame(rows).sort_values(["weighted_accuracy", "rmse"], ascending=[False, True])
    return best, diagnostics


def predict(df: pd.DataFrame, model: KMeansClusterModel) -> pd.DataFrame:
    x = transform_features(df, model.preprocessor)
    labels = model.kmeans.predict(x)
    y_pred = model.cluster_values_[labels]
    return pd.DataFrame({TIME_COL: df[TIME_COL], TARGET_COL: y_pred})