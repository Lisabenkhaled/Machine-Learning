# Explication détaillée — `reference_model.py` et `train_reference_model.py`

## Variables du projet
- Index: `DELIVERY_START` (date/heure de livraison).
- Cible: `spot_id_delta = Intraday - SPOT`.
- Variables explicatives candidates:
  - `load_forecast`
  - `coal_power_available`, `gas_power_available`, `nucelear_power_available` (corrigée en `nuclear_power_available` en amont dans le pipeline features)
  - `wind_power_forecasts_average`, `solar_power_forecasts_average`
  - `wind_power_forecasts_std`, `solar_power_forecasts_std`
  - `predicted_spot_price`
  - + variables calendaires et lags déjà créés à l'étape features.

## Ce que fait exactement `train/reference_model.py`

### 1) `select_feature_columns(df)`
- Prend uniquement les colonnes **numériques**.
- Exclut explicitement la cible `spot_id_delta`.
- `DELIVERY_START` n'entre pas dans la régression (datetime).

### 2) `fit_preprocessor(df, feature_cols)`
- Calcule les statistiques de prétraitement **sur train estimation uniquement**:
  - médiane par feature (imputation des NaN);
  - moyenne et écart-type par feature (standardisation).
- Les valeurs non finies (`inf`, `-inf`) sont converties en `NaN`.
- Si une colonne est entièrement manquante, sa médiane est forcée à `0.0` (évite les erreurs numériques).
- Remplace les écarts-types nuls par 1 pour éviter la division par zéro.

### 3) `transform_features(df, prep)`
- Applique les stats du train:
  - NaN -> médiane train;
  - standardisation `(x - mean_train) / std_train`.

### 4) `fit_ridge(train_df, valid_df, alphas, scorer)`
- Entraîne un modèle pour chaque `alpha` de la grille.
- Utilise une résolution Ridge stable `(X'X + alpha I)^(-1)X'y` avec un plancher numérique `alpha >= 1e-8` pour éviter les erreurs de convergence de type SVD/OLS.
- Intercept fixé à la moyenne de `y_train` (cohérent avec X standardisé).
- Évalue sur validation:
  - `weighted_accuracy` (métrique principale),
  - RMSE,
  - MAE.
- Sélectionne le meilleur alpha:
  1. WA max,
  2. en cas d'égalité: RMSE min.
- Retourne:
  - le meilleur modèle,
  - un tableau diagnostics pour la grille.

### 5) `predict(df, model)`
- Applique le même prétraitement (médianes/moyennes/std train).
- Produit `y_pred = X_standardisé @ beta + intercept`.
- Renvoie un DataFrame avec:
  - `DELIVERY_START`
  - `spot_id_delta` prédit.

## Ce que fait exactement `scripts/train_reference_model.py`
1. Charge:
   - `train_estimation.csv`
   - `train_validation.csv`
   - `test_features.csv`
2. Entraîne **un alpha fixe de référence** (`--reference-alpha`, défaut `1.0`).
3. Génère les prédictions validation/test (`predict`).
4. Applique **une calibration fixe de référence** (`--calibration`, défaut `positive_clip`).
5. Calcule les métriques validation: WA, RMSE, MAE.
6. Écrit les sorties:
   - `ridge_alpha_grid.csv`
   - `ridge_valid_predictions.csv`
   - `ridge_test_submission.csv`
   - `reference_model_metrics.md`

