# Guide de cadrage — workflow Machine Learning attendu 

1. `X_train` + `y_train` sont fusionnés sur `DELIVERY_START`.
2. Le train complet est scindé en:
   - `train_estimation.csv` (apprentissage)
   - `train_validation.csv` (validation)
3. `X_test` est conservé à part (`test_features.csv`).

## 1) Rôle des fichiers `data/processed`

- `train_estimation.csv`: échantillon d'estimation (train interne).
- `train_validation.csv`: échantillon de validation temporelle.
- `test_features.csv`: features pour le test final (pas de cible).
- `feature_description.md`: stats descriptives des variables.
- `baseline_valid_predictions.csv` / `baseline_test_submission.csv`: benchmark naïf (delta positif).
- `baseline_metrics.md`: métriques du benchmark (WA/RMSE/MAE) + contrôle temporel.

## 2) Enoncé

1. **Split train/test**: déjà fait, en gardant l'ordre temporel.
2. **Split train/validation**: déjà fait (chronologique).
3. **Features**: déjà faites (calendaires + sin/cos + lags 1h/24h + stats descriptives).
4. **Modèle simple de référence**: à faire via logistique (classification du signe) ou OLS (régression).
5. **Au moins un non supervisé**: KMeans avec mapping cluster -> signe majoritaire + comparaison WA.
6. **Au moins un supervisé**: SVM et méthode ensembliste (Random Forest) + grid search + validation croisée temporelle.
7. **Interprétation**: importance des variables (permutation importance, puis SHAP/LIME si dispo).
8. **(Optionnel) deep learning**: non obligatoire.
9. **Comparaison finale**: tableau des performances et discussion.

## Étape 4 — Modèle de référence 
Un modèle de référence de **régression Ridge** a été ajouté:

- Pipeline: sélection des features numériques, imputation médiane, standardisation, régression Ridge.
- Paramètres de référence figés pour la reproductibilité (ex. `alpha=1.0`).
- Critère principal: **Weighted Accuracy (WA)** sur l'échantillon de validation.
- Calibration de signe figée (ex. `positive_clip`) pour une baseline stable dans le temps.
- Sorties prévues:
  - `reference_model_metrics.md`
  - `ridge_alpha_grid.csv`
  - `ridge_valid_predictions.csv`
  - `ridge_test_submission.csv`

Pourquoi ce choix:
- plus robuste qu'une OLS pure en présence de colinéarité;
- reste interprétable et défendable académiquement;
