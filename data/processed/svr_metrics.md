# Modèle SVR

Métrique prioritaire : Weighted Accuracy sur validation chronologique.

## Meilleurs hyperparamètres
- kernel = rbf
- C = 10.0
- epsilon = 0.01625
- gamma = 0.1

## Validation
- WA = 0.635879
- RMSE = 17.052754
- MAE = 11.416306

## Recherche (CV temporelle)
- WA CV mean/std = 0.532538 ± 0.051666
- Gap train-valid WA = 0.371724 ± 0.055708
- Grille large puis fine autour du meilleur triplet
- Préprocessing identique au modèle de référence (imputation/standardisation sur train uniquement)

