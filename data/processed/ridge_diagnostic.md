# Diagnostic du modèle de référence Ridge

## 1) Vérification look-ahead bias (fuite temporelle)

- Fin du train estimation : `2022-12-30 12:00:00`
- Début de la validation : `2022-12-30 13:00:00`
- **✅  Pas de fuite temporelle**

## 2) Distribution des prédictions brutes (sans contrainte)

- Prédictions positives : 409 (19.3%)
- Prédictions négatives : 1712 (80.7%)

## 3) Comparaison des métriques de validation

| Modèle | WA | RMSE | MAE |
|---|---|---|---|
| Baseline naïve (always positive) | 0.658359 | — | — |
| Ridge brut (sans contrainte) | 0.466138 | 18.027602 | 12.672909 |
| Ridge + positive_clip | 0.658359 | 16.783985 | 11.300921 |

## 4) Interprétation

⚠️  Le Ridge brut obtient une WA de 0.4661, en dessous de la baseline naïve (0.6584). La contrainte positive_clip récupère 0.1922 points. Cela suggère que le modèle Ridge n'apprend pas de signal utile au-delà du biais haussier du marché. Envisager : enrichir les features, élargir la grille alpha, ou reconsidérer la formulation du problème.

## 5) Conclusion sur le modèle de référence

- ✅ Pas de fuite temporelle : le split train/validation est correct.
- ⚠️  Le Ridge sans contrainte (0.4661) ne dépasse pas la baseline naïve (0.6584) : la WA observée précédemment était portée par la contrainte de positivité, pas par le modèle.