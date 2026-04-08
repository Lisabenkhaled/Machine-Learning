#!/usr/bin/env python3
from __future__ import annotations

"""Diagnostic du modèle de référence Ridge.

Ce script répond à deux questions :
1. La WA de 0.658 vient-elle du modèle lui-même ou uniquement de la contrainte
   de positivité (positive_clip) ?
2. Y a-t-il un look-ahead bias (fuite temporelle) dans le pipeline ?

Entrées attendues :
- data/processed/train_estimation.csv
- data/processed/train_validation.csv

Sortie :
- data/processed/ridge_diagnostic.md
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_positive_clip(y: np.ndarray) -> np.ndarray:
    return np.where(y <= 0, 0.1, y)


def sign_stats(y_pred: np.ndarray) -> dict:
    """Statistiques sur les signes des prédictions brutes."""
    n_pos = int((y_pred > 0).sum())
    n_neg = int((y_pred <= 0).sum())
    return {
        "n_positif": n_pos,
        "n_negatif": n_neg,
        "pct_positif": 100 * n_pos / len(y_pred),
        "pct_negatif": 100 * n_neg / len(y_pred),
    }


def leakage_check(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict:
    """Vérifie l'ordre temporel strict entre train et validation."""
    t_train_max = pd.to_datetime(train_df[TIME_COL]).max()
    t_valid_min = pd.to_datetime(valid_df[TIME_COL]).min()
    overlap = t_valid_min <= t_train_max
    return {
        "train_max": str(t_train_max),
        "valid_min": str(t_valid_min),
        "overlap_detecte": overlap,
        "statut": "⚠️  FUITE TEMPORELLE DÉTECTÉE" if overlap else "✅  Pas de fuite temporelle",
    }


def naive_always_positive_wa(y_true: np.ndarray) -> float:
    """WA de la baseline naïve always-positive (pour comparaison)."""
    y_naive = np.ones_like(y_true)
    return float(weighted_accuracy(y_true, y_naive))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnostic Ridge : raw vs calibré + vérification look-ahead.")
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--alpha", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.data_dir / "train_estimation.csv")
    valid_df = pd.read_csv(args.data_dir / "train_validation.csv")

    y_valid = valid_df[TARGET_COL].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # 1) Vérification look-ahead bias
    # ------------------------------------------------------------------
    leak = leakage_check(train_df, valid_df)

    # ------------------------------------------------------------------
    # 2) Entraînement Ridge (alpha fixe)
    # ------------------------------------------------------------------
    model, _ = fit_ridge(
        train_df,
        valid_df,
        alphas=[args.alpha],
        scorer=weighted_accuracy,
    )
    valid_pred_df = predict(valid_df, model)
    y_raw = valid_pred_df[TARGET_COL].to_numpy(dtype=float)
    y_clipped = apply_positive_clip(y_raw)

    # ------------------------------------------------------------------
    # 3) Métriques RAW (sans contrainte)
    # ------------------------------------------------------------------
    wa_raw = float(weighted_accuracy(y_valid, y_raw))
    rmse_raw = float(np.sqrt(np.mean((y_valid - y_raw) ** 2)))
    mae_raw = float(np.mean(np.abs(y_valid - y_raw)))
    signs_raw = sign_stats(y_raw)

    # ------------------------------------------------------------------
    # 4) Métriques CLIPPED (avec contrainte positive_clip)
    # ------------------------------------------------------------------
    wa_clip = float(weighted_accuracy(y_valid, y_clipped))
    rmse_clip = float(np.sqrt(np.mean((y_valid - y_clipped) ** 2)))
    mae_clip = float(np.mean(np.abs(y_valid - y_clipped)))

    # ------------------------------------------------------------------
    # 5) Baseline naïve always-positive (plancher de comparaison)
    # ------------------------------------------------------------------
    wa_naive = naive_always_positive_wa(y_valid)

    # ------------------------------------------------------------------
    # 6) Interprétation automatique
    # ------------------------------------------------------------------
    delta_wa = wa_clip - wa_raw

    if wa_raw >= wa_clip - 0.005:
        interpretation = (
            "Le modèle Ridge génère une WA similaire avec ou sans contrainte. "
            "La contrainte positive_clip n'est pas la source principale de la performance : "
            "le modèle apprend effectivement quelque chose."
        )
    elif wa_raw >= wa_naive - 0.01:
        interpretation = (
            f"Le Ridge brut obtient une WA ({wa_raw:.4f}) proche de la baseline naïve ({wa_naive:.4f}). "
            "La contrainte positive_clip améliore la WA de {delta_wa:.4f} points. "
            "Le modèle prédit correctement la tendance haussière générale mais n'ajoute "
            "pas encore beaucoup de signal au-delà de la baseline."
        )
    else:
        interpretation = (
            f"⚠️  Le Ridge brut obtient une WA de {wa_raw:.4f}, en dessous de la baseline naïve ({wa_naive:.4f}). "
            f"La contrainte positive_clip récupère {delta_wa:.4f} points. "
            "Cela suggère que le modèle Ridge n'apprend pas de signal utile au-delà du biais "
            "haussier du marché. Envisager : enrichir les features, élargir la grille alpha, "
            "ou reconsidérer la formulation du problème."
        )

    # ------------------------------------------------------------------
    # 7) Rapport markdown
    # ------------------------------------------------------------------
    lines = [
        "# Diagnostic du modèle de référence Ridge\n",
        "## 1) Vérification look-ahead bias (fuite temporelle)\n",
        f"- Fin du train estimation : `{leak['train_max']}`",
        f"- Début de la validation : `{leak['valid_min']}`",
        f"- **{leak['statut']}**\n",
        "## 2) Distribution des prédictions brutes (sans contrainte)\n",
        f"- Prédictions positives : {signs_raw['n_positif']} ({signs_raw['pct_positif']:.1f}%)",
        f"- Prédictions négatives : {signs_raw['n_negatif']} ({signs_raw['pct_negatif']:.1f}%)\n",
        "## 3) Comparaison des métriques de validation\n",
        "| Modèle | WA | RMSE | MAE |",
        "|---|---|---|---|",
        f"| Baseline naïve (always positive) | {wa_naive:.6f} | — | — |",
        f"| Ridge brut (sans contrainte) | {wa_raw:.6f} | {rmse_raw:.6f} | {mae_raw:.6f} |",
        f"| Ridge + positive_clip | {wa_clip:.6f} | {rmse_clip:.6f} | {mae_clip:.6f} |\n",
        "## 4) Interprétation\n",
        interpretation,
        "\n## 5) Conclusion sur le modèle de référence\n",
    ]

    # Conclusion sur ce qu'il faut retenir
    if not leak["overlap_detecte"]:
        lines.append("- ✅ Pas de fuite temporelle : le split train/validation est correct.")
    else:
        lines.append("- ⚠️  Fuite temporelle détectée : revoir le split.")

    if wa_raw > wa_naive:
        lines.append(
            f"- ✅ Le Ridge sans contrainte ({wa_raw:.4f}) dépasse la baseline naïve ({wa_naive:.4f}) : "
            "le modèle apporte du signal réel."
        )
    else:
        lines.append(
            f"- ⚠️  Le Ridge sans contrainte ({wa_raw:.4f}) ne dépasse pas la baseline naïve ({wa_naive:.4f}) : "
            "la WA observée précédemment était portée par la contrainte de positivité, pas par le modèle."
        )

    report = "\n".join(lines)
    out_path = args.out_dir / "ridge_diagnostic.md"
    out_path.write_text(report, encoding="utf-8")

    # Affichage console
    print("=" * 60)
    print("DIAGNOSTIC RIDGE")
    print("=" * 60)
    print(f"Look-ahead bias : {leak['statut']}")
    print(f"Prédictions positives brutes : {signs_raw['pct_positif']:.1f}%")
    print()
    print(f"{'Modèle':<35} {'WA':>10} {'RMSE':>10} {'MAE':>10}")
    print("-" * 65)
    print(f"{'Baseline naïve (always positive)':<35} {wa_naive:>10.6f} {'—':>10} {'—':>10}")
    print(f"{'Ridge brut (sans contrainte)':<35} {wa_raw:>10.6f} {rmse_raw:>10.6f} {mae_raw:>10.6f}")
    print(f"{'Ridge + positive_clip':<35} {wa_clip:>10.6f} {rmse_clip:>10.6f} {mae_clip:>10.6f}")
    print()
    print("Interprétation :")
    print(interpretation)
    print()
    print(f"Rapport écrit : {out_path}")


if __name__ == "__main__":
    main()
