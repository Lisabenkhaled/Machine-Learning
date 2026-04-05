#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train.benchmark import always_positive_baseline
from train.describe import build_feature_stats, write_markdown_report
from train.features import build_features
from train.io import TARGET_COL, TIME_COL, load_raw_data, merge_train_xy
from train.metrics import weighted_accuracy
from train.split import chronological_train_valid_split


EXPECTED_TRAIN_START = "2022-01-01 01:00:00"  # 2022-01-01 02:00:00+01:00 en UTC
EXPECTED_TRAIN_END = "2023-03-29 21:00:00"    # 2023-03-29 23:00:00+02:00 en UTC
EXPECTED_TEST_START = "2023-04-01 22:00:00"   # 2023-04-02 00:00:00+02:00 en UTC
EXPECTED_TEST_END = "2023-10-24 21:00:00"     # 2023-10-24 23:00:00+02:00 en UTC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prépare les données et décrit les features.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dossier des CSV sources")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"), help="Dossier de sortie")
    parser.add_argument("--valid-fraction", type=float, default=0.2, help="Fraction de validation chronologique")
    return parser.parse_args()


def _check_date_ranges(train_df, test_df) -> str:
    t0, t1 = train_df[TIME_COL].min(), train_df[TIME_COL].max()
    s0, s1 = test_df[TIME_COL].min(), test_df[TIME_COL].max()

    expected_lines = [
        f"Train observé : {t0} -> {t1}",
        f"Train attendu : {EXPECTED_TRAIN_START} -> {EXPECTED_TRAIN_END} (UTC)",
        f"Test observé  : {s0} -> {s1}",
        f"Test attendu  : {EXPECTED_TEST_START} -> {EXPECTED_TEST_END} (UTC)",
    ]

    if s0 <= t1:
        raise ValueError("Le test commence avant la fin du train: ordre temporel invalide")

    return "\n".join(expected_lines)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data(args.data_dir)

    train_full = merge_train_xy(raw.x_train, raw.y_train)
    train_full = build_features(train_full)
    test_full = build_features(raw.x_test)

    split = chronological_train_valid_split(train_full, valid_fraction=args.valid_fraction)

    split.train_df.to_csv(args.out_dir / "train_estimation.csv", index=False)
    split.valid_df.to_csv(args.out_dir / "train_validation.csv", index=False)
    test_full.to_csv(args.out_dir / "test_features.csv", index=False)

    stats = build_feature_stats(split.train_df)
    write_markdown_report(stats, args.out_dir / "feature_description.md")

    # Baseline demandé par l'énoncé: delta toujours positif.
    baseline_valid = always_positive_baseline(split.valid_df)
    wa = weighted_accuracy(split.valid_df[TARGET_COL].to_numpy(), baseline_valid[TARGET_COL].to_numpy())
    y_true = split.valid_df[TARGET_COL].to_numpy()
    y_pred = baseline_valid[TARGET_COL].to_numpy()
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    baseline_valid.to_csv(args.out_dir / "baseline_valid_predictions.csv", index=False)
    baseline_test = always_positive_baseline(test_full)
    baseline_test.to_csv(args.out_dir / "baseline_test_submission.csv", index=False)

    range_report = _check_date_ranges(train_full, test_full)

    metrics_text = (
        "# Baseline: always positive\n\n"
        f"Validation WA: {wa:.6f}\n\n"
        f"Validation RMSE: {rmse:.6f}\n\n"
        f"Validation MAE: {mae:.6f}\n\n"
        "## Contrôle des plages temporelles\n\n"
        f"{range_report}\n"
    )
    (args.out_dir / "baseline_metrics.md").write_text(metrics_text, encoding="utf-8")

    print("Fichiers générés :")
    print(f"- {args.out_dir / 'train_estimation.csv'}")
    print(f"- {args.out_dir / 'train_validation.csv'}")
    print(f"- {args.out_dir / 'test_features.csv'}")
    print(f"- {args.out_dir / 'feature_description.md'}")
    print(f"- {args.out_dir / 'baseline_valid_predictions.csv'}")
    print(f"- {args.out_dir / 'baseline_test_submission.csv'}")
    print(f"- {args.out_dir / 'baseline_metrics.md'}")


if __name__ == "__main__":
    main()
