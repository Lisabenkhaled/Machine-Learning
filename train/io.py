from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TIME_COL = "DELIVERY_START"
TARGET_COL = "spot_id_delta"


@dataclass
class RawData:
    x_train: pd.DataFrame
    y_train: pd.DataFrame
    x_test: pd.DataFrame


def _robust_read_csv(path: Path) -> pd.DataFrame:
    """Lecture robuste pour CSV séparés par `,` ou `;` avec colonnes parfois mal formatées."""
    # sep=None + engine=python détecte automatiquement , ; \t ...
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    # Cas pathologique: tout est lu dans une seule colonne texte.
    if df.shape[1] == 1 and "," in df.columns[0]:
        df = pd.read_csv(path, sep=",", engine="python")
        df.columns = [str(c).strip() for c in df.columns]

    return df


def load_raw_data(data_dir: Path) -> RawData:
    """Charge les 3 fichiers utiles du challenge depuis `data/`."""
    x_train = _robust_read_csv(data_dir / "X_train.csv")
    y_train = _robust_read_csv(data_dir / "y_train.csv")
    x_test = _robust_read_csv(data_dir / "X_test.csv")
    return RawData(x_train=x_train, y_train=y_train, x_test=x_test)


def merge_train_xy(x_train: pd.DataFrame, y_train: pd.DataFrame) -> pd.DataFrame:
    """Jointure de référence demandée dans l'énoncé."""
    return x_train.merge(y_train, on=TIME_COL, how="inner").sort_values(TIME_COL).reset_index(drop=True)
