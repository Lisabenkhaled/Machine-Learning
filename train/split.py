from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TrainValidSplit:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame


def chronological_train_valid_split(df: pd.DataFrame, valid_fraction: float = 0.2) -> TrainValidSplit:
    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction doit être dans ]0,1[.")

    cut = int(len(df) * (1 - valid_fraction))
    train_df = df.iloc[:cut].copy()
    valid_df = df.iloc[cut:].copy()
    return TrainValidSplit(train_df=train_df, valid_df=valid_df)
