from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

TIME_COL = "DELIVERY_START"


def normalize_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit DELIVERY_START en datetime timezone-aware puis UTC naive."""
    out = df.copy()

    # Rend la conversion robuste aux chaînes non typées.
    out[TIME_COL] = pd.to_datetime(out[TIME_COL], errors="coerce", utc=True)
    invalid = int(out[TIME_COL].isna().sum())
    if invalid > 0:
        raise ValueError(f"{invalid} lignes ont un DELIVERY_START invalide après parsing datetime")

    # Simplifie la suite (lags, split, export) avec timestamps UTC naïfs.
    out[TIME_COL] = out[TIME_COL].dt.tz_convert("UTC").dt.tz_localize(None)
    return out


def fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "nucelear_power_available" in out.columns and "nuclear_power_available" not in out.columns:
        out = out.rename(columns={"nucelear_power_available": "nuclear_power_available"})
    return out


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out[TIME_COL].dt.hour
    out["dayofweek"] = out[TIME_COL].dt.dayofweek
    out["month"] = out[TIME_COL].dt.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    return out


def add_lag_features(df: pd.DataFrame, columns: Iterable[str], lags: Iterable[int]) -> pd.DataFrame:
    out = df.copy().sort_values(TIME_COL).reset_index(drop=True)
    for col in columns:
        if col not in out.columns:
            continue
        for lag in lags:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de feature engineering demandé pour la série temporelle."""
    out = normalize_datetime(df)
    out = fix_column_names(out)
    out = add_datetime_features(out)
    out = add_lag_features(
        out,
        columns=[
            "load_forecast",
            "gas_power_available",
            "wind_power_forecasts_average",
            "solar_power_forecasts_average",
            "wind_power_forecasts_std",
            "solar_power_forecasts_std",
        ],
        lags=[1, 24],
    )
    return out
