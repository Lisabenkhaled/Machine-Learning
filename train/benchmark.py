from __future__ import annotations

import numpy as np
import pandas as pd

from train.io import TARGET_COL, TIME_COL


def always_positive_baseline(df: pd.DataFrame, positive_value: float = 1.0) -> pd.DataFrame:
    """Benchmark simple: prédire un delta toujours positif."""
    out = pd.DataFrame({TIME_COL: df[TIME_COL], TARGET_COL: np.full(len(df), positive_value)})
    return out
