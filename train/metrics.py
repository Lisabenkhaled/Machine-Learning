from __future__ import annotations

import numpy as np


def weighted_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Accuracy = précision du signe pondérée par |y_true|."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true et y_pred doivent avoir la même forme")

    correct_sign = (np.sign(y_true) == np.sign(y_pred)).astype(float)
    weights = np.abs(y_true)
    denom = weights.sum()

    if denom == 0:
        return float(correct_sign.mean())
    return float((correct_sign * weights).sum() / denom)
