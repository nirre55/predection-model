import numpy as np


def compute_equity_curve(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    signals = np.where(y_pred == y_true, 1, -1)
    return np.cumsum(signals)


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / (np.abs(peak) + 1e-9) * 100.0
    return float(drawdown.min())
