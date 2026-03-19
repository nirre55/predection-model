import collections

import numpy as np
import pandas as pd


def _normalize_window(window_array: np.ndarray, ref_close: float) -> np.ndarray:
    return window_array / ref_close


def build_dataset(
    df: pd.DataFrame, window: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    ohlcv = df[["open", "high", "low", "close", "volume"]].values
    close = df["close"].values
    open_ = df["open"].values

    n_samples = len(df) - window - 2
    X = np.empty((n_samples, window * 5), dtype="float64")
    y = np.empty(n_samples, dtype="int64")

    for j, i in enumerate(range(window + 1, len(df) - 1)):
        window_slice = ohlcv[i - window : i].flatten()
        ref_close = close[i - window - 1]
        X[j] = _normalize_window(window_slice, ref_close)
        y[j] = 1 if close[i + 1] > open_[i + 1] else 0

    return X, y


def build_inference_features(
    candles: collections.deque, window: int = 50
) -> np.ndarray:
    candles_list = list(candles)
    ref_close = candles_list[0]["close"]
    feature_candles = candles_list[1:]

    window_array = np.array(
        [[c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in feature_candles],
        dtype="float64",
    ).flatten()

    X = _normalize_window(window_array, ref_close).reshape(1, -1)
    return X
