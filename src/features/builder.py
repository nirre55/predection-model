import collections

import numpy as np
import pandas as pd

from src.features.indicators import (
    compute_rsi,
    compute_macd_histogram,
    compute_bollinger_pct,
    compute_atr_normalized,
    compute_volume_ratio,
)


def _normalize_window(window_array: np.ndarray, ref_close: float) -> np.ndarray:
    return window_array / ref_close


def build_dataset(
    df: pd.DataFrame, window: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    ohlcv = np.asarray(df[["open", "high", "low", "close", "volume"]].values, dtype="float64")
    close = np.asarray(df["close"].values, dtype="float64")
    open_ = np.asarray(df["open"].values, dtype="float64")
    high = np.asarray(df["high"].values, dtype="float64")
    low = np.asarray(df["low"].values, dtype="float64")
    volume = np.asarray(df["volume"].values, dtype="float64")

    # Indicateurs calculés sur la série complète (causaux, pas de look-ahead)
    rsi = compute_rsi(close)
    macd_hist = compute_macd_histogram(close)
    bb_pct = compute_bollinger_pct(close)
    atr_norm = compute_atr_normalized(high, low, close)
    vol_ratio = compute_volume_ratio(volume)

    n_samples = len(df) - window - 2
    j_arr = np.arange(n_samples)

    # Indices des fenêtres : sample j → lignes [j+1 .. j+window]
    window_rows = (j_arr + 1)[:, None] + np.arange(window)  # (n_samples, window)

    # OHLCV normalisé par ref_close (bougie juste avant la fenêtre)
    ref_closes = close[j_arr]  # close[j] = close[i-window-1]
    ohlcv_windows = ohlcv[window_rows]  # (n_samples, window, 5)
    ohlcv_norm = (ohlcv_windows / ref_closes[:, None, None]).reshape(n_samples, window * 5)

    # Fenêtres d'indicateurs
    rsi_win = rsi[window_rows]        # (n_samples, window)
    macd_win = macd_hist[window_rows]
    bb_win = bb_pct[window_rows]
    atr_win = atr_norm[window_rows]
    vol_win = vol_ratio[window_rows]

    # Concaténation : OHLCV(window*5) + 5 indicateurs(window chacun) = window*10
    X = np.concatenate(
        [ohlcv_norm, rsi_win, macd_win, bb_win, atr_win, vol_win], axis=1
    )

    # Target : prochaine bougie (j + window + 2)
    target_idx = j_arr + window + 2
    y = (close[target_idx] > open_[target_idx]).astype("int64")

    return X, y


def build_inference_features(
    candles: collections.deque, window: int = 50
) -> np.ndarray:
    candles_list = list(candles)  # window+1 éléments
    ref_close = candles_list[0]["close"]
    feature_candles = candles_list[1:]  # window éléments

    # Tableaux pour le calcul des indicateurs (window+1 bougies pour le warmup)
    close_arr = np.array([c["close"] for c in candles_list])
    high_arr = np.array([c["high"] for c in candles_list])
    low_arr = np.array([c["low"] for c in candles_list])
    volume_arr = np.array([c["volume"] for c in candles_list])

    # Calcul des indicateurs, on prend les window dernières valeurs
    rsi_arr = compute_rsi(close_arr)[-window:]
    macd_arr = compute_macd_histogram(close_arr)[-window:]
    bb_arr = compute_bollinger_pct(close_arr)[-window:]
    atr_arr = compute_atr_normalized(high_arr, low_arr, close_arr)[-window:]
    vol_arr = compute_volume_ratio(volume_arr)[-window:]

    # OHLCV normalisé
    ohlcv_window = np.array(
        [[c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in feature_candles],
        dtype="float64",
    )
    ohlcv_norm = (ohlcv_window / ref_close).flatten()

    X = np.concatenate([ohlcv_norm, rsi_arr, macd_arr, bb_arr, atr_arr, vol_arr])
    return X.reshape(1, -1)
