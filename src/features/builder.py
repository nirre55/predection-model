import collections
from collections.abc import Callable
from datetime import datetime

import numpy as np
import pandas as pd

from src.features.indicators import (
    compute_rsi,
    compute_macd_histogram,
    compute_bollinger_pct,
    compute_atr_normalized,
    compute_volume_ratio,
)
from src.features import time_features as tf

# Registre des indicateurs disponibles (même interface qu'ablation.py)
INDICATOR_REGISTRY: dict[str, Callable[..., np.ndarray]] = {
    "rsi": lambda df: compute_rsi(np.asarray(df["close"].values, dtype="float64")),
    "macd": lambda df: compute_macd_histogram(np.asarray(df["close"].values, dtype="float64")),
    "bb": lambda df: compute_bollinger_pct(np.asarray(df["close"].values, dtype="float64")),
    "atr": lambda df: compute_atr_normalized(
        np.asarray(df["high"].values, dtype="float64"),
        np.asarray(df["low"].values, dtype="float64"),
        np.asarray(df["close"].values, dtype="float64"),
    ),
    "vol": lambda df: compute_volume_ratio(np.asarray(df["volume"].values, dtype="float64")),
}

ALL_INDICATORS = ["rsi", "macd", "bb", "atr", "vol"]


def build_dataset(
    df: pd.DataFrame,
    window: int = 50,
    indicators: list[str] | None = None,
    include_time: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construit X, y pour l'entraînement.

    Args:
        df:           DataFrame OHLCV avec colonne open_time.
        window:       Taille de la fenêtre (nombre de bougies).
        indicators:   Liste d'indicateurs à inclure (None = tous les 5).
        include_time: Si True, ajoute 5 features temporelles (heure/jour).

    Returns:
        X: (n_samples, n_features)
        y: (n_samples,) — 1 si bougie verte, 0 sinon
    """
    active = indicators if indicators is not None else ALL_INDICATORS

    ohlcv = np.asarray(df[["open", "high", "low", "close", "volume"]].values, dtype="float64")
    close = np.asarray(df["close"].values, dtype="float64")
    open_ = np.asarray(df["open"].values, dtype="float64")

    # Indicateurs sur la série complète (causaux)
    ind_arrays = [INDICATOR_REGISTRY[name](df) for name in active]

    n_samples = len(df) - window - 2
    j_arr = np.arange(n_samples)
    window_rows = (j_arr + 1)[:, None] + np.arange(window)  # (n_samples, window)

    # OHLCV normalisé par ref_close
    ref_closes = close[j_arr]
    ohlcv_windows = ohlcv[window_rows]
    ohlcv_norm = (ohlcv_windows / ref_closes[:, None, None]).reshape(n_samples, window * 5)

    parts: list[np.ndarray] = [ohlcv_norm]
    for arr in ind_arrays:
        parts.append(arr[window_rows])

    # Features temporelles (scalaires par sample — heure de la bougie cible)
    if include_time and "open_time" in df.columns:
        target_idx = j_arr + window + 2
        target_times = df["open_time"].iloc[target_idx].reset_index(drop=True)
        time_feats = tf.from_timestamps(target_times)  # (n_samples, 5)
        parts.append(time_feats)

    X = np.concatenate(parts, axis=1)

    target_idx = j_arr + window + 2
    y = (close[target_idx] > open_[target_idx]).astype("int64")

    return X, y


def build_dataset_with_target_times(
    df: pd.DataFrame,
    window: int = 50,
    indicators: list[str] | None = None,
    include_time: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Comme build_dataset mais retourne aussi les timestamps des bougies cibles.
    Utilisé par ablation_schedule.py pour filtrer par slot horaire.

    Returns:
        X, y, target_open_times (np.ndarray de datetime64)
    """
    active = indicators if indicators is not None else ALL_INDICATORS

    ohlcv = np.asarray(df[["open", "high", "low", "close", "volume"]].values, dtype="float64")
    close = np.asarray(df["close"].values, dtype="float64")
    open_ = np.asarray(df["open"].values, dtype="float64")

    ind_arrays = [INDICATOR_REGISTRY[name](df) for name in active]

    n_samples = len(df) - window - 2
    j_arr = np.arange(n_samples)
    window_rows = (j_arr + 1)[:, None] + np.arange(window)

    ref_closes = close[j_arr]
    ohlcv_windows = ohlcv[window_rows]
    ohlcv_norm = (ohlcv_windows / ref_closes[:, None, None]).reshape(n_samples, window * 5)

    parts: list[np.ndarray] = [ohlcv_norm]
    for arr in ind_arrays:
        parts.append(arr[window_rows])

    target_idx = j_arr + window + 2
    target_times_series = df["open_time"].iloc[target_idx].reset_index(drop=True)

    if include_time:
        time_feats = tf.from_timestamps(target_times_series)
        parts.append(time_feats)

    X = np.concatenate(parts, axis=1)
    y = (close[target_idx] > open_[target_idx]).astype("int64")
    target_open_times = np.asarray(target_times_series.values)

    return X, y, target_open_times


def build_inference_features(
    candles: collections.deque,
    window: int = 50,
    indicators: list[str] | None = None,
    predict_time: datetime | None = None,
) -> np.ndarray:
    """
    Construit les features pour une prédiction en temps réel.

    Args:
        candles:      Deque de window+1 bougies (dict avec open/high/low/close/volume).
        window:       Taille de la fenêtre.
        indicators:   Liste d'indicateurs (None = tous les 5, pour compat. ascendante).
        predict_time: Datetime UTC de la bougie à prédire — pour les features temporelles.
    """
    active = indicators if indicators is not None else ALL_INDICATORS

    candles_list = list(candles)  # window+1 éléments
    ref_close = candles_list[0]["close"]
    feature_candles = candles_list[1:]  # window éléments

    close_arr = np.array([c["close"] for c in candles_list])
    high_arr = np.array([c["high"] for c in candles_list])
    low_arr = np.array([c["low"] for c in candles_list])
    volume_arr = np.array([c["volume"] for c in candles_list])

    # Calcul des indicateurs demandés uniquement
    ind_map = {
        "rsi": lambda: compute_rsi(close_arr)[-window:],
        "macd": lambda: compute_macd_histogram(close_arr)[-window:],
        "bb": lambda: compute_bollinger_pct(close_arr)[-window:],
        "atr": lambda: compute_atr_normalized(high_arr, low_arr, close_arr)[-window:],
        "vol": lambda: compute_volume_ratio(volume_arr)[-window:],
    }

    ohlcv_window = np.array(
        [[c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in feature_candles],
        dtype="float64",
    )
    ohlcv_norm = (ohlcv_window / ref_close).flatten()

    parts: list[np.ndarray] = [ohlcv_norm]
    for name in active:
        parts.append(ind_map[name]())

    if predict_time is not None:
        parts.append(tf.from_datetime(predict_time))

    X = np.concatenate(parts)
    return X.reshape(1, -1)
