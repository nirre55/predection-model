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
    compute_mfi,
    compute_volume_delta,
    compute_cci,
    compute_body_ratio,
    compute_upper_wick,
    compute_lower_wick,
    compute_streak,
)
from src.features import time_features as tf

# Registre des indicateurs disponibles
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
    # Nouveaux indicateurs
    "mfi": lambda df: compute_mfi(
        np.asarray(df["high"].values, dtype="float64"),
        np.asarray(df["low"].values, dtype="float64"),
        np.asarray(df["close"].values, dtype="float64"),
        np.asarray(df["volume"].values, dtype="float64"),
    ),
    "vdelta": lambda df: compute_volume_delta(
        np.asarray(df["open"].values, dtype="float64"),
        np.asarray(df["high"].values, dtype="float64"),
        np.asarray(df["low"].values, dtype="float64"),
        np.asarray(df["close"].values, dtype="float64"),
        np.asarray(df["volume"].values, dtype="float64"),
    ),
    "cci": lambda df: compute_cci(
        np.asarray(df["high"].values, dtype="float64"),
        np.asarray(df["low"].values, dtype="float64"),
        np.asarray(df["close"].values, dtype="float64"),
    ),
    "body": lambda df: compute_body_ratio(
        np.asarray(df["open"].values, dtype="float64"),
        np.asarray(df["high"].values, dtype="float64"),
        np.asarray(df["low"].values, dtype="float64"),
        np.asarray(df["close"].values, dtype="float64"),
    ),
    "uwik": lambda df: compute_upper_wick(
        np.asarray(df["open"].values, dtype="float64"),
        np.asarray(df["high"].values, dtype="float64"),
        np.asarray(df["low"].values, dtype="float64"),
        np.asarray(df["close"].values, dtype="float64"),
    ),
    "lwik": lambda df: compute_lower_wick(
        np.asarray(df["open"].values, dtype="float64"),
        np.asarray(df["high"].values, dtype="float64"),
        np.asarray(df["low"].values, dtype="float64"),
        np.asarray(df["close"].values, dtype="float64"),
    ),
    "streak": lambda df: compute_streak(
        np.asarray(df["close"].values, dtype="float64"),
        np.asarray(df["open"].values, dtype="float64"),
    ),
}

ALL_INDICATORS = ["rsi", "macd", "bb", "atr", "vol", "mfi", "vdelta", "cci", "body", "uwik", "lwik", "streak"]


def build_multitf_context(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    target_times: np.ndarray,
) -> np.ndarray:
    """
    Construit 5 features contextuelles multi-timeframe alignées sur target_times.

    Args:
        df_1h:        DataFrame 1h avec colonnes open_time, open, high, low, close, volume.
        df_4h:        DataFrame 4h avec colonnes open_time, open, high, low, close, volume.
        target_times: Array de timestamps (datetime64 ou int64 nanoseconds UTC).

    Returns:
        ctx: (n_samples, 5) — [trend_1h, trend_4h, momentum_1h, volatility_regime_1h, rsi_1h]
    """
    times_1h_ns = np.asarray(df_1h["open_time"].values, dtype="int64")
    times_4h_ns = np.asarray(df_4h["open_time"].values, dtype="int64")
    target_times_ns = np.asarray(target_times).astype("int64")

    close_1h = np.asarray(df_1h["close"].values, dtype="float64")
    high_1h = np.asarray(df_1h["high"].values, dtype="float64")
    low_1h = np.asarray(df_1h["low"].values, dtype="float64")

    sma20_1h = pd.Series(close_1h).rolling(20, min_periods=1).mean().values
    trend_1h_arr = (close_1h > sma20_1h).astype("float64")
    rsi_1h_arr = compute_rsi(close_1h)
    atr_1h_arr = compute_atr_normalized(high_1h, low_1h, close_1h)

    close_4h = np.asarray(df_4h["close"].values, dtype="float64")
    sma20_4h = pd.Series(close_4h).rolling(20, min_periods=1).mean().values
    trend_4h_arr = (close_4h > sma20_4h).astype("float64")

    # Utiliser uniquement les bougies 1h/4h FERMEES au moment de la prediction.
    # Une bougie 1h ouverte a t ferme a t+1h, donc on soustrait 1h/4h pour
    # garantir que la bougie selectionnee est entierement close.
    ONE_HOUR_NS  = np.int64(3_600_000_000_000)
    FOUR_HOUR_NS = np.int64(14_400_000_000_000)
    idx_1h = np.searchsorted(times_1h_ns, target_times_ns - ONE_HOUR_NS,  side="right") - 1
    idx_4h = np.searchsorted(times_4h_ns, target_times_ns - FOUR_HOUR_NS, side="right") - 1

    idx_1h = np.clip(idx_1h, 0, len(df_1h) - 1)
    idx_4h = np.clip(idx_4h, 0, len(df_4h) - 1)

    f_trend_1h = trend_1h_arr[idx_1h]
    f_trend_4h = trend_4h_arr[idx_4h]

    idx_1h_lag = np.maximum(0, idx_1h - 3)
    f_momentum_1h = np.clip(
        (close_1h[idx_1h] - close_1h[idx_1h_lag]) / close_1h[idx_1h_lag],
        -0.1, 0.1,
    )

    f_volatility_1h = atr_1h_arr[idx_1h]
    f_rsi_1h = rsi_1h_arr[idx_1h]

    return np.column_stack([f_trend_1h, f_trend_4h, f_momentum_1h, f_volatility_1h, f_rsi_1h])


def build_dataset(
    df: pd.DataFrame,
    window: int = 50,
    indicators: list[str] | None = None,
    include_time: bool = False,
    df_1h: pd.DataFrame | None = None,
    df_4h: pd.DataFrame | None = None,
    min_move_pct: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construit X, y pour l'entraînement.

    Args:
        df:           DataFrame OHLCV avec colonne open_time.
        window:       Taille de la fenêtre (nombre de bougies).
        indicators:   Liste d'indicateurs à inclure (None = tous).
        include_time: Si True, ajoute 5 features temporelles (heure/jour).
        df_1h:        DataFrame 1h pour les features multi-timeframe (optionnel).
        df_4h:        DataFrame 4h pour les features multi-timeframe (optionnel).
        min_move_pct: Seuil minimum de mouvement de prix — bougies en-dessous masquées.

    Returns:
        X: (n_samples, n_features)
        y: (n_samples,) — 1 si bougie verte, 0 sinon
    """
    active = indicators if indicators is not None else ALL_INDICATORS

    ohlcv = np.asarray(df[["open", "high", "low", "close", "volume"]].values, dtype="float64")
    close = np.asarray(df["close"].values, dtype="float64")
    open_ = np.asarray(df["open"].values, dtype="float64")

    ind_arrays = [INDICATOR_REGISTRY[name](df) for name in active]

    n_samples = len(df) - window - 1
    j_arr = np.arange(n_samples)
    window_rows = (j_arr + 1)[:, None] + np.arange(window)

    ref_closes = close[j_arr]
    ohlcv_windows = ohlcv[window_rows]
    ohlcv_norm = (ohlcv_windows / ref_closes[:, None, None]).reshape(n_samples, window * 5)

    parts: list[np.ndarray] = [ohlcv_norm]
    for arr in ind_arrays:
        parts.append(arr[window_rows])

    # target = bougie qui ouvre immediatement apres la derniere feature (j+window+1)
    target_idx = j_arr + window + 1

    if include_time and "open_time" in df.columns:
        target_times = df["open_time"].iloc[target_idx].reset_index(drop=True)
        time_feats = tf.from_timestamps(target_times)
        parts.append(time_feats)

    X = np.concatenate(parts, axis=1)

    y = (close[target_idx] > open_[target_idx]).astype("int64")

    if df_1h is not None and df_4h is not None:
        target_times_mtf = np.asarray(df["open_time"].iloc[target_idx].reset_index(drop=True).values)
        ctx = build_multitf_context(df_1h, df_4h, target_times_mtf)
        X = np.concatenate([X, ctx], axis=1)

    if min_move_pct > 0.0:
        price_move_pct = np.abs(close[target_idx] - open_[target_idx]) / open_[target_idx]
        valid_mask = price_move_pct >= min_move_pct
        X, y = X[valid_mask], y[valid_mask]

    return X, y


def build_dataset_with_target_times(
    df: pd.DataFrame,
    window: int = 50,
    indicators: list[str] | None = None,
    include_time: bool = False,
    df_1h: pd.DataFrame | None = None,
    df_4h: pd.DataFrame | None = None,
    min_move_pct: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Comme build_dataset mais retourne aussi les timestamps des bougies cibles.
    """
    active = indicators if indicators is not None else ALL_INDICATORS

    ohlcv = np.asarray(df[["open", "high", "low", "close", "volume"]].values, dtype="float64")
    close = np.asarray(df["close"].values, dtype="float64")
    open_ = np.asarray(df["open"].values, dtype="float64")

    ind_arrays = [INDICATOR_REGISTRY[name](df) for name in active]

    n_samples = len(df) - window - 1
    j_arr = np.arange(n_samples)
    window_rows = (j_arr + 1)[:, None] + np.arange(window)

    ref_closes = close[j_arr]
    ohlcv_windows = ohlcv[window_rows]
    ohlcv_norm = (ohlcv_windows / ref_closes[:, None, None]).reshape(n_samples, window * 5)

    parts: list[np.ndarray] = [ohlcv_norm]
    for arr in ind_arrays:
        parts.append(arr[window_rows])

    # target = bougie qui ouvre immediatement apres la derniere feature (j+window+1)
    target_idx = j_arr + window + 1
    target_times_series = df["open_time"].iloc[target_idx].reset_index(drop=True)

    if include_time:
        time_feats = tf.from_timestamps(target_times_series)
        parts.append(time_feats)

    X = np.concatenate(parts, axis=1)
    y = (close[target_idx] > open_[target_idx]).astype("int64")
    target_open_times = np.asarray(target_times_series.values)

    if df_1h is not None and df_4h is not None:
        ctx = build_multitf_context(df_1h, df_4h, np.asarray(target_times_series.values))
        X = np.concatenate([X, ctx], axis=1)

    if min_move_pct > 0.0:
        price_move_pct = np.abs(close[target_idx] - open_[target_idx]) / open_[target_idx]
        valid_mask = price_move_pct >= min_move_pct
        X, y, target_open_times = X[valid_mask], y[valid_mask], target_open_times[valid_mask]

    return X, y, target_open_times


def build_inference_features(
    candles: collections.deque,
    window: int = 50,
    indicators: list[str] | None = None,
    predict_time: datetime | None = None,
    df_1h: pd.DataFrame | None = None,
    df_4h: pd.DataFrame | None = None,
) -> np.ndarray:
    """
    Construit les features pour une prédiction en temps réel.
    """
    active = indicators if indicators is not None else ALL_INDICATORS

    candles_list = list(candles)
    ref_close = candles_list[0]["close"]
    feature_candles = candles_list[1:]

    close_arr = np.array([c["close"] for c in candles_list])
    high_arr = np.array([c["high"] for c in candles_list])
    low_arr = np.array([c["low"] for c in candles_list])
    open_arr = np.array([c["open"] for c in candles_list])
    volume_arr = np.array([c["volume"] for c in candles_list])

    ind_map = {
        "rsi":    lambda: compute_rsi(close_arr)[-window:],
        "macd":   lambda: compute_macd_histogram(close_arr)[-window:],
        "bb":     lambda: compute_bollinger_pct(close_arr)[-window:],
        "atr":    lambda: compute_atr_normalized(high_arr, low_arr, close_arr)[-window:],
        "vol":    lambda: compute_volume_ratio(volume_arr)[-window:],
        "mfi":    lambda: compute_mfi(high_arr, low_arr, close_arr, volume_arr)[-window:],
        "vdelta": lambda: compute_volume_delta(open_arr, high_arr, low_arr, close_arr, volume_arr)[-window:],
        "cci":    lambda: compute_cci(high_arr, low_arr, close_arr)[-window:],
        "body":   lambda: compute_body_ratio(open_arr, high_arr, low_arr, close_arr)[-window:],
        "uwik":   lambda: compute_upper_wick(open_arr, high_arr, low_arr, close_arr)[-window:],
        "lwik":   lambda: compute_lower_wick(open_arr, high_arr, low_arr, close_arr)[-window:],
        "streak": lambda: compute_streak(close_arr, open_arr)[-window:],
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

    if df_1h is not None and df_4h is not None and predict_time is not None:
        target_ts = np.array([pd.Timestamp(predict_time).value])
        ctx = build_multitf_context(df_1h, df_4h, target_ts)
        X = np.concatenate([X, ctx.flatten()])
    elif df_1h is not None or df_4h is not None:
        X = np.concatenate([X, np.array([0.5, 0.5, 0.0, 0.0, 0.5])])

    return X.reshape(1, -1)
