"""
Features basées sur le timeframe Daily (D1) et le Fear & Greed Index.

Toutes les fonctions respectent le no-look-ahead :
  - Pour chaque target_time T, seules les bougies D1 fermées AVANT T sont utilisées.
  - Le F&G du jour J est utilisé pour prédire les bougies du jour J+1 (publié à minuit).

Features D1 (7 par sample) :
  0 - d1_rsi:         RSI(14) sur les clôtures daily [0, 1]
  1 - d1_above_ma20:  close > SMA(20) daily → 0 ou 1
  2 - d1_above_ma50:  close > SMA(50) daily → 0 ou 1
  3 - d1_momentum_5d: (close - close_5d) / close_5d, clippé [-0.5, 0.5]
  4 - d1_atr:         ATR(14) normalisé par close
  5 - d1_vol_regime:  volume / SMA(20_vol), clippé [0, 5]
  6 - d1_day_return:  (close_d1 - open_d1) / open_d1 — direction du jour courant

Features Fear & Greed (4 par sample) :
  0 - fg_value:   score Fear & Greed / 100 [0, 1]
  1 - fg_ma7:     MA(7) du score / 100 [0, 1]
  2 - fg_delta:   fg_value - fg_ma7 — tendance du sentiment [-1, 1]
  3 - fg_regime:  -1=extreme fear, 0=neutre, +1=extreme greed

Features Session (4 par sample) :
  0 - is_asian:       heure UTC in [0, 8)
  1 - is_london:      heure UTC in [7, 16)
  2 - is_ny:          heure UTC in [13, 22)
  3 - is_ln_ny_overlap: heure UTC in [13, 16)
"""

import numpy as np
import pandas as pd

from src.features.indicators import compute_rsi, compute_atr_normalized


# ── D1 Features ───────────────────────────────────────────────────────────────

def build_d1_features(
    df_1d: pd.DataFrame,
    target_times: np.ndarray,
) -> np.ndarray:
    """
    Construit 7 features D1 par candle M5 cible.

    Pour chaque target_time T (en datetime64 UTC), on utilise la DERNIERE bougie D1
    dont open_time < T (bougie entièrement fermée avant T).
    """
    df = df_1d.sort_values("open_time").reset_index(drop=True)
    times_ns   = np.asarray(df["open_time"].values, dtype="int64")
    close_arr  = np.asarray(df["close"].values,     dtype="float64")
    open_arr   = np.asarray(df["open"].values,      dtype="float64")
    high_arr   = np.asarray(df["high"].values,      dtype="float64")
    low_arr    = np.asarray(df["low"].values,       dtype="float64")
    vol_arr    = np.asarray(df["volume"].values,    dtype="float64")

    # Pre-compute indicators on all D1 data
    rsi_all = compute_rsi(close_arr)
    atr_all = compute_atr_normalized(high_arr, low_arr, close_arr)

    sma20 = np.asarray(pd.Series(close_arr).rolling(20, min_periods=1).mean().values, dtype="float64")
    sma50 = np.asarray(pd.Series(close_arr).rolling(50, min_periods=1).mean().values, dtype="float64")
    vol_sma20 = np.asarray(pd.Series(vol_arr).rolling(20, min_periods=1).mean().values, dtype="float64")

    target_ns = np.asarray(target_times).astype("int64")
    n = len(target_ns)
    out = np.zeros((n, 7), dtype="float64")
    out[:, 0] = 0.5   # neutral RSI default
    out[:, 3] = 0.0   # neutral momentum
    out[:, 4] = 0.01  # small ATR default
    out[:, 5] = 1.0   # neutral vol regime

    ONE_DAY_NS = np.int64(86_400_000_000_000)

    for i, t_ns in enumerate(target_ns):
        # A D1 candle opened at day D closes at day D+1 00:00 UTC.
        # For target time T, the last CLOSED D1 candle is the one from day D-1.
        # Round T down to midnight of its day, then find the candle before that.
        day_start = (t_ns // ONE_DAY_NS) * ONE_DAY_NS
        idx = int(np.searchsorted(times_ns, day_start, side="left")) - 1
        if idx < 1:
            continue

        out[i, 0] = float(rsi_all[idx])
        out[i, 1] = float(close_arr[idx] > sma20[idx])
        out[i, 2] = float(close_arr[idx] > sma50[idx])

        # 5-day momentum
        idx_5d = max(0, idx - 5)
        if close_arr[idx_5d] > 0:
            out[i, 3] = np.clip((close_arr[idx] - close_arr[idx_5d]) / close_arr[idx_5d], -0.5, 0.5)

        out[i, 4] = float(atr_all[idx])

        # Volume regime
        if vol_sma20[idx] > 0:
            out[i, 5] = float(np.clip(vol_arr[idx] / vol_sma20[idx], 0.0, 5.0))

        # Intraday return of the current D1 candle (direction of the day so far)
        if open_arr[idx] > 0:
            out[i, 6] = float(np.clip((close_arr[idx] - open_arr[idx]) / open_arr[idx], -0.1, 0.1))

    return np.nan_to_num(out, nan=0.0)


# ── Fear & Greed Features ─────────────────────────────────────────────────────

def build_fg_features(
    df_fg: pd.DataFrame,
    target_times: np.ndarray,
) -> np.ndarray:
    """
    Construit 4 features Fear & Greed par candle cible.

    df_fg doit contenir : date (datetime64 UTC, midnight), fg_value (0-100).
    Le F&G du jour J est considéré disponible dès minuit UTC du jour J
    (publié quotidiennement par alternative.me).
    """
    df = df_fg.sort_values("date").reset_index(drop=True)
    times_ns = np.asarray(df["date"].values, dtype="int64")
    fg_arr   = np.asarray(df["fg_value"].values, dtype="float64") / 100.0

    fg_ma7 = np.asarray(pd.Series(fg_arr).rolling(7, min_periods=1).mean().values, dtype="float64")

    target_ns = np.asarray(target_times).astype("int64")
    n = len(target_ns)
    out = np.full((n, 4), 0.5, dtype="float64")
    out[:, 2] = 0.0
    out[:, 3] = 0.0

    ONE_DAY_NS = np.int64(86_400_000_000_000)
    for i, t_ns in enumerate(target_ns):
        # F&G for day D is published at 00:00 UTC of day D.
        # Use F&G from day D-1 (the last one strictly before midnight of today).
        day_start = (t_ns // ONE_DAY_NS) * ONE_DAY_NS
        idx = int(np.searchsorted(times_ns, day_start, side="left")) - 1
        if idx < 0:
            continue

        val  = fg_arr[idx]
        ma7  = fg_ma7[idx]
        delta = float(np.clip(val - ma7, -1.0, 1.0))
        regime = 0.0
        if val < 0.25:
            regime = -1.0
        elif val > 0.75:
            regime = 1.0

        out[i, 0] = float(val)
        out[i, 1] = float(ma7)
        out[i, 2] = delta
        out[i, 3] = regime

    return np.nan_to_num(out, nan=0.5)


# ── Session Features ──────────────────────────────────────────────────────────

def build_session_features(target_times: np.ndarray) -> np.ndarray:
    """
    Construit 4 features de session de trading par candle cible.

    Returns (n, 4):
      0 - is_asian:          hour UTC in [0, 8)
      1 - is_london:         hour UTC in [7, 16)
      2 - is_ny:             hour UTC in [13, 22)
      3 - is_ln_ny_overlap:  hour UTC in [13, 16)
    """
    dt = pd.to_datetime(target_times, utc=True)
    hours = np.asarray(dt.hour.values, dtype="float64")

    is_asian   = ((hours >= 0)  & (hours < 8)).astype("float64")
    is_london  = ((hours >= 7)  & (hours < 16)).astype("float64")
    is_ny      = ((hours >= 13) & (hours < 22)).astype("float64")
    is_overlap = ((hours >= 13) & (hours < 16)).astype("float64")

    return np.column_stack([is_asian, is_london, is_ny, is_overlap])
