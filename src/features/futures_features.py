"""
Futures market features with strict no-look-ahead-bias guarantee.

All functions accept:
  - a DataFrame of futures data (sorted by time ascending)
  - target_times: numpy array of candle open times (datetime64 UTC) we are predicting

For each target time T, ONLY data strictly before T is used.
"""

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling z-score for each element i using the previous `window` values."""
    n = len(arr)
    result = np.zeros(n, dtype="float64")
    for i in range(n):
        lo = max(0, i - window + 1)
        chunk = arr[lo : i + 1]
        std = chunk.std()
        if std == 0 or np.isnan(std):
            result[i] = 0.0
        else:
            result[i] = (arr[i] - chunk.mean()) / std
    return result


def _vectorized_rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
    """Vectorized rolling z-score (faster for large arrays)."""
    s = pd.Series(arr)
    roll = s.rolling(window=window, min_periods=2)
    mean = roll.mean().to_numpy(copy=True)
    std = roll.std(ddof=0).to_numpy(copy=True)
    std[std == 0] = np.nan
    z = (arr - mean) / std
    return np.nan_to_num(z, nan=0.0)


# ── CVD Features ──────────────────────────────────────────────────────────────

def build_cvd_features(
    df_taker: pd.DataFrame,
    target_times: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """
    Compute Cumulative Volume Delta features per target candle.

    For target time T, uses only candles with open_time < T (i.e., closed before T starts).
    T-1 = candle at T minus one 5-minute bar.

    Features (5 per sample):
      0 - cvd_ratio_last:    taker_buy / total of candle T-1  [0..1]
      1 - cvd_ratio_mean_N:  rolling mean of cvd_ratio over last N candles before T
      2 - cvd_momentum:      cvd_ratio_last - cvd_ratio_mean_N
      3 - cvd_acceleration:  cvd_ratio[T-1] - cvd_ratio[T-2]
      4 - cvd_cum_delta:     normalized cumulative delta over last N candles

    Returns:
        np.ndarray of shape (len(target_times), 5)
    """
    # Ensure sorted
    df = df_taker.sort_values("open_time").reset_index(drop=True)

    times_ns = np.asarray(df["open_time"].values, dtype="int64")
    taker_buy = np.asarray(df["taker_buy_volume"].values, dtype="float64")
    total_vol = np.asarray(df["total_volume"].values, dtype="float64")

    # cvd_ratio per candle — avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        cvd_ratio = np.where(total_vol > 0, taker_buy / total_vol, 0.5)

    # delta per candle: (2*taker - total) / total  -> in [-1, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(total_vol > 0, (2.0 * taker_buy - total_vol) / total_vol, 0.0)

    target_ns = np.asarray(target_times).astype("int64")
    n = len(target_ns)
    out = np.full((n, 5), 0.0, dtype="float64")

    for i, t_ns in enumerate(target_ns):
        # index of last candle strictly before T
        idx = int(np.searchsorted(times_ns, t_ns, side="left")) - 1

        if idx < 1:
            # Not enough data — use neutral defaults
            out[i] = [0.5, 0.5, 0.0, 0.0, 0.0]
            continue

        # T-1 candle
        r_last = cvd_ratio[idx]
        r_prev = cvd_ratio[idx - 1]

        # Rolling window
        lo = max(0, idx - window + 1)
        ratio_window = cvd_ratio[lo : idx + 1]
        delta_window = delta[lo : idx + 1]

        r_mean = float(ratio_window.mean())
        cum_delta = float(delta_window.sum()) / max(len(delta_window), 1)

        out[i, 0] = r_last                   # cvd_ratio_last
        out[i, 1] = r_mean                   # cvd_ratio_mean_N
        out[i, 2] = r_last - r_mean          # cvd_momentum
        out[i, 3] = r_last - r_prev          # cvd_acceleration
        out[i, 4] = np.clip(cum_delta, -1.0, 1.0)  # cvd_cum_delta

    # Fill any residual NaN
    out = np.nan_to_num(out, nan=0.5)
    # cvd_ratio cols should be 0-1, zero-delta is 0
    out[:, 0] = np.clip(out[:, 0], 0.0, 1.0)
    out[:, 1] = np.clip(out[:, 1], 0.0, 1.0)
    return out


# ── Open Interest Features ────────────────────────────────────────────────────

def build_oi_features(
    df_oi: pd.DataFrame,
    target_times: np.ndarray,
) -> np.ndarray:
    """
    Compute open interest change features per target candle.

    Uses the last OI snapshot AT OR BEFORE target_time (searchsorted right - 1).

    Features (4 per sample):
      0 - oi_change_1h:      (oi_now - oi_1h_ago) / oi_1h_ago
      1 - oi_change_4h:      (oi_now - oi_4h_ago) / oi_4h_ago
      2 - oi_zscore:         z-score of oi_change_1h over rolling 30 periods
      3 - oi_price_momentum: sign of oi_change_1h (-1, 0, +1)

    Returns:
        np.ndarray of shape (len(target_times), 4)
    """
    df = df_oi.sort_values("open_time").reset_index(drop=True)
    times_ns = np.asarray(df["open_time"].values, dtype="int64")
    oi_arr = np.asarray(df["open_interest"].values, dtype="float64")

    ONE_HOUR_NS  = np.int64(3_600_000_000_000)
    FOUR_HOUR_NS = np.int64(14_400_000_000_000)

    target_ns = np.asarray(target_times).astype("int64")
    n = len(target_ns)

    # Pre-compute oi_change_1h for all OI rows (for rolling z-score)
    n_oi = len(oi_arr)
    oi_change_1h_all = np.zeros(n_oi, dtype="float64")
    for i in range(n_oi):
        t = times_ns[i]
        idx_1h = int(np.searchsorted(times_ns, t - ONE_HOUR_NS, side="right")) - 1
        if idx_1h >= 0 and oi_arr[idx_1h] != 0:
            oi_change_1h_all[i] = (oi_arr[i] - oi_arr[idx_1h]) / oi_arr[idx_1h]

    # Rolling z-score over 30 OI periods
    oi_zscore_all = _vectorized_rolling_zscore(oi_change_1h_all, window=30)

    out = np.zeros((n, 4), dtype="float64")

    for i, t_ns in enumerate(target_ns):
        # Last closed OI snapshot at or before T
        idx = int(np.searchsorted(times_ns, t_ns, side="right")) - 1

        if idx < 0:
            out[i] = [0.0, 0.0, 0.0, 0.0]
            continue

        oi_now = oi_arr[idx]
        t_now = times_ns[idx]

        # OI 1h ago
        idx_1h = int(np.searchsorted(times_ns, t_now - ONE_HOUR_NS, side="right")) - 1
        if idx_1h >= 0 and oi_arr[idx_1h] != 0:
            ch_1h = (oi_now - oi_arr[idx_1h]) / oi_arr[idx_1h]
        else:
            ch_1h = 0.0

        # OI 4h ago
        idx_4h = int(np.searchsorted(times_ns, t_now - FOUR_HOUR_NS, side="right")) - 1
        if idx_4h >= 0 and oi_arr[idx_4h] != 0:
            ch_4h = (oi_now - oi_arr[idx_4h]) / oi_arr[idx_4h]
        else:
            ch_4h = 0.0

        zscore = float(oi_zscore_all[idx])
        sign = float(np.sign(ch_1h))

        out[i, 0] = np.clip(ch_1h, -0.5, 0.5)
        out[i, 1] = np.clip(ch_4h, -0.5, 0.5)
        out[i, 2] = np.clip(zscore, -3.0, 3.0)
        out[i, 3] = sign

    return np.nan_to_num(out, nan=0.0)


# ── Funding Rate Features ─────────────────────────────────────────────────────

def build_funding_features(
    df_funding: pd.DataFrame,
    target_times: np.ndarray,
) -> np.ndarray:
    """
    Compute funding rate features per target candle.

    Funding rates are published every 8h and are always known at prediction time
    (no look-ahead bias).

    Features (3 per sample):
      0 - funding_rate:   current funding rate, clipped to [-0.005, 0.005]
      1 - funding_zscore: z-score over rolling 90 periods (~30 days)
      2 - funding_sign:   sign of funding rate (+1 longs paying, -1 shorts paying)

    Returns:
        np.ndarray of shape (len(target_times), 3)
    """
    df = df_funding.sort_values("funding_time").reset_index(drop=True)
    times_ns = np.asarray(df["funding_time"].values, dtype="int64")
    rate_arr = np.asarray(df["funding_rate"].values, dtype="float64")

    # Pre-compute rolling z-score over all funding rate rows
    rate_zscore_all = _vectorized_rolling_zscore(rate_arr, window=90)

    target_ns = np.asarray(target_times).astype("int64")
    n = len(target_ns)
    out = np.zeros((n, 3), dtype="float64")

    for i, t_ns in enumerate(target_ns):
        # Last funding rate at or before T
        idx = int(np.searchsorted(times_ns, t_ns, side="right")) - 1

        if idx < 0:
            out[i] = [0.0, 0.0, 0.0]
            continue

        rate = float(np.clip(rate_arr[idx], -0.005, 0.005))
        zscore = float(np.clip(rate_zscore_all[idx], -3.0, 3.0))
        sign = float(np.sign(rate_arr[idx]))

        out[i, 0] = rate
        out[i, 1] = zscore
        out[i, 2] = sign

    return np.nan_to_num(out, nan=0.0)
