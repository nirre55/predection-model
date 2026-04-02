import numpy as np
import pandas as pd


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI normalisé [0, 1] via Wilder smoothing (EWM)."""
    s = pd.Series(close)
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = np.asarray(gain.ewm(alpha=1.0 / period, adjust=False).mean().values, dtype="float64")
    avg_loss = np.asarray(loss.ewm(alpha=1.0 / period, adjust=False).mean().values, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[0] = 50.0
    return np.clip(rsi, 0.0, 100.0) / 100.0


def compute_macd_histogram(
    close: np.ndarray, fast: int = 12, slow: int = 26, signal_period: int = 9
) -> np.ndarray:
    """Histogramme MACD normalisé par le close (scale-invariant)."""
    s = pd.Series(close)
    macd_line = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    histogram = np.asarray((macd_line - macd_line.ewm(span=signal_period, adjust=False).mean()).values, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(close > 0, histogram / close, 0.0)


def compute_bollinger_pct(
    close: np.ndarray, period: int = 20, num_std: float = 2.0
) -> np.ndarray:
    """Position %B dans les bandes de Bollinger, clampée à [-1, 2]."""
    s = pd.Series(close)
    ma = s.rolling(period, min_periods=1).mean()
    std = s.rolling(period, min_periods=1).std(ddof=0).fillna(0.0)
    band_width = np.asarray((2 * num_std * std).values, dtype="float64")
    lower = np.asarray((ma - num_std * std).values, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_b = np.where(
            band_width > 0,
            (close - lower) / band_width,
            0.5,
        )
    return np.clip(pct_b, -1.0, 2.0)


def compute_atr_normalized(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """ATR normalisé par le close via Wilder smoothing (EWM)."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    atr = np.asarray(pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().values, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(close > 0, atr / close, 0.0)


def compute_volume_ratio(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Volume courant / SMA(volume, period), clampé à [0, 10]."""
    ma = np.asarray(pd.Series(volume).rolling(period, min_periods=1).mean().values, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ma > 0, volume / ma, 1.0)
    return np.clip(ratio, 0.0, 10.0)


# ---------------------------------------------------------------------------
# Nouveaux indicateurs
# ---------------------------------------------------------------------------

def compute_mfi(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 14
) -> np.ndarray:
    """Money Flow Index [0, 1] — RSI pondéré par le volume."""
    typical_price = (high + low + close) / 3.0
    money_flow = typical_price * volume

    prev_tp = np.roll(typical_price, 1)
    prev_tp[0] = typical_price[0]

    pos_flow = np.where(typical_price > prev_tp, money_flow, 0.0)
    neg_flow = np.where(typical_price < prev_tp, money_flow, 0.0)

    pos_mf = np.asarray(pd.Series(pos_flow).rolling(period, min_periods=1).sum().values, dtype="float64")
    neg_mf = np.asarray(pd.Series(neg_flow).rolling(period, min_periods=1).sum().values, dtype="float64")

    with np.errstate(divide="ignore", invalid="ignore"):
        mfr = np.where(neg_mf > 0, pos_mf / neg_mf, 100.0)

    mfi = 100.0 - 100.0 / (1.0 + mfr)
    mfi[0] = 50.0
    return np.clip(mfi, 0.0, 100.0) / 100.0


def compute_volume_delta(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    """Volume delta directionnel normalisé [-1, 1].

    Mesure la pression acheteuse/vendeuse : volume × direction pondérée par le corps.
    """
    candle_range = high - low
    with np.errstate(divide="ignore", invalid="ignore"):
        body_pct = np.where(candle_range > 0, (close - open_) / candle_range, 0.0)

    delta = volume * np.clip(body_pct, -1.0, 1.0)

    vol_sma = np.asarray(pd.Series(volume).rolling(20, min_periods=1).mean().values, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.where(vol_sma > 0, delta / vol_sma, 0.0)

    return np.clip(normalized, -10.0, 10.0)


def compute_cci(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20
) -> np.ndarray:
    """CCI normalisé à [-1, 1] — déviation du prix par rapport à sa moyenne."""
    typical_price = (high + low + close) / 3.0
    s = pd.Series(typical_price)
    tp_ma = np.asarray(s.rolling(period, min_periods=1).mean().values, dtype="float64")
    tp_std = np.asarray(s.rolling(period, min_periods=1).std(ddof=0).fillna(0.0).values, dtype="float64")

    with np.errstate(divide="ignore", invalid="ignore"):
        cci = np.where(tp_std > 0, (typical_price - tp_ma) / (0.015 * tp_std), 0.0)

    return np.clip(cci / 200.0, -1.0, 1.0)


def compute_body_ratio(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Ratio corps / range total [0, 1]. 1.0 = corps pur, 0.0 = doji total."""
    total_range = high - low
    body = np.abs(close - open_)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(total_range > 0, body / total_range, 0.5)
    return np.clip(ratio, 0.0, 1.0)


def compute_upper_wick(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Ratio mèche haute / range [0, 1] — pression vendeuse (rejet des hauts)."""
    total_range = high - low
    upper_wick = high - np.maximum(open_, close)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(total_range > 0, upper_wick / total_range, 0.0)
    return np.clip(ratio, 0.0, 1.0)


def compute_lower_wick(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Ratio mèche basse / range [0, 1] — pression acheteuse (rejet des bas)."""
    total_range = high - low
    lower_wick = np.minimum(open_, close) - low
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(total_range > 0, lower_wick / total_range, 0.0)
    return np.clip(ratio, 0.0, 1.0)


def compute_adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """ADX normalisé [0, 1] — force de la tendance (0=range, 1=trend fort)."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

    prev_high = np.roll(high, 1); prev_high[0] = high[0]
    prev_low  = np.roll(low, 1);  prev_low[0]  = low[0]
    dm_plus  = np.where((high - prev_high) > (prev_low - low), np.maximum(high - prev_high, 0.0), 0.0)
    dm_minus = np.where((prev_low - low) > (high - prev_high), np.maximum(prev_low - low, 0.0), 0.0)
    dm_plus[0] = 0.0; dm_minus[0] = 0.0

    alpha = 1.0 / period
    atr_s = np.asarray(pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values,     dtype="float64")
    dmp_s = np.asarray(pd.Series(dm_plus).ewm(alpha=alpha, adjust=False).mean().values, dtype="float64")
    dmm_s = np.asarray(pd.Series(dm_minus).ewm(alpha=alpha, adjust=False).mean().values,dtype="float64")

    with np.errstate(divide="ignore", invalid="ignore"):
        di_plus  = np.where(atr_s > 0, 100.0 * dmp_s / atr_s, 0.0)
        di_minus = np.where(atr_s > 0, 100.0 * dmm_s / atr_s, 0.0)
        di_sum   = di_plus + di_minus
        dx       = np.where(di_sum > 0, 100.0 * np.abs(di_plus - di_minus) / di_sum, 0.0)

    adx = np.asarray(pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values, dtype="float64")
    return np.clip(adx / 100.0, 0.0, 1.0)


def compute_stoch_k(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14
) -> np.ndarray:
    """Stochastic %K normalisé [0, 1]."""
    s_high = np.asarray(pd.Series(high).rolling(k_period, min_periods=1).max().values, dtype="float64")
    s_low  = np.asarray(pd.Series(low).rolling(k_period,  min_periods=1).min().values, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(s_high - s_low > 0, (close - s_low) / (s_high - s_low), 0.5)
    return np.clip(k, 0.0, 1.0)


def compute_streak(close: np.ndarray, open_: np.ndarray) -> np.ndarray:
    """Nombre de bougies consécutives dans la même direction, normalisé [-1, 1].

    Positif = streak haussier, négatif = streak baissier.
    """
    direction = (close > open_).astype("int64")
    sign = np.where(direction, 1, -1)

    changes = np.concatenate([[True], direction[1:] != direction[:-1]])
    group_starts = np.where(changes)[0]  # positions de début de chaque groupe
    group_id = np.cumsum(changes) - 1    # index 0-based du groupe

    within_pos = np.arange(len(direction)) - group_starts[group_id]
    streak = sign * (within_pos + 1).astype("float64")
    return np.clip(streak / 10.0, -1.0, 1.0)
