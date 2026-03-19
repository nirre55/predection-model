import numpy as np
import pandas as pd


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI normalisé [0, 1] via Wilder smoothing (EWM)."""
    s = pd.Series(close)
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean().values
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[0] = 50.0  # premier diff = NaN → valeur neutre
    return np.clip(rsi, 0.0, 100.0) / 100.0


def compute_macd_histogram(
    close: np.ndarray, fast: int = 12, slow: int = 26, signal_period: int = 9
) -> np.ndarray:
    """Histogramme MACD normalisé par le close (scale-invariant)."""
    s = pd.Series(close)
    macd_line = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    histogram = (macd_line - macd_line.ewm(span=signal_period, adjust=False).mean()).values
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(close > 0, histogram / close, 0.0)


def compute_bollinger_pct(
    close: np.ndarray, period: int = 20, num_std: float = 2.0
) -> np.ndarray:
    """Position %B dans les bandes de Bollinger, clampée à [-1, 2]."""
    s = pd.Series(close)
    ma = s.rolling(period, min_periods=1).mean()
    std = s.rolling(period, min_periods=1).std(ddof=0).fillna(0.0)
    band_width = (2 * num_std * std).values
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_b = np.where(
            band_width > 0,
            (close - (ma - num_std * std).values) / band_width,
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
    atr = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(close > 0, atr / close, 0.0)


def compute_volume_ratio(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Volume courant / SMA(volume, period), clampé à [0, 10]."""
    ma = pd.Series(volume).rolling(period, min_periods=1).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ma > 0, volume / ma, 1.0)
    return np.clip(ratio, 0.0, 10.0)
