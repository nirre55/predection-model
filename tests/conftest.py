import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_df():
    rng = np.random.default_rng(42)
    n = 200
    close = 30000.0 + np.cumsum(rng.normal(0, 100, n))
    open_ = close + rng.normal(0, 50, n)
    high = np.maximum(open_, close) + rng.uniform(10, 200, n)
    low = np.minimum(open_, close) - rng.uniform(10, 200, n)
    volume = rng.uniform(1.0, 100.0, n)

    timestamps = pd.date_range(
        start="2024-01-01", periods=n, freq="5min", tz="UTC"
    )
    return pd.DataFrame(
        {
            "open_time": timestamps,
            "open": open_.astype("float64"),
            "high": high.astype("float64"),
            "low": low.astype("float64"),
            "close": close.astype("float64"),
            "volume": volume.astype("float64"),
        }
    )


@pytest.fixture
def sample_multitf_dfs():
    """DataFrames 1h et 4h couvrant la même période que sample_ohlcv_df."""
    rng = np.random.default_rng(42)
    # 1h : ~17 bougies pour couvrir 200 bougies 5m (200 * 5min / 60min ≈ 17h)
    n_1h = 30
    close_1h = 30000.0 + np.cumsum(rng.normal(0, 300, n_1h))
    open_1h = close_1h + rng.normal(0, 100, n_1h)
    high_1h = np.maximum(open_1h, close_1h) + rng.uniform(50, 500, n_1h)
    low_1h = np.minimum(open_1h, close_1h) - rng.uniform(50, 500, n_1h)
    ts_1h = pd.date_range(start="2024-01-01", periods=n_1h, freq="1h", tz="UTC")
    df_1h = pd.DataFrame({
        "open_time": ts_1h, "open": open_1h.astype("float64"),
        "high": high_1h.astype("float64"), "low": low_1h.astype("float64"),
        "close": close_1h.astype("float64"), "volume": rng.uniform(10, 1000, n_1h),
    })
    # 4h : ~8 bougies
    n_4h = 15
    close_4h = 30000.0 + np.cumsum(rng.normal(0, 600, n_4h))
    open_4h = close_4h + rng.normal(0, 200, n_4h)
    high_4h = np.maximum(open_4h, close_4h) + rng.uniform(100, 1000, n_4h)
    low_4h = np.minimum(open_4h, close_4h) - rng.uniform(100, 1000, n_4h)
    ts_4h = pd.date_range(start="2024-01-01", periods=n_4h, freq="4h", tz="UTC")
    df_4h = pd.DataFrame({
        "open_time": ts_4h, "open": open_4h.astype("float64"),
        "high": high_4h.astype("float64"), "low": low_4h.astype("float64"),
        "close": close_4h.astype("float64"), "volume": rng.uniform(100, 10000, n_4h),
    })
    return df_1h, df_4h
