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
