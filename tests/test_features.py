import collections

import numpy as np
import pytest

from src.features.builder import build_dataset, build_inference_features

# window=50 : OHLCV(50*5) + 5 indicateurs(50 chacun) = 500 features
EXPECTED_FEATURES = 500


def test_build_dataset_shape(sample_ohlcv_df):
    X, y = build_dataset(sample_ohlcv_df, window=50)
    # n = 200 - 50 - 2 = 148
    assert X.shape == (148, EXPECTED_FEATURES), f"Expected (148, {EXPECTED_FEATURES}), got {X.shape}"
    assert y.shape == (148,), f"Expected (148,), got {y.shape}"


def test_build_dataset_no_nan(sample_ohlcv_df):
    X, y = build_dataset(sample_ohlcv_df, window=50)
    assert not np.isnan(X).any(), "X contains NaN values"


def test_build_dataset_binary_labels(sample_ohlcv_df):
    _, y = build_dataset(sample_ohlcv_df, window=50)
    assert set(np.unique(y)).issubset({0, 1}), "y contains values other than 0 and 1"


def test_build_dataset_next_candle_label(sample_ohlcv_df):
    window = 50
    X, y = build_dataset(sample_ohlcv_df, window=window)
    close = sample_ohlcv_df["close"].values
    open_ = sample_ohlcv_df["open"].values

    # j=0 : target = window+2
    expected_y0 = 1 if close[window + 2] > open_[window + 2] else 0
    assert y[0] == expected_y0, f"Expected y[0]={expected_y0}, got {y[0]}"


def test_build_inference_features_shape():
    window = 50
    candles = collections.deque(maxlen=window + 1)
    rng = np.random.default_rng(0)
    for _ in range(window + 1):
        candles.append({
            "open": rng.uniform(29000, 31000),
            "high": rng.uniform(31000, 32000),
            "low": rng.uniform(28000, 29000),
            "close": rng.uniform(29000, 31000),
            "volume": rng.uniform(1, 100),
        })

    X = build_inference_features(candles, window=window)
    assert X.shape == (1, EXPECTED_FEATURES), f"Expected (1, {EXPECTED_FEATURES}), got {X.shape}"
