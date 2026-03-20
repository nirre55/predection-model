import collections

import numpy as np
import pytest

from src.features.builder import build_dataset, build_dataset_with_target_times, build_inference_features

# window=50 : OHLCV(50*5) + 5 indicateurs(50 chacun) = 500 features
BASE_FEATURES = 500        # sans multi-TF
MTF_FEATURES = 5           # features contextuelles 1h/4h
EXPECTED_FEATURES = BASE_FEATURES  # rétro-compatibilité
EXPECTED_FEATURES_WITH_MTF = BASE_FEATURES + MTF_FEATURES


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


def test_build_dataset_with_multitf_shape(sample_ohlcv_df, sample_multitf_dfs):
    df_1h, df_4h = sample_multitf_dfs
    X, y = build_dataset(sample_ohlcv_df, window=50, df_1h=df_1h, df_4h=df_4h)
    assert X.shape == (148, EXPECTED_FEATURES_WITH_MTF), (
        f"Expected (148, {EXPECTED_FEATURES_WITH_MTF}), got {X.shape}"
    )
    assert not np.isnan(X).any(), "X contains NaN values with multi-TF features"


def test_build_dataset_target_filtering(sample_ohlcv_df):
    X_unfiltered, y_unfiltered = build_dataset(sample_ohlcv_df, window=50)
    X_filtered, y_filtered = build_dataset(sample_ohlcv_df, window=50, min_move_pct=0.001)
    assert len(y_filtered) < len(y_unfiltered), (
        "Filtered dataset should have fewer samples than unfiltered"
    )
    # Vérifier que les targets filtrés ont |move| >= 0.001
    close = sample_ohlcv_df["close"].values
    open_ = sample_ohlcv_df["open"].values
    window = 50
    n_samples = len(sample_ohlcv_df) - window - 2
    j_arr = np.arange(n_samples)
    target_idx = j_arr + window + 2
    price_move_pct = np.abs(close[target_idx] - open_[target_idx]) / open_[target_idx]
    valid_mask = price_move_pct >= 0.001
    assert len(y_filtered) == int(valid_mask.sum()), (
        f"Expected {valid_mask.sum()} filtered samples, got {len(y_filtered)}"
    )


def test_build_inference_features_with_multitf_shape(sample_multitf_dfs):
    df_1h, df_4h = sample_multitf_dfs
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

    # predict_time=None mais df_1h/df_4h fournis → 5 valeurs neutres ajoutées (shape 505)
    X = build_inference_features(candles, window=window, df_1h=df_1h, df_4h=df_4h)
    assert X.shape == (1, EXPECTED_FEATURES_WITH_MTF), (
        f"Expected (1, {EXPECTED_FEATURES_WITH_MTF}), got {X.shape}"
    )
