import numpy as np
import pytest

from src.features.builder import build_dataset
from src.backtest.walk_forward import run_walk_forward


def test_walk_forward_returns_correct_structure(sample_ohlcv_df):
    X, y = build_dataset(sample_ohlcv_df, window=50)
    price_moves = np.ones(len(y), dtype="float64")

    results = run_walk_forward(X, y, price_moves, n_splits=5, model_type="lgbm")

    # 5 fold dicts + 1 global dict
    assert len(results) == 6, f"Expected 6 dicts, got {len(results)}"

    fold_results = [r for r in results if r["fold"] != "global"]
    assert len(fold_results) == 5

    global_result = next(r for r in results if r["fold"] == "global")
    assert "accuracy" in global_result
    assert "profit_factor" in global_result


def test_walk_forward_fold_keys(sample_ohlcv_df):
    X, y = build_dataset(sample_ohlcv_df, window=50)
    price_moves = np.ones(len(y), dtype="float64")

    results = run_walk_forward(X, y, price_moves, n_splits=5, model_type="lgbm")

    for r in results:
        assert "accuracy" in r, f"'accuracy' missing in fold {r.get('fold')}"
        assert "profit_factor" in r, f"'profit_factor' missing in fold {r.get('fold')}"
