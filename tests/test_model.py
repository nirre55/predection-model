import numpy as np
import pytest

from src.features.builder import build_dataset
from src.model.trainer import train, _PrefitCalibratedClassifier


def test_train_returns_calibrated_model(sample_ohlcv_df):
    X, y = build_dataset(sample_ohlcv_df, window=50)
    model = train(X, y, model_type="lgbm")
    assert isinstance(model, _PrefitCalibratedClassifier)


def test_predict_proba_sums_to_one(sample_ohlcv_df):
    X, y = build_dataset(sample_ohlcv_df, window=50)
    model = train(X, y, model_type="lgbm")

    n = len(X)
    X_test = X[int(n * 0.8):]
    proba = model.predict_proba(X_test)

    sums = proba.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6), "predict_proba rows do not sum to 1"


def test_predict_proba_in_range(sample_ohlcv_df):
    X, y = build_dataset(sample_ohlcv_df, window=50)
    model = train(X, y, model_type="lgbm")

    n = len(X)
    X_test = X[int(n * 0.8):]
    proba = model.predict_proba(X_test)

    assert (proba >= 0).all() and (proba <= 1).all(), "Probabilities outside [0, 1]"
