import collections
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.live.predictor import LivePredictor, Prediction

_NOW = datetime.now(timezone.utc)


def _make_mock_deque(window: int = 50) -> collections.deque:
    rng = np.random.default_rng(1)
    candles = collections.deque(maxlen=window + 1)
    for _ in range(window + 1):
        candles.append({
            "open": rng.uniform(29000, 31000),
            "high": rng.uniform(31000, 32000),
            "low": rng.uniform(28000, 29000),
            "close": rng.uniform(29000, 31000),
            "volume": rng.uniform(1, 100),
        })
    return candles


def _make_predictor(window: int, mock_model) -> LivePredictor:
    """Crée un LivePredictor avec un modèle mocké (sans scheduler)."""
    meta = {"window": window, "indicators": None, "include_time": False}
    with patch("src.live.predictor.sched.has_schedule", return_value=False), \
         patch("src.live.predictor.serializer.load_model", return_value=(mock_model, meta)):
        return LivePredictor(
            symbol="BTCUSDT",
            interval="5m",
            window=window,
            model_path="models/model_calibrated.pkl",
        )


def test_predict_returns_prediction_dataclass():
    window = 50
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.35, 0.65]])
    meta = {"window": window, "indicators": None, "include_time": False}

    predictor = _make_predictor(window, mock_model)
    candles = _make_mock_deque(window)
    pred = predictor.predict(candles, mock_model, meta, predict_time=_NOW)

    assert isinstance(pred, Prediction)


def test_predict_direction_vert():
    window = 50
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.30, 0.70]])
    meta = {"window": window, "indicators": None, "include_time": False}

    predictor = _make_predictor(window, mock_model)
    candles = _make_mock_deque(window)
    pred = predictor.predict(candles, mock_model, meta, predict_time=_NOW)

    assert pred.direction == "VERT"
    assert 0.0 <= pred.probability <= 1.0


def test_predict_direction_rouge():
    window = 50
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.72, 0.28]])
    meta = {"window": window, "indicators": None, "include_time": False}

    predictor = _make_predictor(window, mock_model)
    candles = _make_mock_deque(window)
    pred = predictor.predict(candles, mock_model, meta, predict_time=_NOW)

    assert pred.direction == "ROUGE"
    assert 0.0 <= pred.probability <= 1.0


def test_predict_direction_in_valid_set():
    window = 50
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.50, 0.50]])
    meta = {"window": window, "indicators": None, "include_time": False}

    predictor = _make_predictor(window, mock_model)
    candles = _make_mock_deque(window)
    pred = predictor.predict(candles, mock_model, meta, predict_time=_NOW)

    assert pred.direction in {"VERT", "ROUGE"}
    assert 0.0 <= pred.probability <= 1.0
