import time
import collections
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

from src.model import serializer
from src.features import builder
from src.data.fetcher import fetch_klines


@dataclass
class Prediction:
    candle_open: datetime
    candle_close: datetime
    direction: str
    probability: float
    predicted_at: datetime


class LivePredictor:
    def __init__(
        self,
        symbol: str,
        interval: str,
        window: int,
        model_path: str = "models/model_calibrated.pkl",
    ):
        self.symbol = symbol
        self.interval = interval
        self.window = window
        self.model, self.metadata = serializer.load_model(model_path)
        self.buffer: collections.deque = collections.deque(maxlen=window + 1)

    def _init_buffer(self):
        df = fetch_klines(self.symbol, self.interval, limit=self.window + 2)
        # Ignore the last candle (currently open)
        closed_candles = df.iloc[:-1]
        for _, row in closed_candles.tail(self.window + 1).iterrows():
            self.buffer.append(
                {
                    "open_time": row["open_time"],
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                }
            )

    def predict(self, candles: collections.deque) -> Prediction:
        X = builder.build_inference_features(candles, self.window)
        proba = self.model.predict_proba(X)[0]
        prob_green = proba[1]
        prob_red = proba[0]

        if prob_green >= 0.5:
            direction = "VERT"
            probability = prob_green
        else:
            direction = "ROUGE"
            probability = prob_red

        now = datetime.now(timezone.utc)
        # La bougie prédite est celle qui vient de s'ouvrir (= bougie en cours)
        candle_open = _current_candle_open(now)
        candle_close = candle_open + timedelta(minutes=5)

        return Prediction(
            candle_open=candle_open,
            candle_close=candle_close,
            direction=direction,
            probability=probability,
            predicted_at=now,
        )

    def start(self):
        self._init_buffer()
        print(f"Buffer initialisé avec {len(self.buffer)} bougies. En attente...")

        while True:
            now = datetime.now(timezone.utc)
            next_trigger = _next_trigger_time(now)
            delay = (next_trigger - now).total_seconds()
            if delay > 0:
                time.sleep(delay)

            # Fetch the last closed candle only
            df = fetch_klines(self.symbol, self.interval, limit=2)
            last_closed = df.iloc[-2]  # ignore the currently open candle
            self.buffer.append(
                {
                    "open_time": last_closed["open_time"],
                    "open": last_closed["open"],
                    "high": last_closed["high"],
                    "low": last_closed["low"],
                    "close": last_closed["close"],
                    "volume": last_closed["volume"],
                }
            )

            pred = self.predict(self.buffer)
            other_direction = "ROUGE" if pred.direction == "VERT" else "VERT"
            other_prob = 1.0 - pred.probability

            open_str = pred.candle_open.strftime("%H:%M")
            close_str = pred.candle_close.strftime("%H:%M")
            print(
                f"[{open_str}→{close_str}] Prédiction : {pred.direction} à {pred.probability:.0%}"
                f" | {other_direction} à {other_prob:.0%}"
            )


def _current_candle_open(now: datetime) -> datetime:
    """Retourne l'ouverture de la bougie M5 en cours (plancher à la minute multiple de 5)."""
    current_5 = (now.minute // 5) * 5
    return now.replace(minute=current_5, second=0, microsecond=0)


def _next_trigger_time(now: datetime) -> datetime:
    """Prochain déclenchement : prochaine minute multiple de 5 + 2s de buffer."""
    next_5 = ((now.minute // 5) + 1) * 5
    if next_5 >= 60:
        next_open = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        next_open = now.replace(minute=next_5, second=0, microsecond=0)
    return next_open + timedelta(seconds=2)
