import time
import collections
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

from src.model import serializer
from src.model import scheduler as sched
from src.features import builder
from src.data.fetcher import fetch_klines


@dataclass
class Prediction:
    candle_open: datetime
    candle_close: datetime
    direction: str
    probability: float
    predicted_at: datetime
    model_slot: str = "default"


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
        self.window = window  # fenêtre par défaut (peut être surchargée par le scheduler)

        # Utilise le scheduler si schedule.json existe, sinon modèle unique
        if sched.has_schedule():
            self._scheduler = sched.ModelScheduler()
            self._single_model = None
            self._single_meta: dict = {}
            print("[Scheduler] models/schedule.json detecte — routing temporel active.")
        else:
            self._scheduler = None
            self._single_model, self._single_meta = serializer.load_model(model_path)
            print(f"[Scheduler] Modele unique : {model_path}")

        self.buffer: collections.deque = collections.deque(maxlen=window + 1)

    def _get_model_and_meta(self, dt: datetime) -> tuple:
        if self._scheduler is not None:
            return self._scheduler.get_model(dt)
        return self._single_model, self._single_meta

    def _active_window(self, meta: dict) -> int:
        return meta.get("window", self.window)

    def _active_indicators(self, meta: dict) -> list[str] | None:
        return meta.get("indicators", None)

    def _init_buffer(self, window: int):
        df = fetch_klines(self.symbol, self.interval, limit=window + 2)
        closed_candles = df.iloc[:-1]
        buf: collections.deque = collections.deque(maxlen=window + 1)
        for _, row in closed_candles.tail(window + 1).iterrows():
            buf.append({
                "open_time": row["open_time"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            })
        return buf

    def predict(self, candles: collections.deque, model, meta: dict, predict_time: datetime) -> Prediction:
        win = self._active_window(meta)
        indicators = self._active_indicators(meta)
        include_time = meta.get("include_time", False)
        pt = predict_time if include_time else None

        X = builder.build_inference_features(candles, window=win, indicators=indicators, predict_time=pt)
        proba = model.predict_proba(X)[0]
        prob_green = proba[1]
        prob_red = proba[0]

        if prob_green >= 0.5:
            direction, probability = "VERT", prob_green
        else:
            direction, probability = "ROUGE", prob_red

        now = datetime.now(timezone.utc)
        candle_open = _current_candle_open(now)
        candle_close = candle_open + timedelta(minutes=5)

        slot = meta.get("slot", "default")
        return Prediction(
            candle_open=candle_open,
            candle_close=candle_close,
            direction=direction,
            probability=probability,
            predicted_at=now,
            model_slot=slot,
        )

    def start(self):
        # Initialise le buffer avec la fenêtre par défaut
        # (sera recalibré au besoin si le scheduler change de window)
        now = datetime.now(timezone.utc)
        model, meta = self._get_model_and_meta(now)
        active_window = self._active_window(meta)

        self.buffer = self._init_buffer(active_window)
        print(f"Buffer initialise avec {len(self.buffer)} bougies (window={active_window}). En attente...")

        if self._scheduler is not None:
            print(self._scheduler.describe(now))

        while True:
            now = datetime.now(timezone.utc)
            next_trigger = _next_trigger_time(now)
            delay = (next_trigger - now).total_seconds()
            if delay > 0:
                time.sleep(delay)

            now = datetime.now(timezone.utc)
            model, meta = self._get_model_and_meta(now)
            active_window = self._active_window(meta)

            # Resize le buffer si la fenêtre a changé (nouveau slot)
            current_maxlen = self.buffer.maxlen or (active_window + 1)
            if current_maxlen != active_window + 1:
                print(f"[Scheduler] Changement de fenetre : {current_maxlen - 1} -> {active_window}")
                self.buffer = self._init_buffer(active_window)

            df = fetch_klines(self.symbol, self.interval, limit=2)
            last_closed = df.iloc[-2]
            self.buffer.append({
                "open_time": last_closed["open_time"],
                "open": last_closed["open"],
                "high": last_closed["high"],
                "low": last_closed["low"],
                "close": last_closed["close"],
                "volume": last_closed["volume"],
            })

            # La bougie prédite s'ouvre maintenant
            pred_time = _current_candle_open(now)
            pred = self.predict(self.buffer, model, meta, predict_time=pred_time)

            other_direction = "ROUGE" if pred.direction == "VERT" else "VERT"
            other_prob = 1.0 - pred.probability

            open_str = pred.candle_open.strftime("%H:%M")
            close_str = pred.candle_close.strftime("%H:%M")
            slot_str = f" [{pred.model_slot}]" if pred.model_slot != "default" else ""
            print(
                f"[{open_str}->{close_str}]{slot_str} "
                f"{pred.direction} {pred.probability:.0%} | "
                f"{other_direction} {other_prob:.0%}"
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
