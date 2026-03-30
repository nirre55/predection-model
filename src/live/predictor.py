import csv
import time as _time
import time
import collections
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.model import serializer
from src.model import scheduler as sched
from src.features import builder
from src.data.fetcher import fetch_klines

PREDICTIONS_CSV = Path("models/predictions.csv")
_CSV_FIELDS = [
    "predicted_at", "candle_open", "candle_close", "slot",
    "predicted_direction", "probability_pct",
    "other_direction", "other_probability_pct", "confidence_pct",
    "actual_direction", "result",
]


def _append_csv(pred: "Prediction", confidence: float,
                actual_direction: str, result: str) -> None:
    is_new = not PREDICTIONS_CSV.exists()
    other_direction = "ROUGE" if pred.direction == "VERT" else "VERT"
    with PREDICTIONS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "predicted_at":          pred.predicted_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "candle_open":           pred.candle_open.strftime("%Y-%m-%d %H:%M"),
            "candle_close":          pred.candle_close.strftime("%Y-%m-%d %H:%M"),
            "slot":                  pred.model_slot,
            "predicted_direction":   pred.direction,
            "probability_pct":       f"{pred.probability * 100:.2f}",
            "other_direction":       other_direction,
            "other_probability_pct": f"{(1 - pred.probability) * 100:.2f}",
            "confidence_pct":        f"{confidence:.2f}",
            "actual_direction":      actual_direction,
            "result":                result,
        })


@dataclass
class Prediction:
    candle_open: datetime
    candle_close: datetime
    direction: str
    probability: float
    predicted_at: datetime
    model_slot: str = "default"


@dataclass
class _Pending:
    """Prédiction en attente de vérification (bougie pas encore fermée)."""
    pred: Prediction
    confidence: float


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
        self._context_dfs: dict[str, pd.DataFrame | None] = {"1h": None, "4h": None}
        self._last_context_refresh: float = 0.0

    def _get_model_and_meta(self, dt: datetime) -> tuple:
        if self._scheduler is not None:
            return self._scheduler.get_model(dt)
        return self._single_model, self._single_meta

    def _active_window(self, meta: dict) -> int:
        return meta.get("window", self.window)

    def _active_indicators(self, meta: dict) -> list[str] | None:
        return meta.get("indicators", None)

    def _refresh_context(self) -> None:
        import pandas as pd
        self._context_dfs["1h"] = fetch_klines(self.symbol, "1h", limit=50)
        self._context_dfs["4h"] = fetch_klines(self.symbol, "4h", limit=20)
        self._last_context_refresh = _time.time()

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

        df_1h = self._context_dfs.get("1h") if meta.get("multitf_enabled", False) else None
        df_4h = self._context_dfs.get("4h") if meta.get("multitf_enabled", False) else None
        X = builder.build_inference_features(candles, window=win, indicators=indicators, predict_time=pt, df_1h=df_1h, df_4h=df_4h)
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
        now = datetime.now(timezone.utc)
        model, meta = self._get_model_and_meta(now)
        active_window = self._active_window(meta)

        self.buffer = self._init_buffer(active_window)
        print(f"Buffer initialise avec {len(self.buffer)} bougies (window={active_window}). En attente...")

        # Refresh initial du contexte multi-timeframe si activé
        if sched.has_schedule() or meta.get("multitf_enabled", False):
            if meta.get("multitf_enabled", False):
                self._refresh_context()

        if self._scheduler is not None:
            print(self._scheduler.describe(now))

        pending: _Pending | None = None

        try:
            while True:
                now = datetime.now(timezone.utc)
                next_trigger = _next_trigger_time(now)
                delay = (next_trigger - now).total_seconds()
                if delay > 0:
                    time.sleep(delay)

                now = datetime.now(timezone.utc)
                model, meta = self._get_model_and_meta(now)
                active_window = self._active_window(meta)

                if meta.get("multitf_enabled", False):
                    if _time.time() - self._last_context_refresh > 3600:
                        self._refresh_context()

                current_maxlen = self.buffer.maxlen or (active_window + 1)
                if current_maxlen != active_window + 1:
                    print(f"[Scheduler] Changement de fenetre : {current_maxlen - 1} -> {active_window}")
                    self.buffer = self._init_buffer(active_window)

                df = fetch_klines(self.symbol, self.interval, limit=2)
                last_closed = df.iloc[-2]

                # --- Résoudre la prédiction précédente ---
                if pending is not None:
                    actual_green = float(last_closed["close"]) > float(last_closed["open"])
                    actual_dir = "VERT" if actual_green else "ROUGE"
                    result = "WIN" if actual_dir == pending.pred.direction else "LOSS"
                    _append_csv(pending.pred, pending.confidence, actual_dir, result)
                    result_str = "WIN" if result == "WIN" else "LOSS"
                    print(f"  --> Bougie fermee : {actual_dir} | {result_str}")

                # --- Mettre à jour le buffer ---
                self.buffer.append({
                    "open_time": last_closed["open_time"],
                    "open": last_closed["open"],
                    "high": last_closed["high"],
                    "low": last_closed["low"],
                    "close": last_closed["close"],
                    "volume": last_closed["volume"],
                })

                # --- Nouvelle prédiction ---
                pred_time = _current_candle_open(now)
                pred = self.predict(self.buffer, model, meta, predict_time=pred_time)

                other_direction = "ROUGE" if pred.direction == "VERT" else "VERT"
                other_prob = 1.0 - pred.probability
                confidence = abs(pred.probability - 0.5) * 200
                conf_str = f"conf={confidence:.1f}%"

                open_str = pred.candle_open.strftime("%H:%M")
                close_str = pred.candle_close.strftime("%H:%M")
                slot_str = f" [{pred.model_slot}]" if pred.model_slot != "default" else ""
                print(
                    f"[{open_str}->{close_str}]{slot_str} "
                    f"{pred.direction} {pred.probability:.2%} | "
                    f"{other_direction} {other_prob:.2%} | {conf_str}"
                )

                # Seuil de confiance par slot (optionnel)
                min_conf = meta.get("min_confidence_pct", 0.0)
                if confidence < min_conf:
                    print(f"  [SKIP] Confiance {confidence:.1f}% < seuil {min_conf:.1f}% — trade ignore")
                    pending = None
                    continue

                pending = _Pending(pred=pred, confidence=confidence)

        except KeyboardInterrupt:
            # Écrire la dernière prédiction comme PENDING si le programme est arrêté
            if pending is not None:
                _append_csv(pending.pred, pending.confidence, "PENDING", "PENDING")
                print("\n[CSV] Derniere prediction sauvegardee comme PENDING.")


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
