from __future__ import annotations

import argparse
import collections
import csv
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.fetcher import fetch_klines  # noqa: E402
from src.features import builder  # noqa: E402
from src.model.serializer import load_model  # noqa: E402

DEFAULT_MODEL_PATH = "models/model_calibrated.pkl"
DEFAULT_OUTPUT_DIR = Path("utils/generated_predictions")
CSV_FIELDS = [
    "predicted_at",
    "candle_open",
    "candle_close",
    "slot",
    "predicted_direction",
    "probability_pct",
    "other_direction",
    "other_probability_pct",
    "confidence_pct",
    "actual_direction",
    "result",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rejoue le modele sur une plage de dates et genere un CSV type predictions.csv."
    )
    parser.add_argument("--start-date", required=True, help="Date debut au format YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Date fin au format YYYY-MM-DD")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbole Binance (defaut: BTCUSDT)")
    parser.add_argument("--interval", default="5m", help="Intervalle (defaut: 5m)")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Dossier de sortie pour le CSV genere",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Historique additionnel a telecharger avant la date debut pour construire les features",
    )
    return parser.parse_args()


def parse_date_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)


def to_utc_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def load_schedule_entries() -> list[dict]:
    schedule_path = Path("models/schedule.json")
    if schedule_path.exists():
        import json

        return json.loads(schedule_path.read_text(encoding="utf-8"))
    return [{"slot": "default", "default": True, "model_path": DEFAULT_MODEL_PATH}]


def get_slot(dt: pd.Timestamp, schedule: list[dict]) -> dict:
    dow = dt.dayofweek
    hour = dt.hour
    for slot in schedule:
        if slot.get("default"):
            continue
        if dow in slot["dow"] and slot["hour_start"] <= hour < slot["hour_end"]:
            return slot
    default = next((slot for slot in schedule if slot.get("default")), None)
    if default is None:
        raise RuntimeError("Aucun slot de schedule et aucun modele par defaut disponibles.")
    return default


def collect_model_specs(schedule: list[dict]) -> tuple[dict[str, dict], bool]:
    specs_by_path: dict[str, dict] = {}
    needs_multitf = False

    for slot in schedule:
        model_path = slot["model_path"].replace("\\", "/")
        if model_path in specs_by_path:
            continue
        _model, meta = load_model(model_path)
        spec = {
            "model_path": model_path,
            "window": int(meta.get("window", 50)),
            "indicators": list(meta.get("indicators", ["rsi", "macd", "atr"])),
            "include_time": bool(meta.get("include_time", False)),
            "multitf_enabled": bool(meta.get("multitf_enabled", False)),
            "slot": meta.get("slot", "default"),
        }
        specs_by_path[model_path] = spec
        needs_multitf = needs_multitf or spec["multitf_enabled"]
    return specs_by_path, needs_multitf


def build_multitf_asof(df_5m: pd.DataFrame, dt: pd.Timestamp, rule: str) -> pd.DataFrame:
    history = df_5m[df_5m["open_time"] < dt].copy()
    if history.empty:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

    history = history.set_index("open_time")
    grouped = history.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    grouped = grouped.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return grouped[["open_time", "open", "high", "low", "close", "volume"]]


def replay_predictions(
    df_5m: pd.DataFrame,
    schedule: list[dict],
    specs_by_path: dict[str, dict],
    start_dt: datetime,
    end_dt_exclusive: datetime,
) -> list[dict]:
    model_cache: dict[str, tuple] = {}
    rows: list[dict] = []
    df_5m = df_5m.sort_values("open_time").reset_index(drop=True)
    open_times_ns = np.asarray(df_5m["open_time"].values, dtype="datetime64[ns]").astype("int64")

    for slot in schedule:
        model_path = slot["model_path"].replace("\\", "/")
        if model_path not in model_cache:
            model_cache[model_path] = load_model(model_path)

    for dt in pd.date_range(start=start_dt, end=end_dt_exclusive - timedelta(minutes=5), freq="5min", tz="UTC"):
        candle_idx = int(np.searchsorted(open_times_ns, dt.value, side="left"))
        if candle_idx >= len(df_5m):
            continue
        candle_row = df_5m.iloc[candle_idx]
        if to_utc_timestamp(candle_row["open_time"]) != dt:
            continue

        slot = get_slot(dt, schedule)
        model_path = slot["model_path"].replace("\\", "/")
        spec = specs_by_path[model_path]
        window = spec["window"]
        if candle_idx < window + 1:
            continue

        model, meta = model_cache[model_path]
        hist = df_5m.iloc[candle_idx - (window + 1) : candle_idx]
        candles = [
            {
                "open_time": row["open_time"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            for _, row in hist.iterrows()
        ]
        predict_time = dt.to_pydatetime() if spec["include_time"] else None
        df_1h_asof = None
        df_4h_asof = None
        if spec["multitf_enabled"]:
            df_1h_asof = build_multitf_asof(df_5m, dt, "1h")
            df_4h_asof = build_multitf_asof(df_5m, dt, "4h")

        X_sample = builder.build_inference_features(
            collections.deque(candles, maxlen=window + 1),
            window=window,
            indicators=spec["indicators"],
            predict_time=predict_time,
            df_1h=df_1h_asof,
            df_4h=df_4h_asof,
        )

        actual_green = bool(float(candle_row["close"]) > float(candle_row["open"]))
        actual_direction = "VERT" if actual_green else "ROUGE"

        proba = model.predict_proba(X_sample)[0]
        prob_green = float(proba[1])
        prob_red = float(proba[0])
        if prob_green >= 0.5:
            predicted_direction = "VERT"
            predicted_probability = prob_green
            other_direction = "ROUGE"
            other_probability = prob_red
        else:
            predicted_direction = "ROUGE"
            predicted_probability = prob_red
            other_direction = "VERT"
            other_probability = prob_green

        confidence = abs(predicted_probability - 0.5) * 200
        result = "WIN" if predicted_direction == actual_direction else "LOSS"
        predicted_at = dt + timedelta(seconds=2)
        candle_close = dt + timedelta(minutes=5)

        rows.append(
            {
                "predicted_at": predicted_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "candle_open": dt.strftime("%Y-%m-%d %H:%M"),
                "candle_close": candle_close.strftime("%Y-%m-%d %H:%M"),
                "slot": meta.get("slot", slot.get("slot", "default")),
                "predicted_direction": predicted_direction,
                "probability_pct": f"{predicted_probability * 100:.2f}",
                "other_direction": other_direction,
                "other_probability_pct": f"{other_probability * 100:.2f}",
                "confidence_pct": f"{confidence:.2f}",
                "actual_direction": actual_direction,
                "result": result,
            }
        )

    return rows


def save_predictions(rows: list[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    args = parse_args()
    start_dt = parse_date_utc(args.start_date)
    end_dt = parse_date_utc(args.end_date)
    if end_dt < start_dt:
        raise ValueError("end-date doit etre superieure ou egale a start-date.")
    end_dt_exclusive = end_dt + timedelta(days=1)

    history_start = start_dt - timedelta(days=max(args.lookback_days, 1))
    start_str = history_start.strftime("%Y-%m-%d")
    end_str = end_dt_exclusive.strftime("%Y-%m-%d")

    print(f"Telechargement des bougies {args.symbol} {args.interval} du {start_str} au {end_str}...")
    df_5m = fetch_klines(args.symbol, args.interval, start_str=start_str, end_str=end_str)
    if df_5m.empty:
        raise ValueError("Aucune donnee 5m recuperee pour cette plage.")

    schedule = load_schedule_entries()
    specs_by_path, needs_multitf = collect_model_specs(schedule)
    if needs_multitf:
        print("Contexte 1h/4h reconstruit depuis les bougies 5m pour chaque prediction...")

    print("Generation des predictions historiques...")
    rows = replay_predictions(
        df_5m=df_5m,
        schedule=schedule,
        specs_by_path=specs_by_path,
        start_dt=start_dt,
        end_dt_exclusive=end_dt_exclusive,
    )
    if not rows:
        raise ValueError("Aucune prediction generee sur cette plage. Essaie d'augmenter lookback-days.")

    output_dir = Path(args.output_dir)
    output_name = f"predictions_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"
    output_path = save_predictions(rows, output_dir / output_name)

    print(f"Predictions generees : {len(rows)}")
    print(f"Fichier CSV          : {output_path}")


if __name__ == "__main__":
    main()
