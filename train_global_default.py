"""
Entraîne le modèle global par défaut avec les nouveaux indicateurs.
Sauvegarde dans models/model_calibrated.pkl et met à jour schedule.json.
"""
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.data.cache import load_klines, save_klines
from src.data.fetcher import fetch_klines as fetch_api
from src.features.builder import build_dataset_with_target_times
from src.backtest.walk_forward import run_walk_forward
from src.model.trainer import train
from src.model.serializer import save_model

SYMBOL = "BTCUSDT"
INTERVAL = "5m"
START = "2 years ago UTC"
BEST_INDICATORS = ["rsi", "macd", "atr", "mfi", "vdelta", "body", "streak"]
INCLUDE_TIME = True
MIN_MOVE_PCT = 0.003
WINDOWS = [20, 50, 100]
MODEL_TYPES = ["lgbm", "xgb"]
N_SPLITS = 5
SCHEDULE_PATH = Path("models/schedule.json")


def parse_start(start: str):
    import re
    m = re.match(r"(\d+)\s+(year|month|day)s?\s+ago", start, re.I)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        offsets = {"year": pd.DateOffset(years=n), "month": pd.DateOffset(months=n), "day": pd.DateOffset(days=n)}
        return pd.Timestamp.now(tz="UTC") - offsets[unit]
    ts = pd.Timestamp(start)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts


print(f"Chargement des données {SYMBOL} {INTERVAL}...")
df = load_klines(SYMBOL, INTERVAL)
cutoff = parse_start(START)
df = df[df["open_time"] >= cutoff].reset_index(drop=True)
print(f"{len(df)} bougies depuis {cutoff.date()}")

try:
    df_1h = load_klines(SYMBOL, "1h")
except FileNotFoundError:
    print("Fetch 1h...")
    df_1h = fetch_api(SYMBOL, "1h", start_str="5 years ago UTC")
    save_klines(df_1h, SYMBOL, "1h")

try:
    df_4h = load_klines(SYMBOL, "4h")
except FileNotFoundError:
    print("Fetch 4h...")
    df_4h = fetch_api(SYMBOL, "4h", start_str="5 years ago UTC")
    save_klines(df_4h, SYMBOL, "4h")

best_score = -1
best_result = None

for window in WINDOWS:
    X, y, _ = build_dataset_with_target_times(
        df, window=window, indicators=BEST_INDICATORS,
        include_time=INCLUDE_TIME, df_1h=df_1h, df_4h=df_4h,
        min_move_pct=MIN_MOVE_PCT,
    )
    price_moves = np.abs(y.astype("float64") - 0.5) * 2

    for model_type in MODEL_TYPES:
        try:
            wf = run_walk_forward(X, y, price_moves, n_splits=N_SPLITS, model_type=model_type)
            g = next(r for r in wf if r["fold"] == "global")
            acc = g.get("accuracy", 0.0)
            pf = g.get("profit_factor", 0.0)
            score = acc * 0.6 + min(pf, 2.0) / 2.0 * 0.4
            print(f"  {model_type:<6} win={window:<4} acc={acc:.4f} pf={pf:.4f} score={score:.4f} n={len(X)}")

            if score > best_score:
                best_score = score
                best_result = {
                    "model_type": model_type,
                    "window": window,
                    "accuracy": acc,
                    "profit_factor": pf,
                    "score": score,
                    "n_samples": len(X),
                }
        except Exception as e:
            print(f"  ERREUR {model_type} win={window}: {e}")

assert best_result is not None, "Aucun modele entraine avec succes — verifiez les donnees et les parametres."
print(f"\nMeilleur global: {best_result['model_type']} win={best_result['window']} score={best_result['score']:.4f}")

X, y, _ = build_dataset_with_target_times(
    df, window=best_result["window"], indicators=BEST_INDICATORS,
    include_time=INCLUDE_TIME, df_1h=df_1h, df_4h=df_4h,
    min_move_pct=MIN_MOVE_PCT,
)
final_model = train(X, y, model_type=best_result["model_type"])

meta = {
    "symbol": SYMBOL,
    "interval": INTERVAL,
    "window": best_result["window"],
    "indicators": BEST_INDICATORS,
    "include_time": INCLUDE_TIME,
    "multitf_enabled": True,
    "min_move_pct": MIN_MOVE_PCT,
    "model_type": best_result["model_type"],
    "accuracy_wf": round(best_result["accuracy"], 6),
    "profit_factor_wf": round(best_result["profit_factor"], 6),
    "n_features": X.shape[1],
    "n_samples": len(X),
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "slot": "default",
}
save_model(final_model, meta, path="models/model_calibrated.pkl")
print(f"Modele global sauvegarde -> models/model_calibrated.pkl (n_features={X.shape[1]})")

# Mise à jour du schedule.json — remplace l'entrée default
schedule = json.loads(SCHEDULE_PATH.read_text())
schedule = [s for s in schedule if not s.get("default")]
schedule.append({
    "slot": "default",
    "default": True,
    "model_path": "models/model_calibrated.pkl",
    "window": best_result["window"],
    "model_type": best_result["model_type"],
    "accuracy_wf": round(best_result["accuracy"], 6),
    "profit_factor_wf": round(best_result["profit_factor"], 6),
    "n_samples": len(X),
})
SCHEDULE_PATH.write_text(json.dumps(schedule, indent=2, ensure_ascii=False))
print(f"schedule.json mis a jour avec le nouveau modele par defaut (window={best_result['window']})")
