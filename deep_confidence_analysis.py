"""
Analyse approfondie du seuil de confiance optimal.
Teste des seuils de 5% a 60% par pas de 2.5%.
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
from src.model.serializer import load_model

SYMBOL = "BTCUSDT"
INTERVAL = "5m"
SCHEDULE_PATH = Path("models/schedule.json")
BEST_INDICATORS = ["rsi", "macd", "atr", "mfi", "vdelta", "body", "streak"]
FEE_RT = 0.002
TEST_RATIO = 0.20
THRESHOLDS = [round(x * 2.5, 1) for x in range(0, 25)]  # 0 -> 60%
POSITION_USD = 1000.0
START = "2 years ago UTC"

import re

def parse_start(start):
    m = re.match(r"(\d+)\s+(year|month|day)s?\s+ago", start, re.I)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        offsets = {"year": pd.DateOffset(years=n), "month": pd.DateOffset(months=n), "day": pd.DateOffset(days=n)}
        return pd.Timestamp.now(tz="UTC") - offsets[unit]
    ts = pd.Timestamp(start)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts


def load_schedule():
    return json.loads(SCHEDULE_PATH.read_text())


def get_slot(dt, schedule):
    dow = dt.dayofweek
    hour = dt.hour
    for slot in schedule:
        if slot.get("default"):
            continue
        if dow in slot["dow"] and slot["hour_start"] <= hour < slot["hour_end"]:
            return slot
    return next((s for s in schedule if s.get("default")), None)


class ModelCache:
    def __init__(self):
        self._cache = {}

    def get(self, model_path):
        path = model_path.replace("\\", "/")
        if path not in self._cache:
            self._cache[path] = load_model(path)
        return self._cache[path]


print(f"Chargement {SYMBOL} {INTERVAL}...")
df = load_klines(SYMBOL, INTERVAL)
cutoff = parse_start(START)
df = df[df["open_time"] >= cutoff].reset_index(drop=True)
print(f"{len(df)} bougies depuis {cutoff.date()}")

print("Chargement MTF...")
df_1h = load_klines(SYMBOL, "1h")
df_4h = load_klines(SYMBOL, "4h")

schedule = load_schedule()
model_cache = ModelCache()

windows = set(s["window"] for s in schedule if s.get("window") is not None)
matrices = {}
for w in sorted(windows):
    print(f"  features window={w}...", end=" ", flush=True)
    X, y, times = build_dataset_with_target_times(
        df, window=w, indicators=BEST_INDICATORS, include_time=True,
        df_1h=df_1h, df_4h=df_4h,
    )
    matrices[w] = (X, y, times)
    print(f"{len(X)} samples")

ref_window = min(matrices.keys())
_, _, ref_times = matrices[ref_window]
ref_dt = pd.to_datetime(ref_times, utc=True)
test_cutoff = ref_dt[int(len(ref_dt) * (1 - TEST_RATIO))]
print(f"\nTest set : {test_cutoff.date()} -> {ref_dt[-1].date()}")

close_arr = np.asarray(df["close"].values, dtype="float64")
open_arr = np.asarray(df["open"].values, dtype="float64")

print("Simulation en cours...")
rows = []
for i, ts in enumerate(ref_times):
    if ref_dt[i] < test_cutoff:
        continue
    dt = ref_dt[i]
    slot = get_slot(dt, schedule)
    if slot is None or slot.get("window") is None:
        continue

    window = slot["window"]
    model_path = slot["model_path"].replace("\\", "/")
    X_w, y_w, times_w = matrices[window]
    dt_w = pd.to_datetime(times_w, utc=True)
    match = np.where(dt_w == dt)[0]
    if len(match) == 0:
        continue
    j = match[0]

    model, meta = model_cache.get(model_path)
    try:
        proba = model.predict_proba(X_w[j:j+1])[0]
    except Exception:
        continue

    prob_green = float(proba[1])
    predicted = "VERT" if prob_green >= 0.5 else "ROUGE"
    prob_pred = prob_green if predicted == "VERT" else float(proba[0])
    confidence = abs(prob_pred - 0.5) * 200

    target_idx = j + window + 2
    if target_idx >= len(df):
        continue

    actual_open = open_arr[target_idx]
    actual_close = close_arr[target_idx]
    actual_dir = "VERT" if actual_close > actual_open else "ROUGE"
    result = "WIN" if actual_dir == predicted else "LOSS"

    price_move_pct = abs(actual_close - actual_open) / actual_open
    if result == "WIN":
        pnl_usd = POSITION_USD * price_move_pct - POSITION_USD * FEE_RT
    else:
        pnl_usd = -(POSITION_USD * price_move_pct + POSITION_USD * FEE_RT)

    rows.append({
        "slot": slot["slot"],
        "confidence_pct": round(float(confidence), 3),
        "result": result,
        "pnl_usd": round(float(pnl_usd), 4),
        "price_move_pct": round(float(price_move_pct) * 100, 4),
    })

all_trades = pd.DataFrame(rows)
print(f"{len(all_trades)} trades pre-calcules.")

print("\n" + "=" * 85)
print(f"{'Seuil':>7} {'Trades':>8} {'Win%':>8} {'P&L':>13} {'PF':>8} {'AvgMove%':>10} {'AvgWin$':>10}")
print("-" * 85)

results = []
for thr in THRESHOLDS:
    trades = all_trades[all_trades["confidence_pct"] >= thr]
    if len(trades) < 10:
        break
    wins = trades[trades["result"] == "WIN"]
    losses = trades[trades["result"] == "LOSS"]
    win_pct = len(wins) / len(trades) * 100
    total_pnl = trades["pnl_usd"].sum()
    gross_profit = wins["pnl_usd"].sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses["pnl_usd"].sum()) if len(losses) > 0 else 1e-9
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_move = trades["price_move_pct"].mean()
    avg_win = wins["pnl_usd"].mean() if len(wins) > 0 else 0.0

    marker = " <-- PROFITABLE" if total_pnl > 0 else ""
    print(f"{thr:>6.1f}% {len(trades):>8} {win_pct:>7.2f}% {total_pnl:>+12.2f} {pf:>8.4f} {avg_move:>9.3f}% {avg_win:>+9.2f}{marker}")
    results.append({
        "threshold": thr,
        "n_trades": len(trades),
        "win_pct": round(win_pct, 2),
        "total_pnl": round(total_pnl, 2),
        "profit_factor": round(pf, 4),
        "avg_move_pct": round(avg_move, 4),
        "avg_win_usd": round(avg_win, 4),
    })

# Analyse par slot aux seuils clés
print("\n=== WIN RATE PAR SLOT (seuils 20%, 35%, 50%) ===")
for thr in [20.0, 35.0, 50.0]:
    trades = all_trades[all_trades["confidence_pct"] >= thr]
    if len(trades) < 10:
        continue
    print(f"\nSeuil >= {thr}% ({len(trades)} trades):")
    for slot_name, grp in trades.groupby("slot"):
        w = (grp["result"] == "WIN").sum()
        pnl = grp["pnl_usd"].sum()
        print(f"  {slot_name:<22}: {len(grp):>5} trades, WR={w/len(grp)*100:.1f}%, P&L=${pnl:+.2f}")

Path("models/deep_confidence_analysis.json").write_text(
    json.dumps(results, indent=2)
)
print("\nAnalyse sauvegardee -> models/deep_confidence_analysis.json")
