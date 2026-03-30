"""
Backtest en mode options binaires (payout 90%, loss=-100%).
Break-even a 52.63% win rate.
Analyse le seuil de confiance optimal pour maximiser le P&L total.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.data.cache import load_klines
from src.features.builder import build_dataset_with_target_times
from src.model.serializer import load_model

SYMBOL = "BTCUSDT"
INTERVAL = "5m"
SCHEDULE_PATH = Path("models/schedule.json")
BEST_INDICATORS = ["rsi", "macd", "atr", "mfi", "vdelta", "body", "streak"]
WIN_PAYOUT = 0.90   # gain net si WIN
LOSS_PAYOUT = -1.0  # perte nette si LOSS
STAKE = 100.0       # mise par trade ($)
TEST_RATIO = 0.20
START = "2 years ago UTC"
THRESHOLDS = [round(x * 2.5, 1) for x in range(0, 25)]  # 0 -> 60%

import re

def parse_start(s):
    m = re.match(r"(\d+)\s+(year|month|day)s?\s+ago", s, re.I)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        off = {"year": pd.DateOffset(years=n), "month": pd.DateOffset(months=n), "day": pd.DateOffset(days=n)}
        return pd.Timestamp.now(tz="UTC") - off[unit]
    ts = pd.Timestamp(s)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts


def load_schedule():
    return json.loads(SCHEDULE_PATH.read_text())


def get_slot(dt, schedule):
    for slot in schedule:
        if slot.get("default"):
            continue
        if dt.dayofweek in slot["dow"] and slot["hour_start"] <= dt.hour < slot["hour_end"]:
            return slot
    return next((s for s in schedule if s.get("default")), None)


class ModelCache:
    def __init__(self): self._cache = {}
    def get(self, p):
        p = p.replace("\\", "/")
        if p not in self._cache: self._cache[p] = load_model(p)
        return self._cache[p]


print("Chargement donnees...")
df = load_klines(SYMBOL, INTERVAL)
df = df[df["open_time"] >= parse_start(START)].reset_index(drop=True)
print(f"{len(df)} bougies")

df_1h = load_klines(SYMBOL, "1h")
df_4h = load_klines(SYMBOL, "4h")

schedule = load_schedule()
model_cache = ModelCache()

windows = set(s["window"] for s in schedule if s.get("window") is not None)
matrices = {}
for w in sorted(windows):
    print(f"  window={w}...", end=" ", flush=True)
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

# --- Simulation ---
rows = []
for i, ts in enumerate(ref_times):
    if ref_dt[i] < test_cutoff:
        continue
    dt = ref_dt[i]
    slot = get_slot(dt, schedule)
    if slot is None or slot.get("window") is None:
        continue

    window = slot["window"]
    X_w, y_w, times_w = matrices[window]
    dt_w = pd.to_datetime(times_w, utc=True)
    match = np.where(dt_w == dt)[0]
    if len(match) == 0:
        continue
    j = match[0]

    try:
        model, meta = model_cache.get(slot["model_path"].replace("\\", "/"))
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

    actual_dir = "VERT" if close_arr[target_idx] > open_arr[target_idx] else "ROUGE"
    result = "WIN" if actual_dir == predicted else "LOSS"

    pnl = STAKE * WIN_PAYOUT if result == "WIN" else STAKE * LOSS_PAYOUT
    rows.append({
        "slot": slot["slot"],
        "confidence_pct": round(confidence, 3),
        "result": result,
        "pnl": round(pnl, 2),
    })

all_trades = pd.DataFrame(rows)
n_days = (ref_dt[-1] - test_cutoff).days
print(f"{len(all_trades)} trades sur {n_days} jours ({len(all_trades)/n_days:.0f} trades/jour)")

BREAK_EVEN_WR = 1.0 / (1.0 + WIN_PAYOUT)  # 52.63%

print(f"\nPayout: +{WIN_PAYOUT*100:.0f}% WIN / {LOSS_PAYOUT*100:.0f}% LOSS")
print(f"Break-even win rate: {BREAK_EVEN_WR*100:.2f}%")
print(f"Mise par trade: ${STAKE:.0f}")
print()
print("=" * 100)
print(f"{'Seuil':>7} {'Trades':>8} {'Trades/j':>9} {'Win%':>8} {'Edge/trade':>11} {'P&L total':>12} {'ROI%':>8} {'Sharpe':>8}")
print("-" * 100)

results = []
for thr in THRESHOLDS:
    t = all_trades[all_trades["confidence_pct"] >= thr]
    if len(t) < 20:
        break
    wins = (t["result"] == "WIN").sum()
    wr = wins / len(t)
    edge = wr * WIN_PAYOUT + (1 - wr) * LOSS_PAYOUT   # E[P&L] per $1 stake
    total_pnl = t["pnl"].sum()
    roi_pct = total_pnl / (STAKE * len(t)) * 100       # ROI sur capital total misé
    trades_per_day = len(t) / n_days

    # Sharpe simplifié (E / std de la distribution de paiements)
    pnl_per_stake = t["pnl"] / STAKE
    sharpe = (pnl_per_stake.mean() / pnl_per_stake.std() * np.sqrt(len(t) / n_days * 365)) if pnl_per_stake.std() > 0 else 0

    marker = ""
    if total_pnl == max(all_trades[all_trades["confidence_pct"] >= r]["pnl"].sum() for r in THRESHOLDS if len(all_trades[all_trades["confidence_pct"] >= r]) >= 20):
        marker = " <- MAX PROFIT"

    print(f"{thr:>6.1f}% {len(t):>8} {trades_per_day:>8.1f} {wr*100:>7.2f}%  {edge*100:>+9.2f}%  ${total_pnl:>+10.0f}  {roi_pct:>7.2f}%  {sharpe:>7.3f}{marker}")
    results.append({
        "threshold_pct": thr,
        "n_trades": int(len(t)),
        "trades_per_day": round(trades_per_day, 1),
        "win_rate_pct": round(wr * 100, 2),
        "edge_pct": round(edge * 100, 4),
        "total_pnl_usd": round(total_pnl, 2),
        "roi_pct": round(roi_pct, 4),
        "sharpe": round(sharpe, 4),
    })

# --- Meilleur seuil ---
best = max(results, key=lambda r: r["total_pnl_usd"])
print(f"\nMeilleur seuil pour max profit total : {best['threshold_pct']}% ({best['n_trades']} trades, WR={best['win_rate_pct']}%)")

# --- Par slot au meilleur seuil ---
print(f"\n=== DETAIL PAR SLOT (seuil >= {best['threshold_pct']}%) ===")
t_best = all_trades[all_trades["confidence_pct"] >= best["threshold_pct"]]
for slot_name, grp in t_best.groupby("slot"):
    w = (grp["result"] == "WIN").sum()
    wr_s = w / len(grp)
    pnl_s = grp["pnl"].sum()
    edge_s = wr_s * WIN_PAYOUT + (1 - wr_s) * LOSS_PAYOUT
    print(f"  {slot_name:<22}: {len(grp):>5} trades, WR={wr_s*100:.1f}%, edge={edge_s*100:+.2f}%, P&L=${pnl_s:+.0f}")

# --- Simulation Kelly/Fixed-fraction ---
print(f"\n=== SIMULATION GESTION DE CAPITAL (seuil >= {best['threshold_pct']}%) ===")
t_best = t_best.sort_index()
capital = 1000.0
for idx, row in t_best.iterrows():
    stake = capital * 0.02  # 2% par trade (Kelly conservateur)
    if row["result"] == "WIN":
        capital += stake * WIN_PAYOUT
    else:
        capital += stake * LOSS_PAYOUT
print(f"  Capital initial: $1,000")
print(f"  Capital final:   ${capital:.2f}")
print(f"  ROI:             {(capital/1000-1)*100:.1f}%")
print(f"  Periode:         {n_days} jours")

Path("models/binary_options_backtest.json").write_text(json.dumps(results, indent=2))
print("\nRapport sauvegarde -> models/binary_options_backtest.json")
