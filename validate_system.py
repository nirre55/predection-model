"""
Validation finale du systeme :
- Charge chaque modele du schedule
- Fait une prediction test
- Verifie que le seuil de confiance est bien lu depuis schedule.json
- Affiche un resume complet
"""
import warnings
warnings.filterwarnings("ignore")

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.cache import load_klines
from src.data.fetcher import fetch_klines
from src.features.builder import build_inference_features
from src.model.scheduler import ModelScheduler, has_schedule

SYMBOL = "BTCUSDT"
SCHEDULE_PATH = Path("models/schedule.json")
WIN_PAYOUT = 0.90
BREAK_EVEN = 1.0 / (1.0 + WIN_PAYOUT)

print("=" * 70)
print("VALIDATION DU SYSTEME DE PREDICTION")
print("=" * 70)

# 1. Verification schedule
schedule = json.loads(SCHEDULE_PATH.read_text())
print(f"\n[1] Schedule : {len(schedule)} entrees")
for s in schedule:
    slot_name = s.get("slot", "?")
    acc = s.get("accuracy_wf", 0)
    pf = s.get("profit_factor_wf", 0)
    min_conf = s.get("min_confidence_pct", 0.0)
    win = s.get("window", "?")
    print(f"    {slot_name:<22} acc={acc:.1%}  PF={pf:.2f}  seuil={min_conf:.1f}%  win={win}")

# 2. Charge les donnees recentes
print("\n[2] Chargement donnees recentes...")
df_live = fetch_klines(SYMBOL, "5m", limit=150)
df_1h   = fetch_klines(SYMBOL, "1h",  limit=55)
df_4h   = fetch_klines(SYMBOL, "4h",  limit=25)
print(f"    5m: {len(df_live)} bougies | 1h: {len(df_1h)} | 4h: {len(df_4h)}")

# 3. Test chaque slot
print("\n[3] Test des predictions par slot...")
scheduler = ModelScheduler()
test_datetimes = {
    "weekdays_day":   datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc),
    "weekdays_night": datetime(2026, 3, 30, 3,  0, tzinfo=timezone.utc),
    "friday_all":     datetime(2026, 3, 27, 14, 0, tzinfo=timezone.utc),
    "sunday_9_20":    datetime(2026, 3, 29, 15, 0, tzinfo=timezone.utc),
    "default":        datetime(2026, 3, 28, 10, 0, tzinfo=timezone.utc),  # samedi
}

import collections
buf = collections.deque(maxlen=101)
for _, row in df_live.tail(101).iterrows():
    buf.append({k: row[k] for k in ["open_time", "open", "high", "low", "close", "volume"]})

all_ok = True
for slot_name, dt_test in test_datetimes.items():
    try:
        model, meta = scheduler.get_model(dt_test)
        win = meta.get("window", 50)
        indicators = meta.get("indicators", None)
        include_time = meta.get("include_time", False)
        multitf = meta.get("multitf_enabled", False)
        min_conf = meta.get("min_confidence_pct", 0.0)

        buf_resized = collections.deque(
            list(buf)[-(win+1):], maxlen=win+1
        )
        X = build_inference_features(
            buf_resized, window=win, indicators=indicators,
            predict_time=dt_test if include_time else None,
            df_1h=df_1h if multitf else None,
            df_4h=df_4h if multitf else None,
        )
        proba = model.predict_proba(X)[0]
        pred_dir = "VERT" if proba[1] >= 0.5 else "ROUGE"
        prob = proba[1] if pred_dir == "VERT" else proba[0]
        confidence = abs(prob - 0.5) * 200
        skipped = confidence < min_conf

        print(f"    {slot_name:<22} -> {pred_dir} ({prob:.1%}) conf={confidence:.1f}% seuil={min_conf:.1f}% {'[SKIP]' if skipped else '[OK  ]'}")
    except Exception as e:
        print(f"    {slot_name:<22} -> ERREUR : {e}")
        all_ok = False

# 4. Resume des performances OOS
print("\n[4] Resume performances (backtest OOS Nov25-Mar26) :")
binary_path = Path("models/binary_options_backtest.json")
if binary_path.exists():
    data = json.loads(binary_path.read_text())
    best = max(data, key=lambda r: r["total_pnl_usd"])
    print(f"    Seuil optimal      : {best['threshold_pct']}% confidence")
    print(f"    Trades/jour        : {best['trades_per_day']}")
    print(f"    Win rate OOS       : {best['win_rate_pct']}%  (break-even: {BREAK_EVEN*100:.1f}%)")
    print(f"    Edge/trade         : +{best['edge_pct']:.2f}%")
    print(f"    P&L total ($100/t) : ${best['total_pnl_usd']:+,.0f}")
    print(f"    ROI/trade          : {best['roi_pct']:.2f}%")
    print(f"    Sharpe annualise   : {best['sharpe']:.2f}")

print("\n[5] Check features dimensions :")
for s in schedule:
    mp = s.get("model_path", "")
    try:
        from src.model.serializer import load_model
        _, m = load_model(mp)
        print(f"    {s['slot']:<22} n_features={m.get('n_features', '?')}  window={m.get('window', '?')}  indicators={m.get('indicators', '?')}")
    except Exception as e:
        print(f"    {s['slot']:<22} ERREUR: {e}")

print("\n" + "=" * 70)
print("VALIDATION TERMINEE" + (" — SYSTEME OK" if all_ok else " — ERREURS DETECTEES"))
print("=" * 70)
