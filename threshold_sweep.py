"""
Sweep du seuil de filtrage pour le target engineering.
Mesure le win rate out-of-sample et le nombre de samples pour
chaque valeur de min_move_pct : [0.0005, 0.001, 0.0015, 0.002].

Usage:
    python threshold_sweep.py
    python threshold_sweep.py --start "1 year ago UTC"
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

from src.data.cache import load_klines
from src.data.fetcher import fetch_klines
from src.data.cache import save_klines
from src.features.builder import build_dataset_with_target_times
from src.model.trainer import train

SYMBOL = "BTCUSDT"
THRESHOLDS = [0.0, 0.0005, 0.001, 0.0015, 0.002]
RESULTS_PATH = Path("models/threshold_sweep_results.json")
BEST_INDICATORS = ["rsi", "macd", "atr", "mfi", "vdelta", "body", "streak"]
WINDOW = 50
INCLUDE_TIME = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None)
    args = parser.parse_args()

    df = load_klines(SYMBOL, "5m")
    if args.start:
        import pandas as pd
        import re
        # Gérer le format relatif "N year(s)/month(s)/day(s) ago UTC"
        m = re.match(r"(\d+)\s+(year|month|day)s?\s+ago", args.start, re.I)
        if m:
            n, unit = int(m.group(1)), m.group(2).lower()
            offsets = {"year": pd.DateOffset(years=n), "month": pd.DateOffset(months=n), "day": pd.DateOffset(days=n)}
            start_ts = pd.Timestamp.now(tz="UTC") - offsets[unit]
        else:
            start_ts = pd.Timestamp(args.start, tz="UTC") if "UTC" in args.start else pd.Timestamp(args.start).tz_localize("UTC")
        df = df[df["open_time"] >= start_ts]

    # Charger 1h/4h
    try:
        df_1h = load_klines(SYMBOL, "1h")
    except FileNotFoundError:
        df_1h = fetch_klines(SYMBOL, "1h", start_str="5 years ago UTC")
        save_klines(df_1h, SYMBOL, "1h")
    try:
        df_4h = load_klines(SYMBOL, "4h")
    except FileNotFoundError:
        df_4h = fetch_klines(SYMBOL, "4h", start_str="5 years ago UTC")
        save_klines(df_4h, SYMBOL, "4h")

    results = []
    for thr in THRESHOLDS:
        X, y, _ = build_dataset_with_target_times(
            df, window=WINDOW, indicators=BEST_INDICATORS,
            include_time=INCLUDE_TIME, df_1h=df_1h, df_4h=df_4h,
            min_move_pct=thr,
        )
        n_total_before_filter = len(df) - WINDOW - 2  # approximatif
        n_samples = len(y)
        # Split 80/20 strict out-of-sample
        split = int(n_samples * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        if len(X_train) < 200 or len(X_test) < 50:
            print(f"  thr={thr:.4f}: pas assez de samples ({n_samples}), skipped")
            continue
        model = train(X_train, y_train, model_type="lgbm")
        y_pred = model.predict(X_test)
        win_rate = float(np.mean(y_pred == y_test))
        print(f"  thr={thr:.4f} | n_samples={n_samples:>6} | win_rate_oos={win_rate:.4f}")
        results.append({"min_move_pct": thr, "n_samples": n_samples, "win_rate_oos": win_rate})

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResultats sauvegardes dans {RESULTS_PATH}")
    best = max(results, key=lambda r: r["win_rate_oos"])
    print(f"Meilleur seuil : {best['min_move_pct']} (win_rate={best['win_rate_oos']:.4f}, n={best['n_samples']})")

if __name__ == "__main__":
    main()
