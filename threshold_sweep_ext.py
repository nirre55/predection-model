"""Extension du threshold sweep pour thr=[0.002, 0.003, 0.004, 0.005]."""
import warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

from src.data.cache import load_klines
from src.features.builder import build_dataset_with_target_times
from src.model.trainer import train
import pandas as pd
import re

SYMBOL = "BTCUSDT"
THRESHOLDS = [0.002, 0.003, 0.004, 0.005]
BEST_INDICATORS = ["rsi", "macd", "atr", "mfi", "vdelta", "body", "streak"]
WINDOW = 50
INCLUDE_TIME = True
START = "1 year ago UTC"

df = load_klines(SYMBOL, "5m")
m = re.match(r"(\d+)\s+(year|month|day)s?\s+ago", START, re.I)
assert m is not None, f"Cannot parse START: {START}"
n, unit = int(m.group(1)), m.group(2).lower()
offsets = {"year": pd.DateOffset(years=n), "month": pd.DateOffset(months=n), "day": pd.DateOffset(days=n)}
start_ts = pd.Timestamp.now(tz="UTC") - offsets[unit]
df = df[df["open_time"] >= start_ts]

try:
    df_1h = load_klines(SYMBOL, "1h")
    df_4h = load_klines(SYMBOL, "4h")
except FileNotFoundError:
    df_1h = df_4h = None

for thr in THRESHOLDS:
    X, y, _ = build_dataset_with_target_times(
        df, window=WINDOW, indicators=BEST_INDICATORS,
        include_time=INCLUDE_TIME, df_1h=df_1h, df_4h=df_4h,
        min_move_pct=thr,
    )
    n_samples = len(y)
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
