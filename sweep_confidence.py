"""
Sweep du seuil de confiance de 0.5 a 5.0 par pas de 0.5.
Affiche un tableau comparatif : trades, win rate, P&L, profit factor.

Usage:
    python sweep_confidence.py
    python sweep_confidence.py --position 500
    python sweep_confidence.py --start "1 year ago UTC"
"""

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from src.data.cache import load_klines, save_klines
from src.data.fetcher import fetch_klines as fetch_klines_api
from src.features.builder import build_dataset_with_target_times
from src.model.serializer import load_model

SYMBOL           = "BTCUSDT"
INTERVAL         = "5m"
SCHEDULE_PATH    = Path("models/schedule.json")
BEST_INDICATORS  = ["rsi", "macd", "atr", "mfi", "vdelta", "body", "streak"]
FEE_RT           = 0.002
TEST_RATIO       = 0.20
CANDLES_PER_YEAR = 365 * 24 * 12

THRESHOLDS = [round(x * 0.5, 1) for x in range(1, 11)]  # 0.5, 1.0, ..., 5.0


# ---------------------------------------------------------------------------
# Helpers (copies from backtest_schedule.py)
# ---------------------------------------------------------------------------

def _parse_start(start: str) -> pd.Timestamp:
    from datetime import timedelta
    import re
    now = datetime.now(timezone.utc)
    s = start.lower().replace("utc", "").strip()
    m = re.match(r"(\d+)\s+(year|month|week|day)s?\s+ago", s)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit == "year":
            cutoff = now.replace(year=now.year - n)
        elif unit == "month":
            month, year = now.month - n, now.year
            while month <= 0:
                month += 12
                year  -= 1
            cutoff = now.replace(year=year, month=month)
        elif unit == "week":
            cutoff = now - timedelta(weeks=n)
        else:
            cutoff = now - timedelta(days=n)
        return pd.Timestamp(cutoff)
    ts = pd.Timestamp(start)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts


def load_schedule() -> list[dict]:
    return json.loads(SCHEDULE_PATH.read_text())


def get_slot(dt: pd.Timestamp, schedule: list[dict]) -> dict | None:
    dow  = dt.dayofweek
    hour = dt.hour
    for slot in schedule:
        if slot.get("default"):
            continue
        if dow in slot["dow"] and slot["hour_start"] <= hour < slot["hour_end"]:
            return slot
    return next((s for s in schedule if s.get("default")), None)


class ModelCache:
    def __init__(self):
        self._cache: dict[str, tuple] = {}

    def get(self, model_path: str):
        path = model_path.replace("\\", "/")
        if path not in self._cache:
            self._cache[path] = load_model(path)
        return self._cache[path]


def _load_mtf_data(symbol: str) -> tuple:
    try:
        df_1h = load_klines(symbol, "1h")
    except FileNotFoundError:
        print("  Fetch donnees 1h...")
        df_1h = fetch_klines_api(symbol, "1h", start_str="5 years ago UTC")
        save_klines(df_1h, symbol, "1h")
    try:
        df_4h = load_klines(symbol, "4h")
    except FileNotFoundError:
        print("  Fetch donnees 4h...")
        df_4h = fetch_klines_api(symbol, "4h", start_str="5 years ago UTC")
        save_klines(df_4h, symbol, "4h")
    return df_1h, df_4h


def build_matrices(df: pd.DataFrame, df_1h=None, df_4h=None) -> dict[int, tuple]:
    schedule = load_schedule()
    windows = set(s["window"] for s in schedule if not s.get("default"))
    matrices = {}
    for w in sorted(windows):
        print(f"  Construction features window={w}...", end=" ", flush=True)
        X, y, times = build_dataset_with_target_times(
            df, window=w, indicators=BEST_INDICATORS, include_time=True,
            df_1h=df_1h, df_4h=df_4h,
        )
        matrices[w] = (X, y, times)
        print(f"{len(X)} samples")
    return matrices


# ---------------------------------------------------------------------------
# Simulation (sans filtrage confiance -- on pre-calcule tout)
# ---------------------------------------------------------------------------

def simulate_all(
    df: pd.DataFrame,
    matrices: dict[int, tuple],
    schedule: list[dict],
    model_cache: ModelCache,
    test_cutoff: pd.Timestamp,
    position_usd: float,
) -> pd.DataFrame:
    """Simule tous les trades du test set, confiance incluse, sans seuil."""
    ref_window = min(matrices.keys())
    _, _, ref_times = matrices[ref_window]
    ref_dt = pd.to_datetime(ref_times, utc=True)
    test_mask_ref = ref_dt >= test_cutoff

    close_arr = np.asarray(df["close"].values, dtype="float64")
    open_arr  = np.asarray(df["open"].values,  dtype="float64")

    rows = []

    for i, ts in enumerate(ref_times):
        if not test_mask_ref[i]:
            continue

        dt = pd.Timestamp(ts, tz="UTC") if not hasattr(ts, "tz") else ts.tz_convert("UTC")
        slot = get_slot(dt, schedule)
        if slot is None:
            continue

        window     = slot["window"]
        model_path = slot["model_path"].replace("\\", "/")

        X_w, y_w, times_w = matrices[window]
        dt_w = pd.to_datetime(times_w, utc=True)
        match = np.where(dt_w == dt)[0]
        if len(match) == 0:
            continue
        j = match[0]

        model, meta = model_cache.get(model_path)
        X_sample = X_w[j : j + 1]

        try:
            proba = model.predict_proba(X_sample)[0]
        except Exception:
            continue

        prob_green = float(proba[1])
        prob_red   = float(proba[0])
        predicted  = "VERT" if prob_green >= 0.5 else "ROUGE"
        prob_pred  = prob_green if predicted == "VERT" else prob_red
        confidence = abs(prob_pred - 0.5) * 200

        target_idx = j + window + 2
        if target_idx >= len(df):
            continue

        actual_open  = open_arr[target_idx]
        actual_close = close_arr[target_idx]
        actual_dir   = "VERT" if actual_close > actual_open else "ROUGE"
        result       = "WIN" if actual_dir == predicted else "LOSS"

        price_move_pct = abs(actual_close - actual_open) / actual_open
        if result == "WIN":
            pnl_usd = position_usd * price_move_pct - position_usd * FEE_RT
        else:
            pnl_usd = -(position_usd * price_move_pct + position_usd * FEE_RT)

        rows.append({
            "confidence_pct": round(confidence, 4),
            "result":         result,
            "pnl_usd":        round(float(pnl_usd), 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metriques pour un seuil donne
# ---------------------------------------------------------------------------

def metrics_for_threshold(all_trades: pd.DataFrame, threshold: float, position_usd: float) -> dict:
    trades = all_trades[all_trades["confidence_pct"] >= threshold]
    if trades.empty:
        return {
            "threshold": threshold,
            "n_trades":      0,
            "win_rate_pct":  0.0,
            "total_pnl_usd": 0.0,
            "profit_factor": 0.0,
            "sharpe":        0.0,
        }

    wins   = trades[trades["result"] == "WIN"]
    losses = trades[trades["result"] == "LOSS"]

    gross_profit  = wins["pnl_usd"].sum()
    gross_loss    = abs(losses["pnl_usd"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    returns = trades["pnl_usd"] / position_usd
    std_r   = returns.std()
    sharpe  = (returns.mean() / std_r * np.sqrt(CANDLES_PER_YEAR)) if std_r > 0 else 0.0

    return {
        "threshold":     threshold,
        "n_trades":      int(len(trades)),
        "win_rate_pct":  round(len(wins) / len(trades) * 100, 2),
        "total_pnl_usd": round(trades["pnl_usd"].sum(), 2),
        "profit_factor": round(profit_factor, 4),
        "sharpe":        round(sharpe, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sweep seuil de confiance 0.5 -> 5.0")
    parser.add_argument("--position", type=float, default=1000.0,
                        help="Taille de position en USD (defaut: $1000)")
    parser.add_argument("--start", type=str, default=None,
                        help="Date de debut des donnees (ex: '1 year ago UTC')")
    args = parser.parse_args()

    print(f"Chargement des donnees {SYMBOL} {INTERVAL}...")
    df = load_klines(SYMBOL, INTERVAL)
    if args.start:
        cutoff_data = _parse_start(args.start)
        df = df[df["open_time"] >= cutoff_data].reset_index(drop=True)
        print(f"Filtre depuis {cutoff_data.date()} : {len(df)} bougies.")
    else:
        print(f"{len(df)} bougies chargees.")

    schedule    = load_schedule()
    model_cache = ModelCache()

    print("\nChargement donnees multi-timeframe...")
    df_1h, df_4h = _load_mtf_data(SYMBOL)

    print("\nConstruction des matrices de features...")
    matrices = build_matrices(df, df_1h=df_1h, df_4h=df_4h)

    ref_times  = matrices[min(matrices.keys())][2]
    ref_dt     = pd.to_datetime(ref_times, utc=True)
    test_cutoff = ref_dt[int(len(ref_dt) * (1 - TEST_RATIO))]
    print(f"\nTest set : {test_cutoff.date()} -> {ref_dt[-1].date()} ({TEST_RATIO*100:.0f}% des donnees)")
    print(f"Position : ${args.position:.0f} | Frais : {FEE_RT*100:.1f}% RT\n")

    print("Simulation (tous trades, sans seuil)...")
    all_trades = simulate_all(df, matrices, schedule, model_cache, test_cutoff, args.position)
    print(f"{len(all_trades)} trades pre-calcules.\n")

    # --- Tableau comparatif ---
    results = [metrics_for_threshold(all_trades, t, args.position) for t in THRESHOLDS]

    print("=" * 72)
    print("SWEEP SEUIL DE CONFIANCE (0.5% -> 5.0%)")
    print("=" * 72)
    print(f"{'Seuil':>7} {'Trades':>8} {'Win%':>8} {'P&L':>12} {'PF':>8} {'Sharpe':>8}")
    print("-" * 72)

    best_pnl = max(r["total_pnl_usd"] for r in results)
    best_pf  = max(r["profit_factor"] for r in results if r["profit_factor"] < float("inf"))

    for r in results:
        marker = ""
        if r["total_pnl_usd"] == best_pnl and r["n_trades"] > 0:
            marker += " <- meilleur P&L"
        elif r["profit_factor"] == best_pf and r["n_trades"] > 0:
            marker += " <- meilleur PF"

        pnl_str = f"${r['total_pnl_usd']:+.2f}"
        pf_str  = f"{r['profit_factor']:.4f}" if r["profit_factor"] < 999 else "inf"
        print(
            f"{r['threshold']:>6.1f}%"
            f" {r['n_trades']:>8}"
            f" {r['win_rate_pct']:>7.2f}%"
            f" {pnl_str:>12}"
            f" {pf_str:>8}"
            f" {r['sharpe']:>8.4f}"
            f"{marker}"
        )

    # Sauvegarde CSV
    out_path = Path("models/sweep_confidence.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\n[Rapport] {out_path}")


if __name__ == "__main__":
    main()
