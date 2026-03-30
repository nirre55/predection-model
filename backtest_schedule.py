"""
Backtest historique du schedule.

Simule les trades sur le dernier 20% des donnees (hors-echantillon strict).
Pour chaque bougie du test set :
  - Identifie le slot horaire via schedule.json
  - Charge le modele correspondant
  - Predit la direction (VERT / ROUGE)
  - Compare avec le resultat reel de Binance
  - Calcule le P&L avec frais Binance 0.1% entry + 0.1% exit = 0.2% round-trip

Metriques produites :
  - Win rate global et par slot
  - P&L total et par slot (position fixe $1000 par trade)
  - Profit factor (gains / pertes bruts)
  - Max drawdown
  - Sharpe ratio annualise
  - Equity curve  -> models/backtest_equity.csv
  - Resume        -> models/backtest_report.json

Usage:
    python backtest_schedule.py
    python backtest_schedule.py --confidence 2.0   # trades si conf > 2%
    python backtest_schedule.py --position 500     # $500 par trade
    python backtest_schedule.py --start "1 year ago UTC"
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

SYMBOL          = "BTCUSDT"
INTERVAL        = "5m"
SCHEDULE_PATH   = Path("models/schedule.json")
BEST_INDICATORS = ["rsi", "macd", "atr", "mfi", "vdelta", "body", "streak"]
FEE_RT          = 0.002   # 0.1% entree + 0.1% sortie = 0.2% round-trip
TEST_RATIO      = 0.20    # dernier 20% des donnees = hors-echantillon
CANDLES_PER_YEAR = 365 * 24 * 12   # bougies 5m par an ≈ 105 120


# ---------------------------------------------------------------------------
# Helpers
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
    dow  = dt.dayofweek   # 0=Lundi, 6=Dimanche
    hour = dt.hour
    for slot in schedule:
        if slot.get("default"):
            continue
        if dow in slot["dow"] and slot["hour_start"] <= hour < slot["hour_end"]:
            return slot
    return next((s for s in schedule if s.get("default")), None)


# ---------------------------------------------------------------------------
# Chargement des modeles (mis en cache)
# ---------------------------------------------------------------------------

class ModelCache:
    def __init__(self):
        self._cache: dict[str, tuple] = {}

    def get(self, model_path: str):
        path = model_path.replace("\\", "/")
        if path not in self._cache:
            self._cache[path] = load_model(path)
        return self._cache[path]


# ---------------------------------------------------------------------------
# Construction des features par window
# ---------------------------------------------------------------------------

def _load_mtf_data(symbol: str) -> tuple:
    """Charge les données 1h et 4h pour le contexte multi-timeframe."""
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
    """Pre-calcule X, y, target_times pour chaque window utilise dans le schedule."""
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
# Simulation trade par trade
# ---------------------------------------------------------------------------

def simulate(
    df: pd.DataFrame,
    matrices: dict[int, tuple],
    schedule: list[dict],
    model_cache: ModelCache,
    test_cutoff: pd.Timestamp,
    confidence_threshold: float,
    position_usd: float,
) -> pd.DataFrame:
    """
    Pour chaque sample du test set, predit et calcule le P&L.
    Retourne un DataFrame avec une ligne par trade tente.
    """
    # On travaille sur window=20 pour avoir le test set de reference
    # (le plus grand nombre de samples)
    ref_window = min(matrices.keys())
    _, _, ref_times = matrices[ref_window]

    # Index du test set base sur la date de coupure
    ref_dt = pd.to_datetime(ref_times, utc=True)
    test_mask_ref = ref_dt >= test_cutoff

    # Recupere le close et open reels du df pour chaque target
    close_arr = np.asarray(df["close"].values, dtype="float64")
    open_arr  = np.asarray(df["open"].values,  dtype="float64")

    rows = []
    skipped = 0

    for i, ts in enumerate(ref_times):
        if not test_mask_ref[i]:
            continue

        dt = pd.Timestamp(ts, tz="UTC") if not hasattr(ts, "tz") else ts.tz_convert("UTC")
        slot = get_slot(dt, schedule)
        if slot is None:
            continue

        window     = slot["window"]
        model_path = slot["model_path"].replace("\\", "/")

        # Retrouver l'index correspondant dans la matrice du bon window
        X_w, y_w, times_w = matrices[window]
        dt_w = pd.to_datetime(times_w, utc=True)
        match = np.where(dt_w == dt)[0]
        if len(match) == 0:
            skipped += 1
            continue
        j = match[0]

        model, meta = model_cache.get(model_path)
        X_sample = X_w[j : j + 1]

        try:
            proba = model.predict_proba(X_sample)[0]
        except Exception:
            skipped += 1
            continue

        prob_green = float(proba[1])
        prob_red   = float(proba[0])
        predicted  = "VERT" if prob_green >= 0.5 else "ROUGE"
        prob_pred  = prob_green if predicted == "VERT" else prob_red
        confidence = abs(prob_pred - 0.5) * 200

        # Seuil de confiance
        if confidence < confidence_threshold:
            skipped += 1
            continue

        # Resultat reel
        # target_idx dans df : sample j du window w -> j + window + 2
        target_idx = j + window + 2
        if target_idx >= len(df):
            skipped += 1
            continue

        actual_open  = open_arr[target_idx]
        actual_close = close_arr[target_idx]
        actual_dir   = "VERT" if actual_close > actual_open else "ROUGE"
        result       = "WIN" if actual_dir == predicted else "LOSS"

        # P&L : variation de prix nette des frais
        price_move_pct = abs(actual_close - actual_open) / actual_open
        if result == "WIN":
            pnl_usd = position_usd * price_move_pct - position_usd * FEE_RT
        else:
            pnl_usd = -(position_usd * price_move_pct + position_usd * FEE_RT)

        rows.append({
            "candle_open":      dt.strftime("%Y-%m-%d %H:%M"),
            "slot":             slot["slot"],
            "model_type":       slot["model_type"],
            "window":           window,
            "predicted":        predicted,
            "probability":      round(prob_pred, 4),
            "confidence_pct":   round(confidence, 2),
            "actual":           actual_dir,
            "result":           result,
            "price_move_pct":   round(float(price_move_pct) * 100, 4),
            "pnl_usd":          round(float(pnl_usd), 4),
        })

    if skipped:
        print(f"  ({skipped} samples ignores : seuil conf ou hors-index)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metriques
# ---------------------------------------------------------------------------

def compute_metrics(trades: pd.DataFrame, position_usd: float) -> dict:
    if trades.empty:
        return {}

    wins  = trades[trades["result"] == "WIN"]
    losses = trades[trades["result"] == "LOSS"]

    total_pnl    = trades["pnl_usd"].sum()
    gross_profit = wins["pnl_usd"].sum()
    gross_loss   = abs(losses["pnl_usd"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Equity curve
    equity = trades["pnl_usd"].cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd   = drawdown.min()

    # Sharpe annualise (sur rendements par trade)
    returns = trades["pnl_usd"] / position_usd
    mean_r  = returns.mean()
    std_r   = returns.std()
    # Annualisation : bougies 5m, on trade ~fraction du temps
    trades_per_year = CANDLES_PER_YEAR * (len(trades) / max(len(trades), 1))
    sharpe = (mean_r / std_r * np.sqrt(CANDLES_PER_YEAR)) if std_r > 0 else 0.0

    return {
        "n_trades":       int(len(trades)),
        "n_wins":         int(len(wins)),
        "n_losses":       int(len(losses)),
        "win_rate_pct":   round(len(wins) / len(trades) * 100, 2),
        "total_pnl_usd":  round(total_pnl, 2),
        "gross_profit":   round(gross_profit, 2),
        "gross_loss":     round(gross_loss, 2),
        "profit_factor":  round(profit_factor, 4),
        "max_drawdown_usd": round(max_dd, 2),
        "sharpe_ratio":   round(sharpe, 4),
        "avg_pnl_usd":    round(trades["pnl_usd"].mean(), 4),
        "best_trade_usd": round(trades["pnl_usd"].max(), 4),
        "worst_trade_usd":round(trades["pnl_usd"].min(), 4),
    }


def compute_slot_metrics(trades: pd.DataFrame) -> list[dict]:
    rows = []
    for slot_name, group in trades.groupby("slot"):
        wins = (group["result"] == "WIN").sum()
        m = {
            "slot":           slot_name,
            "n_trades":       int(len(group)),
            "win_rate_pct":   round(wins / len(group) * 100, 2),
            "total_pnl_usd":  round(group["pnl_usd"].sum(), 2),
            "profit_factor":  round(
                group[group["pnl_usd"] > 0]["pnl_usd"].sum() /
                max(abs(group[group["pnl_usd"] < 0]["pnl_usd"].sum()), 1e-9),
                4,
            ),
            "avg_confidence": round(group["confidence_pct"].mean(), 2),
        }
        rows.append(m)
    return sorted(rows, key=lambda r: r["total_pnl_usd"], reverse=True)


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def print_report(metrics: dict, slot_metrics: list[dict],
                 test_cutoff: pd.Timestamp, confidence_threshold: float,
                 position_usd: float) -> None:
    print("\n" + "=" * 70)
    print("BACKTEST SCHEDULE -- RAPPORT COMPLET")
    print("=" * 70)
    print(f"Periode test  : {test_cutoff.date()} -> aujourd'hui (dernier 20%)")
    print(f"Position      : ${position_usd:.0f} par trade")
    print(f"Frais         : {FEE_RT*100:.1f}% round-trip (Binance 0.1%+0.1%)")
    print(f"Seuil conf.   : {confidence_threshold:.1f}%")
    print()
    print(f"  Trades total     : {metrics['n_trades']}")
    print(f"  Win rate         : {metrics['win_rate_pct']:.2f}%")
    print(f"  P&L total        : ${metrics['total_pnl_usd']:+.2f}")
    print(f"  Profit factor    : {metrics['profit_factor']:.4f}")
    print(f"  Max drawdown     : ${metrics['max_drawdown_usd']:.2f}")
    print(f"  Sharpe (annuel)  : {metrics['sharpe_ratio']:.4f}")
    print(f"  Gain moyen/trade : ${metrics['avg_pnl_usd']:+.4f}")
    print(f"  Meilleur trade   : ${metrics['best_trade_usd']:+.4f}")
    print(f"  Pire trade       : ${metrics['worst_trade_usd']:+.4f}")

    print()
    print(f"{'Slot':<22} {'Trades':>7} {'Win%':>7} {'P&L':>10} {'PF':>7} {'Conf%':>7}")
    print("-" * 70)
    for s in slot_metrics:
        pnl_str = f"${s['total_pnl_usd']:+.2f}"
        print(
            f"{s['slot']:<22} {s['n_trades']:>7} {s['win_rate_pct']:>6.2f}%"
            f" {pnl_str:>10} {s['profit_factor']:>7.4f} {s['avg_confidence']:>6.2f}%"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backtest historique du schedule")
    parser.add_argument("--confidence", type=float, default=0.0,
                        help="Seuil de confiance minimum en %% (defaut: 0 = tous les trades)")
    parser.add_argument("--position", type=float, default=1000.0,
                        help="Taille de position en USD par trade (defaut: $1000)")
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

    schedule = load_schedule()
    model_cache = ModelCache()

    print("\nChargement donnees multi-timeframe...")
    df_1h, df_4h = _load_mtf_data(SYMBOL)

    print("\nConstruction des matrices de features...")
    matrices = build_matrices(df, df_1h=df_1h, df_4h=df_4h)

    # Date de coupure test set = 80e percentile chronologique
    ref_times = matrices[min(matrices.keys())][2]
    ref_dt = pd.to_datetime(ref_times, utc=True)
    test_cutoff = ref_dt[int(len(ref_dt) * (1 - TEST_RATIO))]
    print(f"\nTest set : {test_cutoff.date()} -> {ref_dt[-1].date()} ({TEST_RATIO*100:.0f}% des donnees)")
    print(f"Seuil confiance : {args.confidence:.1f}%  |  Position : ${args.position:.0f}\n")

    print("Simulation en cours...")
    trades = simulate(df, matrices, schedule, model_cache,
                      test_cutoff, args.confidence, args.position)

    if trades.empty:
        print("Aucun trade effectue -- essaie de baisser --confidence")
        return

    metrics      = compute_metrics(trades, args.position)
    slot_metrics = compute_slot_metrics(trades)

    print_report(metrics, slot_metrics, test_cutoff, args.confidence, args.position)

    # Sauvegarde
    Path("models").mkdir(exist_ok=True)

    equity_df = trades[["candle_open", "slot", "result", "pnl_usd"]].copy()
    equity_df["equity_usd"] = trades["pnl_usd"].cumsum()
    equity_df.to_csv("models/backtest_equity.csv", index=False)

    report = {
        "generated_at":        datetime.now(timezone.utc).isoformat(),
        "test_period_start":   str(test_cutoff.date()),
        "confidence_threshold":args.confidence,
        "position_usd":        args.position,
        "fee_rt_pct":          FEE_RT * 100,
        "global":              metrics,
        "by_slot":             slot_metrics,
    }
    Path("models/backtest_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    print(f"\n[Rapport] models/backtest_equity.csv")
    print(f"[Rapport] models/backtest_report.json")


if __name__ == "__main__":
    main()
