"""
Comparatif des performances sur différentes périodes d'entraînement.
Ne modifie PAS les modèles en production.

Compare 4 périodes :
  - 5 ans  (modèle live actuel)
  - 1 an
  - 6 mois
  - 3 mois

Pour chaque période :
  - Score global (meilleur combo : rsi+macd+atr, window=50)
  - Score par slot horaire (window=20, LGBM vs XGBoost)

Sorties :
  models/comparison_global.csv
  models/comparison_slots.csv
  models/comparison_report.json

Usage:
    python compare_periods.py
"""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from src.data.cache import load_klines
from src.features.builder import build_dataset_with_target_times
from src.backtest.walk_forward import run_walk_forward

# --- Configuration ---

SYMBOL   = "BTCUSDT"
INTERVAL = "5m"
N_SPLITS = 5

PERIODS = [
    ("5_ans_live",  None),
    ("1_an",        "1 year ago UTC"),
    ("6_mois",      "6 months ago UTC"),
    ("3_mois",      "3 months ago UTC"),
]

BEST_INDICATORS = ["rsi", "macd", "atr"]
GLOBAL_WINDOW   = 50
SLOT_WINDOW     = 20
MODEL_TYPES     = ["lgbm", "xgb"]

TIME_SLOTS = [
    ("sunday_9_20",    [6],       9,  20),
    ("sunday_0_9",     [6],       0,   9),
    ("sunday_20_24",   [6],      20,  24),
    ("friday_all",     [4],       0,  24),
    ("saturday_7_14",  [5],       7,  14),
    ("saturday_0_7",   [5],       0,   7),
    ("saturday_14_24", [5],      14,  24),
    ("weekdays_day",   [0,1,2,3], 7,  20),
    ("weekdays_night", [0,1,2,3], 0,   7),
    ("weekdays_eve",   [0,1,2,3],20,  24),
]

MIN_SAMPLES = 300
OUT_DIR = Path("models")


# --- Helpers ---

def filter_df(df: pd.DataFrame, start: str | None) -> pd.DataFrame:
    if start is None:
        return df
    cutoff = pd.Timestamp(start)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    return df[df["open_time"] >= cutoff].reset_index(drop=True)


def wf_score(X: np.ndarray, y: np.ndarray, model_type: str) -> dict:
    price_moves = np.ones(len(y), dtype="float64")  # proxy uniforme
    wf = run_walk_forward(X, y, price_moves, n_splits=N_SPLITS, model_type=model_type)
    g = next(r for r in wf if r["fold"] == "global")
    acc = g.get("accuracy", 0.0)
    pf  = g.get("profit_factor", 0.0)
    score = acc * 0.6 + min(pf, 2.0) / 2.0 * 0.4
    return {"accuracy": round(acc, 6), "profit_factor": round(pf, 6), "score": round(score, 6)}


def slot_mask(target_times: np.ndarray, dow_list: list[int],
              h_start: int, h_end: int) -> np.ndarray:
    dt   = pd.to_datetime(target_times, utc=True)
    dow  = dt.dayofweek.values
    hour = dt.hour.values
    return np.isin(dow, dow_list) & (hour >= h_start) & (hour < h_end)


# --- Section 1 : comparatif global ---

def run_global_comparison(df_full: pd.DataFrame) -> list[dict]:
    rows = []
    for period_name, start in PERIODS:
        df = filter_df(df_full, start)
        n_candles = len(df)

        print(f"\n  [{period_name}] {n_candles} bougies — global window={GLOBAL_WINDOW}")

        try:
            X, y, _ = build_dataset_with_target_times(
                df, window=GLOBAL_WINDOW,
                indicators=BEST_INDICATORS, include_time=False,
            )
        except Exception as e:
            print(f"    ERREUR build: {e}")
            continue

        for mt in MODEL_TYPES:
            try:
                metrics = wf_score(X, y, mt)
                print(
                    f"    {mt:<6} acc={metrics['accuracy']:.4f}  "
                    f"pf={metrics['profit_factor']:.4f}  score={metrics['score']:.4f}  n={len(X)}"
                )
                rows.append({
                    "period":        period_name,
                    "n_candles":     n_candles,
                    "n_samples":     len(X),
                    "model_type":    mt,
                    "window":        GLOBAL_WINDOW,
                    "indicators":    "+".join(BEST_INDICATORS),
                    "accuracy":      metrics["accuracy"],
                    "profit_factor": metrics["profit_factor"],
                    "score":         metrics["score"],
                })
            except Exception as e:
                print(f"    ERREUR {mt}: {e}")

    return rows


# --- Section 2 : comparatif par slot horaire ---

def run_slot_comparison(df_full: pd.DataFrame) -> list[dict]:
    rows = []
    for period_name, start in PERIODS:
        df = filter_df(df_full, start)
        print(f"\n  [{period_name}] {len(df)} bougies — slots window={SLOT_WINDOW}")

        try:
            X_all, y_all, target_times = build_dataset_with_target_times(
                df, window=SLOT_WINDOW,
                indicators=BEST_INDICATORS, include_time=True,
            )
        except Exception as e:
            print(f"    ERREUR build: {e}")
            continue

        for slot_name, dow_list, h_start, h_end in TIME_SLOTS:
            mask = slot_mask(target_times, dow_list, h_start, h_end)
            X_s, y_s = X_all[mask], y_all[mask]

            if len(X_s) < MIN_SAMPLES:
                print(f"    {slot_name:<22} — skip ({len(X_s)} samples)")
                continue

            best_score = -np.inf
            best_mt    = ""
            for mt in MODEL_TYPES:
                try:
                    metrics = wf_score(X_s, y_s, mt)
                    if metrics["score"] > best_score:
                        best_score = metrics["score"]
                        best_mt    = mt
                    rows.append({
                        "period":        period_name,
                        "slot":          slot_name,
                        "model_type":    mt,
                        "n_samples":     len(X_s),
                        "accuracy":      metrics["accuracy"],
                        "profit_factor": metrics["profit_factor"],
                        "score":         metrics["score"],
                    })
                except Exception as e:
                    print(f"    ERREUR {slot_name} {mt}: {e}")

            # Afficher le meilleur
            best_row = max(
                (r for r in rows
                 if r["period"] == period_name and r["slot"] == slot_name),
                key=lambda r: r["score"],
                default=None,
            )
            if best_row:
                marker = " <--" if best_row["score"] > 0.508 else ""
                print(
                    f"    {slot_name:<22} best={best_row['model_type']:<6} "
                    f"acc={best_row['accuracy']:.4f}  pf={best_row['profit_factor']:.4f}  "
                    f"score={best_row['score']:.4f}  n={len(X_s)}{marker}"
                )

    return rows


# --- Section 3 : rapport synthétique ---

def print_global_summary(global_rows: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("COMPARATIF GLOBAL — rsi+macd+atr, window=50")
    print("=" * 90)
    print(f"{'Periode':<16} {'Model':<6} {'N candl':>8} {'N samp':>8} {'Accuracy':>10} {'PF':>8} {'Score':>8}")
    print("-" * 90)

    # Grouper par periode pour afficher le meilleur
    by_period: dict[str, list[dict]] = {}
    for r in global_rows:
        by_period.setdefault(r["period"], []).append(r)

    for period_name, _ in PERIODS:
        rows = by_period.get(period_name, [])
        if not rows:
            continue
        best = max(rows, key=lambda r: r["score"])
        for r in rows:
            marker = " <-- BEST" if r == best else ""
            print(
                f"{r['period']:<16} {r['model_type']:<6} {r['n_candles']:>8} "
                f"{r['n_samples']:>8} {r['accuracy']:>10.4f} {r['profit_factor']:>8.4f} "
                f"{r['score']:>8.4f}{marker}"
            )
        print()


def print_slot_summary(slot_rows: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("COMPARATIF PAR SLOT — meilleur modèle par slot et par période")
    print("=" * 90)

    # Pour chaque slot : meilleur score par période
    slot_names = [s[0] for s in TIME_SLOTS]
    period_names = [p[0] for p in PERIODS]

    # Header
    header = f"{'Slot':<22}"
    for p in period_names:
        header += f" {p[:10]:>12}"
    print(header)
    print("-" * 90)

    for slot_name in slot_names:
        line = f"{slot_name:<22}"
        for period_name in period_names:
            candidates = [
                r for r in slot_rows
                if r["period"] == period_name and r["slot"] == slot_name
            ]
            if candidates:
                best = max(candidates, key=lambda r: r["score"])
                cell = f"{best['model_type']}:{best['score']:.4f}"
            else:
                cell = "N/A"
            line += f" {cell:>12}"
        print(line)


# --- Main ---

def main():
    print(f"Chargement des données {SYMBOL} {INTERVAL}...")
    df_full = load_klines(SYMBOL, INTERVAL)
    print(f"{len(df_full)} bougies totales en cache.\n")

    print("=" * 60)
    print("SECTION 1 : COMPARATIF GLOBAL")
    print("=" * 60)
    global_rows = run_global_comparison(df_full)

    print("\n" + "=" * 60)
    print("SECTION 2 : COMPARATIF PAR SLOT HORAIRE")
    print("=" * 60)
    slot_rows = run_slot_comparison(df_full)

    # --- Affichage synthétique ---
    print_global_summary(global_rows)
    print_slot_summary(slot_rows)

    # --- Sauvegarde CSV ---
    OUT_DIR.mkdir(exist_ok=True)

    global_csv = OUT_DIR / "comparison_global.csv"
    slot_csv   = OUT_DIR / "comparison_slots.csv"
    report_json = OUT_DIR / "comparison_report.json"

    pd.DataFrame(global_rows).to_csv(global_csv, index=False)
    pd.DataFrame(slot_rows).to_csv(slot_csv, index=False)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "periods":       [p[0] for p in PERIODS],
        "global":        global_rows,
        "slots":         slot_rows,
    }
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n[Rapport] {global_csv}")
    print(f"[Rapport] {slot_csv}")
    print(f"[Rapport] {report_json}")


if __name__ == "__main__":
    main()
