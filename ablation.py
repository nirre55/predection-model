"""
Ablation study automatisé — combinaisons d'indicateurs avec LightGBM.
Teste chaque combinaison, conserve le meilleur modèle calibré.

Usage:
    python ablation.py
    python ablation.py --start "1 year ago UTC"
    python ablation.py --start "1 Jan 2024 UTC"
    python ablation.py --start "2025-01-01"
"""

import json
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from src.data.cache import load_klines
from src.features.indicators import (
    compute_rsi,
    compute_macd_histogram,
    compute_bollinger_pct,
    compute_atr_normalized,
    compute_volume_ratio,
)
from src.backtest.walk_forward import run_walk_forward
from src.model.trainer import train
from src.model.serializer import save_model

WINDOW = 50
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
N_SPLITS = 5
RESULTS_PATH = Path("models/ablation_results.json")

# --- Définition des indicateurs disponibles ---

INDICATOR_REGISTRY: dict[str, Callable[..., np.ndarray]] = {
    "rsi": lambda df: compute_rsi(np.asarray(df["close"].values, dtype="float64")),
    "macd": lambda df: compute_macd_histogram(np.asarray(df["close"].values, dtype="float64")),
    "bb": lambda df: compute_bollinger_pct(np.asarray(df["close"].values, dtype="float64")),
    "atr": lambda df: compute_atr_normalized(
        np.asarray(df["high"].values, dtype="float64"),
        np.asarray(df["low"].values, dtype="float64"),
        np.asarray(df["close"].values, dtype="float64"),
    ),
    "vol": lambda df: compute_volume_ratio(np.asarray(df["volume"].values, dtype="float64")),
}


def build_X_y(df: pd.DataFrame, active_indicators: list[str], window: int = WINDOW):
    """Construit X, y et price_moves pour une combinaison d'indicateurs donnée."""
    ohlcv = np.asarray(df[["open", "high", "low", "close", "volume"]].values, dtype="float64")
    close = np.asarray(df["close"].values, dtype="float64")
    open_ = np.asarray(df["open"].values, dtype="float64")

    # Indicateurs actifs pré-calculés
    ind_arrays = [INDICATOR_REGISTRY[name](df) for name in active_indicators]

    n_samples = len(df) - window - 2
    j_arr = np.arange(n_samples)
    window_rows = (j_arr + 1)[:, None] + np.arange(window)  # (n_samples, window)

    # OHLCV normalisé
    ref_closes = close[j_arr]
    ohlcv_windows = ohlcv[window_rows]
    ohlcv_norm = (ohlcv_windows / ref_closes[:, None, None]).reshape(n_samples, window * 5)

    # Fenêtres d'indicateurs
    parts = [ohlcv_norm]
    for arr in ind_arrays:
        parts.append(arr[window_rows])

    X = np.concatenate(parts, axis=1)

    target_idx = j_arr + window + 2
    y = (close[target_idx] > open_[target_idx]).astype("int64")

    price_moves = np.abs(close[target_idx] - open_[target_idx])
    return X, y, price_moves


def run_ablation(start: str | None = None):
    print(f"Chargement des données {SYMBOL} {INTERVAL}...")
    df = load_klines(SYMBOL, INTERVAL)
    if start is not None:
        cutoff = pd.Timestamp(start).tz_localize("UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start)
        df = df[df["open_time"] >= cutoff].reset_index(drop=True)
        print(f"Filtre appliqué depuis {cutoff.date()} : {len(df)} bougies.")
    print(f"{len(df)} bougies chargées.\n")

    indicator_names = list(INDICATOR_REGISTRY.keys())

    # Toutes les combinaisons : baseline (aucun indicateur) + 1, 2, 3, 4, 5 indicateurs
    combos: list[tuple] = [()]  # baseline OHLCV seul
    for r in range(1, len(indicator_names) + 1):
        combos.extend(combinations(indicator_names, r))

    results = []
    best_score = -np.inf
    best_combo = None
    best_model = None
    best_meta = None

    print(f"{'Combo':<40} {'Accuracy':>10} {'PF':>10} {'Features':>10}")
    print("-" * 75)

    for combo in combos:
        combo_name = "+".join(combo) if combo else "OHLCV_seul"
        active = list(combo)

        try:
            X, y, price_moves = build_X_y(df, active)

            wf_results = run_walk_forward(X, y, price_moves, n_splits=N_SPLITS, model_type="lgbm")
            global_r = next(r for r in wf_results if r["fold"] == "global")

            acc = global_r.get("accuracy", 0.0)
            pf = global_r.get("profit_factor", 0.0)
            n_feat = X.shape[1]

            # Score combiné : accuracy pondérée + profit factor
            score = acc * 0.6 + min(pf, 2.0) / 2.0 * 0.4

            print(f"{combo_name:<40} {acc:>10.4f} {pf:>10.4f} {n_feat:>10}")

            results.append({
                "combo": combo_name,
                "indicators": active,
                "n_features": n_feat,
                "accuracy": round(acc, 6),
                "profit_factor": round(pf, 6),
                "score": round(score, 6),
            })

            if score > best_score:
                best_score = score
                best_combo = combo_name
                # Réentraîner sur toutes les données pour sauvegarder le meilleur modèle
                best_model = train(X, y, model_type="lgbm")
                best_meta = {
                    "symbol": SYMBOL,
                    "interval": INTERVAL,
                    "window": WINDOW,
                    "indicators": active,
                    "n_features": n_feat,
                    "combo": combo_name,
                    "accuracy_wf": round(acc, 6),
                    "profit_factor_wf": round(pf, 6),
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                    "n_samples": len(X),
                }

        except Exception as e:
            print(f"{combo_name:<40} ERREUR: {e}")

    # Sauvegarder le meilleur modèle
    if best_model is not None and best_meta is not None:
        save_model(best_model, best_meta)
        print(f"\n[OK] Meilleur modele sauvegarde : {best_combo} (score={best_score:.4f})")

    # Rapport JSON trié par score
    results.sort(key=lambda r: r["score"], reverse=True)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[Rapport] {RESULTS_PATH}")

    # Top 5
    print("\n=== TOP 5 COMBINAISONS ===")
    print(f"{'Rang':<5} {'Combo':<35} {'Accuracy':>10} {'PF':>10} {'Score':>8}")
    print("-" * 75)
    for i, r in enumerate(results[:5], 1):
        print(f"{i:<5} {r['combo']:<35} {r['accuracy']:>10.4f} {r['profit_factor']:>10.4f} {r['score']:>8.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ablation study sur les indicateurs")
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Date de début du filtrage (ex: '1 year ago UTC', '2024-01-01'). "
             "Défaut : toutes les données en cache.",
    )
    args = parser.parse_args()
    run_ablation(start=args.start)
