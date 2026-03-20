"""
Ablation temporelle — entraîne et compare LGBM vs XGBoost par slot horaire.

Pour chaque slot (dimanche 9-20h, vendredi, samedi 7-14h, etc.) :
  - Filtre les samples dont la bougie cible tombe dans ce slot
  - Teste windows [20, 50, 100] × model_types [lgbm, xgb]
  - Conserve le meilleur modèle par slot
  - Génère models/schedule.json utilisé par le live predictor

Usage:
    python ablation_schedule.py
    python ablation_schedule.py --start "1 year ago UTC"
    python ablation_schedule.py --start "1 Jan 2024 UTC"
    python ablation_schedule.py --start "2025-01-01"
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
from src.model.trainer import train
from src.model.serializer import save_model

# --- Configuration ---

SYMBOL = "BTCUSDT"
INTERVAL = "5m"
N_SPLITS = 5
BEST_INDICATORS = ["rsi", "macd", "atr"]   # meilleur combo de l'ablation globale
INCLUDE_TIME = True                          # ajouter features heure/jour
WINDOWS = [20, 50, 100]
MODEL_TYPES = ["lgbm", "xgb"]
SCHEDULE_DIR = Path("models/schedule")
SCHEDULE_PATH = Path("models/schedule.json")
MIN_SAMPLES = 500   # minimum de samples pour qu'un slot soit entraînable

# --- Définition des slots horaires ---
# dow: 0=Lundi, 1=Mardi, 2=Mercredi, 3=Jeudi, 4=Vendredi, 5=Samedi, 6=Dimanche

TIME_SLOTS = [
    # (slot_name, dow_list, hour_start, hour_end, preferred_model)
    # Préférence issue de la demande utilisateur — l'ablation peut la confirmer ou corriger
    ("sunday_9_20",     [6],          9,  20, "lgbm"),
    ("sunday_0_9",      [6],          0,   9, "xgb"),
    ("sunday_20_24",    [6],         20,  24, "xgb"),
    ("friday_all",      [4],          0,  24, "xgb"),
    ("saturday_7_14",   [5],          7,  14, "lgbm"),
    ("saturday_0_7",    [5],          0,   7, "xgb"),
    ("saturday_14_24",  [5],         14,  24, "xgb"),
    ("weekdays_day",    [0,1,2,3],    7,  20, "lgbm"),
    ("weekdays_night",  [0,1,2,3],    0,   7, "xgb"),
    ("weekdays_eve",    [0,1,2,3],   20,  24, "xgb"),
]


def mask_for_slot(
    target_times: np.ndarray,
    dow_list: list[int],
    hour_start: int,
    hour_end: int,
) -> np.ndarray:
    """Retourne un masque booléen sur target_times."""
    dt = pd.to_datetime(target_times, utc=True)
    dow = dt.dayofweek.values         # 0=Lundi, 6=Dimanche
    hour = dt.hour.values
    in_dow = np.isin(dow, dow_list)
    in_hour = (hour >= hour_start) & (hour < hour_end)
    return in_dow & in_hour


def run_slot(
    slot_name: str,
    X_all: np.ndarray,
    y_all: np.ndarray,
    target_times: np.ndarray,
    dow_list: list[int],
    hour_start: int,
    hour_end: int,
    preferred_model: str,
    window: int,
) -> dict | None:
    """
    Teste LGBM et XGBoost sur un slot horaire, retourne le meilleur résultat.
    """
    mask = mask_for_slot(target_times, dow_list, hour_start, hour_end)
    X_slot, y_slot = X_all[mask], y_all[mask]

    if len(X_slot) < MIN_SAMPLES:
        print(f"  [{slot_name}] Trop peu de samples ({len(X_slot)}), slot ignoré.")
        return None

    price_moves = np.abs(y_slot.astype("float64") - 0.5) * 2  # proxy simple

    best_score = -np.inf
    best_result = None

    for model_type in MODEL_TYPES:
        try:
            wf = run_walk_forward(X_slot, y_slot, price_moves, n_splits=N_SPLITS, model_type=model_type)
            g = next(r for r in wf if r["fold"] == "global")
            acc = g.get("accuracy", 0.0)
            pf = g.get("profit_factor", 0.0)
            score = acc * 0.6 + min(pf, 2.0) / 2.0 * 0.4
            label = "PREFERE" if model_type == preferred_model else "      "
            print(f"    {label} {model_type:<6} win={window:<4} acc={acc:.4f} pf={pf:.4f} score={score:.4f} n={len(X_slot)}")

            if score > best_score:
                best_score = score
                best_result = {
                    "slot": slot_name,
                    "model_type": model_type,
                    "window": window,
                    "accuracy": round(acc, 6),
                    "profit_factor": round(pf, 6),
                    "score": round(score, 6),
                    "n_samples": int(len(X_slot)),
                    "dow": dow_list,
                    "hour_start": hour_start,
                    "hour_end": hour_end,
                }
        except Exception as e:
            print(f"    ERREUR {model_type} win={window}: {e}")

    return best_result


def run_schedule_ablation(start: str | None = None):
    print(f"Chargement des données {SYMBOL} {INTERVAL}...")
    df = load_klines(SYMBOL, INTERVAL)
    if start is not None:
        cutoff = pd.Timestamp(start).tz_localize("UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start)
        df = df[df["open_time"] >= cutoff].reset_index(drop=True)
        print(f"Filtre appliqué depuis {cutoff.date()} : {len(df)} bougies.")
    print(f"{len(df)} bougies chargées.\n")

    SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)

    # Résultats par slot : {slot_name: best_result_across_windows}
    slot_best: dict[str, dict] = {}

    for slot_name, dow_list, h_start, h_end, preferred in TIME_SLOTS:
        print(f"\n=== Slot : {slot_name} (dow={dow_list} {h_start}h-{h_end}h) ===")

        best_for_slot: dict | None = None

        for window in WINDOWS:
            X_all, y_all, target_times = build_dataset_with_target_times(
                df,
                window=window,
                indicators=BEST_INDICATORS,
                include_time=INCLUDE_TIME,
            )

            result = run_slot(
                slot_name, X_all, y_all, target_times,
                dow_list, h_start, h_end, preferred, window,
            )

            if result is not None:
                if best_for_slot is None or result["score"] > best_for_slot["score"]:
                    best_for_slot = result

        if best_for_slot is None:
            print(f"  --> Slot {slot_name} ignoré (pas assez de données).")
            continue

        # Réentraîner le meilleur modèle sur toutes les données du slot
        print(
            f"  --> Meilleur: {best_for_slot['model_type']} win={best_for_slot['window']} "
            f"score={best_for_slot['score']:.4f}"
        )

        X_all, y_all, target_times = build_dataset_with_target_times(
            df,
            window=best_for_slot["window"],
            indicators=BEST_INDICATORS,
            include_time=INCLUDE_TIME,
        )
        mask = mask_for_slot(target_times, dow_list, h_start, h_end)
        X_slot, y_slot = X_all[mask], y_all[mask]

        final_model = train(X_slot, y_slot, model_type=best_for_slot["model_type"])
        model_path = str(SCHEDULE_DIR / f"{slot_name}.pkl")
        meta = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "window": best_for_slot["window"],
            "indicators": BEST_INDICATORS,
            "include_time": INCLUDE_TIME,
            "n_features": X_slot.shape[1],
            "combo": "+".join(BEST_INDICATORS),
            "model_type": best_for_slot["model_type"],
            "slot": slot_name,
            "dow": dow_list,
            "hour_start": h_start,
            "hour_end": h_end,
            "accuracy_wf": best_for_slot["accuracy"],
            "profit_factor_wf": best_for_slot["profit_factor"],
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_samples": int(len(X_slot)),
        }
        save_model(final_model, meta, path=model_path)
        best_for_slot["model_path"] = model_path
        slot_best[slot_name] = best_for_slot
        print(f"  --> Sauvegarde: {model_path}")

    # Générer schedule.json
    schedule = []
    for slot_name, dow_list, h_start, h_end, _ in TIME_SLOTS:
        if slot_name not in slot_best:
            continue
        info = slot_best[slot_name]
        schedule.append({
            "slot": slot_name,
            "dow": dow_list,
            "hour_start": h_start,
            "hour_end": h_end,
            "model_type": info["model_type"],
            "model_path": info["model_path"],
            "accuracy_wf": info["accuracy"],
            "profit_factor_wf": info["profit_factor"],
            "score": info["score"],
            "window": info["window"],
            "n_samples": info["n_samples"],
        })

    # Modèle par défaut = meilleur modèle global existant
    schedule.append({
        "slot": "default",
        "default": True,
        "model_path": "models/model_calibrated.pkl",
    })

    SCHEDULE_PATH.write_text(json.dumps(schedule, indent=2, ensure_ascii=False))
    print(f"\n[Rapport] Schedule sauvegarde : {SCHEDULE_PATH}")

    # Tableau de synthèse
    print("\n=== RESUME PAR SLOT ===")
    print(f"{'Slot':<22} {'Model':<6} {'Win':>4} {'Acc':>8} {'PF':>8} {'Score':>7} {'N':>7}")
    print("-" * 70)
    for slot_name, info in slot_best.items():
        print(
            f"{slot_name:<22} {info['model_type']:<6} {info['window']:>4} "
            f"{info['accuracy']:>8.4f} {info['profit_factor']:>8.4f} "
            f"{info['score']:>7.4f} {info['n_samples']:>7}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ablation temporelle par slot horaire")
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Date de début du filtrage (ex: '1 year ago UTC', '2024-01-01'). "
             "Défaut : toutes les données en cache.",
    )
    args = parser.parse_args()
    run_schedule_ablation(start=args.start)
