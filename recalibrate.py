"""
Recalibration rapide — remplace la couche IsotonicRegression par Sigmoid (Platt)
sans réentraîner les modèles de base (LGBM/XGBoost).

Usage:
    python recalibrate.py
"""

import warnings
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from src.data.cache import load_klines
from src.features.builder import build_dataset_with_target_times
from src.features.builder import build_dataset

SYMBOL = "BTCUSDT"
INTERVAL = "5m"

# Chemin du modèle global
GLOBAL_MODEL = Path("models/model_calibrated.pkl")
# Modèles schedule
SCHEDULE_DIR = Path("models/schedule")


def recalibrate_artifact(artifact: dict, X_cal: np.ndarray, y_cal: np.ndarray) -> dict:
    """
    Remplace la couche de calibration d'un artifact par une LogisticRegression.
    Ne touche pas au modèle de base.
    """
    model = artifact["model"]
    base = model.base_estimator

    raw_proba = base.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(raw_proba, y_cal)

    model._lr = lr
    # Supprimer l'ancien attribut isotonique si présent
    if hasattr(model, "_ir"):
        del model._ir

    artifact["model"] = model
    return artifact


def get_cal_data(
    df,
    window: int,
    indicators: list[str],
    include_time: bool,
    dow_filter: list[int] | None = None,
    hour_start: int = 0,
    hour_end: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Construit les données de calibration pour un slot donné."""
    import pandas as pd

    if dow_filter is not None or (hour_start, hour_end) != (0, 24):
        X_all, y_all, target_times = build_dataset_with_target_times(
            df, window=window, indicators=indicators, include_time=include_time,
        )
        dt = pd.to_datetime(target_times, utc=True)
        dow = dt.dayofweek.values
        hour = dt.hour.values
        in_dow = np.isin(dow, dow_filter) if dow_filter else np.ones(len(X_all), dtype=bool)
        in_hour = (hour >= hour_start) & (hour < hour_end)
        mask = in_dow & in_hour
        X_slot, y_slot = X_all[mask], y_all[mask]
    else:
        X_slot, y_slot = build_dataset(df, window=window, indicators=indicators, include_time=include_time)

    # 20% du milieu pour la calibration (équivalent au split dans trainer.py)
    n = len(X_slot)
    i60 = int(n * 0.6)
    i80 = int(n * 0.8)
    return X_slot[i60:i80], y_slot[i60:i80]


def main():
    print(f"Chargement des données {SYMBOL} {INTERVAL}...")
    df = load_klines(SYMBOL, INTERVAL)
    print(f"{len(df)} bougies chargées.\n")

    models_to_recal: list[tuple[Path, dict | None]] = []

    # Modèle global
    if GLOBAL_MODEL.exists():
        models_to_recal.append((GLOBAL_MODEL, None))

    # Modèles schedule
    if SCHEDULE_DIR.exists():
        for pkl in sorted(SCHEDULE_DIR.glob("*.pkl")):
            artifact = joblib.load(pkl)
            slot_info = {
                "dow": artifact.get("dow"),
                "hour_start": artifact.get("hour_start", 0),
                "hour_end": artifact.get("hour_end", 24),
            }
            models_to_recal.append((pkl, slot_info))

    for path, slot_info in models_to_recal:
        artifact = joblib.load(path)
        window = artifact.get("window", 50)
        indicators = artifact.get("indicators") or ["rsi", "macd", "bb", "atr", "vol"]
        include_time = artifact.get("include_time", False)

        dow = slot_info["dow"] if slot_info else None
        h_start = slot_info["hour_start"] if slot_info else 0
        h_end = slot_info["hour_end"] if slot_info else 24

        X_cal, y_cal = get_cal_data(df, window, indicators, include_time, dow, h_start, h_end)

        print(f"Recalibration {path.name} (window={window}, n_cal={len(X_cal)})...", end=" ")

        # Avant
        model = artifact["model"]
        raw_sample = model.base_estimator.predict_proba(X_cal[:100])[:, 1]

        artifact = recalibrate_artifact(artifact, X_cal, y_cal)

        # Après
        new_model = artifact["model"]
        cal_sample = new_model.predict_proba(X_cal[:100])[:, 1]

        joblib.dump(artifact, path)
        print(
            f"OK | raw std={raw_sample.std():.4f} | "
            f"cal std={cal_sample.std():.4f} | "
            f"cal range=[{cal_sample.min():.3f}, {cal_sample.max():.3f}]"
        )

    print("\n[OK] Tous les modeles recalibres avec sigmoid (Platt scaling).")
    print("Relance: python cli.py live")


if __name__ == "__main__":
    main()
