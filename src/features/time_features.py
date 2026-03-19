"""
Encodage cyclique des features temporelles (heure du jour, jour de semaine).
Sin/cos pour préserver la continuité (ex: 23h → 0h est proche, pas loin).
"""

import numpy as np
import pandas as pd
from datetime import datetime


def _encode(hours: np.ndarray, dow: np.ndarray) -> np.ndarray:
    """
    Encode heure + jour de semaine en 5 features:
      hour_sin, hour_cos, dow_sin, dow_cos, is_weekend
    """
    return np.column_stack([
        np.sin(2 * np.pi * hours / 24.0),
        np.cos(2 * np.pi * hours / 24.0),
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
        (dow >= 5).astype("float64"),  # 1 = samedi ou dimanche
    ])


def from_timestamps(open_times: pd.Series) -> np.ndarray:
    """
    Construit (n, 5) features temporelles à partir d'une Series de timestamps.
    open_times doit être convertible en DatetimeTZDtype UTC.
    """
    dt = pd.to_datetime(open_times, utc=True)
    hours = np.asarray(
        (dt.dt.hour + dt.dt.minute / 60.0).values, dtype="float64"
    )
    dow = np.asarray(dt.dt.dayofweek.values, dtype="float64")  # 0=Lundi, 6=Dimanche
    return _encode(hours, dow)


def from_datetime(dt: datetime) -> np.ndarray:
    """Retourne (5,) pour un seul instant datetime (UTC recommandé)."""
    hours = float(dt.hour) + float(dt.minute) / 60.0
    dow = float(dt.weekday())  # 0=Lundi, 6=Dimanche
    return _encode(np.array([hours]), np.array([dow]))[0]
