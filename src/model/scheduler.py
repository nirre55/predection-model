"""
Scheduler temps-réel : sélectionne le bon modèle selon le jour et l'heure.

Format schedule.json :
[
  {"slot": "sunday_9_20", "dow": [6], "hour_start": 9, "hour_end": 20,
   "model_path": "models/schedule/sunday_9_20.pkl"},
  ...
  {"slot": "default", "default": true, "model_path": "models/model_calibrated.pkl"}
]

dow : 0=Lundi, 1=Mardi, 2=Mercredi, 3=Jeudi, 4=Vendredi, 5=Samedi, 6=Dimanche
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.model.serializer import load_model

SCHEDULE_PATH = Path("models/schedule.json")


def has_schedule() -> bool:
    """Retourne True si un fichier schedule.json existe."""
    return SCHEDULE_PATH.exists()


class ModelScheduler:
    """
    Charge et met en cache les modèles.
    Sélectionne le bon modèle en fonction de l'heure UTC courante.
    """

    def __init__(self, schedule_path: str | Path = SCHEDULE_PATH):
        self._schedule: list[dict] = json.loads(Path(schedule_path).read_text())
        self._cache: dict[str, tuple] = {}

    def get_model(self, dt: datetime | None = None):
        """
        Retourne (model, metadata) pour le datetime donné (UTC).
        Si dt est None, utilise datetime.now(UTC).
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        dow = dt.weekday()   # 0=Lundi … 6=Dimanche
        hour = dt.hour

        for slot in self._schedule:
            if slot.get("default"):
                continue
            dow_list: list[int] = slot["dow"]
            h_start: int = slot["hour_start"]
            h_end: int = slot["hour_end"]
            if dow in dow_list and h_start <= hour < h_end:
                return self._load_with_slot(slot)

        # Aucun slot spécifique → modèle par défaut
        default = next((s for s in self._schedule if s.get("default")), None)
        if default:
            return self._load_with_slot(default)

        raise RuntimeError("Aucun slot de schedule trouvé et pas de modèle par défaut.")

    def describe(self, dt: datetime | None = None) -> str:
        """Retourne une description textuelle du modèle actif."""
        if dt is None:
            dt = datetime.now(timezone.utc)
        dow = dt.weekday()
        hour = dt.hour
        day_names = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

        for slot in self._schedule:
            if slot.get("default"):
                continue
            if dow in slot["dow"] and slot["hour_start"] <= hour < slot["hour_end"]:
                return (
                    f"Slot '{slot['slot']}' — {day_names[dow]} {hour:02d}h "
                    f"({slot['model_path']})"
                )
        return f"Slot 'default' — {day_names[dow]} {hour:02d}h"

    def _load_with_slot(self, slot: dict) -> tuple:
        """Charge le modèle et fusionne la config du slot dans la metadata."""
        path = slot["model_path"]
        if path not in self._cache:
            self._cache[path] = load_model(path)
        model, meta = self._cache[path]
        # Fusionne les paramètres live du slot (min_confidence_pct, window, etc.)
        # dans la meta du modèle pour qu'ils soient accessibles au predictor.
        merged = dict(meta)
        for k, v in slot.items():
            if k not in ("model_path", "default"):
                merged.setdefault(k, v)
        # min_confidence_pct du slot prend toujours la priorité
        if "min_confidence_pct" in slot:
            merged["min_confidence_pct"] = slot["min_confidence_pct"]
        return model, merged
