"""
Recherche le marché "Bitcoin Up or Down" 5 minutes actif sur Polymarket.

Stratégie (v2 — slug-based) :
  Les marchés BTC 5m ont un slug prévisible :
    btc-updown-5m-{floored_epoch}
  où floored_epoch = now_unix - (now_unix % 300)

  On interroge directement l'API Gamma avec ce slug → pas de pagination.
  Si le marché courant est déjà expiré ou pas encore actif, on essaie
  le slot suivant (+5 min).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import requests

logger = logging.getLogger(__name__)

_GAMMA_HOST = "https://gamma-api.polymarket.com"
_SESSION = requests.Session()
_SESSION.headers.update({"Accept": "application/json"})

INTERVAL_SEC = 300  # 5 minutes


@dataclass
class BtcMarket:
    condition_id: str
    question: str
    up_token_id: str
    down_token_id: str
    window_open: datetime    # UTC — début du créneau 5 min
    window_close: datetime   # UTC — fin du créneau 5 min (= résolution)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _floored_epoch(now: datetime) -> int:
    """Epoch Unix arrondi au dernier multiple de 300 secondes."""
    ts = int(now.timestamp())
    return ts - (ts % INTERVAL_SEC)


def _fetch_market_by_slug(slug: str, retries: int = 3) -> dict | None:
    """Interroge Gamma API et retourne le premier marché correspondant, ou None.
    Retente automatiquement sur erreur réseau (pas de fallback vers un autre slot)."""
    for attempt in range(1, retries + 1):
        try:
            resp = _SESSION.get(
                f"{_GAMMA_HOST}/markets",
                params={"slug": slug},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0]
            return None
        except Exception as exc:
            logger.warning("Gamma API error (slug=%s, attempt=%d/%d): %s", slug, attempt, retries, exc)
            if attempt < retries:
                import time as _time
                _time.sleep(1)
    return None


def _extract_tokens(market: dict) -> tuple[str, str] | None:
    """
    Extrait (up_token_id, down_token_id) depuis les champs Gamma :
      - clobTokenIds : liste de token IDs dans le même ordre que outcomes
      - outcomes     : liste de strings, ex. ["Up", "Down"]
    """
    raw_ids = market.get("clobTokenIds") or []
    outcomes = market.get("outcomes") or []

    # clobTokenIds peut être une string JSON ou une liste
    if isinstance(raw_ids, str):
        import json
        try:
            raw_ids = json.loads(raw_ids)
        except Exception:
            raw_ids = []
    if isinstance(outcomes, str):
        import json
        try:
            outcomes = json.loads(outcomes)
        except Exception:
            outcomes = []

    if len(raw_ids) < 2 or len(outcomes) < 2:
        return None

    up_id = down_id = None
    for tid, outcome in zip(raw_ids, outcomes):
        o = str(outcome).lower()
        if o == "up":
            up_id = str(tid)
        elif o == "down":
            down_id = str(tid)

    if up_id and down_id:
        return up_id, down_id
    return None


# ── Recherche principale ──────────────────────────────────────────────────────

def find_btc_5m_market(now: datetime | None = None) -> BtcMarket | None:
    """
    Retourne le marché BTC Up/Down 5 min actif pour la prochaine bougie.

    Essaie d'abord le slot courant (floored_epoch), puis le suivant (+5 min)
    si le slot courant est déjà expiré ou n'accepte plus d'ordres.

    Args:
        now: datetime UTC de référence (défaut: maintenant).

    Returns:
        BtcMarket ou None si introuvable.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    current_epoch = _floored_epoch(now)

    # Slot courant en priorité ; si fermé (acceptingOrders=False), essayer le suivant
    for offset in (0, 1):
        epoch = current_epoch + offset * INTERVAL_SEC
        slug = f"btc-updown-5m-{epoch}"

        market = _fetch_market_by_slug(slug)  # retente 3x sur erreur réseau
        if market is None:
            logger.warning("Slug %s introuvable apres retries", slug)
            continue

        # Vérifier que le marché accepte des ordres
        if not market.get("acceptingOrders") and not market.get("accepting_orders"):
            logger.debug("Slug %s : acceptingOrders=False — essai slot suivant", slug)
            continue

        tokens = _extract_tokens(market)
        if tokens is None:
            logger.warning("Impossible d'extraire UP/DOWN tokens pour %s", slug)
            continue

        up_id, down_id = tokens

        # Fenêtre temporelle depuis l'epoch du slug
        window_open  = datetime.fromtimestamp(epoch, tz=timezone.utc)
        window_close = window_open + timedelta(seconds=INTERVAL_SEC)

        # Le marché doit se fermer dans le futur
        delta = (window_close - now).total_seconds()
        if delta <= 0:
            logger.debug("Slug %s déjà expiré (delta=%.0fs)", slug, delta)
            continue

        question = market.get("question") or slug

        btc_market = BtcMarket(
            condition_id=market.get("conditionId") or market.get("condition_id") or "",
            question=question,
            up_token_id=up_id,
            down_token_id=down_id,
            window_open=window_open,
            window_close=window_close,
        )

        logger.info(
            "Marche BTC 5m : '%s' | ferme %s UTC | up=%s... | down=%s...",
            question[:60],
            window_close.strftime("%H:%M:%S"),
            up_id[:10],
            down_id[:10],
        )
        return btc_market

    logger.warning(
        "Aucun marche BTC 5m actif trouve (now=%s)",
        now.strftime("%H:%M:%S UTC"),
    )
    return None
