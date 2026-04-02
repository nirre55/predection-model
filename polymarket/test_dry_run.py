"""
Test de validation en dry run — aucun ordre réel n'est envoyé.

Ce script vérifie :
  1. Connexion au CLOB Polymarket (ping)
  2. Lecture de la balance USDC
  3. Recherche du marché BTC 5m actif
  4. Chargement du modèle ML + prédiction sur les données récentes
  5. Simulation d'un trade (dry run)

Exécution :
    python -m polymarket.test_dry_run
"""
from __future__ import annotations

import sys
import os
import logging
from datetime import datetime, timezone

# Fix encodage console Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_dry_run")

PASS = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"

errors: list[str] = []


def check(label: str, fn):
    """Exécute fn(), affiche le résultat, retourne la valeur ou None si erreur."""
    try:
        result = fn()
        print(f"  {PASS} {label}: {result}")
        return result
    except Exception as exc:
        print(f"  {FAIL} {label}: {exc}")
        errors.append(f"{label}: {exc}")
        return None


def main() -> None:
    print("\n===========================================")
    print("  Polymarket Bot - Test de validation (DRY RUN)")
    print("===========================================\n")

    now = datetime.now(timezone.utc)
    print(f"Heure UTC : {now.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── 1. Connexion CLOB ──────────────────────────────────────────────────
    print("1. Connexion au CLOB Polymarket")
    from polymarket.client import build_client

    client = check("build_client()", build_client)
    if client:
        check("get_ok()", lambda: client.get_ok())

    # ── 2. Balance USDC ────────────────────────────────────────────────────
    print("\n2. Balance USDC")
    from polymarket.trader import PolyTrader

    trader = PolyTrader(bet_pct=5.0, min_bet_usd=1.0, dry_run=True)
    balance = check("get_usdc_balance()", trader.get_usdc_balance)
    if balance is not None:
        bet = trader._compute_bet_size(balance)
        print(f"  {PASS} Mise calculée (5%% de ${balance:.2f}) = ${bet:.4f}")

    # ── 3. Recherche marché BTC 5m ─────────────────────────────────────────
    print("\n3. Marché BTC Up/Down 5m (Polymarket Gamma)")
    from polymarket.market_finder import find_btc_5m_market

    market = check("find_btc_5m_market()", lambda: find_btc_5m_market(now))
    if market:
        print(f"  {PASS} Question  : {market.question[:80]}")
        print(f"  {PASS} Up token  : {market.up_token_id[:16]}…")
        print(f"  {PASS} Down token: {market.down_token_id[:16]}…")
        print(f"  {PASS} Fenetre   : {market.window_open.strftime('%H:%M')} -> {market.window_close.strftime('%H:%M UTC')}")
    else:
        print(f"  {WARN} Marché introuvable — normal si hors heures de trading")

    # ── 4. Modèle ML + prédiction ──────────────────────────────────────────
    print("\n4. Modèle ML + prédiction")

    def load_and_predict():
        import collections
        from src.model import scheduler as sched, serializer
        from src.features import builder
        from src.data.fetcher import fetch_klines
        from src.live.predictor import _current_candle_open

        if sched.has_schedule():
            scheduler = sched.ModelScheduler()
            model, meta = scheduler.get_model(now)
            slot = meta.get("slot", "default")
        else:
            model, meta = serializer.load_model("models/model_calibrated.pkl")
            slot = "default"

        win = meta.get("window", 50)
        df = fetch_klines("BTCUSDT", "5m", limit=win + 2)
        closed = df.iloc[:-1]
        buf: collections.deque = collections.deque(maxlen=win + 1)
        for _, row in closed.tail(win + 1).iterrows():
            buf.append({
                "open_time": row["open_time"],
                "open":   row["open"],
                "high":   row["high"],
                "low":    row["low"],
                "close":  row["close"],
                "volume": row["volume"],
            })

        indicators = meta.get("indicators", None)
        include_time = meta.get("include_time", False)
        predict_time = _current_candle_open(now) if include_time else None

        df_1h = df_4h = None
        if meta.get("multitf_enabled"):
            df_1h = fetch_klines("BTCUSDT", "1h", limit=50)
            df_4h = fetch_klines("BTCUSDT", "4h", limit=20)

        X = builder.build_inference_features(
            buf, window=win, indicators=indicators,
            predict_time=predict_time, df_1h=df_1h, df_4h=df_4h,
        )
        proba = model.predict_proba(X)[0]
        prob_green = float(proba[1])
        direction = "VERT" if prob_green >= 0.5 else "ROUGE"
        confidence = abs(prob_green - 0.5) * 200
        return f"{direction} | prob_green={prob_green:.2%} | conf={confidence:.1f}% | slot={slot}"

    pred_str = check("predict()", load_and_predict)

    # ── 5. Simulation d'un trade ───────────────────────────────────────────
    print("\n5. Simulation de trade (dry run)")

    if market and balance is not None:
        result = check(
            "place_trade(UP, dry_run=True)",
            lambda: trader.place_trade("UP", market),
        )
        if result:
            print(f"  {PASS} Résultat : {result}")
    else:
        print(f"  {WARN} Skipped — marché ou balance indisponible")

    # ── Résumé ─────────────────────────────────────────────────────────────
    print("\n===========================================")
    if errors:
        print(f"  {FAIL} {len(errors)} erreur(s) détectée(s) :")
        for e in errors:
            print(f"     • {e}")
        print()
        sys.exit(1)
    else:
        print(f"  {PASS} Tous les tests passés — bot prêt !")
        print("  Pour démarrer en dry run : python -m polymarket.bot --dry-run")
        print("  Pour démarrer en live    : python -m polymarket.bot --live")
    print()


if __name__ == "__main__":
    main()
