"""
Bot principal — intègre le modèle de prédiction avec les ordres Polymarket.

Usage:
    python -m polymarket.bot [--bet-pct 5] [--min-bet 1] [--dry-run] [--live]

Logique:
    - SKIP (confiance < seuil)  → aucun trade
    - VERT (prob_green >= 0.5)  → position UP  (achète le token "Up")
    - ROUGE (prob_red  >= 0.5)  → position DOWN (achète le token "Down")

Minimum de position : 1 USD (configurable via --min-bet).
"""
from __future__ import annotations

import argparse
import collections
import csv
import logging
import sys
import time as _time
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("polymarket.bot")

# ── Imports projet ────────────────────────────────────────────────────────────
from src.model import scheduler as sched
from src.model import serializer
from src.features import builder
from src.data.fetcher import fetch_klines
from src.live.predictor import Prediction, _current_candle_open, _next_trigger_time

from polymarket.market_finder import find_btc_5m_market
from polymarket.trader import PolyTrader, TradeResult

# ── CSV de log des trades ─────────────────────────────────────────────────────
TRADES_CSV = Path("polymarket/trades_log.csv")
_TRADE_FIELDS = [
    "timestamp", "candle_open", "candle_close",
    "trade_direction", "model_direction", "model_prob_pct", "confidence_pct",
    "market_up_pct", "market_down_pct",
    "amount_usd", "price", "order_id", "status",
    "actual_direction", "result",
    "error", "slot",
]


@dataclass
class _PendingTrade:
    """Trade placé en attente de résolution (bougie pas encore fermée)."""
    pred: Prediction
    trade: TradeResult
    confidence: float
    market_up_pct: float
    market_down_pct: float


def _append_trade_csv(
    pending: _PendingTrade,
    actual_direction: str,
    result: str,
) -> None:
    """Écrit une ligne dans le CSV une fois le résultat connu."""
    is_new = not TRADES_CSV.exists()
    with TRADES_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_TRADE_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({
            "timestamp":        pending.trade.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "candle_open":      pending.pred.candle_open.strftime("%Y-%m-%d %H:%M"),
            "candle_close":     pending.pred.candle_close.strftime("%Y-%m-%d %H:%M"),
            "trade_direction":  pending.trade.direction,
            "model_direction":  pending.pred.direction,
            "model_prob_pct":   f"{pending.pred.probability * 100:.2f}",
            "confidence_pct":   f"{pending.confidence:.2f}",
            "market_up_pct":    f"{pending.market_up_pct:.1f}",
            "market_down_pct":  f"{pending.market_down_pct:.1f}",
            "amount_usd":       f"{pending.trade.amount_usd:.4f}",
            "price":            f"{pending.trade.price:.4f}",
            "order_id":         pending.trade.order_id,
            "status":           pending.trade.status,
            "actual_direction": actual_direction,
            "result":           result,
            "error":            pending.trade.error,
            "slot":             pending.pred.model_slot,
        })


# ── Bot ───────────────────────────────────────────────────────────────────────

class PolyBot:
    """
    Boucle principale du bot.

    Combine le LivePredictor du modèle ML et le PolyTrader pour
    exécuter des ordres sur Polymarket à chaque bougie M5.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "5m",
        window: int = 50,
        bet_pct: float = 5.0,
        min_bet_usd: float = 1.0,
        dry_run: bool = True,
        entry_delay_s: int = 180,
        market_min_pct: float = 40.0,
    ):
        self.symbol = symbol
        self.interval = interval
        self.window = window

        # ── Modèle ML ──
        if sched.has_schedule():
            self._scheduler = sched.ModelScheduler()
            self._single_model = None
            self._single_meta: dict = {}
            logger.info("Scheduler détecté — routing temporel activé.")
        else:
            self._scheduler = None
            self._single_model, self._single_meta = serializer.load_model(
                "models/model_calibrated.pkl"
            )
            logger.info("Modèle unique chargé.")

        self.buffer: collections.deque = collections.deque(maxlen=window + 1)
        self._context_dfs: dict = {"1h": None, "4h": None}
        self._last_context_refresh: float = 0.0

        # ── Paramètres de filtrage ──
        self.entry_delay_s  = entry_delay_s   # secondes à attendre avant d'entrer (défaut 180)
        self.market_min_pct = market_min_pct  # cote minimale du marché pour notre direction (%)

        # ── Trader Polymarket ──
        self.trader = PolyTrader(
            bet_pct=bet_pct,
            min_bet_usd=min_bet_usd,
            dry_run=dry_run,
        )

        mode_label = "DRY RUN" if dry_run else "LIVE"
        logger.info(
            "PolyBot prêt [%s] | bet=%.1f%% | min=$%.2f | delai=%ds | min_cote=%.0f%%",
            mode_label, bet_pct, min_bet_usd, entry_delay_s, market_min_pct,
        )

    # ── helpers (mêmes que LivePredictor) ────────────────────────────────────

    def _get_model_and_meta(self, dt: datetime) -> tuple:
        if self._scheduler is not None:
            return self._scheduler.get_model(dt)
        return self._single_model, self._single_meta

    def _active_window(self, meta: dict) -> int:
        return meta.get("window", self.window)

    def _refresh_context(self) -> None:
        self._context_dfs["1h"] = fetch_klines(self.symbol, "1h", limit=50)
        self._context_dfs["4h"] = fetch_klines(self.symbol, "4h", limit=20)
        self._last_context_refresh = _time.time()

    def _init_buffer(self, window: int) -> collections.deque:
        df = fetch_klines(self.symbol, self.interval, limit=500)
        closed = df.iloc[:-1]
        buf: collections.deque = collections.deque(maxlen=window + 1)
        for _, row in closed.tail(window + 1).iterrows():
            buf.append({
                "open_time": row["open_time"],
                "open":  row["open"],
                "high":  row["high"],
                "low":   row["low"],
                "close": row["close"],
                "volume": row["volume"],
            })
        return buf

    def _predict(
        self, candles: collections.deque, model, meta: dict, predict_time: datetime
    ) -> Prediction:
        win = self._active_window(meta)
        indicators = meta.get("indicators", None)
        include_time = meta.get("include_time", False)
        pt = predict_time if include_time else None

        df_1h = self._context_dfs.get("1h") if meta.get("multitf_enabled") else None
        df_4h = self._context_dfs.get("4h") if meta.get("multitf_enabled") else None

        X = builder.build_inference_features(
            candles, window=win, indicators=indicators,
            predict_time=pt, df_1h=df_1h, df_4h=df_4h,
        )
        proba = model.predict_proba(X)[0]
        prob_green, prob_red = float(proba[1]), float(proba[0])

        if prob_green >= 0.5:
            direction, probability = "VERT", prob_green
        else:
            direction, probability = "ROUGE", prob_red

        now = datetime.now(timezone.utc)
        candle_open = _current_candle_open(now)
        return Prediction(
            candle_open=candle_open,
            candle_close=candle_open + timedelta(minutes=5),
            direction=direction,
            probability=probability,
            predicted_at=now,
            model_slot=meta.get("slot", "default"),
        )

    # ── boucle principale ────────────────────────────────────────────────────

    def run(self) -> None:
        now = datetime.now(timezone.utc)
        model, meta = self._get_model_and_meta(now)
        active_window = self._active_window(meta)

        self.buffer = self._init_buffer(active_window)
        logger.info("Buffer initialisé : %d bougies (window=%d).", len(self.buffer), active_window)

        if meta.get("multitf_enabled"):
            self._refresh_context()

        if self._scheduler:
            logger.info(self._scheduler.describe(now))

        pending: _PendingTrade | None = None

        try:
            while True:
                now = datetime.now(timezone.utc)
                next_trigger = _next_trigger_time(now)
                delay = (next_trigger - now).total_seconds()
                if delay > 0:
                    time.sleep(delay)

                now = datetime.now(timezone.utc)
                model, meta = self._get_model_and_meta(now)
                active_window = self._active_window(meta)

                if meta.get("multitf_enabled"):
                    if _time.time() - self._last_context_refresh > 3600:
                        self._refresh_context()

                # Redimensionner le buffer si le slot a changé
                if (self.buffer.maxlen or active_window + 1) != active_window + 1:
                    logger.info("Changement de fenêtre → réinitialisation du buffer.")
                    self.buffer = self._init_buffer(active_window)

                # Récupérer la dernière bougie fermée
                df = fetch_klines(self.symbol, self.interval, limit=2)
                last_closed = df.iloc[-2]

                # ── Résoudre le trade précédent (WIN / LOSS) ──────────────────
                if pending is not None:
                    actual_green = float(last_closed["close"]) > float(last_closed["open"])
                    actual_dir   = "UP" if actual_green else "DOWN"
                    result_str   = "WIN" if actual_dir == pending.trade.direction else "LOSS"

                    _append_trade_csv(pending, actual_dir, result_str)

                    pnl_sign = "+" if result_str == "WIN" else "-"
                    logger.info(
                        ">>> Bougie fermee : %s | %s%s | $%s",
                        actual_dir, pnl_sign, result_str,
                        f"{pending.trade.amount_usd:.2f}",
                    )
                    pending = None

                self.buffer.append({
                    "open_time": last_closed["open_time"],
                    "open":  last_closed["open"],
                    "high":  last_closed["high"],
                    "low":   last_closed["low"],
                    "close": last_closed["close"],
                    "volume": last_closed["volume"],
                })

                # ── Prédiction ──
                pred_time = _current_candle_open(now)
                pred = self._predict(self.buffer, model, meta, predict_time=pred_time)

                other_dir = "ROUGE" if pred.direction == "VERT" else "VERT"
                other_prob = 1.0 - pred.probability
                confidence = abs(pred.probability - 0.5) * 200
                min_conf = meta.get("min_confidence_pct", 0.0)

                open_str  = pred.candle_open.strftime("%H:%M")
                close_str = pred.candle_close.strftime("%H:%M")
                slot_str  = f" [{pred.model_slot}]" if pred.model_slot != "default" else ""

                logger.info(
                    "[%s→%s]%s %s %.2f%% | %s %.2f%% | conf=%.1f%%",
                    open_str, close_str, slot_str,
                    pred.direction, pred.probability * 100,
                    other_dir, other_prob * 100,
                    confidence,
                )

                # ── Filtre confiance ──
                if confidence < min_conf:
                    logger.info(
                        "SKIP — confiance %.1f%% < seuil %.1f%%", confidence, min_conf
                    )
                    continue

                # ── Recherche du marché Polymarket (tôt, pour avoir le token_id) ──
                market = find_btc_5m_market(now)
                if market is None:
                    logger.warning("SKIP — marché BTC 5m introuvable sur Polymarket.")
                    continue

                trade_dir = "UP" if pred.direction == "VERT" else "DOWN"

                # ── Condition 1 : attendre 3 minutes après l'ouverture du marché ──
                entry_time = market.window_open + timedelta(seconds=self.entry_delay_s)
                wait_s = (entry_time - datetime.now(timezone.utc)).total_seconds()
                if wait_s > 0:
                    logger.info(
                        "En attente %ds (entree a %s UTC)...",
                        int(wait_s), entry_time.strftime("%H:%M:%S"),
                    )
                    time.sleep(wait_s)

                # Vérifier que le marché est encore ouvert
                now_after_wait = datetime.now(timezone.utc)
                if now_after_wait >= market.window_close - timedelta(seconds=10):
                    logger.warning("SKIP — marché fermé avant l'entrée.")
                    continue

                # ── Condition 2 : cotes du marché vs direction du modèle ──
                up_pct, down_pct = self.trader.get_market_odds(market)
                our_pct   = up_pct   if trade_dir == "UP" else down_pct
                their_pct = down_pct if trade_dir == "UP" else up_pct

                logger.info(
                    "Modele=%s (%.1f%%) | Marche: UP=%.1f%% DOWN=%.1f%%",
                    trade_dir, our_pct, up_pct, down_pct,
                )

                if our_pct < self.market_min_pct:
                    logger.info(
                        "SKIP — marche donne %.1f%% pour %s (seuil=%.0f%%) — "
                        "consensus oppose trop fort (adversaire=%.1f%%)",
                        our_pct, trade_dir, self.market_min_pct, their_pct,
                    )
                    continue

                # ── Passage d'ordre ──
                result = self.trader.place_trade(direction=trade_dir, market=market)
                logger.info("%s", result)

                # ── Stocker en pending pour résolution à la prochaine bougie ──
                if result.status not in ("ERROR",):
                    pending = _PendingTrade(
                        pred=pred,
                        trade=result,
                        confidence=confidence,
                        market_up_pct=up_pct,
                        market_down_pct=down_pct,
                    )

        except KeyboardInterrupt:
            if pending is not None:
                _append_trade_csv(pending, "PENDING", "PENDING")
                logger.info("Derniere position sauvegardee comme PENDING.")
            logger.info("Bot arrete par l'utilisateur.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Bot Polymarket BTC Up/Down 5m")
    parser.add_argument("--bet-pct",      type=float, default=5.0,
                        help="Pourcentage de la balance par trade (défaut: 5%%)")
    parser.add_argument("--min-bet",      type=float, default=1.0,
                        help="Mise minimale en USD (défaut: 1.0)")
    parser.add_argument("--entry-delay",  type=int,   default=180,
                        help="Secondes a attendre avant d'entrer dans le marche (défaut: 180)")
    parser.add_argument("--market-min",   type=float, default=40.0,
                        help="Cote minimale du marche pour notre direction en %% (défaut: 40)")
    parser.add_argument("--dry-run",      action="store_true", default=True,
                        help="Mode simulation — aucun ordre réel (défaut: activé)")
    parser.add_argument("--live",         action="store_true", default=False,
                        help="Mode live — place des ordres réels (désactive --dry-run)")
    args = parser.parse_args()

    dry_run = not args.live  # --live désactive le dry_run

    if not dry_run:
        print("\n⚠️  MODE LIVE ACTIVÉ — Des ordres réels vont être placés !\n")
        confirm = input("Tapez 'OUI' pour confirmer : ").strip()
        if confirm != "OUI":
            print("Annulé.")
            sys.exit(0)

    bot = PolyBot(
        bet_pct=args.bet_pct,
        min_bet_usd=args.min_bet,
        dry_run=dry_run,
        entry_delay_s=args.entry_delay,
        market_min_pct=args.market_min,
    )
    bot.run()


if __name__ == "__main__":
    main()
