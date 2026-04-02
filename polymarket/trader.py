"""
Gestion des ordres Polymarket.

- DRY_RUN=True  : simule les ordres sans rien envoyer au réseau.
- DRY_RUN=False : place des ordres réels sur le CLOB Polymarket.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from py_clob_client.clob_types import (
    AssetType,
    BalanceAllowanceParams,
    MarketOrderArgs,
    OrderType,
)
# OrderType.FAK = Fill-And-Kill (ordre de marché natif Polymarket)
from py_clob_client.order_builder.constants import BUY

from polymarket.client import build_client
from polymarket.market_finder import BtcMarket

logger = logging.getLogger(__name__)

_COLLATERAL_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC.e Polygon


@dataclass
class TradeResult:
    direction: str          # "UP" ou "DOWN"
    amount_usd: float
    token_id: str
    order_id: str = ""
    status: str = "DRY_RUN"
    price: float = 0.0
    error: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __str__(self) -> str:
        if self.error:
            return (
                f"[TRADE ERROR] {self.direction} ${self.amount_usd:.2f} | "
                f"token={self.token_id[:8]}… | {self.error}"
            )
        return (
            f"[TRADE {self.status}] {self.direction} ${self.amount_usd:.2f} | "
            f"price={self.price:.4f} | token={self.token_id[:8]}… | "
            f"order_id={self.order_id or 'N/A'}"
        )


class PolyTrader:
    """
    Passe des ordres sur Polymarket en fonction des prédictions du modèle.

    Args:
        bet_pct: % de la balance USDC à miser par trade (ex: 5 = 5 %).
        min_bet_usd: Mise minimale en USD (défaut: 1.0).
        dry_run: Si True, simule sans envoyer d'ordre réel.
    """

    def __init__(
        self,
        bet_pct: float = 5.0,
        min_bet_usd: float = 1.0,
        dry_run: bool = True,
    ):
        self.bet_pct = bet_pct
        self.min_bet_usd = min_bet_usd
        self.dry_run = dry_run
        self._client = build_client()
        mode = "DRY RUN" if dry_run else "LIVE"
        logger.info("PolyTrader initialisé [%s] | bet=%.1f%% | min=$%.2f", mode, bet_pct, min_bet_usd)

    def get_usdc_balance(self) -> float:
        """Retourne la balance USDC disponible (en USD)."""
        try:
            sig_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "1"))
            resp: dict = self._client.get_balance_allowance(  # type: ignore[assignment]
                BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL,  # type: ignore[arg-type]
                    signature_type=sig_type,
                )
            )
            raw = float(resp.get("balance", 0))
            # USDC a 6 décimales sur Polygon
            return raw / 1e6
        except Exception as exc:
            logger.error("Impossible de récupérer la balance : %s", exc)
            return 0.0

    def _compute_bet_size(self, balance: float) -> float:
        """Calcule la mise en USD, applique le minimum."""
        bet = balance * self.bet_pct / 100.0
        return max(bet, self.min_bet_usd)

    def get_market_odds(self, market: BtcMarket) -> tuple[float, float]:
        """
        Retourne (up_pct, down_pct) — les probabilités implicites du marché
        pour les côtés UP et DOWN (valeurs entre 0 et 100).

        Utilise le prix midpoint du CLOB (moyenne bid/ask).
        En cas d'erreur, retourne (50.0, 50.0).
        """
        try:
            up_raw   = self._client.get_midpoint(market.up_token_id)
            down_raw = self._client.get_midpoint(market.down_token_id)
            # Le CLOB retourne {'mid': '0.52'} — on extrait la valeur
            up_mid   = float(up_raw["mid"]   if isinstance(up_raw, dict)   else up_raw)
            down_mid = float(down_raw["mid"] if isinstance(down_raw, dict) else down_raw)
            # Les midpoints sont entre 0 et 1 sur Polymarket
            up_pct   = round(up_mid * 100, 1)
            down_pct = round(down_mid * 100, 1)
            logger.info(
                "Cotes marche : UP=%.1f%% | DOWN=%.1f%%", up_pct, down_pct
            )
            return up_pct, down_pct
        except Exception as exc:
            logger.warning("Impossible de lire les cotes : %s — skip filtre", exc)
            return 50.0, 50.0

    def place_trade(
        self,
        direction: str,
        market: BtcMarket,
    ) -> TradeResult:
        """
        Place un ordre de marché.

        Args:
            direction: "UP" ou "DOWN"
            market: Marché BTC 5m actif retourné par market_finder.

        Returns:
            TradeResult avec les détails de l'ordre.
        """
        token_id = market.up_token_id if direction == "UP" else market.down_token_id

        balance = self.get_usdc_balance()
        amount = self._compute_bet_size(balance)

        logger.info(
            "Signal %s | balance=$%.2f | mise=$%.2f | token=%s…",
            direction, balance, amount, token_id[:8],
        )

        if self.dry_run:
            return TradeResult(
                direction=direction,
                amount_usd=amount,
                token_id=token_id,
                status="DRY_RUN",
                price=0.5,  # prix inconnu en dry run
            )

        order_args = MarketOrderArgs(
            token_id=token_id,
            amount=amount,
            side=BUY,
            price=0.99,
            order_type=OrderType.FAK,  # type: ignore[arg-type]
        )
        signed_order = self._client.create_market_order(order_args)

        # Retry sur 425 "service not ready" (marché pas encore prêt)
        last_exc: Exception | None = None
        for attempt in range(1, 6):
            try:
                resp: dict = self._client.post_order(signed_order, OrderType.FAK)  # type: ignore[assignment]

                order_id = resp.get("orderID") or resp.get("id") or ""
                status = resp.get("status", "UNKNOWN")
                price = float(resp.get("price", 0))

                result = TradeResult(
                    direction=direction,
                    amount_usd=amount,
                    token_id=token_id,
                    order_id=order_id,
                    status=status,
                    price=price,
                )
                logger.info("Ordre placé : %s", result)
                return result

            except Exception as exc:
                last_exc = exc
                if "425" in str(exc) or "service not ready" in str(exc).lower():
                    logger.warning("425 service not ready (tentative %d/5) — retry dans 2s", attempt)
                    import time as _t; _t.sleep(2)
                else:
                    break

        logger.error("Erreur ordre %s : %s", direction, last_exc)
        return TradeResult(
            direction=direction,
            amount_usd=amount,
            token_id=token_id,
            status="ERROR",
            error=str(last_exc),
        )
