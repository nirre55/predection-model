"""
Fetcher for Binance Futures public REST API.
No authentication required — public market data only.
"""

import time
from datetime import timezone

import pandas as pd
import requests


_FAPI_BASE = "https://fapi.binance.com"
_FUTURES_DATA_BASE = "https://fapi.binance.com"


def _parse_start_str(start_str: str | None) -> int | None:
    """Convert human-readable or ISO start_str to millisecond timestamp."""
    if start_str is None:
        return None
    try:
        # Try direct ISO parsing first
        ts = pd.Timestamp(start_str, tz="UTC")
        return int(ts.timestamp() * 1000)
    except Exception:
        pass
    # Handle relative expressions like "2 years ago UTC"
    try:
        import dateparser
        dt = dateparser.parse(start_str, settings={"RETURN_AS_TIMEZONE_AWARE": True})
        if dt is not None:
            return int(dt.timestamp() * 1000)
    except ImportError:
        pass
    # Fallback: manual parsing of "N unit ago UTC"
    import re
    m = re.match(r"(\d+)\s+(year|month|week|day|hour|minute)s?\s+ago", start_str, re.I)
    if m:
        amount = int(m.group(1))
        unit = m.group(2).lower()
        delta_map = {
            "year":   365 * 24 * 3600,
            "month":  30  * 24 * 3600,
            "week":   7   * 24 * 3600,
            "day":    24  * 3600,
            "hour":   3600,
            "minute": 60,
        }
        seconds = amount * delta_map.get(unit, 0)
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        return now_ms - seconds * 1000
    raise ValueError(f"Cannot parse start_str: {start_str!r}")


def _parse_end_str(end_str: str | None) -> int | None:
    if end_str is None:
        return None
    ts = pd.Timestamp(end_str, tz="UTC")
    return int(ts.timestamp() * 1000)


def fetch_taker_volume(
    symbol: str,
    interval: str,
    start_str: str | None = None,
    end_str: str | None = None,
) -> pd.DataFrame:
    """
    Fetch taker buy volume from Binance Futures klines (GET /fapi/v1/klines).

    Returns a DataFrame with columns:
        open_time         datetime64[ns, UTC]
        taker_buy_volume  float64
        total_volume      float64

    Args:
        symbol:    e.g. "BTCUSDT"
        interval:  e.g. "5m", "1h"
        start_str: e.g. "2 years ago UTC" or "2023-01-01"
        end_str:   e.g. "2025-01-01" (defaults to now)
    """
    url = f"{_FAPI_BASE}/fapi/v1/klines"
    start_ms = _parse_start_str(start_str)
    end_ms = _parse_end_str(end_str)
    limit = 1000

    all_rows: list[tuple] = []

    while True:
        params: dict = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            klines = resp.json()
        except requests.HTTPError as exc:
            raise RuntimeError(f"HTTP error fetching futures klines: {exc}") from exc

        if not klines:
            break

        for k in klines:
            # k[0]  = open_time (ms)
            # k[5]  = total volume (base asset)
            # k[9]  = taker buy base asset volume
            all_rows.append((int(k[0]), float(k[9]), float(k[5])))

        last_open_ms = int(klines[-1][0])
        if len(klines) < limit:
            break

        # Advance start to next candle
        start_ms = last_open_ms + 1
        if end_ms is not None and start_ms > end_ms:
            break

        time.sleep(0.1)

    if not all_rows:
        return pd.DataFrame(columns=["open_time", "taker_buy_volume", "total_volume"])

    df = pd.DataFrame(all_rows, columns=["open_time", "taker_buy_volume", "total_volume"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
    return df


def fetch_open_interest_hist(
    symbol: str,
    period: str = "1h",
    start_str: str | None = None,
    end_str: str | None = None,
) -> pd.DataFrame:
    """
    Fetch historical open interest from Binance Futures
    (GET /futures/data/openInterestHist).

    Returns a DataFrame with columns:
        open_time      datetime64[ns, UTC]
        open_interest  float64

    Args:
        symbol:    e.g. "BTCUSDT"
        period:    e.g. "1h", "4h" (supported: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
        start_str: start time string
        end_str:   end time string
    """
    url = f"{_FAPI_BASE}/futures/data/openInterestHist"
    start_ms = _parse_start_str(start_str)
    end_ms = _parse_end_str(end_str)
    limit = 500

    all_rows: list[tuple] = []

    while True:
        params: dict = {"symbol": symbol, "period": period, "limit": limit}
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as exc:
            raise RuntimeError(f"HTTP error fetching OI history: {exc}") from exc

        if not data:
            break

        for item in data:
            # timestamp field: "timestamp" (ms)
            # sumOpenInterest: open interest in base asset
            ts_ms = int(item.get("timestamp", 0))
            oi = float(item.get("sumOpenInterest", 0.0))
            all_rows.append((ts_ms, oi))

        last_ts_ms = int(data[-1].get("timestamp", 0))
        if len(data) < limit:
            break

        start_ms = last_ts_ms + 1
        if end_ms is not None and start_ms > end_ms:
            break

        time.sleep(0.1)

    if not all_rows:
        return pd.DataFrame(columns=["open_time", "open_interest"])

    df = pd.DataFrame(all_rows, columns=["open_time", "open_interest"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
    return df


def fetch_funding_rates(
    symbol: str,
    start_str: str | None = None,
    end_str: str | None = None,
) -> pd.DataFrame:
    """
    Fetch historical funding rates from Binance Futures
    (GET /fapi/v1/fundingRate).

    Funding rates are settled every 8 hours.

    Returns a DataFrame with columns:
        funding_time  datetime64[ns, UTC]
        funding_rate  float64

    Args:
        symbol:    e.g. "BTCUSDT"
        start_str: start time string
        end_str:   end time string
    """
    url = f"{_FAPI_BASE}/fapi/v1/fundingRate"
    start_ms = _parse_start_str(start_str)
    end_ms = _parse_end_str(end_str)
    limit = 1000

    all_rows: list[tuple] = []

    while True:
        params: dict = {"symbol": symbol, "limit": limit}
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as exc:
            raise RuntimeError(f"HTTP error fetching funding rates: {exc}") from exc

        if not data:
            break

        for item in data:
            ts_ms = int(item.get("fundingTime", 0))
            rate = float(item.get("fundingRate", 0.0))
            all_rows.append((ts_ms, rate))

        last_ts_ms = int(data[-1].get("fundingTime", 0))
        if len(data) < limit:
            break

        start_ms = last_ts_ms + 1
        if end_ms is not None and start_ms > end_ms:
            break

        time.sleep(0.1)

    if not all_rows:
        return pd.DataFrame(columns=["funding_time", "funding_rate"])

    df = pd.DataFrame(all_rows, columns=["funding_time", "funding_rate"])
    df["funding_time"] = pd.to_datetime(df["funding_time"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="funding_time").sort_values("funding_time").reset_index(drop=True)
    return df
