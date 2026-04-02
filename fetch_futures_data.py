"""
Fetch and cache Binance Futures public data.

Run once (or periodically to update):
    python fetch_futures_data.py

Saves to data/raw/:
    BTCUSDT_taker_5m.parquet   — taker buy volume at 5-minute granularity
    BTCUSDT_oi_1h.parquet      — open interest history at 1h granularity
    BTCUSDT_funding.parquet    — funding rates (every 8h)
"""

import sys
from pathlib import Path

import pandas as pd

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.data.fetcher_futures import (
    fetch_taker_volume,
    fetch_open_interest_hist,
    fetch_funding_rates,
)
from src.data.cache import save_futures_data, load_futures_data

SYMBOL   = "BTCUSDT"
START    = "2 years ago UTC"
START_OI = "29 days ago UTC"   # Binance openInterestHist only keeps ~30 days


def _update_or_fetch(symbol: str, data_type: str, fetch_fn, fetch_kwargs: dict, fallback_start: str = START) -> pd.DataFrame:
    """
    Load existing cache, determine the last timestamp, and fetch only new data.
    Falls back to full fetch if no cache exists.
    """
    try:
        existing = load_futures_data(symbol, data_type)
        time_col = existing.columns[0]
        last_ts = existing[time_col].max()
        # Advance by 1 second to avoid re-fetching the last candle
        start_ms = int(last_ts.timestamp() * 1000) + 1_000
        start_str = pd.Timestamp(start_ms, unit="ms", tz="UTC").isoformat()
        print(f"  Updating {data_type} from {start_str} ...")
        new_df = fetch_fn(symbol=symbol, start_str=start_str, **fetch_kwargs)
        if new_df.empty:
            print(f"  No new data for {data_type}.")
            return existing
        path = save_futures_data(new_df, symbol, data_type)
        result = load_futures_data(symbol, data_type)
        print(f"  Saved {len(new_df)} new rows -> {path}  (total: {len(result)})")
        return result
    except FileNotFoundError:
        print(f"  No cache for {data_type}. Fetching full history from {fallback_start} ...")
        df = fetch_fn(symbol=symbol, start_str=fallback_start, **fetch_kwargs)
        if df.empty:
            print(f"  WARNING: No data returned for {data_type}.")
            return df
        path = save_futures_data(df, symbol, data_type)
        print(f"  Saved {len(df)} rows -> {path}")
        return df


def main() -> None:
    print("=" * 60)
    print("Fetching Binance Futures public data")
    print(f"Symbol : {SYMBOL}  |  Start : {START}")
    print("=" * 60)

    # 1. Taker buy volume at 5m granularity
    print("\n[1/3] Taker volume (5m) ...")
    df_taker = _update_or_fetch(
        symbol=SYMBOL,
        data_type="taker_5m",
        fetch_fn=fetch_taker_volume,
        fetch_kwargs={"interval": "5m"},
    )
    if not df_taker.empty:
        print(f"  Range: {df_taker['open_time'].min()} -> {df_taker['open_time'].max()}")

    # 2. Open interest history at 1h granularity
    # NOTE: Binance only keeps ~30 days of OI history; START_OI caps the lookback.
    print("\n[2/3] Open interest history (1h) ...")
    df_oi = _update_or_fetch(
        symbol=SYMBOL,
        data_type="oi_1h",
        fetch_fn=fetch_open_interest_hist,
        fetch_kwargs={"period": "1h"},
        fallback_start=START_OI,
    )
    if not df_oi.empty:
        print(f"  Range: {df_oi['open_time'].min()} -> {df_oi['open_time'].max()}")

    # 3. Funding rates
    print("\n[3/3] Funding rates ...")
    df_funding = _update_or_fetch(
        symbol=SYMBOL,
        data_type="funding",
        fetch_fn=fetch_funding_rates,
        fetch_kwargs={},
    )
    if not df_funding.empty:
        print(f"  Range: {df_funding['funding_time'].min()} -> {df_funding['funding_time'].max()}")

    print("\n" + "=" * 60)
    print("Done. All futures data cached in data/raw/")
    print("=" * 60)


if __name__ == "__main__":
    main()
