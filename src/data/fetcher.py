import os
import time
import functools

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

load_dotenv()


def _with_retry(max_attempts: int = 3, backoff: float = 2.0):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            import requests.exceptions

            last_exc: Exception = RuntimeError("No attempts made")
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except (BinanceAPIException, requests.exceptions.ConnectionError) as exc:
                    last_exc = exc
                    wait = backoff ** attempt
                    time.sleep(wait)
            raise last_exc

        return wrapper

    return decorator


def _get_client() -> Client:
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    return Client(api_key, api_secret)


def _parse_raw(klines: list) -> pd.DataFrame:
    df = pd.DataFrame(
        klines,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
        ],
    )
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype("float64")
    return df


@_with_retry(max_attempts=3, backoff=2.0)
def fetch_klines(
    symbol: str,
    interval: str,
    start_str: str | None = None,
    end_str: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    client = _get_client()

    # Quand seul limit est fourni (mode live), on utilise get_klines pour avoir
    # les dernières bougies récentes — get_historical_klines partirait de start_str.
    if limit is not None and start_str is None:
        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
        )
    else:
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_str or "5 years ago UTC",
            end_str=end_str,
            limit=limit,
        )

    return _parse_raw(list(klines))
