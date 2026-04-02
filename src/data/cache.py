from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")


def _get_path(symbol: str, interval: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR / f"{symbol}_{interval}.parquet"


def save_klines(df: pd.DataFrame, symbol: str, interval: str) -> Path:
    path = _get_path(symbol, interval)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset="open_time").sort_values("open_time")
        combined.to_parquet(path, index=False)
    else:
        df.to_parquet(path, index=False)
    return path


def load_klines(symbol: str, interval: str) -> pd.DataFrame:
    path = _get_path(symbol, interval)
    return pd.read_parquet(path)


# ── Futures data cache ────────────────────────────────────────────────────────

def _get_futures_path(symbol: str, data_type: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR / f"{symbol}_{data_type}.parquet"


def save_futures_data(df: pd.DataFrame, symbol: str, data_type: str) -> Path:
    """
    Save futures data (taker volume, OI, funding) to parquet.
    Merges with existing file if present, deduplicating on the first column.

    Args:
        df:        DataFrame to save.
        symbol:    e.g. "BTCUSDT"
        data_type: e.g. "taker_5m", "oi_1h", "funding"
    """
    path = _get_futures_path(symbol, data_type)
    time_col = df.columns[0]  # open_time or funding_time

    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = (
            combined
            .drop_duplicates(subset=time_col)
            .sort_values(time_col)
            .reset_index(drop=True)
        )
        combined.to_parquet(path, index=False)
    else:
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        df_sorted.to_parquet(path, index=False)

    return path


def load_futures_data(symbol: str, data_type: str) -> pd.DataFrame:
    """
    Load futures data from parquet cache.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    path = _get_futures_path(symbol, data_type)
    if not path.exists():
        raise FileNotFoundError(
            f"Futures data not found: {path}. "
            f"Run fetch_futures_data.py first."
        )
    return pd.read_parquet(path)
