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
