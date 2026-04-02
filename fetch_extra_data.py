"""
Télécharge les données supplémentaires pour le modèle v3 :
  1. Bougies daily (D1) — BTCUSDT depuis 4 ans (Binance, gratuit)
  2. Fear & Greed Index — depuis 2018 (alternative.me, gratuit)

Usage:
    python fetch_extra_data.py
"""

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))

from src.data.cache import save_klines, load_klines
from src.data.fetcher import fetch_klines

SYMBOL = "BTCUSDT"
FG_PATH = Path("data/raw/fear_greed.parquet")
FG_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── D1 OHLCV ──────────────────────────────────────────────────────────────────

def fetch_daily_data() -> pd.DataFrame:
    print("[1/2] Bougies daily (1d) ...")
    try:
        existing = load_klines(SYMBOL, "1d")
        last_ts  = existing["open_time"].max()
        start_str = (last_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"  Mise à jour depuis {start_str} ...")
        new_df = fetch_klines(SYMBOL, "1d", start_str=start_str)
        if new_df.empty:
            print("  Aucune nouvelle bougie D1.")
            return existing
        save_klines(new_df, SYMBOL, "1d")
        result = load_klines(SYMBOL, "1d")
        print(f"  +{len(new_df)} bougies  ->  total={len(result)}")
        return result
    except FileNotFoundError:
        print("  Téléchargement complet depuis 4 ans ...")
        df = fetch_klines(SYMBOL, "1d", start_str="4 years ago UTC")
        save_klines(df, SYMBOL, "1d")
        print(f"  Sauvegardé {len(df)} bougies D1")
        return df


# ── Fear & Greed Index ────────────────────────────────────────────────────────

def fetch_fear_greed() -> pd.DataFrame:
    print("[2/2] Fear & Greed Index (alternative.me) ...")
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception as exc:
        print(f"  ERREUR téléchargement F&G: {exc}")
        if FG_PATH.exists():
            return pd.read_parquet(FG_PATH)
        return pd.DataFrame(columns=["date", "fg_value"])

    rows = []
    for item in data:
        ts   = int(item.get("timestamp", 0))
        val  = int(item.get("value", 50))
        rows.append((ts, val))

    df = pd.DataFrame(rows, columns=["timestamp", "fg_value"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.normalize()
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    df = df[["date", "fg_value"]]
    df.to_parquet(FG_PATH, index=False)
    print(f"  Sauvegardé {len(df)} jours F&G  ({df['date'].min()} -> {df['date'].max()})")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Téléchargement données supplémentaires v3")
    print("=" * 60)

    df_1d = fetch_daily_data()
    if not df_1d.empty:
        print(f"  D1 range: {df_1d['open_time'].min()} -> {df_1d['open_time'].max()}")

    df_fg = fetch_fear_greed()
    if not df_fg.empty:
        print(f"  F&G range: {df_fg['date'].min()} -> {df_fg['date'].max()}")

    print("\n" + "=" * 60)
    print("Terminé. Données dans data/raw/")
    print("=" * 60)


if __name__ == "__main__":
    main()
