#!/usr/bin/env python
"""
analyze_candles.py -- Analyse des points communs entre des bougies Polymarket

Usage:
    python analyze_candles.py <markets.txt>

Format du fichier (une ligne par marche):
    btc-updown-5m-1775571300
    # commentaires ignores

Le script fetche directement depuis Binance uniquement les bougies necessaires,
pas tout l'historique. Aucun fichier local requis.
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

import numpy as np
import pandas as pd
import requests

# ------------------------------------------------------------------ constants

BINANCE_URL  = "https://api.binance.com/api/v3/klines"
FG_URL       = "https://api.alternative.me/fng/?limit=0&format=json"
SYMBOL       = "BTCUSDT"
WINDOW       = 50        # bougies de contexte avant chaque cible
MONTHLY_CSV  = "sim_A_payoutA_MM1_flat_fixed.csv"
ONE_SEC_NS  = 1_000_000_000
ONE_MIN_NS  = 60  * ONE_SEC_NS
ONE_HOUR_NS = 60  * ONE_MIN_NS
ONE_DAY_NS  = 24  * ONE_HOUR_NS


# ------------------------------------------------------------------ parsing

def parse_market_id(line: str):
    """'btc-updown-5m-1775571300' -> (asset, direction, timeframe, ts_seconds)"""
    parts = line.strip().split("-")
    ts_s  = int(parts[-1])
    tf    = parts[-2]
    direction = "-".join(parts[1:-2])
    asset = parts[0].upper()
    return asset, direction, tf, ts_s


def ts_to_human(ts_s: int) -> str:
    dt = datetime.fromtimestamp(ts_s, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC (%A)")


# ------------------------------------------------------------------ Binance fetch

def _fetch_klines_raw(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch klines from Binance for a specific ms range. Returns OHLCV DataFrame."""
    rows = []
    params = {
        "symbol":    symbol,
        "interval":  interval,
        "startTime": start_ms,
        "endTime":   end_ms,
        "limit":     1000,
    }
    while True:
        resp = requests.get(BINANCE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        rows.extend(data)
        if len(data) < 1000:
            break
        params["startTime"] = int(data[-1][0]) + 1
        time.sleep(0.08)

    if not rows:
        return pd.DataFrame(columns=["ts_ns", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=[
        "open_time_ms", "open", "high", "low", "close", "volume",
        "close_time", "qvol", "ntrades", "tbvol", "tbqvol", "ignore",
    ])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    df["ts_ns"] = df["open_time_ms"].astype("int64") * 1_000_000   # ms -> ns
    return df[["ts_ns", "open", "high", "low", "close", "volume"]].sort_values("ts_ns").reset_index(drop=True)


def fetch_5m(timestamps_s: list[int]) -> pd.DataFrame:
    """
    Fetch 5m candles covering all given timestamps plus WINDOW candles of context before each.
    Makes the minimum number of API calls needed.
    """
    # Range needed: earliest start - WINDOW*5 min  to  latest end + 5 min
    min_ts_ms = (min(timestamps_s) - WINDOW * 5 * 60) * 1000
    max_ts_ms = (max(timestamps_s) + 5 * 60) * 1000

    print(f"  Fetch 5m  {_ms_to_str(min_ts_ms)} -> {_ms_to_str(max_ts_ms)} ...")
    df = _fetch_klines_raw(SYMBOL, "5m", min_ts_ms, max_ts_ms)
    print(f"  -> {len(df)} bougies 5m")
    return df


def fetch_1d(timestamps_s: list[int]) -> pd.DataFrame:
    """Fetch daily candles from 2 days before the earliest candle to cover J-1 context."""
    min_ts_ms = (min(timestamps_s) - 2 * 24 * 3600) * 1000
    max_ts_ms = (max(timestamps_s) + 24 * 3600) * 1000

    print(f"  Fetch 1d  {_ms_to_str(min_ts_ms)} -> {_ms_to_str(max_ts_ms)} ...")
    df = _fetch_klines_raw(SYMBOL, "1d", min_ts_ms, max_ts_ms)
    print(f"  -> {len(df)} bougies 1d")
    return df


def fetch_fear_greed() -> pd.DataFrame:
    """Fetch full Fear & Greed history (lightweight ~50 KB)."""
    print("  Fetch Fear & Greed ...")
    try:
        resp = requests.get(FG_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception as exc:
        print(f"  F&G indisponible: {exc}")
        return pd.DataFrame(columns=["ts_ns", "fg_value"])

    rows = []
    for item in data:
        ts_s  = int(item["timestamp"])
        # Normalize to midnight UTC
        day_ns = (ts_s * ONE_SEC_NS // ONE_DAY_NS) * ONE_DAY_NS
        rows.append({"ts_ns": day_ns, "fg_value": int(item["value"])})

    df = (pd.DataFrame(rows)
          .drop_duplicates(subset="ts_ns")
          .sort_values("ts_ns")
          .reset_index(drop=True))
    print(f"  -> {len(df)} jours F&G")
    return df


def _ms_to_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


# ------------------------------------------------------------------ indicators

def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return float("nan")
    d = np.diff(closes[-(period + 1):])
    ag = np.where(d > 0,  d, 0.0).mean()
    al = np.where(d < 0, -d, 0.0).mean()
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


def _atr(w: pd.DataFrame, period: int = 14) -> float:
    if len(w) < period + 1:
        return float("nan")
    h = np.asarray(w["high"], dtype=float); l = np.asarray(w["low"], dtype=float); c = np.asarray(w["close"], dtype=float)
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]),
                               np.abs(l[1:] - c[:-1])))
    return float(tr[-period:].mean())


def _sma(v: np.ndarray, p: int) -> float:
    return float("nan") if len(v) < p else float(v[-p:].mean())


def _bollinger(closes: np.ndarray, period: int = 20) -> dict:
    if len(closes) < period:
        return {}
    sl = closes[-period:]; mid = sl.mean(); std = sl.std()
    upper = mid + 2 * std; lower = mid - 2 * std; rng = upper - lower
    price = closes[-1]
    return {
        "bb_width_pct": round(rng / mid * 100, 4) if mid else 0,
        "bb_position":  round((price - lower) / rng, 4) if rng else 0.5,
    }


def _macd(closes: np.ndarray) -> dict:
    if len(closes) < 35:
        return {}
    def ema(arr, span):
        a = 2 / (span + 1); r = [arr[0]]
        for v in arr[1:]: r.append(r[-1] * (1 - a) + v * a)
        return np.array(r)
    line   = ema(closes, 12) - ema(closes, 26)
    signal = ema(line, 9)
    hist   = line - signal
    return {"macd_hist": round(float(hist[-1]), 4),
            "macd_trend": "bullish" if hist[-1] > 0 else "bearish"}


def _stoch(w: pd.DataFrame, k: int = 14) -> float:
    if len(w) < k:
        return float("nan")
    sl = w.iloc[-k:]
    lo = sl["low"].min(); hi = sl["high"].max()
    c  = w["close"].iloc[-1]
    return round((c - lo) / (hi - lo) * 100, 2) if hi != lo else 50.0


def _candle_type(o: float, h: float, l: float, c: float) -> str:
    total = h - l
    if total < o * 0.0002: return "flat"
    body  = abs(c - o); ratio = body / total
    upper = h - max(o, c); lower = min(o, c) - l
    if ratio < 0.1:
        if upper > 2 * lower: return "shooting_star"
        if lower > 2 * upper: return "hammer"
        return "doji"
    if ratio > 0.7: return "strong_bull" if c > o else "strong_bear"
    return "bull" if c > o else "bear"


def _session(hour: int) -> str:
    if hour < 8:  return "asian"
    if hour < 13: return "london"
    if hour < 17: return "overlap"
    if hour < 22: return "new_york"
    return "late"


def _streak(w: pd.DataFrame):
    dirs = (np.asarray(w["close"], dtype=float) > np.asarray(w["open"], dtype=float)).astype(int)
    last = dirs[-1]; n = 0
    for d in reversed(dirs):
        if d == last: n += 1
        else: break
    return n, "bull" if last else "bear"


# ------------------------------------------------------------------ context helpers

def _d1_context(df_d1: "pd.DataFrame | None", ts_ns: int) -> dict:
    if df_d1 is None or len(df_d1) == 0:
        return {}
    day_start = (ts_ns // ONE_DAY_NS) * ONE_DAY_NS
    idx = int(np.searchsorted(np.asarray(df_d1["ts_ns"], dtype=np.int64), day_start, side="left")) - 1
    if idx < 1:
        return {}
    row    = df_d1.iloc[idx]
    closes = np.asarray(df_d1["close"], dtype=float)[: idx + 1]
    result = {
        "d1_return_pct": round((float(row["close"]) - float(row["open"])) / float(row["open"]) * 100, 4),
        "d1_rsi":        round(_rsi(closes, 14), 2),
    }
    if len(closes) >= 20:
        s20 = _sma(closes, 20)
        result["d1_above_sma20"]    = bool(float(row["close"]) > s20)
        result["d1_dist_sma20_pct"] = round((float(row["close"]) - s20) / s20 * 100, 4)
    return result


def _fear_greed_ctx(df_fg: "pd.DataFrame | None", ts_ns: int) -> dict:
    if df_fg is None or len(df_fg) == 0:
        return {}
    day_start = (ts_ns // ONE_DAY_NS) * ONE_DAY_NS
    idx = int(np.searchsorted(np.asarray(df_fg["ts_ns"], dtype=np.int64), day_start, side="left")) - 1
    if idx < 0:
        return {}
    v = float(df_fg.iloc[idx]["fg_value"])
    if   v < 25: regime = "extreme_fear"
    elif v < 45: regime = "fear"
    elif v < 55: regime = "neutral"
    elif v < 75: regime = "greed"
    else:        regime = "extreme_greed"
    return {"fg_value": round(v, 1), "fg_regime": regime}


# ------------------------------------------------------------------ feature extraction

def extract_features(df_5m: pd.DataFrame, target_ts_ns: int,
                     df_1d=None, df_fg=None) -> "dict | None":
    """Find the candle at target_ts_ns and compute all features."""
    timestamps = np.asarray(df_5m["ts_ns"], dtype=np.int64)
    idx = int(np.searchsorted(timestamps, target_ts_ns))

    # Allow small tolerance (< 5 min)
    for candidate in (idx, idx - 1):
        if 0 <= candidate < len(timestamps):
            if abs(int(timestamps[candidate]) - target_ts_ns) < 5 * ONE_MIN_NS:
                idx = candidate
                break
    else:
        return None

    if idx < WINDOW:
        return None

    candle = df_5m.iloc[idx]
    window = df_5m.iloc[idx - WINDOW: idx]
    closes = np.asarray(window["close"], dtype=float)

    o  = float(candle["open"]);   h = float(candle["high"])
    l  = float(candle["low"]);    c = float(candle["close"])
    vol = float(candle["volume"])

    total  = h - l
    body   = c - o
    upper  = h - max(o, c)
    lower  = min(o, c) - l

    ts_utc = pd.Timestamp(target_ts_ns, unit="ns", tz="UTC")
    ts_et  = ts_utc.tz_convert("America/New_York")
    ts_pd  = ts_utc   # alias conserve pour compatibilite
    streak_n, streak_dir = _streak(window)

    prev_close = float(closes[-1])
    highs = np.asarray(window["high"], dtype=float)
    lows  = np.asarray(window["low"],  dtype=float)

    h_utc = ts_utc.hour
    h_et  = ts_et.hour

    def _hour_range(h, step):
        start = (h // step) * step
        end   = start + step
        return f"{start:02d}h-{end % 24:02d}h" if end != 24 else f"{start:02d}h-00h"

    feat: dict = {
        "datetime_utc": ts_utc.strftime("%Y-%m-%d %H:%M"),
        "datetime_et":  ts_et.strftime("%Y-%m-%d %H:%M"),

        # ══ PRE-CANDLE — connus avant l'ouverture ══════════════════
        # -- UTC
        "hour_utc":        h_utc,
        "day_of_week":     ts_utc.strftime("%A"),
        "session":         _session(h_utc),
        "hour_range_6h":   _hour_range(h_utc, 6),
        "hour_range_4h":   _hour_range(h_utc, 4),
        "hour_range_3h":   _hour_range(h_utc, 3),
        "hour_range_8h":   _hour_range(h_utc, 8),
        # -- ET (America/New_York, gere EDT/EST automatiquement)
        "hour_et":         h_et,
        "day_of_week_et":  ts_et.strftime("%A"),
        "session_et":      _session(h_et),
        "hour_range_et_6h": _hour_range(h_et, 6),
        "hour_range_et_4h": _hour_range(h_et, 4),
        "hour_range_et_3h": _hour_range(h_et, 3),
        "hour_range_et_8h": _hour_range(h_et, 8),

        # indicateurs sur bougies precedentes fermees
        "rsi_14":          round(_rsi(closes, 14), 2),
        "stoch_k":         _stoch(window, 14),
        "atr_14":          round(_atr(window, 14), 4),
        "prev_streak":     streak_n,
        "prev_streak_dir": streak_dir,
    }

    # SMA comparee a la derniere bougie fermee (pre-candle)
    if len(closes) >= 20:
        s20 = _sma(closes, 20)
        feat["prev_above_sma20"]    = bool(prev_close > s20)
        feat["prev_dist_sma20_pct"] = round((prev_close - s20) / s20 * 100, 4)
    if len(closes) >= 50:
        s50 = _sma(closes, 50)
        feat["prev_above_sma50"]    = bool(prev_close > s50)
        feat["prev_dist_sma50_pct"] = round((prev_close - s50) / s50 * 100, 4)

    feat.update(_bollinger(closes, 20))
    feat.update(_macd(closes))

    # Volatility regime: ATR court vs ATR long (>1 = volatilite en hausse)
    atr_short = _atr(window.iloc[-14:].copy().reset_index(drop=True)
                     if len(window) >= 14 else window, 7)
    atr_long  = _atr(window, 28)
    if atr_long and atr_long > 0:
        feat["vol_regime"] = round(atr_short / atr_long, 4)

    # Distance du prix precedent aux extremes des 20 dernieres bougies
    if len(highs) >= 20:
        high20 = highs[-20:].max()
        low20  = lows[-20:].min()
        feat["dist_high20_pct"] = round((prev_close - high20) / high20 * 100, 4)
        feat["dist_low20_pct"]  = round((prev_close - low20)  / low20  * 100, 4)

    # Momentum des 10 dernieres bougies
    if len(closes) >= 11:
        feat["momentum_10"] = round((prev_close - closes[-11]) / closes[-11] * 100, 4)

    # Volume moyen des bougies precedentes (contexte de liquidite)
    avg_vol = float(window["volume"].mean())
    feat["prev_avg_vol_ratio"] = round(
        float(window["volume"].iloc[-1]) / avg_vol, 4
    ) if avg_vol > 0 else None

    feat.update(_d1_context(df_1d, target_ts_ns))
    feat.update(_fear_greed_ctx(df_fg, target_ts_ns))

    # ══ POST-CANDLE — connus seulement apres fermeture ═════════════
    feat["direction"]        = "bull" if c > o else "bear"
    feat["candle_type"]      = _candle_type(o, h, l, c)
    feat["body_pct"]         = round(body / o * 100, 4)
    feat["body_abs_pct"]     = round(abs(body) / o * 100, 4)
    feat["range_pct"]        = round(total / o * 100, 4)
    feat["upper_wick_ratio"] = round(upper / total, 4) if total else 0
    feat["lower_wick_ratio"] = round(lower / total, 4) if total else 0
    if avg_vol > 0:
        feat["volume_ratio"] = round(vol / avg_vol, 4)

    return feat


# ------------------------------------------------------------------ stats

# PRE  = connu avant l'ouverture de la bougie (actionnables en live)
# POST = connu seulement apres la fermeture (informationnels)

PRE_CAT_KEYS  = ("day_of_week", "session",
                 "hour_range_8h", "hour_range_6h", "hour_range_4h", "hour_range_3h",
                 "day_of_week_et", "session_et",
                 "hour_range_et_8h", "hour_range_et_6h", "hour_range_et_4h", "hour_range_et_3h",
                 "macd_trend", "prev_streak_dir", "fg_regime")
POST_CAT_KEYS = ("direction", "candle_type")

PRE_NUM_KEYS  = ("hour_utc", "hour_et", "rsi_14", "stoch_k", "atr_14",
                 "prev_dist_sma20_pct", "prev_dist_sma50_pct",
                 "bb_width_pct", "bb_position",
                 "vol_regime", "dist_high20_pct", "dist_low20_pct",
                 "momentum_10", "prev_avg_vol_ratio", "prev_streak",
                 "d1_return_pct", "d1_rsi", "fg_value")
POST_NUM_KEYS = ("body_pct", "body_abs_pct", "range_pct",
                 "upper_wick_ratio", "lower_wick_ratio", "volume_ratio")

PRE_BOOL_KEYS  = ("prev_above_sma20", "prev_above_sma50", "d1_above_sma20")
POST_BOOL_KEYS = ()

CAT_KEYS  = PRE_CAT_KEYS  + POST_CAT_KEYS
NUM_KEYS  = PRE_NUM_KEYS  + POST_NUM_KEYS
BOOL_KEYS = PRE_BOOL_KEYS + POST_BOOL_KEYS


def build_stats(features: list) -> dict:
    """Compute all statistics from a features list into a structured dict."""
    def col(k): return [f[k] for f in features if k in f]

    stats: dict = {"n": len(features)}

    for key in CAT_KEYS:
        vals = col(key)
        if vals:
            total = len(vals)
            stats[key] = {k: v / total * 100 for k, v in Counter(vals).items()}

    for key in NUM_KEYS:
        raw = [v for v in col(key)
               if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if raw:
            arr = np.array(raw, dtype=float)
            q25, q75 = np.percentile(arr, [25, 75])
            stats[key] = {
                "n": len(arr), "mean": float(arr.mean()), "median": float(np.median(arr)),
                "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max()),
                "q25": float(q25), "q75": float(q75),
                "_arr": arr,   # kept for zone computations, not written to file
            }

    for key in BOOL_KEYS:
        vals = col(key)
        if vals:
            stats[key] = {"pct_true": sum(vals) / len(vals) * 100, "n": len(vals)}

    return stats


# ------------------------------------------------------------------ individual report

def _bar(n, total, w=24):
    f = int(n / total * w) if total else 0
    return "#" * f + "." * (w - f)


def format_report(features: list, title: str) -> list:
    """Build report lines (also prints to console). Returns list of strings."""
    stats = build_stats(features)
    n = stats["n"]
    lines: list = []

    def p(line: str = ""):
        print(line)
        lines.append(line)

    def cat(label, key):
        d = stats.get(key)
        if not d: return
        p(f"\n  {label}:")
        for k, pct in sorted(d.items(), key=lambda x: -x[1]):
            count = round(pct * n / 100)
            p(f"    {str(k):26s} {_bar(count, n)} {count:3d}  ({pct:.1f}%)")

    def num(label, key):
        d = stats.get(key)
        if not d: return
        p(f"\n  {label}:")
        p(f"    n={d['n']}   moy={d['mean']:.2f}   med={d['median']:.2f}   std={d['std']:.2f}")
        p(f"    min={d['min']:.2f}   Q25={d['q25']:.2f}   Q75={d['q75']:.2f}   max={d['max']:.2f}")

    def boolean(label, key):
        d = stats.get(key)
        if not d: return
        pct = d["pct_true"]
        p(f"\n  {label}: Au-dessus {pct:.1f}%  /  En-dessous {100-pct:.1f}%")

    p("\n" + "=" * 62)
    p(f"  {title}  --  {n} BOUGIES")
    p("=" * 62)

    # ── PRE-CANDLE ──────────────────────────────────────────────────
    p("\n" + "*" * 62)
    p("  [PRE-CANDLE] Features connues avant l'ouverture")
    p("*" * 62)

    p("\n" + "-" * 40)
    p("  TEMPS & SESSION")
    p("-" * 40)
    p("  -- UTC --")
    cat("Jour de la semaine", "day_of_week")
    cat("Session",            "session")
    cat("Tranche 8h",         "hour_range_8h")
    cat("Tranche 6h",         "hour_range_6h")
    cat("Tranche 4h",         "hour_range_4h")
    cat("Tranche 3h",         "hour_range_3h")
    num("Heure (detail)",     "hour_utc")
    p("  -- ET (America/New_York) --")
    cat("Jour (ET)",            "day_of_week_et")
    cat("Session (ET)",         "session_et")
    cat("Tranche 8h (ET)",      "hour_range_et_8h")
    cat("Tranche 6h (ET)",      "hour_range_et_6h")
    cat("Tranche 4h (ET)",      "hour_range_et_4h")
    cat("Tranche 3h (ET)",      "hour_range_et_3h")
    num("Heure ET (detail)",    "hour_et")

    p("\n" + "-" * 40)
    p("  INDICATEURS TECHNIQUES (bougies precedentes)")
    p("-" * 40)
    num("RSI (14)",               "rsi_14")
    rsi = stats.get("rsi_14")
    if rsi:
        arr = rsi["_arr"]
        p(f"    Zones -> <30:{int((arr<30).sum())}  30-50:{int(((arr>=30)&(arr<50)).sum())}"
          f"  50-70:{int(((arr>=50)&(arr<70)).sum())}  >70:{int((arr>70).sum())}")
    num("Stochastique K",         "stoch_k")
    num("ATR (14)",               "atr_14")
    num("Volatility regime",      "vol_regime")
    num("Distance SMA20 (%)",     "prev_dist_sma20_pct")
    num("Distance SMA50 (%)",     "prev_dist_sma50_pct")
    boolean("Au-dessus SMA20",    "prev_above_sma20")
    boolean("Au-dessus SMA50",    "prev_above_sma50")
    cat("Trend MACD",             "macd_trend")
    num("Bollinger largeur (%)",  "bb_width_pct")
    num("Bollinger position",     "bb_position")
    num("Distance plus haut 20",  "dist_high20_pct")
    num("Distance plus bas 20",   "dist_low20_pct")
    num("Momentum 10 bougies (%)", "momentum_10")
    num("Vol derniere bougie",    "prev_avg_vol_ratio")
    num("Streak precedent",       "prev_streak")
    cat("Direction streak",       "prev_streak_dir")

    if "d1_return_pct" in stats:
        p("\n" + "-" * 40)
        p("  CONTEXTE JOURNALIER (D1 J-1)")
        p("-" * 40)
        num("Return D1 (%)",          "d1_return_pct")
        num("RSI daily (14)",         "d1_rsi")
        boolean("Au-dessus SMA20 D1", "d1_above_sma20")

    if "fg_value" in stats:
        p("\n" + "-" * 40)
        p("  FEAR & GREED INDEX")
        p("-" * 40)
        num("Valeur (0-100)", "fg_value")
        cat("Regime",         "fg_regime")

    # ── POST-CANDLE ─────────────────────────────────────────────────
    p("\n" + "*" * 62)
    p("  [POST-CANDLE] Features connues apres fermeture uniquement")
    p("*" * 62)

    p("\n" + "-" * 40)
    p("  ANATOMIE DE LA BOUGIE")
    p("-" * 40)
    cat("Direction",          "direction")
    cat("Type",               "candle_type")
    num("Corps (%)",          "body_pct")
    num("Corps absolu (%)",   "body_abs_pct")
    num("Range total (%)",    "range_pct")
    num("Meche haute",        "upper_wick_ratio")
    num("Meche basse",        "lower_wick_ratio")
    num("Volume ratio",       "volume_ratio")

    p("\n" + "=" * 62 + "\n")
    return lines


# ------------------------------------------------------------------ comparison report

def _danger_label(ratio: float) -> str:
    if ratio >= 2.0:  return "!! EVITER ++"
    if ratio >= 1.5:  return "!! EVITER"
    if ratio >= 1.2:  return "!  risque"
    if ratio <= 0.4:  return "++ PRENDRE ++"
    if ratio <= 0.6:  return "++ PRENDRE"
    if ratio <= 0.8:  return "+  favorable"
    return ""


def format_comparison(stats_a: dict, stats_b: dict,
                      title_a: str, title_b: str) -> list:
    """Side-by-side comparison with danger ratio. [A]=stop_hits, [B]=stop_not_hit."""
    lines: list = []
    na, nb = stats_a["n"], stats_b["n"]
    W = 26

    def p(line: str = ""):
        print(line)
        lines.append(line)

    def row(label, va, vb, fmt=".1f", suffix="", show_ratio=False):
        la = f"{va:{fmt}}{suffix}" if va is not None else "  n/a"
        lb = f"{vb:{fmt}}{suffix}" if vb is not None else "  n/a"
        d  = vb - va if (va is not None and vb is not None) else None
        diff = (("+" if d >= 0 else "") + f"{d:{fmt}}{suffix}") if d is not None else ""
        ratio_str = ""
        if show_ratio and va is not None and vb is not None and vb > 0:
            r = va / vb
            ratio_str = f"  {r:.2f}x  {_danger_label(r)}"
        p(f"    {label:{W}s} {la:>9s}   {lb:>9s}   {diff:>9s}{ratio_str}")

    def section(title, pre=True):
        tag = " [PRE]" if pre else " [POST]"
        p("\n" + "-" * 72)
        p(f"  {title}{tag}")
        p("-" * 72)
        p(f"    {'':26s} {'[A]':>9s}   {'[B]':>9s}   {'diff':>9s}   ratio A/B")

    def compare_cat(label, key, pre=True):
        da = stats_a.get(key, {})
        db = stats_b.get(key, {})
        if not da and not db: return
        p(f"\n  {label}:")
        for k in sorted(set(da) | set(db)):
            va = da.get(k, 0.0); vb = db.get(k, 0.0)
            row(f"  {k}", va, vb, ".1f", "%", show_ratio=(vb > 0))

    def compare_num(label, key):
        da = stats_a.get(key); db = stats_b.get(key)
        if not da and not db: return
        p(f"\n  {label}:")
        for metric in ("mean", "median"):
            va = da[metric] if da else None
            vb = db[metric] if db else None
            row(f"  {metric}", va, vb, ".2f")

    def compare_bool(label, key):
        da = stats_a.get(key); db = stats_b.get(key)
        if not da and not db: return
        va = da["pct_true"] if da else None
        vb = db["pct_true"] if db else None
        p(f"\n  {label}:")
        row("  Au-dessus (%)", va, vb, ".1f", "%", show_ratio=(vb is not None and vb > 0))

    # ── Header ──────────────────────────────────────────────────────
    p("\n" + "=" * 72)
    p(f"  COMPARAISON")
    p(f"  [A] {title_a}  ({na} bougies)  <- stop hits (pertes)")
    p(f"  [B] {title_b}  ({nb} bougies)  <- stop non atteint")
    p(f"  ratio A/B > 1.5 = condition sur-representee dans les pertes -> EVITER")
    p(f"  ratio A/B < 0.7 = condition sous-representee dans les pertes -> PRENDRE")
    p("=" * 72)

    # ── PRE-CANDLE ──────────────────────────────────────────────────
    section("TEMPS & SESSION")
    p("  -- UTC --")
    compare_cat("Jour de la semaine", "day_of_week")
    compare_cat("Session",            "session")
    compare_cat("Tranche 8h",         "hour_range_8h")
    compare_cat("Tranche 6h",         "hour_range_6h")
    compare_cat("Tranche 4h",         "hour_range_4h")
    compare_cat("Tranche 3h",         "hour_range_3h")
    compare_num("Heure (detail)",     "hour_utc")
    p("  -- ET (America/New_York) --")
    compare_cat("Jour (ET)",            "day_of_week_et")
    compare_cat("Session (ET)",         "session_et")
    compare_cat("Tranche 8h (ET)",      "hour_range_et_8h")
    compare_cat("Tranche 6h (ET)",      "hour_range_et_6h")
    compare_cat("Tranche 4h (ET)",      "hour_range_et_4h")
    compare_cat("Tranche 3h (ET)",      "hour_range_et_3h")
    compare_num("Heure ET (detail)",    "hour_et")

    section("INDICATEURS TECHNIQUES (bougies precedentes)")
    compare_num("RSI (14)",            "rsi_14")
    ra = stats_a.get("rsi_14"); rb = stats_b.get("rsi_14")
    if ra or rb:
        p(f"\n  RSI zones:")
        for lbl, cond in (("<30",   lambda a: a < 30),
                          ("30-50", lambda a: (a >= 30) & (a < 50)),
                          ("50-70", lambda a: (a >= 50) & (a < 70)),
                          (">70",   lambda a: a > 70)):
            va = float(cond(ra["_arr"]).mean() * 100) if ra else None
            vb = float(cond(rb["_arr"]).mean() * 100) if rb else None
            row(f"  {lbl}", va, vb, ".1f", "%", show_ratio=(vb is not None and vb > 0))
    compare_num("Stochastique K",      "stoch_k")
    compare_num("ATR (14)",            "atr_14")
    compare_num("Volatility regime",   "vol_regime")
    compare_num("Dist SMA20 (%)",      "prev_dist_sma20_pct")
    compare_bool("Au-dessus SMA20",    "prev_above_sma20")
    compare_cat("Trend MACD",          "macd_trend")
    compare_num("Bollinger largeur",   "bb_width_pct")
    compare_num("Bollinger position",  "bb_position")
    compare_num("Dist plus haut 20",   "dist_high20_pct")
    compare_num("Dist plus bas 20",    "dist_low20_pct")
    compare_num("Momentum 10",         "momentum_10")
    compare_num("Vol derniere bougie", "prev_avg_vol_ratio")
    compare_num("Streak precedent",    "prev_streak")
    compare_cat("Direction streak",    "prev_streak_dir")

    if "d1_return_pct" in stats_a or "d1_return_pct" in stats_b:
        section("CONTEXTE JOURNALIER (D1 J-1)")
        compare_num("Return D1 (%)",      "d1_return_pct")
        compare_num("RSI daily (14)",     "d1_rsi")
        compare_bool("Au-dessus SMA20",   "d1_above_sma20")

    if "fg_value" in stats_a or "fg_value" in stats_b:
        section("FEAR & GREED INDEX")
        compare_num("Valeur (0-100)", "fg_value")
        compare_cat("Regime",         "fg_regime")

    # ── POST-CANDLE ─────────────────────────────────────────────────
    section("ANATOMIE DE LA BOUGIE", pre=False)
    compare_cat("Direction",          "direction",    pre=False)
    compare_cat("Type de bougie",     "candle_type",  pre=False)
    compare_num("Corps absolu (%)",   "body_abs_pct")
    compare_num("Range total (%)",    "range_pct")
    compare_num("Meche haute",        "upper_wick_ratio")
    compare_num("Meche basse",        "lower_wick_ratio")
    compare_num("Volume ratio",       "volume_ratio")

    # ── Recommandations auto ────────────────────────────────────────
    p("\n" + "=" * 72)
    p("  RECOMMANDATIONS (basees sur les features PRE-CANDLE uniquement)")
    p("=" * 72)

    candidates = []
    for key in PRE_CAT_KEYS:
        da = stats_a.get(key, {}); db = stats_b.get(key, {})
        for k in set(da) | set(db):
            va = da.get(k, 0.0); vb = db.get(k, 0.0)
            if vb > 2.0:   # ignore categories with < 2% presence
                candidates.append((f"{key}={k}", va / vb, va, vb))
    for key in PRE_BOOL_KEYS:
        da = stats_a.get(key); db = stats_b.get(key)
        if da and db and db["pct_true"] > 5:
            r = da["pct_true"] / db["pct_true"]
            candidates.append((f"{key}=True", r, da["pct_true"], db["pct_true"]))

    to_avoid = sorted([c for c in candidates if c[1] >= 1.2], key=lambda x: -x[1])[:8]
    to_take  = sorted([c for c in candidates if c[1] <= 0.80], key=lambda x: x[1])[:8]

    p("\n  CONDITIONS A EVITER (sur-representees dans [A]=pertes):")
    if to_avoid:
        for name, ratio, va, vb in to_avoid:
            p(f"    {name:<35s}  A={va:.1f}%  B={vb:.1f}%  ratio={ratio:.2f}x  {_danger_label(ratio)}")
    else:
        p("    Aucune condition clairement dangereuse identifiee.")

    p("\n  CONDITIONS FAVORABLES (sous-representees dans [A]=pertes):")
    if to_take:
        for name, ratio, va, vb in to_take:
            p(f"    {name:<35s}  A={va:.1f}%  B={vb:.1f}%  ratio={ratio:.2f}x  {_danger_label(ratio)}")
    else:
        p("    Aucune condition clairement favorable identifiee.")

    p("\n" + "=" * 72 + "\n")
    return lines


# ------------------------------------------------------------------ helpers

def _parse_monthly_folder(folder: Path):
    """Walk month subfolders (YYYY-MM), load MONTHLY_CSV from each, combine losses/wins."""
    losses, wins = [], []
    months_ok = []
    for month_dir in sorted(d for d in folder.iterdir() if d.is_dir()):
        csv_path = month_dir / MONTHLY_CSV
        if not csv_path.exists():
            print(f"  [SKIP] {month_dir.name}: {MONTHLY_CSV} introuvable")
            continue
        try:
            ml, mw = _parse_csv(csv_path, _verbose=False)
            print(f"  {month_dir.name}: {len(ml)} losses  {len(mw)} wins")
            losses.extend(ml)
            wins.extend(mw)
            months_ok.append(month_dir.name)
        except Exception as e:
            print(f"  [ERR] {month_dir.name}: {e}")
    print(f"\n  Total: {len(months_ok)} mois  |  {len(losses)} losses  {len(wins)} wins")
    return losses, wins, months_ok


def _parse_csv(path: Path, *, _verbose: bool = True):
    """Parse a trading journal CSV -> (losses, wins) as lists of (label, asset, direction, tf, ts_s)."""
    df = pd.read_csv(path)
    required = {"time", "result"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV doit avoir les colonnes: {required}. Colonnes: {list(df.columns)}")
    losses, wins = [], []
    for row_number in range(1, len(df) + 1):
        row = df.iloc[row_number - 1]
        ts = pd.Timestamp(str(row["time"]))
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        ts_s = int(ts.timestamp())
        trade_number = row_number
        if "trade_number" in df.columns and not pd.isna(row["trade_number"]):
            trade_number = int(float(str(row["trade_number"])))
        lbl = f"row-{trade_number}"
        entry = (lbl, "BTC", "updown", "5m", ts_s)
        if str(row["result"]).lower() == "loss":
            losses.append(entry)
        else:
            wins.append(entry)
    if _verbose:
        print(f"  CSV: {len(losses)} losses, {len(wins)} wins")
    return losses, wins


def _parse_file(path: Path) -> list:
    lines = [
        l.split("#")[0].strip()
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.split("#")[0].strip()
    ]
    markets = []
    for line in lines:
        try:
            markets.append((line, *parse_market_id(line)))
        except Exception as e:
            print(f"  [SKIP] Format invalide '{line}': {e}")
    return markets


def _extract_all(markets: list, df_5m, df_1d, df_fg, label: str) -> list:
    features_list = []
    skipped = []
    print(f"\nExtraction [{label}]...")
    for line, asset, direction, tf, ts_s in markets:
        ts_ns = ts_s * ONE_SEC_NS
        feat  = extract_features(df_5m, ts_ns, df_1d=df_1d, df_fg=df_fg)
        if feat is None:
            print(f"  [SKIP] {line}  ({ts_to_human(ts_s)})")
            skipped.append(line)
            continue
        feat["market_id"] = line; feat["asset"] = asset; feat["direction_market"] = direction
        print(f"  [OK]  {line:<42} {ts_to_human(ts_s)}")
        features_list.append(feat)
    if skipped:
        print(f"  {len(skipped)} ignore(s)")
    return features_list


def _save(lines: list, path: Path):
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {path}")


# ------------------------------------------------------------------ main

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    paths = [Path(a) for a in sys.argv[1:3]]
    for p in paths:
        if not p.exists():
            print(f"Fichier introuvable: {p}")
            sys.exit(1)

    # -- Monthly folder mode: directory of YYYY-MM subfolders
    if len(paths) == 1 and paths[0].is_dir():
        print(f"\nMode mensuel: {paths[0]}")
        losses, wins, months = _parse_monthly_folder(paths[0])
        if not losses or not wins:
            print("Dossier: aucune donnee losses/wins trouvee")
            sys.exit(1)
        all_ts = [ts_s for _, _, _, _, ts_s in losses + wins]
        print(f"\nFetch des donnees ({len(all_ts)} trades sur {len(months)} mois)...\n")
        df_5m = fetch_5m(all_ts)
        df_1d = fetch_1d(all_ts)
        df_fg = fetch_fear_greed()
        fa = _extract_all(losses, df_5m, df_1d, df_fg, "losses")
        fb = _extract_all(wins,   df_5m, df_1d, df_fg, "wins")
        stem = paths[0].name
        pd.DataFrame(fa).to_csv(f"features_{stem}_losses.csv", index=False)
        pd.DataFrame(fb).to_csv(f"features_{stem}_wins.csv",   index=False)
        _save(format_report(fa, f"LOSSES ({len(months)} mois)"), Path(f"report_{stem}_losses.txt"))
        _save(format_report(fb, f"WINS ({len(months)} mois)"),   Path(f"report_{stem}_wins.txt"))
        stats_a = build_stats(fa); stats_b = build_stats(fb)
        print("\n" + "=" * 68 + "\n  RAPPORT COMPARATIF\n" + "=" * 68)
        comp_lines = format_comparison(stats_a, stats_b,
                                       f"LOSSES ({len(months)} mois)", f"WINS ({len(months)} mois)")
        _save(comp_lines, Path(f"report_{stem}_comparison.txt"))
        return

    # -- CSV mode: single CSV with result column -> auto split losses vs wins
    if len(paths) == 1 and paths[0].suffix.lower() == ".csv":
        losses, wins = _parse_csv(paths[0])
        if not losses or not wins:
            print("CSV: impossible de separer losses/wins (verifier colonne 'result')")
            sys.exit(1)
        all_ts = [ts_s for _, _, _, _, ts_s in losses + wins]
        print(f"\nFetch des donnees ({len(all_ts)} trades)...\n")
        df_5m = fetch_5m(all_ts)
        df_1d = fetch_1d(all_ts)
        df_fg = fetch_fear_greed()
        fa = _extract_all(losses, df_5m, df_1d, df_fg, "losses")
        fb = _extract_all(wins,   df_5m, df_1d, df_fg, "wins")
        stem = paths[0].stem
        pd.DataFrame(fa).to_csv(f"features_{stem}_losses.csv", index=False)
        pd.DataFrame(fb).to_csv(f"features_{stem}_wins.csv",   index=False)
        _save(format_report(fa, "LOSSES"), Path(f"report_{stem}_losses.txt"))
        _save(format_report(fb, "WINS"),   Path(f"report_{stem}_wins.txt"))
        stats_a = build_stats(fa); stats_b = build_stats(fb)
        print("\n" + "=" * 68 + "\n  RAPPORT COMPARATIF\n" + "=" * 68)
        comp_lines = format_comparison(stats_a, stats_b, "LOSSES", "WINS")
        _save(comp_lines, Path(f"report_{stem}_comparison.txt"))
        return

    # -- TXT mode: one or two market-ID files
    all_markets = [_parse_file(p) for p in paths]
    for p, m in zip(paths, all_markets):
        if not m:
            print(f"Aucun marche valide dans {p}")
            sys.exit(1)

    # -- fetch: combine timestamps from all files for a single API call
    all_ts = [ts_s for markets in all_markets for _, _, _, _, ts_s in markets]
    print(f"\nFetch des donnees ({len(all_ts)} marches)...\n")
    df_5m = fetch_5m(all_ts)
    df_1d = fetch_1d(all_ts)
    df_fg = fetch_fear_greed()

    # -- extract per file
    all_features = []
    for p, markets in zip(paths, all_markets):
        label = p.stem
        feats = _extract_all(markets, df_5m, df_1d, df_fg, label)
        if not feats:
            print(f"Aucune bougie trouvee dans {p}")
            sys.exit(1)
        all_features.append((label, feats))

    # -- individual reports + CSV
    print()
    for label, feats in all_features:
        suffix = f"_{label}" if len(all_features) > 1 else ""
        pd.DataFrame(feats).to_csv(f"features{suffix}.csv", index=False)
        report_lines = format_report(feats, label.upper())
        _save(report_lines, Path(f"report{suffix}.txt"))

    # -- comparison report (only if 2 files)
    if len(all_features) == 2:
        (la, fa), (lb, fb) = all_features
        stats_a = build_stats(fa)
        stats_b = build_stats(fb)
        print("\n" + "=" * 68)
        print("  RAPPORT COMPARATIF")
        print("=" * 68)
        comp_lines = format_comparison(stats_a, stats_b, la.upper(), lb.upper())
        _save(comp_lines, Path("report_comparison.txt"))


if __name__ == "__main__":
    main()
