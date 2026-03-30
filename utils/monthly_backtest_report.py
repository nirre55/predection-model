"""
Rapport mensuel detaille du schedule, mois par mois.

Chaque mois est backteste en solo puis exporte dans son propre dossier avec :
  - trades.csv
  - report.json (format proche de analyze_predictions.py)

Un rapport maitre est aussi genere pour resumer l'ensemble des mois analyses.

Sorties :
  - utils/raports/monthly_backtest/months/YYYY-MM/trades.csv
  - utils/raports/monthly_backtest/months/YYYY-MM/report.json
  - utils/raports/monthly_backtest/summary/monthly_backtest_master_report.json
  - utils/raports/monthly_backtest/summary/monthly_backtest_summary.csv
  - utils/raports/monthly_backtest/summary/monthly_backtest_by_slot.csv
  - utils/raports/monthly_backtest/summary/monthly_backtest_money_management.csv
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from datetime import UTC, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtest_schedule import INTERVAL, SYMBOL, ModelCache, get_slot, load_schedule  # noqa: E402
from src.data.cache import load_klines  # noqa: E402
from src.features import builder  # noqa: E402
from src.model.serializer import load_model  # noqa: E402
from utils.analyze_predictions import (  # noqa: E402
    build_money_management_report,
    compute_streaks,
    load_config,
    simulate_flat,
)

DEFAULT_MODEL_PATH = "models/model_calibrated.pkl"
DEFAULT_CONFIG_PATH = Path("utils/config_money_management.yaml")
OUTPUT_DIR = Path("utils/raports/monthly_backtest")
MONTHS_DIR = OUTPUT_DIR / "months"
SUMMARY_DIR = OUTPUT_DIR / "summary"
MAX_MONTHS_CAP = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genere un rapport detaille par mois")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.0,
        help="Seuil de confiance minimum en %% (defaut: 0 = tous les trades)",
    )
    parser.add_argument(
        "--position",
        type=float,
        default=1000.0,
        help="Taille de position en USD par trade (defaut: $1000)",
    )
    parser.add_argument(
        "--max-years",
        type=int,
        default=5,
        help="Nombre maximal d'annees retrospectives a analyser (defaut: 5)",
    )
    parser.add_argument(
        "--max-months",
        type=int,
        default=None,
        help="Limite explicite du nombre de mois (cappee a 60)",
    )
    parser.add_argument(
        "--include-current-month",
        action="store_true",
        help="Inclut le mois courant meme s'il est incomplet",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Chemin vers le YAML de money management",
    )
    return parser.parse_args()


def _month_floor(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1, tz="UTC")


def _next_month(ts: pd.Timestamp) -> pd.Timestamp:
    return ts + pd.offsets.MonthBegin(1)


def _safe_int_ns(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).astype("datetime64[ns]").astype("int64")


def _as_utc_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _clean_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    MONTHS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def build_analysis_months(
    df: pd.DataFrame,
    max_years: int,
    max_months: int | None,
    include_current_month: bool,
) -> list[pd.Timestamp]:
    data_start = _as_utc_timestamp(df["open_time"].min())
    data_end = _as_utc_timestamp(df["open_time"].max())

    last_month = _month_floor(data_end)
    if not include_current_month:
        last_month = last_month - pd.offsets.MonthBegin(1)

    if last_month < _month_floor(data_start):
        return []

    requested_months = max_years * 12
    if max_months is not None:
        requested_months = min(requested_months, max_months)
    requested_months = max(1, min(requested_months, MAX_MONTHS_CAP))

    earliest_allowed = _month_floor(data_start)
    first_candidate = last_month - pd.offsets.MonthBegin(requested_months - 1)
    first_month = max(first_candidate, earliest_allowed)

    months = []
    current = first_month
    while current <= last_month:
        months.append(current)
        current = _next_month(current)
    return months


def normalize_schedule(schedule: list[dict]) -> list[dict]:
    if schedule:
        return schedule
    return [{"slot": "default", "default": True, "model_path": DEFAULT_MODEL_PATH}]


def collect_model_specs(schedule: list[dict]) -> tuple[list[dict], dict[str, dict], bool]:
    specs_by_path: dict[str, dict] = {}
    needs_multitf = False

    for slot in schedule:
        model_path = slot["model_path"].replace("\\", "/")
        if model_path in specs_by_path:
            continue
        _, meta = load_model(model_path)
        indicators = meta.get("indicators", ["rsi", "macd", "atr"])
        spec = {
            "model_path": model_path,
            "window": int(meta.get("window", 50)),
            "indicators": list(indicators),
            "include_time": bool(meta.get("include_time", False)),
            "multitf_enabled": bool(meta.get("multitf_enabled", False)),
        }
        specs_by_path[model_path] = spec
        needs_multitf = needs_multitf or spec["multitf_enabled"]

    return schedule, specs_by_path, needs_multitf


def build_multitf_asof(df_5m: pd.DataFrame, dt: pd.Timestamp, rule: str) -> pd.DataFrame:
    history = df_5m[df_5m["open_time"] < dt].copy()
    if history.empty:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

    history = history.set_index("open_time")
    grouped = history.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    grouped = grouped.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return grouped[["open_time", "open", "high", "low", "close", "volume"]]


def aggregate_ohlcv(df_5m: pd.DataFrame, rule: str) -> pd.DataFrame:
    base = df_5m.sort_values("open_time").set_index("open_time")
    grouped = base.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    grouped = grouped.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return grouped[["open_time", "open", "high", "low", "close", "volume"]]


def simulate_all_trades(
    df_5m: pd.DataFrame,
    df_1h_full: pd.DataFrame | None,
    df_4h_full: pd.DataFrame | None,
    schedule: list[dict],
    specs_by_path: dict[str, dict],
    model_cache: ModelCache,
    month_starts: list[pd.Timestamp],
    confidence_threshold: float,
    stake: float,
    win_profit: float,
    loss_profit: float,
) -> pd.DataFrame:
    month_lookup = {
        month.strftime("%Y-%m"): (month, _next_month(month))
        for month in month_starts
    }
    selected_months = set(month_lookup.keys())

    df_5m = df_5m.sort_values("open_time").reset_index(drop=True)
    open_times_ns = np.asarray(df_5m["open_time"].values, dtype="datetime64[ns]").astype("int64")
    times_1h_ns = None
    times_4h_ns = None
    if df_1h_full is not None:
        times_1h_ns = np.asarray(df_1h_full["open_time"].values, dtype="datetime64[ns]").astype("int64")
    if df_4h_full is not None:
        times_4h_ns = np.asarray(df_4h_full["open_time"].values, dtype="datetime64[ns]").astype("int64")

    rows: list[dict] = []
    skipped = 0

    for ts in pd.to_datetime(df_5m["open_time"], utc=True):
        month_key = ts.strftime("%Y-%m")
        if month_key not in selected_months:
            continue

        candle_idx = int(np.searchsorted(open_times_ns, ts.value, side="left"))
        if candle_idx >= len(df_5m):
            skipped += 1
            continue
        candle_row = df_5m.iloc[candle_idx]
        if _as_utc_timestamp(candle_row["open_time"]) != ts:
            skipped += 1
            continue

        slot = get_slot(ts, schedule)
        if slot is None:
            skipped += 1
            continue

        model_path = slot["model_path"].replace("\\", "/")
        spec = specs_by_path[model_path]
        window = spec["window"]
        if candle_idx < window + 1:
            skipped += 1
            continue

        model, meta = model_cache.get(model_path)
        hist = df_5m.iloc[candle_idx - (window + 1) : candle_idx]
        candles = [
            {
                "open_time": row["open_time"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            for _, row in hist.iterrows()
        ]
        predict_time = ts.to_pydatetime() if spec["include_time"] else None
        df_1h_asof = None
        df_4h_asof = None
        if spec["multitf_enabled"]:
            cutoff_1h = int(np.searchsorted(times_1h_ns, ts.value, side="left")) if times_1h_ns is not None else 0
            cutoff_4h = int(np.searchsorted(times_4h_ns, ts.value, side="left")) if times_4h_ns is not None else 0
            df_1h_asof = df_1h_full.iloc[:cutoff_1h] if df_1h_full is not None else None
            df_4h_asof = df_4h_full.iloc[:cutoff_4h] if df_4h_full is not None else None

        X_sample = builder.build_inference_features(
            collections.deque(candles, maxlen=window + 1),
            window=window,
            indicators=spec["indicators"],
            predict_time=predict_time,
            df_1h=df_1h_asof,
            df_4h=df_4h_asof,
        )

        try:
            proba = model.predict_proba(X_sample)[0]
        except Exception:
            skipped += 1
            continue

        prob_green = float(proba[1])
        prob_red = float(proba[0])
        predicted = "VERT" if prob_green >= 0.5 else "ROUGE"
        prob_pred = prob_green if predicted == "VERT" else prob_red
        confidence = abs(prob_pred - 0.5) * 200

        if confidence < confidence_threshold:
            skipped += 1
            continue

        actual_open = float(candle_row["open"])
        actual_close = float(candle_row["close"])
        actual_dir = "VERT" if actual_close > actual_open else "ROUGE"
        result = "WIN" if actual_dir == predicted else "LOSS"
        price_move_pct = abs(actual_close - actual_open) / actual_open

        pnl_usd = stake * win_profit if result == "WIN" else stake * loss_profit

        month_start, month_end = month_lookup[month_key]
        rows.append(
            {
                "month": month_key,
                "month_start": month_start.strftime("%Y-%m-%d"),
                "month_end_exclusive": month_end.strftime("%Y-%m-%d"),
                "candle_open": ts.strftime("%Y-%m-%d %H:%M"),
                "slot": slot["slot"],
                "model_type": meta.get("model_type", slot.get("model_type", "default")),
                "window": window,
                "predicted": predicted,
                "probability": round(prob_pred, 4),
                "confidence_pct": round(confidence, 2),
                "actual": actual_dir,
                "result": result,
                "price_move_pct": round(float(price_move_pct) * 100, 4),
                "pnl_usd": round(float(pnl_usd), 4),
            }
        )

    if skipped:
        print(f"  ({skipped} samples ignores : seuil conf, index ou prediction impossible)")

    return pd.DataFrame(rows)


def compute_trade_metrics(trades: pd.DataFrame, stake: float) -> dict:
    if trades.empty:
        return {}

    wins = trades[trades["result"] == "WIN"]
    losses = trades[trades["result"] == "LOSS"]

    total_pnl = float(trades["pnl_usd"].sum())
    gross_profit = float(wins["pnl_usd"].sum())
    gross_loss = abs(float(losses["pnl_usd"].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    equity = trades["pnl_usd"].cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

    returns = trades["pnl_usd"] / max(stake, 1e-9)
    mean_r = float(returns.mean())
    std_r = float(returns.std())
    sharpe = (mean_r / std_r * np.sqrt(len(trades))) if std_r > 0 else 0.0

    return {
        "n_trades": int(len(trades)),
        "n_wins": int(len(wins)),
        "n_losses": int(len(losses)),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2),
        "total_pnl_usd": round(total_pnl, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown_usd": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 4),
        "avg_pnl_usd": round(float(trades["pnl_usd"].mean()), 4),
        "best_trade_usd": round(float(trades["pnl_usd"].max()), 4),
        "worst_trade_usd": round(float(trades["pnl_usd"].min()), 4),
    }


def compute_trade_slot_metrics(trades: pd.DataFrame) -> list[dict]:
    rows = []
    for slot_name, group in trades.groupby("slot"):
        wins = int((group["result"] == "WIN").sum())
        gross_profit = float(group[group["pnl_usd"] > 0]["pnl_usd"].sum())
        gross_loss = abs(float(group[group["pnl_usd"] < 0]["pnl_usd"].sum()))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        rows.append(
            {
                "slot": slot_name,
                "n_trades": int(len(group)),
                "win_rate_pct": round(wins / len(group) * 100, 2),
                "total_pnl_usd": round(float(group["pnl_usd"].sum()), 2),
                "profit_factor": round(profit_factor, 4),
                "avg_confidence": round(float(group["confidence_pct"].mean()), 2),
            }
        )
    return sorted(rows, key=lambda r: r["total_pnl_usd"], reverse=True)


def build_month_report(
    month_key: str,
    month_trades: pd.DataFrame,
    month_dir: Path,
    config: dict,
    config_path: Path,
    stake: float,
    win_profit: float,
    loss_profit: float,
    confidence_threshold: float,
) -> dict:
    results = month_trades["result"].tolist()
    general = config["general"]

    if results:
        winning_trades = sum(result == "WIN" for result in results)
        losing_trades = sum(result == "LOSS" for result in results)
        win_rate = (winning_trades / len(results)) * 100
        max_win_streak, current_win_streak = compute_streaks(results, "WIN")
        max_loss_streak, current_loss_streak = compute_streaks(results, "LOSS")
        flat = simulate_flat(
            results,
            general["starting_capital"],
            config["strategies"]["flat_fixed_stake"],
            general["win_profit"],
            general["loss_profit"],
        )
        money_management = build_money_management_report(
            results,
            general["starting_capital"],
            config,
        )
    else:
        winning_trades = 0
        losing_trades = 0
        win_rate = 0.0
        max_win_streak = 0
        current_win_streak = 0
        max_loss_streak = 0
        current_loss_streak = 0
        flat = simulate_flat(
            [],
            general["starting_capital"],
            config["strategies"]["flat_fixed_stake"],
            general["win_profit"],
            general["loss_profit"],
        )
        money_management = build_money_management_report(
            [],
            general["starting_capital"],
            config,
        )

    backtest_metrics = compute_trade_metrics(month_trades, stake) if not month_trades.empty else {}
    slot_metrics = compute_trade_slot_metrics(month_trades) if not month_trades.empty else []

    return {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "month": month_key,
        "source_csv": str((month_dir / "trades.csv").as_posix()),
        "config_file": str(config_path.as_posix()),
        "metrics": {
            "trades_analyzed": len(results),
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate_pct": round(win_rate, 2),
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "current_win_streak": current_win_streak,
            "current_loss_streak": current_loss_streak,
            "pnl": round(flat.pnl, 2),
            "starting_capital": round(general["starting_capital"], 2),
            "ending_capital": round(flat.ending_capital, 2),
        },
        "assumptions": {
            "stake_per_trade": round(config["strategies"]["flat_fixed_stake"]["base_stake"], 2),
            "win_profit": round(general["win_profit"], 2),
            "loss_profit": round(general["loss_profit"], 2),
            "pending_rows_ignored": True,
            "confidence_threshold_pct": confidence_threshold,
            "stake": stake,
            "win_profit_per_stake": win_profit,
            "loss_profit_per_stake": loss_profit,
        },
        "money_management": money_management,
        "backtest": {
            "global": backtest_metrics,
            "by_slot": slot_metrics,
        },
    }


def summarize_money_management(month_reports_full: list[dict]) -> dict:
    strategy_rows: dict[str, list[dict]] = {}

    for report in month_reports_full:
        month = report["month"]
        strategies = report["money_management"]["strategies"]
        for strategy_name, metrics in strategies.items():
            strategy_rows.setdefault(strategy_name, []).append(
                {
                    "month": month,
                    "pnl": metrics["pnl"],
                    "ending_capital": metrics["ending_capital"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                    "trades_executed": metrics["trades_executed"],
                    "trades_skipped": metrics["trades_skipped"],
                }
            )

    if not strategy_rows:
        return {
            "by_strategy": {},
            "ranking_total_pnl": [],
            "most_consistent_strategy": None,
            "lowest_avg_drawdown_strategy": None,
        }

    aggregate_rows = []
    by_strategy = {}

    for strategy_name, rows in strategy_rows.items():
        df = pd.DataFrame(rows)
        positive_months = int((df["pnl"] > 0).sum())
        negative_months = int((df["pnl"] < 0).sum())
        flat_months = int((df["pnl"] == 0).sum())
        total_pnl = round(float(df["pnl"].sum()), 2)
        avg_pnl = round(float(df["pnl"].mean()), 4)
        median_pnl = round(float(df["pnl"].median()), 4)
        pnl_std = round(float(df["pnl"].std(ddof=0)), 4)
        avg_drawdown = round(float(df["max_drawdown_pct"].mean()), 4)
        max_drawdown = round(float(df["max_drawdown_pct"].max()), 4)
        avg_ending_capital = round(float(df["ending_capital"].mean()), 4)
        best_month_idx = df["pnl"].idxmax()
        worst_month_idx = df["pnl"].idxmin()
        best_row = df.iloc[int(best_month_idx)]
        worst_row = df.iloc[int(worst_month_idx)]

        # Consistency score: favor positive hit-rate, then lower volatility.
        consistency_score = round(
            float((positive_months / len(df)) * 100 - pnl_std / max(abs(avg_pnl), 1.0)),
            4,
        )

        entry = {
            "strategy": strategy_name,
            "months_count": int(len(df)),
            "positive_months": positive_months,
            "negative_months": negative_months,
            "flat_months": flat_months,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "median_pnl": median_pnl,
            "pnl_std": pnl_std,
            "avg_drawdown_pct": avg_drawdown,
            "max_drawdown_pct": max_drawdown,
            "avg_ending_capital": avg_ending_capital,
            "consistency_score": consistency_score,
            "best_month": {
                "month": str(best_row["month"]),
                "pnl": round(float(best_row["pnl"]), 2),
                "ending_capital": round(float(best_row["ending_capital"]), 2),
                "max_drawdown_pct": round(float(best_row["max_drawdown_pct"]), 2),
            },
            "worst_month": {
                "month": str(worst_row["month"]),
                "pnl": round(float(worst_row["pnl"]), 2),
                "ending_capital": round(float(worst_row["ending_capital"]), 2),
                "max_drawdown_pct": round(float(worst_row["max_drawdown_pct"]), 2),
            },
        }
        aggregate_rows.append(entry)
        by_strategy[strategy_name] = entry

    aggregate_df = pd.DataFrame(aggregate_rows)
    ranking_total_pnl = aggregate_df.sort_values(
        ["total_pnl", "positive_months", "avg_drawdown_pct"],
        ascending=[False, False, True],
    ).to_dict(orient="records")

    consistent_idx = aggregate_df.sort_values(
        ["consistency_score", "positive_months", "avg_drawdown_pct"],
        ascending=[False, False, True],
    ).index[0]
    low_drawdown_idx = aggregate_df.sort_values(
        ["avg_drawdown_pct", "max_drawdown_pct", "pnl_std"],
        ascending=[True, True, True],
    ).index[0]

    return {
        "by_strategy": by_strategy,
        "ranking_total_pnl": ranking_total_pnl,
        "most_consistent_strategy": aggregate_df.loc[consistent_idx].to_dict(),
        "lowest_avg_drawdown_strategy": aggregate_df.loc[low_drawdown_idx].to_dict(),
    }


def build_money_management_rows(month_reports_full: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for report in month_reports_full:
        month = report["month"]
        strategies = report["money_management"]["strategies"]
        ranking = {
            item["strategy"]: item["rank"]
            for item in report["money_management"].get("pnl_ranking", [])
        }
        for strategy_name, metrics in strategies.items():
            rows.append(
                {
                    "month": month,
                    "strategy": strategy_name,
                    "rank_in_month": ranking.get(strategy_name),
                    "trades_executed": metrics["trades_executed"],
                    "trades_skipped": metrics["trades_skipped"],
                    "total_staked": metrics["total_staked"],
                    "pnl": metrics["pnl"],
                    "ending_capital": metrics["ending_capital"],
                    "max_capital": metrics["max_capital"],
                    "min_capital": metrics["min_capital"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                }
            )
    return rows


def summarize_master_reports(
    month_reports: list[dict],
    month_reports_full: list[dict],
    summary_rows: list[dict],
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    args: argparse.Namespace,
    config_path: Path,
) -> dict:
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        best_flat_idx = summary_df["flat_pnl"].idxmax()
        worst_flat_idx = summary_df["flat_pnl"].idxmin()
        best_backtest_idx = summary_df["backtest_total_pnl_usd"].idxmax()
        worst_backtest_idx = summary_df["backtest_total_pnl_usd"].idxmin()
        max_win_rate_idx = summary_df["win_rate_pct"].idxmax()
        max_win_streak_idx = summary_df["max_win_streak"].idxmax()
        max_loss_streak_idx = summary_df["max_loss_streak"].idxmax()
        best_trade_idx = summary_df["best_trade_usd"].idxmax()
        worst_trade_idx = summary_df["worst_trade_usd"].idxmin()
    else:
        summary_df = pd.DataFrame()
        best_flat_idx = worst_flat_idx = best_backtest_idx = worst_backtest_idx = None
        max_win_rate_idx = max_win_streak_idx = max_loss_streak_idx = None
        best_trade_idx = worst_trade_idx = None

    def _row_or_none(idx):
        if idx is None or summary_df.empty:
            return None
        return summary_df.loc[idx].to_dict()

    money_management_summary = summarize_money_management(month_reports_full)

    return {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "config_file": str(config_path.as_posix()),
        "analysis": {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "confidence_threshold_pct": args.confidence,
            "position_usd": args.position,
            "max_years_requested": args.max_years,
            "max_months_requested": min(
                MAX_MONTHS_CAP,
                args.max_months if args.max_months is not None else args.max_years * 12,
            ),
            "include_current_month": args.include_current_month,
            "data_start": str(data_start),
            "data_end": str(data_end),
            "months_generated": len(month_reports),
        },
        "summary": {
            "profitable_months_flat": [row["month"] for row in summary_rows if row["flat_pnl"] > 0],
            "negative_months_flat": [row["month"] for row in summary_rows if row["flat_pnl"] < 0],
            "profitable_months_backtest": [
                row["month"] for row in summary_rows if row["backtest_total_pnl_usd"] > 0
            ],
            "negative_months_backtest": [
                row["month"] for row in summary_rows if row["backtest_total_pnl_usd"] < 0
            ],
            "best_month_flat": _row_or_none(best_flat_idx),
            "worst_month_flat": _row_or_none(worst_flat_idx),
            "best_month_backtest": _row_or_none(best_backtest_idx),
            "worst_month_backtest": _row_or_none(worst_backtest_idx),
            "max_win_rate_month": _row_or_none(max_win_rate_idx),
            "max_win_streak_month": _row_or_none(max_win_streak_idx),
            "max_loss_streak_month": _row_or_none(max_loss_streak_idx),
            "max_trade_gain_month": _row_or_none(best_trade_idx),
            "max_trade_loss_month": _row_or_none(worst_trade_idx),
            "avg_flat_pnl": round(float(summary_df["flat_pnl"].mean()), 4) if not summary_df.empty else 0.0,
            "avg_backtest_pnl_usd": round(float(summary_df["backtest_total_pnl_usd"].mean()), 4)
            if not summary_df.empty
            else 0.0,
        },
        "money_management_summary": money_management_summary,
        "months": month_reports,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    general = config["general"]
    stake = float(general["stake"])
    win_profit = float(general["win_profit"])
    loss_profit = float(general["loss_profit"])

    print(f"Chargement des donnees {SYMBOL} {INTERVAL}...")
    df_5m = load_klines(SYMBOL, INTERVAL).sort_values("open_time").reset_index(drop=True)
    data_start = _as_utc_timestamp(df_5m["open_time"].min())
    data_end = _as_utc_timestamp(df_5m["open_time"].max())
    print(f"{len(df_5m)} bougies chargees ({data_start.date()} -> {data_end.date()}).")

    months = build_analysis_months(
        df_5m,
        max_years=args.max_years,
        max_months=args.max_months,
        include_current_month=args.include_current_month,
    )
    if not months:
        raise ValueError("Aucun mois complet disponible pour l'analyse.")

    print(
        f"Periode analysee : {months[0].strftime('%Y-%m')} -> {months[-1].strftime('%Y-%m')} "
        f"({len(months)} mois)"
    )

    schedule = normalize_schedule(load_schedule() if Path("models/schedule.json").exists() else [])
    schedule, specs_by_path, needs_multitf = collect_model_specs(schedule)
    model_cache = ModelCache()
    df_1h_full = None
    df_4h_full = None

    if needs_multitf:
        print("Pre-aggregation du contexte 1h/4h depuis les bougies 5m...")
        df_1h_full = aggregate_ohlcv(df_5m, "1h")
        df_4h_full = aggregate_ohlcv(df_5m, "4h")

    print("\nSimulation mensuelle en cours...")
    trades = simulate_all_trades(
        df_5m=df_5m,
        df_1h_full=df_1h_full,
        df_4h_full=df_4h_full,
        schedule=schedule,
        specs_by_path=specs_by_path,
        model_cache=model_cache,
        month_starts=months,
        confidence_threshold=args.confidence,
        stake=stake,
        win_profit=win_profit,
        loss_profit=loss_profit,
    )

    _clean_output_dir(OUTPUT_DIR)

    month_reports: list[dict] = []
    month_reports_full: list[dict] = []
    summary_rows: list[dict] = []
    slot_rows: list[dict] = []

    for month_start in months:
        month_key = month_start.strftime("%Y-%m")
        month_dir = MONTHS_DIR / month_key
        month_dir.mkdir(parents=True, exist_ok=True)
        month_trades = trades[trades["month"] == month_key].copy()
        month_trades.to_csv(month_dir / "trades.csv", index=False)

        report = build_month_report(
            month_key=month_key,
            month_trades=month_trades,
            month_dir=month_dir,
            config=config,
            config_path=config_path,
            stake=stake,
            win_profit=win_profit,
            loss_profit=loss_profit,
            confidence_threshold=args.confidence,
        )
        (month_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        backtest_global = report["backtest"]["global"]
        summary_rows.append(
            {
                "month": month_key,
                "trades_analyzed": report["metrics"]["trades_analyzed"],
                "win_rate_pct": report["metrics"]["win_rate_pct"],
                "max_win_streak": report["metrics"]["max_win_streak"],
                "max_loss_streak": report["metrics"]["max_loss_streak"],
                "flat_pnl": report["metrics"]["pnl"],
                "flat_ending_capital": report["metrics"]["ending_capital"],
                "best_strategy": (
                    report["money_management"]["pnl_ranking"][0]["strategy"]
                    if report["money_management"]["pnl_ranking"]
                    else None
                ),
                "best_strategy_pnl": (
                    report["money_management"]["pnl_ranking"][0]["pnl"]
                    if report["money_management"]["pnl_ranking"]
                    else 0.0
                ),
                "backtest_total_pnl_usd": backtest_global.get("total_pnl_usd", 0.0),
                "profit_factor": backtest_global.get("profit_factor", 0.0),
                "best_trade_usd": backtest_global.get("best_trade_usd", 0.0),
                "worst_trade_usd": backtest_global.get("worst_trade_usd", 0.0),
                "report_path": str((month_dir / "report.json").as_posix()),
                "trades_path": str((month_dir / "trades.csv").as_posix()),
            }
        )

        for slot_metric in report["backtest"]["by_slot"]:
            slot_rows.append({"month": month_key, **slot_metric})

        month_reports.append(
            {
                "month": month_key,
                "report_path": str((month_dir / "report.json").as_posix()),
                "trades_path": str((month_dir / "trades.csv").as_posix()),
                "metrics": report["metrics"],
                "backtest_global": backtest_global,
            }
        )
        month_reports_full.append(report)

    summary_df = pd.DataFrame(summary_rows)
    slot_df = pd.DataFrame(slot_rows)
    mm_df = pd.DataFrame(build_money_management_rows(month_reports_full))
    summary_csv_path = SUMMARY_DIR / "monthly_backtest_summary.csv"
    slot_csv_path = SUMMARY_DIR / "monthly_backtest_by_slot.csv"
    mm_csv_path = SUMMARY_DIR / "monthly_backtest_money_management.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    slot_df.to_csv(slot_csv_path, index=False)
    mm_df.to_csv(mm_csv_path, index=False)

    master_report = summarize_master_reports(
        month_reports=month_reports,
        month_reports_full=month_reports_full,
        summary_rows=summary_rows,
        data_start=data_start,
        data_end=data_end,
        args=args,
        config_path=config_path,
    )
    master_report_path = SUMMARY_DIR / "monthly_backtest_master_report.json"
    master_report_path.write_text(
        json.dumps(master_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("RAPPORTS MENSUELS GENERES")
    print("=" * 72)
    print(f"Mois generes     : {len(months)}")
    print(f"Dossier racine   : {OUTPUT_DIR}")
    print(f"Dossiers mois    : {MONTHS_DIR}")
    print(f"Dossier resume   : {SUMMARY_DIR}")
    if summary_rows:
        best_flat = master_report["summary"]["best_month_flat"]
        worst_flat = master_report["summary"]["worst_month_flat"]
        mm = master_report["money_management_summary"]
        print(f"Meilleur mois    : {best_flat['month']} (flat pnl={best_flat['flat_pnl']:+.2f})")
        print(f"Pire mois        : {worst_flat['month']} (flat pnl={worst_flat['flat_pnl']:+.2f})")
        print(
            f"Mois positifs    : {len(master_report['summary']['profitable_months_flat'])} "
            f"| Mois negatifs : {len(master_report['summary']['negative_months_flat'])}"
        )
        if mm["ranking_total_pnl"]:
            print(f"Top MM global    : {mm['ranking_total_pnl'][0]['strategy']}")
            print(f"MM plus stable   : {mm['most_consistent_strategy']['strategy']}")
            print(f"MM drawdown min  : {mm['lowest_avg_drawdown_strategy']['strategy']}")

    print(f"\n[Rapport] {summary_csv_path}")
    print(f"[Rapport] {slot_csv_path}")
    print(f"[Rapport] {mm_csv_path}")
    print(f"[Rapport] {master_report_path}")


if __name__ == "__main__":
    main()
