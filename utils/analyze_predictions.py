from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml


DEFAULT_CONFIG = {
    "general": {
        "starting_capital": 100.0,
        "stake": 1.0,
        "win_profit": 0.9,
        "loss_profit": -1.0,
        "reports_dir": "utils/raports",
    },
    "strategies": {
        "flat_fixed_stake": {"enabled": True, "base_stake": 1.0},
        "fixed_fraction": {"enabled": True, "fraction_pct": 5.0},
        "martingale_classic": {"enabled": True, "base_stake": 1.0},
        "martingale_linear": {"enabled": True, "base_stake": 1.0},
        "martingale_limited": {
            "enabled": True,
            "base_stake": 1.0,
            "sequence": [1, 1, 2, 4, 8, 16, 32],
        },
        "anti_martingale": {
            "enabled": True,
            "base_fraction_pct": 5.0,
            "win_multipliers": [
                {"min_win_streak": 2, "multiplier": 1.25},
                {"min_win_streak": 3, "multiplier": 1.5},
            ],
        },
        "reduction_after_losses": {
            "enabled": True,
            "base_fraction_pct": 5.0,
            "loss_steps": [
                {"min_loss_streak": 1, "fraction_pct": 2.5},
                {"min_loss_streak": 2, "fraction_pct": 1.0},
            ],
        },
        "pause_after_losses": {
            "enabled": True,
            "base_fraction_pct": 5.0,
            "pause_after_losses": 5,
            "pause_trades": 3,
        },
        "combined": {
            "enabled": True,
            "base_fraction_pct": 5.0,
            "win_multipliers": [
                {"min_win_streak": 2, "multiplier": 1.25},
                {"min_win_streak": 3, "multiplier": 1.5},
            ],
            "loss_multipliers": [
                {"min_loss_streak": 3, "multiplier": 0.5},
                {"min_loss_streak": 5, "multiplier": 0.2},
            ],
            "pause_after_losses": 7,
            "pause_trades": 3,
        },
    },
}


@dataclass
class StrategyResult:
    name: str
    trades_executed: int
    trades_skipped: int
    total_staked: float
    pnl: float
    ending_capital: float
    max_capital: float
    min_capital: float
    max_drawdown_pct: float


def deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse models/predictions.csv et affiche un rapport de performance."
    )
    parser.add_argument(
        "--csv",
        default="models/predictions.csv",
        help="Chemin vers le fichier predictions.csv",
    )
    parser.add_argument(
        "--config",
        default="utils/config_money_management.yaml",
        help="Chemin vers le fichier YAML de configuration money management",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        config_path.write_text(
            yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        return deepcopy(DEFAULT_CONFIG)

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Le fichier de configuration YAML doit contenir un objet.")
    return deep_merge(DEFAULT_CONFIG, loaded)


def load_results(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "result" not in (reader.fieldnames or []):
            raise ValueError("La colonne 'result' est introuvable dans le CSV.")

        return [
            row["result"].strip().upper()
            for row in reader
            if row.get("result", "").strip().upper() in {"WIN", "LOSS"}
        ]


def compute_streaks(results: list[str], target: str) -> tuple[int, int]:
    max_streak = 0
    current_streak = 0

    for result in results:
        if result == target:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    trailing_streak = 0
    for result in reversed(results):
        if result == target:
            trailing_streak += 1
        else:
            break

    return max_streak, trailing_streak


def format_money(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}${value:.2f}"


def round_result(result: StrategyResult) -> dict:
    return {
        "trades_executed": result.trades_executed,
        "trades_skipped": result.trades_skipped,
        "total_staked": round(result.total_staked, 2),
        "pnl": round(result.pnl, 2),
        "ending_capital": round(result.ending_capital, 2),
        "max_capital": round(result.max_capital, 2),
        "min_capital": round(result.min_capital, 2),
        "max_drawdown_pct": round(result.max_drawdown_pct, 2),
    }


def compute_drawdown(capital: float, peak_capital: float) -> float:
    if peak_capital <= 0:
        return 0.0
    return max(0.0, ((peak_capital - capital) / peak_capital) * 100)


def resolve_multiplier(steps: list[dict], streak: int, default: float = 1.0) -> float:
    multiplier = default
    for step in sorted(steps, key=lambda item: item["min_win_streak"] if "min_win_streak" in item else item["min_loss_streak"]):
        threshold = step.get("min_win_streak", step.get("min_loss_streak", 0))
        if streak >= threshold:
            multiplier = step["multiplier"]
    return multiplier


def resolve_fraction(steps: list[dict], loss_streak: int, default_fraction: float) -> float:
    fraction = default_fraction
    for step in sorted(steps, key=lambda item: item["min_loss_streak"]):
        if loss_streak >= step["min_loss_streak"]:
            fraction = step["fraction_pct"] / 100.0
    return fraction


def simulate_variable_stake(
    results: list[str],
    starting_capital: float,
    stake_getter,
    win_profit: float,
    loss_profit: float,
) -> StrategyResult:
    capital = starting_capital
    peak_capital = starting_capital
    min_capital = starting_capital
    total_staked = 0.0
    trades_executed = 0
    trades_skipped = 0
    max_drawdown_pct = 0.0
    win_streak = 0
    loss_streak = 0
    pause_remaining = 0

    for result in results:
        if pause_remaining > 0:
            pause_remaining -= 1
            trades_skipped += 1
            continue

        requested_stake, extra_pause = stake_getter(capital, win_streak, loss_streak)
        stake = max(0.0, min(requested_stake, capital))
        if stake <= 0:
            trades_skipped += 1
            continue

        pnl = stake * (win_profit if result == "WIN" else loss_profit)
        capital += pnl
        total_staked += stake
        trades_executed += 1

        peak_capital = max(peak_capital, capital)
        min_capital = min(min_capital, capital)
        max_drawdown_pct = max(max_drawdown_pct, compute_drawdown(capital, peak_capital))

        if result == "WIN":
            win_streak += 1
            loss_streak = 0
        else:
            loss_streak += 1
            win_streak = 0

        if extra_pause > 0:
            pause_remaining = extra_pause
            win_streak = 0
            loss_streak = 0

    return StrategyResult(
        name="",
        trades_executed=trades_executed,
        trades_skipped=trades_skipped,
        total_staked=total_staked,
        pnl=capital - starting_capital,
        ending_capital=capital,
        max_capital=peak_capital,
        min_capital=min_capital,
        max_drawdown_pct=max_drawdown_pct,
    )


def simulate_flat(results: list[str], starting_capital: float, cfg: dict, win_profit: float, loss_profit: float) -> StrategyResult:
    result = simulate_variable_stake(
        results,
        starting_capital,
        lambda capital, _w, _l: (cfg["base_stake"], 0),
        win_profit,
        loss_profit,
    )
    result.name = "flat_fixed_stake"
    return result


def simulate_fixed_fraction(results: list[str], starting_capital: float, cfg: dict, win_profit: float, loss_profit: float) -> StrategyResult:
    fraction = cfg["fraction_pct"] / 100.0
    result = simulate_variable_stake(
        results,
        starting_capital,
        lambda capital, _w, _l: (capital * fraction, 0),
        win_profit,
        loss_profit,
    )
    result.name = "fixed_fraction"
    return result


def simulate_martingale(results: list[str], starting_capital: float, cfg: dict, win_profit: float, loss_profit: float) -> StrategyResult:
    base_stake = cfg["base_stake"]
    result = simulate_variable_stake(
        results,
        starting_capital,
        lambda capital, _w, loss_streak: (base_stake * (2**loss_streak), 0),
        win_profit,
        loss_profit,
    )
    result.name = "martingale_classic"
    return result


def simulate_linear_martingale(results: list[str], starting_capital: float, cfg: dict, win_profit: float, loss_profit: float) -> StrategyResult:
    base_stake = cfg["base_stake"]
    result = simulate_variable_stake(
        results,
        starting_capital,
        lambda capital, _w, loss_streak: (base_stake * (loss_streak + 1), 0),
        win_profit,
        loss_profit,
    )
    result.name = "martingale_linear"
    return result


def simulate_limited_martingale(results: list[str], starting_capital: float, cfg: dict, win_profit: float, loss_profit: float) -> StrategyResult:
    sequence = cfg["sequence"]
    base_stake = cfg["base_stake"]

    def stake_getter(capital: float, _w: int, loss_streak: int) -> tuple[float, int]:
        index = min(loss_streak, len(sequence) - 1)
        return base_stake * sequence[index], 0

    result = simulate_variable_stake(
        results,
        starting_capital,
        stake_getter,
        win_profit,
        loss_profit,
    )
    result.name = "martingale_limited"
    return result


def simulate_anti_martingale(results: list[str], starting_capital: float, cfg: dict, win_profit: float, loss_profit: float) -> StrategyResult:
    base_fraction = cfg["base_fraction_pct"] / 100.0
    win_multipliers = cfg["win_multipliers"]

    def stake_getter(capital: float, win_streak: int, _l: int) -> tuple[float, int]:
        multiplier = resolve_multiplier(win_multipliers, win_streak, default=1.0)
        return capital * base_fraction * multiplier, 0

    result = simulate_variable_stake(
        results,
        starting_capital,
        stake_getter,
        win_profit,
        loss_profit,
    )
    result.name = "anti_martingale"
    return result


def simulate_reduction_after_losses(results: list[str], starting_capital: float, cfg: dict, win_profit: float, loss_profit: float) -> StrategyResult:
    base_fraction = cfg["base_fraction_pct"] / 100.0
    loss_steps = cfg["loss_steps"]

    def stake_getter(capital: float, _w: int, loss_streak: int) -> tuple[float, int]:
        fraction = resolve_fraction(loss_steps, loss_streak, base_fraction)
        return capital * fraction, 0

    result = simulate_variable_stake(
        results,
        starting_capital,
        stake_getter,
        win_profit,
        loss_profit,
    )
    result.name = "reduction_after_losses"
    return result


def simulate_pause_after_losses(results: list[str], starting_capital: float, cfg: dict, win_profit: float, loss_profit: float) -> StrategyResult:
    base_fraction = cfg["base_fraction_pct"] / 100.0
    pause_after_losses = cfg["pause_after_losses"]
    pause_trades = cfg["pause_trades"]

    def stake_getter(capital: float, _w: int, loss_streak: int) -> tuple[float, int]:
        extra_pause = pause_trades if loss_streak >= pause_after_losses else 0
        return capital * base_fraction, extra_pause

    result = simulate_variable_stake(
        results,
        starting_capital,
        stake_getter,
        win_profit,
        loss_profit,
    )
    result.name = "pause_after_losses"
    return result


def simulate_combined(results: list[str], starting_capital: float, cfg: dict, win_profit: float, loss_profit: float) -> StrategyResult:
    base_fraction = cfg["base_fraction_pct"] / 100.0
    win_multipliers = cfg["win_multipliers"]
    loss_multipliers = cfg["loss_multipliers"]
    pause_after_losses = cfg["pause_after_losses"]
    pause_trades = cfg["pause_trades"]

    def stake_getter(capital: float, win_streak: int, loss_streak: int) -> tuple[float, int]:
        win_multiplier = resolve_multiplier(win_multipliers, win_streak, default=1.0)
        loss_multiplier = resolve_multiplier(loss_multipliers, loss_streak, default=1.0)
        extra_pause = pause_trades if loss_streak >= pause_after_losses else 0
        return capital * base_fraction * win_multiplier * loss_multiplier, extra_pause

    result = simulate_variable_stake(
        results,
        starting_capital,
        stake_getter,
        win_profit,
        loss_profit,
    )
    result.name = "combined"
    return result


def build_money_management_report(results: list[str], starting_capital: float, config: dict) -> dict:
    general = config["general"]
    strategies_cfg = config["strategies"]
    win_profit = general["win_profit"]
    loss_profit = general["loss_profit"]

    simulators = [
        ("flat_fixed_stake", simulate_flat),
        ("fixed_fraction", simulate_fixed_fraction),
        ("martingale_classic", simulate_martingale),
        ("martingale_linear", simulate_linear_martingale),
        ("martingale_limited", simulate_limited_martingale),
        ("anti_martingale", simulate_anti_martingale),
        ("reduction_after_losses", simulate_reduction_after_losses),
        ("pause_after_losses", simulate_pause_after_losses),
        ("combined", simulate_combined),
    ]

    strategies: list[StrategyResult] = []
    enabled_config: dict[str, dict] = {}

    for strategy_name, simulator in simulators:
        strategy_cfg = strategies_cfg[strategy_name]
        enabled_config[strategy_name] = strategy_cfg
        if not strategy_cfg.get("enabled", True):
            continue
        strategies.append(
            simulator(results, starting_capital, strategy_cfg, win_profit, loss_profit)
        )

    strategies_by_name = {strategy.name: round_result(strategy) for strategy in strategies}
    sorted_by_pnl = sorted(strategies, key=lambda item: item.pnl, reverse=True)

    return {
        "strategies": strategies_by_name,
        "pnl_ranking": [
            {
                "rank": index + 1,
                "strategy": strategy.name,
                "pnl": round(strategy.pnl, 2),
                "ending_capital": round(strategy.ending_capital, 2),
                "max_drawdown_pct": round(strategy.max_drawdown_pct, 2),
            }
            for index, strategy in enumerate(sorted_by_pnl)
        ],
        "config_used": enabled_config,
    }


def build_report(csv_path: Path, config_path: Path, config: dict, results: list[str]) -> dict:
    general = config["general"]
    total_trades = len(results)
    winning_trades = sum(result == "WIN" for result in results)
    losing_trades = sum(result == "LOSS" for result in results)
    win_rate = (winning_trades / total_trades) * 100

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

    return {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source_csv": str(csv_path),
        "config_file": str(config_path),
        "metrics": {
            "trades_analyzed": total_trades,
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
        },
        "money_management": money_management,
    }


def save_report(report: dict, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"predictions_report_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report_path


def print_pnl_comparison(money_management: dict) -> None:
    print("")
    print("Comparatif PnL:")
    for item in money_management["pnl_ranking"]:
        print(
            f"- #{item['rank']} {item['strategy']}: "
            f"{format_money(item['pnl'])} | capital final ${item['ending_capital']:.2f} "
            f"| drawdown max {item['max_drawdown_pct']:.2f}%"
        )


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    config_path = Path(args.config)

    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {csv_path}")

    config = load_config(config_path)
    results = load_results(csv_path)
    if not results:
        raise ValueError("Aucun resultat WIN/LOSS exploitable trouve dans le CSV.")

    report = build_report(csv_path, config_path, config, results)
    metrics = report["metrics"]
    assumptions = report["assumptions"]
    reports_dir = Path(config["general"]["reports_dir"])
    report_path = save_report(report, reports_dir)

    print("=== Rapport predictions.csv ===")
    print(f"Fichier analyse      : {csv_path}")
    print(f"Config utilisee      : {config_path}")
    print(f"Trades analyses      : {metrics['trades_analyzed']}")
    print(f"Trades gagnants      : {metrics['winning_trades']}")
    print(f"Trades perdants      : {metrics['losing_trades']}")
    print(f"Win rate             : {metrics['win_rate_pct']:.2f}%")
    print(f"Serie max de WIN     : {metrics['max_win_streak']}")
    print(f"Serie max de LOSS    : {metrics['max_loss_streak']}")
    print(f"Serie actuelle WIN   : {metrics['current_win_streak']}")
    print(f"Serie actuelle LOSS  : {metrics['current_loss_streak']}")
    print(f"Capital initial      : ${metrics['starting_capital']:.2f}")
    print(f"PnL total flat       : {format_money(metrics['pnl'])}")
    print(f"Capital final flat   : ${metrics['ending_capital']:.2f}")
    print(f"Rapport JSON         : {report_path}")
    print("")
    print("Hypotheses:")
    print(f"- Mise fixe par trade: ${assumptions['stake_per_trade']:.2f}")
    print(f"- Un WIN rapporte    : {format_money(assumptions['win_profit'])}")
    print(f"- Un LOSS rapporte   : {format_money(assumptions['loss_profit'])}")
    print("- Les lignes PENDING sont ignorees")
    print_pnl_comparison(report["money_management"])


if __name__ == "__main__":
    main()
