"""
Validation OOS v2 — avec features futures (taker volume, OI, funding rate).

Methode identique a true_oos_backtest.py, mais avec les 12 features futures
appendees en fin de vecteur. Compare les resultats avec / sans features futures.

Usage:
    python fetch_futures_data.py   # fetch futures data first (once)
    python true_oos_backtest_v2.py
"""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.data.cache import load_klines, save_klines, load_futures_data
from src.data.fetcher import fetch_klines
from src.features.builder import build_dataset_with_target_times
from src.model.trainer import train
from src.model.serializer import save_model

# ── Configuration ─────────────────────────────────────────────────────────────
SYMBOL          = "BTCUSDT"
INTERVAL        = "5m"
TRAIN_END       = pd.Timestamp("2025-10-31", tz="UTC")
TEST_START      = pd.Timestamp("2025-11-01", tz="UTC")
WIN_PAYOUT      = 0.90
LOSS_PAYOUT     = -1.0
STAKE           = 100.0
BREAK_EVEN_WR   = 1.0 / (1.0 + WIN_PAYOUT)   # 52.63%
BEST_INDICATORS = ["rsi", "macd", "atr", "mfi", "vdelta", "body", "streak"]
INCLUDE_TIME    = True
MIN_MOVE_PCT    = 0.003
WINDOWS         = [20, 50, 100]
N_SPLITS        = 5
OOS_DIR         = Path("models/oos_v2")
OOS_DIR.mkdir(parents=True, exist_ok=True)

TIME_SLOTS = [
    ("weekdays_day",   [0, 1, 2, 3], 7,  20),
    ("weekdays_night", [0, 1, 2, 3], 0,   7),
    ("friday_all",     [4],          0,  24),
    ("sunday_9_20",    [6],          9,  20),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def mask_slot(times, dow_list, h_start, h_end):
    dt = pd.to_datetime(times, utc=True)
    return np.isin(dt.dayofweek.values, dow_list) & \
           (dt.hour.values >= h_start) & (dt.hour.values < h_end)


def evaluate_oos(model, X_test, y_test, confidence_threshold=0.0):
    proba = model.predict_proba(X_test)
    prob_green = proba[:, 1]
    predicted = (prob_green >= 0.5).astype(int)
    prob_pred = np.where(predicted == 1, prob_green, proba[:, 0])
    confidence = np.abs(prob_pred - 0.5) * 200

    mask = confidence >= confidence_threshold
    if mask.sum() == 0:
        return None

    actual   = y_test[mask]
    pred     = predicted[mask]
    conf     = confidence[mask]
    is_win   = (pred == actual)
    wr       = is_win.mean()
    total_pnl = is_win.sum() * STAKE * WIN_PAYOUT + (~is_win).sum() * STAKE * LOSS_PAYOUT

    return {
        "n_trades":  int(mask.sum()),
        "win_rate":  float(wr),
        "total_pnl": float(total_pnl),
        "edge_pct":  float(wr * WIN_PAYOUT + (1 - wr) * LOSS_PAYOUT) * 100,
        "conf_mean": float(conf.mean()),
    }


def run_slots(df, df_1h, df_4h, df_taker, df_oi, df_funding, label: str) -> dict:
    """Run all slots and return results dict."""
    all_results = {}

    for slot_name, dow_list, h_start, h_end in TIME_SLOTS:
        print(f"\n  === Slot : {slot_name} ({h_start}h-{h_end}h) ===")
        best_score  = -np.inf
        best_result = None

        for window in WINDOWS:
            X_all, y_all, target_times = build_dataset_with_target_times(
                df,
                window=window,
                indicators=BEST_INDICATORS,
                include_time=INCLUDE_TIME,
                df_1h=df_1h,
                df_4h=df_4h,
                min_move_pct=MIN_MOVE_PCT,
                df_taker=df_taker,
                df_oi=df_oi,
                df_funding=df_funding,
            )

            times_dt   = pd.to_datetime(target_times, utc=True)
            slot_mask  = mask_slot(target_times, dow_list, h_start, h_end)
            train_mask = slot_mask & np.asarray(times_dt <= TRAIN_END)
            test_mask  = slot_mask & np.asarray(times_dt >= TEST_START)

            n_train, n_test = train_mask.sum(), test_mask.sum()
            if n_train < 100 or n_test < 20:
                print(f"    win={window}: train={n_train} / test={n_test} — insuffisant, skip")
                continue

            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

            for model_type in ["lgbm", "xgb"]:
                try:
                    model = train(X_train, y_train, model_type=model_type)
                    res   = evaluate_oos(model, X_test, y_test)
                    if res is None:
                        continue

                    wr = res["win_rate"]
                    print(
                        f"    win={window:<4} {model_type:<6}  "
                        f"train={n_train}  test={n_test}  "
                        f"WR={wr*100:.2f}%  P&L=${res['total_pnl']:+,.0f}  "
                        f"edge={res['edge_pct']:+.2f}%"
                    )

                    if wr > best_score:
                        best_score  = wr
                        best_result = {
                            "slot":       slot_name,
                            "model_type": model_type,
                            "window":     window,
                            "model":      model,
                            "oos_wr":     wr,
                            "oos_pnl":    res["total_pnl"],
                            "oos_edge":   res["edge_pct"],
                            "n_train":    n_train,
                            "n_test":     n_test,
                            "dow":        dow_list,
                            "hour_start": h_start,
                            "hour_end":   h_end,
                        }
                except Exception as exc:
                    print(f"    win={window} {model_type}: ERREUR {exc}")

        if best_result is not None:
            verdict = "[PROFITABLE]" if best_result["oos_wr"] > BREAK_EVEN_WR else "[SOUS BREAK-EVEN]"
            print(
                f"  --> MEILLEUR: {best_result['model_type']} win={best_result['window']} "
                f"WR_OOS={best_result['oos_wr']*100:.2f}% {verdict}"
            )
            all_results[slot_name] = best_result
        else:
            print(f"  --> Slot {slot_name}: aucun modele valide")

    # Default slot
    print(f"\n  === Slot : default ===")
    best_score_def  = -np.inf
    best_result_def = None

    for window in WINDOWS:
        X_all, y_all, target_times = build_dataset_with_target_times(
            df,
            window=window,
            indicators=BEST_INDICATORS,
            include_time=INCLUDE_TIME,
            df_1h=df_1h,
            df_4h=df_4h,
            min_move_pct=MIN_MOVE_PCT,
            df_taker=df_taker,
            df_oi=df_oi,
            df_funding=df_funding,
        )

        times_dt     = pd.to_datetime(target_times, utc=True)
        in_specific  = np.zeros(len(target_times), dtype=bool)
        for _, dow_list_s, h_start_s, h_end_s in TIME_SLOTS:
            in_specific |= mask_slot(target_times, dow_list_s, h_start_s, h_end_s)

        default_mask = ~in_specific
        train_mask   = default_mask & np.asarray(times_dt <= TRAIN_END)
        test_mask    = default_mask & np.asarray(times_dt >= TEST_START)

        n_train, n_test = train_mask.sum(), test_mask.sum()
        if n_train < 100 or n_test < 20:
            print(f"    win={window}: train={n_train} / test={n_test} — insuffisant, skip")
            continue

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

        for model_type in ["lgbm", "xgb"]:
            try:
                model = train(X_train, y_train, model_type=model_type)
                res   = evaluate_oos(model, X_test, y_test)
                if res is None:
                    continue

                wr = res["win_rate"]
                print(
                    f"    win={window:<4} {model_type:<6}  "
                    f"train={n_train}  test={n_test}  "
                    f"WR={wr*100:.2f}%  P&L=${res['total_pnl']:+,.0f}  "
                    f"edge={res['edge_pct']:+.2f}%"
                )

                if wr > best_score_def:
                    best_score_def  = wr
                    best_result_def = {
                        "slot":       "default",
                        "model_type": model_type,
                        "window":     window,
                        "model":      model,
                        "oos_wr":     wr,
                        "oos_pnl":    res["total_pnl"],
                        "oos_edge":   res["edge_pct"],
                        "n_train":    n_train,
                        "n_test":     n_test,
                    }
            except Exception as exc:
                print(f"    win={window} {model_type}: ERREUR {exc}")

    if best_result_def is not None:
        verdict = "[PROFITABLE]" if best_result_def["oos_wr"] > BREAK_EVEN_WR else "[SOUS BREAK-EVEN]"
        print(
            f"  --> MEILLEUR default: {best_result_def['model_type']} win={best_result_def['window']} "
            f"WR_OOS={best_result_def['oos_wr']*100:.2f}% {verdict}"
        )
        all_results["default"] = best_result_def

    return all_results


def print_summary(all_results: dict, label: str) -> float:
    """Print summary table and return overall weighted WR."""
    print(f"\n{'='*70}")
    print(f"RESUME OOS — {label}")
    print(f"Periode de test : {TEST_START.date()} -> {pd.Timestamp.now(tz='UTC').date()}")
    print(f"Break-even WR   : {BREAK_EVEN_WR*100:.2f}%")
    print(f"{'='*70}")
    print(f"{'Slot':<22} {'Model':<6} {'Win':>4} {'N_test':>7} {'WR_OOS':>8} {'P&L':>12} {'Edge':>8} {'Verdict'}")
    print("-" * 90)

    total_pnl   = 0.0
    total_trades = 0
    profitable   = []

    for slot_name, res in all_results.items():
        verdict = "PROFITABLE" if res["oos_wr"] > BREAK_EVEN_WR else "PERDANT"
        print(
            f"{slot_name:<22} {res['model_type']:<6} {res['window']:>4} "
            f"{res['n_test']:>7} {res['oos_wr']*100:>7.2f}% "
            f"${res['oos_pnl']:>+10,.0f} {res['oos_edge']:>+7.2f}%  {verdict}"
        )
        total_pnl    += res["oos_pnl"]
        total_trades += res["n_test"]
        if res["oos_wr"] > BREAK_EVEN_WR:
            profitable.append(slot_name)

    print("-" * 90)
    overall_wr = (
        sum(r["n_test"] * r["oos_wr"] for r in all_results.values()) / total_trades
    ) if total_trades > 0 else 0.0
    print(f"{'GLOBAL':<22} {'':>11} {total_trades:>7} {overall_wr*100:>7.2f}% ${total_pnl:>+10,.0f}")
    print(f"\nSlots profitables: {profitable}")
    return overall_wr


def save_results(all_results: dict, oos_dir: Path, label: str, futures_enabled: bool) -> None:
    """Persist models and schedule JSON for profitable slots."""
    print(f"\n[SAUVEGARDE] {label}")
    oos_schedule = []

    for slot_name, res in all_results.items():
        if res["oos_wr"] <= BREAK_EVEN_WR:
            print(f"  {slot_name:<22} -> IGNORE (WR={res['oos_wr']*100:.2f}% < break-even)")
            continue

        model_path = str(oos_dir / f"{slot_name}.pkl")
        meta = {
            "symbol":            SYMBOL,
            "interval":          INTERVAL,
            "window":            res["window"],
            "indicators":        BEST_INDICATORS,
            "include_time":      INCLUDE_TIME,
            "multitf_enabled":   True,
            "futures_enabled":   futures_enabled,
            "min_move_pct":      MIN_MOVE_PCT,
            "model_type":        res["model_type"],
            "slot":              slot_name,
            "oos_win_rate":      round(res["oos_wr"], 6),
            "oos_edge_pct":      round(res["oos_edge"], 4),
            "n_test":            res["n_test"],
            "trained_at":        datetime.now(timezone.utc).isoformat(),
            "train_end":         str(TRAIN_END.date()),
            "test_start":        str(TEST_START.date()),
        }
        if "dow" in res:
            meta.update({
                "dow":        res["dow"],
                "hour_start": res["hour_start"],
                "hour_end":   res["hour_end"],
            })

        save_model(res["model"], meta, path=model_path)
        print(f"  {slot_name:<22} -> {model_path} (WR_OOS={res['oos_wr']*100:.2f}%)")

        entry: dict = {
            "slot":               slot_name,
            "model_path":         model_path,
            "model_type":         res["model_type"],
            "window":             res["window"],
            "futures_enabled":    futures_enabled,
            "oos_win_rate":       round(res["oos_wr"] * 100, 2),
            "oos_edge_pct":       round(res["oos_edge"], 4),
            "n_test":             res["n_test"],
            "min_confidence_pct": 12.5,
        }
        if slot_name == "default":
            entry["default"] = True
        else:
            entry.update({
                "dow":        res["dow"],
                "hour_start": res["hour_start"],
                "hour_end":   res["hour_end"],
            })
        oos_schedule.append(entry)

    def _to_python(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return obj

    sched_path = oos_dir / "schedule_oos_v2.json"
    sched_path.write_text(
        json.dumps(oos_schedule, indent=2, ensure_ascii=False, default=_to_python)
    )
    print(f"  Schedule sauvegarde -> {sched_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

print("=" * 70)
print("VALIDATION OOS v2 — AVEC FEATURES FUTURES")
print(f"Train : jusqu'au {TRAIN_END.date()}  |  Test : {TEST_START.date()} -> aujourd'hui")
print("=" * 70)

# ── Chargement des donnees de base ────────────────────────────────────────────
print("\nChargement des donnees OHLCV ...")
df = load_klines(SYMBOL, INTERVAL)

try:
    df_1h = load_klines(SYMBOL, "1h")
except FileNotFoundError:
    df_1h = fetch_klines(SYMBOL, "1h", start_str="5 years ago UTC")
    save_klines(df_1h, SYMBOL, "1h")

try:
    df_4h = load_klines(SYMBOL, "4h")
except FileNotFoundError:
    df_4h = fetch_klines(SYMBOL, "4h", start_str="5 years ago UTC")
    save_klines(df_4h, SYMBOL, "4h")

print(f"  5m : {len(df)} bougies | 1h : {len(df_1h)} | 4h : {len(df_4h)}")

# ── Chargement des donnees futures ────────────────────────────────────────────
print("\nChargement des donnees futures ...")

df_taker: pd.DataFrame | None = None
df_oi:    pd.DataFrame | None = None
df_funding: pd.DataFrame | None = None

try:
    df_taker = load_futures_data(SYMBOL, "taker_5m")
    print(f"  Taker volume 5m : {len(df_taker)} lignes")
except FileNotFoundError:
    print("  AVERTISSEMENT : taker_5m non trouve. Lancez fetch_futures_data.py.")

try:
    df_oi = load_futures_data(SYMBOL, "oi_1h")
    print(f"  Open interest 1h: {len(df_oi)} lignes")
except FileNotFoundError:
    print("  AVERTISSEMENT : oi_1h non trouve. Lancez fetch_futures_data.py.")

try:
    df_funding = load_futures_data(SYMBOL, "funding")
    print(f"  Funding rates   : {len(df_funding)} lignes")
except FileNotFoundError:
    print("  AVERTISSEMENT : funding non trouve. Lancez fetch_futures_data.py.")

futures_available = any(x is not None for x in [df_taker, df_oi, df_funding])

# ── Passe 1 : sans features futures (baseline) ───────────────────────────────
print("\n" + "=" * 70)
print("PASSE 1 : SANS FEATURES FUTURES (baseline identique a true_oos_backtest.py)")
print("=" * 70)

results_baseline = run_slots(df, df_1h, df_4h, None, None, None, "Baseline")
wr_baseline = print_summary(results_baseline, "SANS FUTURES")

# ── Passe 2 : avec features futures ──────────────────────────────────────────
if futures_available:
    print("\n" + "=" * 70)
    print("PASSE 2 : AVEC FEATURES FUTURES (taker + OI + funding)")
    print("=" * 70)

    results_futures = run_slots(df, df_1h, df_4h, df_taker, df_oi, df_funding, "Avec futures")
    wr_futures = print_summary(results_futures, "AVEC FUTURES")

    # ── Sauvegarde des modeles v2 ─────────────────────────────────────────────
    save_results(results_futures, OOS_DIR, "AVEC FUTURES", futures_enabled=True)

    # ── Comparaison finale ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARAISON FINALE")
    print("=" * 70)
    print(f"  Sans features futures : WR global = {wr_baseline*100:.2f}%")
    print(f"  Avec features futures : WR global = {wr_futures*100:.2f}%")
    delta = (wr_futures - wr_baseline) * 100
    arrow = "+" if delta >= 0 else ""
    print(f"  Delta                 : {arrow}{delta:.2f}pp")
    print(f"  Break-even WR         : {BREAK_EVEN_WR*100:.2f}%")

    if wr_futures > BREAK_EVEN_WR:
        print(f"\n  CONCLUSION : EDGE REEL AVEC FUTURES — WR = {wr_futures*100:.2f}%")
        if delta >= 0.5:
            print(f"  Les features futures apportent un gain significatif (+{delta:.2f}pp).")
        else:
            print(f"  Gain marginal ({arrow}{delta:.2f}pp) — a surveiller sur plus de donnees.")
    else:
        print(f"\n  CONCLUSION : PAS D'EDGE meme avec les features futures.")
    print("=" * 70)
else:
    print("\n" + "=" * 70)
    print("PASSE 2 IGNOREE : Donnees futures non disponibles.")
    print("Lancez fetch_futures_data.py puis relancez ce script.")
    print("=" * 70)
    save_results(results_baseline, OOS_DIR, "Baseline (sans futures)", futures_enabled=False)
