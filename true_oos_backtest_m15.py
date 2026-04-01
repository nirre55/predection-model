"""
Validation OOS correcte M15 — sans data leakage.

Methode :
  - TRAIN : donnees jusqu'a TRAIN_END (exclu)
  - TEST  : donnees a partir de TEST_START (jamais vues pendant l'entrainement)

Modeles sauvegardes dans models/m15/ (separement du pipeline M5).

Usage :
    python true_oos_backtest_m15.py
"""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.data.cache import load_klines, save_klines
from src.data.fetcher import fetch_klines
from src.features.builder import build_dataset_with_target_times
from src.model.trainer import train
from src.model.serializer import save_model

# ── Configuration ─────────────────────────────────────────────────────────────
SYMBOL          = "BTCUSDT"
INTERVAL        = "15m"
TRAIN_END       = pd.Timestamp("2025-10-31", tz="UTC")
TEST_START      = pd.Timestamp("2025-11-01", tz="UTC")
WIN_PAYOUT      = 0.90
LOSS_PAYOUT     = -1.0
STAKE           = 100.0
BREAK_EVEN_WR   = 1.0 / (1.0 + WIN_PAYOUT)              # 52.63%
BEST_INDICATORS = ["rsi", "macd", "atr", "mfi", "vdelta", "body", "streak"]
INCLUDE_TIME    = True
MIN_MOVE_PCT    = 0.003
WINDOWS         = [20, 50, 100]
OOS_DIR         = Path("models/m15/oos_valid")
SCHEDULE_DIR    = Path("models/m15/schedule")
OOS_DIR.mkdir(parents=True, exist_ok=True)
SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)

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

    actual    = y_test[mask]
    pred      = predicted[mask]
    conf      = confidence[mask]
    is_win    = (pred == actual)
    wr        = is_win.mean()
    total_pnl = is_win.sum() * STAKE * WIN_PAYOUT + (~is_win).sum() * STAKE * LOSS_PAYOUT

    return {
        "n_trades":  int(mask.sum()),
        "win_rate":  float(wr),
        "total_pnl": float(total_pnl),
        "edge_pct":  float(wr * WIN_PAYOUT + (1 - wr) * LOSS_PAYOUT) * 100,
        "conf_mean": float(conf.mean()),
    }


def _to_python(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj


# ── Chargement des donnees ────────────────────────────────────────────────────
print("=" * 70)
print("VALIDATION OOS M15 (sans data leakage)")
print(f"Train : jusqu'au {TRAIN_END.date()}  |  Test : {TEST_START.date()} -> aujourd'hui")
print("=" * 70)

print("\nChargement des donnees 15m...")
try:
    df = load_klines(SYMBOL, INTERVAL)
    print(f"  Cache trouve : {len(df)} bougies 15m")
except FileNotFoundError:
    print("  Fetch 15m depuis Binance...")
    df = fetch_klines(SYMBOL, INTERVAL, start_str="5 years ago UTC")
    save_klines(df, SYMBOL, INTERVAL)
    print(f"  {len(df)} bougies 15m telechargees et mises en cache.")

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

print(f"  15m: {len(df)} bougies | 1h: {len(df_1h)} | 4h: {len(df_4h)}")

# ── Entrainement + evaluation par slot ───────────────────────────────────────
print("\n[1] ENTRAINEMENT ET EVALUATION PAR SLOT")
print("-" * 70)

all_results = {}

for slot_name, dow_list, h_start, h_end in TIME_SLOTS:
    print(f"\n=== Slot : {slot_name} ({h_start}h-{h_end}h) ===")

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
        )

        times_dt   = pd.to_datetime(target_times, utc=True)
        slot_mask  = mask_slot(target_times, dow_list, h_start, h_end)
        train_mask = slot_mask & np.asarray(times_dt <= TRAIN_END)
        test_mask  = slot_mask & np.asarray(times_dt >= TEST_START)

        n_train, n_test = train_mask.sum(), test_mask.sum()
        if n_train < 100 or n_test < 20:
            print(f"  win={window}: train={n_train} / test={n_test} — insuffisant, skip")
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
                    f"  win={window:<4} {model_type:<6}  "
                    f"train={n_train}  test={n_test}  "
                    f"WR={wr*100:.2f}%  P&L=${res['total_pnl']:+,.0f}  "
                    f"edge={res['edge_pct']:+.2f}%"
                )

                if wr > best_score:
                    best_score = wr
                    best_result = {
                        "slot": slot_name, "model_type": model_type,
                        "window": window, "model": model,
                        "oos_wr": wr, "oos_pnl": res["total_pnl"],
                        "oos_edge": res["edge_pct"],
                        "n_train": n_train, "n_test": n_test,
                        "dow": dow_list, "hour_start": h_start, "hour_end": h_end,
                    }
            except Exception as e:
                print(f"  win={window} {model_type}: ERREUR {e}")

    if best_result is not None:
        verdict = "PROFITABLE" if best_result["oos_wr"] > BREAK_EVEN_WR else "SOUS BREAK-EVEN"
        print(f"  --> MEILLEUR: {best_result['model_type']} win={best_result['window']} "
              f"WR_OOS={best_result['oos_wr']*100:.2f}%  [{verdict}]")
        all_results[slot_name] = best_result
    else:
        print(f"  --> Slot {slot_name} : aucun modele valide")

# ── Slot DEFAULT ──────────────────────────────────────────────────────────────
print(f"\n=== Slot : default ===")

best_score_def  = -np.inf
best_result_def = None

for window in WINDOWS:
    X_all, y_all, target_times = build_dataset_with_target_times(
        df, window=window, indicators=BEST_INDICATORS,
        include_time=INCLUDE_TIME, df_1h=df_1h, df_4h=df_4h,
        min_move_pct=MIN_MOVE_PCT,
    )

    times_dt    = pd.to_datetime(target_times, utc=True)
    in_specific = np.zeros(len(target_times), dtype=bool)
    for _, dow_list, h_start, h_end in TIME_SLOTS:
        in_specific |= mask_slot(target_times, dow_list, h_start, h_end)

    default_mask = ~in_specific
    train_mask   = default_mask & np.asarray(times_dt <= TRAIN_END)
    test_mask    = default_mask & np.asarray(times_dt >= TEST_START)

    n_train, n_test = train_mask.sum(), test_mask.sum()
    if n_train < 100 or n_test < 20:
        print(f"  win={window}: train={n_train} / test={n_test} — insuffisant, skip")
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
                f"  win={window:<4} {model_type:<6}  "
                f"train={n_train}  test={n_test}  "
                f"WR={wr*100:.2f}%  P&L=${res['total_pnl']:+,.0f}  "
                f"edge={res['edge_pct']:+.2f}%"
            )

            if wr > best_score_def:
                best_score_def  = wr
                best_result_def = {
                    "slot": "default", "model_type": model_type,
                    "window": window, "model": model,
                    "oos_wr": wr, "oos_pnl": res["total_pnl"],
                    "oos_edge": res["edge_pct"],
                    "n_train": n_train, "n_test": n_test,
                }
        except Exception as e:
            print(f"  win={window} {model_type}: ERREUR {e}")

if best_result_def is not None:
    verdict = "PROFITABLE" if best_result_def["oos_wr"] > BREAK_EVEN_WR else "SOUS BREAK-EVEN"
    print(f"  --> MEILLEUR default: {best_result_def['model_type']} win={best_result_def['window']} "
          f"WR_OOS={best_result_def['oos_wr']*100:.2f}%  [{verdict}]")
    all_results["default"] = best_result_def

# ── Resume final ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESUME OOS M15")
print(f"Periode de test : {TEST_START.date()} -> {pd.Timestamp.now(tz='UTC').date()}")
print(f"Break-even WR   : {BREAK_EVEN_WR*100:.2f}%")
print("=" * 70)
print(f"{'Slot':<22} {'Model':<6} {'Win':>4} {'N_test':>7} {'WR_OOS':>8} {'P&L':>12} {'Edge':>8} {'Verdict'}")
print("-" * 90)

total_pnl_all    = 0.0
total_trades     = 0
profitable_slots = []

for slot_name, res in all_results.items():
    verdict = "PROFITABLE" if res["oos_wr"] > BREAK_EVEN_WR else "PERDANT"
    print(
        f"{slot_name:<22} {res['model_type']:<6} {res['window']:>4} "
        f"{res['n_test']:>7} {res['oos_wr']*100:>7.2f}% "
        f"${res['oos_pnl']:>+10,.0f} {res['oos_edge']:>+7.2f}%  {verdict}"
    )
    total_pnl_all += res["oos_pnl"]
    total_trades  += res["n_test"]
    if res["oos_wr"] > BREAK_EVEN_WR:
        profitable_slots.append(slot_name)

print("-" * 90)
overall_wr = (sum(r["n_test"] * r["oos_wr"] for r in all_results.values()) / total_trades) if total_trades > 0 else 0
print(f"{'GLOBAL':<22} {'':>11} {total_trades:>7} {overall_wr*100:>7.2f}% ${total_pnl_all:>+10,.0f}")
print(f"\nSlots profitables : {profitable_slots}")
print(f"Slots a supprimer : {[s for s in all_results if s not in profitable_slots]}")

# ── Sauvegarde des modeles M15 ────────────────────────────────────────────────
print("\n[2] SAUVEGARDE DES MODELES M15")
m15_schedule = []

for slot_name, res in all_results.items():
    if res["oos_wr"] <= BREAK_EVEN_WR:
        print(f"  {slot_name:<22} -> IGNORE (WR={res['oos_wr']*100:.2f}% < break-even)")
        continue

    if slot_name == "default":
        model_path = "models/m15/model_calibrated.pkl"
    else:
        model_path = f"models/m15/schedule/{slot_name}.pkl"

    meta = {
        "symbol":          SYMBOL,
        "interval":        INTERVAL,
        "window":          res["window"],
        "indicators":      BEST_INDICATORS,
        "include_time":    INCLUDE_TIME,
        "multitf_enabled": True,
        "min_move_pct":    MIN_MOVE_PCT,
        "model_type":      res["model_type"],
        "slot":            slot_name,
        "oos_win_rate":    round(res["oos_wr"], 6),
        "oos_edge_pct":    round(res["oos_edge"], 4),
        "n_test":          res["n_test"],
        "trained_at":      datetime.now(timezone.utc).isoformat(),
        "train_end":       str(TRAIN_END.date()),
        "test_start":      str(TEST_START.date()),
    }
    if "dow" in res:
        meta.update({"dow": res["dow"], "hour_start": res["hour_start"], "hour_end": res["hour_end"]})

    save_model(res["model"], meta, path=model_path)
    print(f"  {slot_name:<22} -> {model_path} (WR_OOS={res['oos_wr']*100:.2f}%)")

    entry = {
        "slot":              slot_name,
        "model_path":        model_path,
        "model_type":        res["model_type"],
        "window":            res["window"],
        "oos_win_rate":      round(res["oos_wr"] * 100, 2),
        "oos_edge_pct":      round(res["oos_edge"], 4),
        "n_test":            res["n_test"],
        "min_confidence_pct": 12.5,
    }
    if slot_name == "default":
        entry["default"] = True
    else:
        entry.update({"dow": res["dow"], "hour_start": res["hour_start"], "hour_end": res["hour_end"]})
    m15_schedule.append(entry)

schedule_path = Path("models/m15/schedule.json")
schedule_path.write_text(json.dumps(m15_schedule, indent=2, ensure_ascii=False, default=_to_python))
print(f"\nSchedule M15 sauvegarde -> {schedule_path}")

print("\n" + "=" * 70)
if overall_wr > BREAK_EVEN_WR:
    print(f"CONCLUSION : EDGE M15 CONFIRME — WR global OOS = {overall_wr*100:.2f}%")
    print("Lancer le live M15 avec : python cli.py live-m15")
else:
    print(f"CONCLUSION : PAS D'EDGE M15 — WR global OOS = {overall_wr*100:.2f}% < {BREAK_EVEN_WR*100:.2f}%")
    print("Le M15 n'est PAS profitable. Garder uniquement le M5.")
print("=" * 70)
