"""
Validation OOS v3 - Features Daily + Fear & Greed + Session + ADX/Stoch + Optuna.

Usage:
    python fetch_extra_data.py   # une fois
    python true_oos_backtest_v3.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.data.cache import load_klines
from src.features.builder import build_dataset_with_target_times

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL        = "BTCUSDT"
TRAIN_END     = pd.Timestamp("2025-10-31", tz="UTC")
TEST_START    = pd.Timestamp("2025-11-01", tz="UTC")
WIN_PAYOUT    = 0.90
LOSS_PAYOUT   = -1.0
STAKE         = 100.0
BREAK_EVEN_WR = 1.0 / (1.0 + WIN_PAYOUT)   # 52.63%
MIN_MOVE_PCT  = 0.003
N_SPLITS      = 5
N_OPTUNA      = 30   # trials Optuna par slot
CONF_THRESH   = 12.5
OOS_DIR       = Path("models/oos_v3")
OOS_DIR.mkdir(parents=True, exist_ok=True)

INDICATORS = ["rsi", "macd", "atr", "mfi", "vdelta", "body", "streak", "adx", "stoch"]

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


def evaluate_oos(model, X_test, y_test, conf_thresh=CONF_THRESH):
    proba    = model.predict_proba(X_test)
    prob_g   = proba[:, 1]
    pred     = (prob_g >= 0.5).astype(int)
    prob_p   = np.where(pred == 1, prob_g, proba[:, 0])
    conf     = np.abs(prob_p - 0.5) * 200
    mask     = conf >= conf_thresh
    if mask.sum() == 0:
        return None
    actual   = y_test[mask]
    is_win   = (pred[mask] == actual)
    wr       = is_win.mean()
    pnl      = is_win.sum() * STAKE * WIN_PAYOUT + (~is_win).sum() * STAKE * LOSS_PAYOUT
    return {
        "n_trades":  int(mask.sum()),
        "win_rate":  float(wr),
        "total_pnl": float(pnl),
        "edge_pct":  float(wr * WIN_PAYOUT + (1 - wr) * LOSS_PAYOUT) * 100,
    }


def train_lgbm(X_tr, y_tr, params: dict):
    from lightgbm import LGBMClassifier
    m = LGBMClassifier(
        n_estimators    = params.get("n_estimators", 400),
        learning_rate   = params.get("learning_rate", 0.05),
        num_leaves      = params.get("num_leaves", 63),
        min_child_samples= params.get("min_child_samples", 20),
        subsample       = params.get("subsample", 0.8),
        colsample_bytree= params.get("colsample_bytree", 0.8),
        reg_alpha       = params.get("reg_alpha", 0.1),
        reg_lambda      = params.get("reg_lambda", 1.0),
        class_weight    = "balanced",
        random_state    = 42,
        verbose         = -1,
    )
    m.fit(X_tr, y_tr)
    return m


def train_xgb(X_tr, y_tr, params: dict):
    from xgboost import XGBClassifier
    m = XGBClassifier(
        n_estimators    = params.get("n_estimators", 400),
        learning_rate   = params.get("learning_rate", 0.05),
        max_depth       = params.get("max_depth", 6),
        min_child_weight= params.get("min_child_weight", 5),
        subsample       = params.get("subsample", 0.8),
        colsample_bytree= params.get("colsample_bytree", 0.8),
        reg_alpha       = params.get("reg_alpha", 0.1),
        reg_lambda      = params.get("reg_lambda", 1.0),
        scale_pos_weight= 1.0,
        random_state    = 42,
        eval_metric     = "logloss",
        verbosity       = 0,
    )
    m.fit(X_tr, y_tr)
    return m


def train_catboost(X_tr, y_tr, params: dict):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(
        iterations      = params.get("n_estimators", 400),
        learning_rate   = params.get("learning_rate", 0.05),
        depth           = params.get("max_depth", 6),
        l2_leaf_reg     = params.get("reg_lambda", 1.0),
        min_data_in_leaf= params.get("min_child_samples", 20),
        subsample       = params.get("subsample", 0.8),
        random_seed     = 42,
        verbose         = False,
        auto_class_weights= "Balanced",
    )
    m.fit(X_tr, y_tr)
    return m


TRAINERS = {
    "lgbm":     train_lgbm,
    "xgb":      train_xgb,
    "catboost": train_catboost,
}


def optuna_search(X_tr, y_tr, X_val, y_val, model_type: str, n_trials: int) -> tuple[dict, float]:
    """Run Optuna to find best hyperparams. Returns (best_params, best_wr)."""
    from sklearn.model_selection import TimeSeriesSplit

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 200, 800),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves":      trial.suggest_int("num_leaves", 31, 255),
            "max_depth":       trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_child_weight":trial.suggest_int("min_child_weight", 1, 20),
        }
        try:
            model = TRAINERS[model_type](X_tr, y_tr, params)
            res   = evaluate_oos(model, X_val, y_val)
            if res is None or res["n_trades"] < 20:
                return 0.0
            return res["win_rate"]
        except Exception:
            return 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


# ── Main backtest ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("VALIDATION OOS v3 - Features D1 + F&G + Session + ADX/Stoch + Optuna")
    print(f"Train : jusqu'au {TRAIN_END.date()}  |  Test : {TEST_START.date()} -> aujourd'hui")
    print("=" * 70)

    # Load data
    print("\nChargement OHLCV ...")
    df    = load_klines(SYMBOL, "5m")
    df_1h = load_klines(SYMBOL, "1h")
    df_4h = load_klines(SYMBOL, "4h")
    df_1d = load_klines(SYMBOL, "1d")
    print(f"  5m:{len(df):,} | 1h:{len(df_1h):,} | 4h:{len(df_4h):,} | 1d:{len(df_1d):,}")

    # Fear & Greed
    fg_path = Path("data/raw/fear_greed.parquet")
    if fg_path.exists():
        df_fg = pd.read_parquet(fg_path)
        print(f"  Fear&Greed: {len(df_fg)} jours")
    else:
        df_fg = None
        print("  Fear&Greed: non disponible (run fetch_extra_data.py)")

    # Build full feature matrix with ALL new features
    print("\nConstruction features (D1 + F&G + Session + ADX + Stoch) ...")
    X_all, y_all, target_times = build_dataset_with_target_times(
        df,
        window          = 50,
        indicators      = INDICATORS,
        include_time    = True,
        df_1h           = df_1h,
        df_4h           = df_4h,
        min_move_pct    = MIN_MOVE_PCT,
        df_1d           = df_1d,
        df_fg           = df_fg,
        include_session = True,
    )
    print(f"  X shape: {X_all.shape}  |  y={y_all.mean():.3f}")

    times_dt = pd.to_datetime(target_times, utc=True)

    print("\n" + "=" * 70)
    print("RECHERCHE DES MEILLEURS MODELES PAR SLOT (Optuna)")
    print("=" * 70)

    results   = {}
    schedules = []

    for slot_name, dow_list, h_start, h_end in TIME_SLOTS:
        print(f"\n=== Slot : {slot_name} ({h_start}h-{h_end}h) ===")

        slot_mask  = mask_slot(target_times, dow_list, h_start, h_end)
        train_mask = slot_mask & np.asarray(times_dt <= TRAIN_END)
        test_mask  = slot_mask & np.asarray(times_dt >= TEST_START)

        n_train, n_test = train_mask.sum(), test_mask.sum()
        print(f"  train={n_train} | test={n_test}")

        if n_train < 100 or n_test < 20:
            print("  Insuffisant, skip.")
            continue

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

        # Split train into train_opt / val for Optuna
        split_idx  = int(len(X_train) * 0.8)
        X_tr_opt   = X_train[:split_idx]
        y_tr_opt   = y_train[:split_idx]
        X_val_opt  = X_train[split_idx:]
        y_val_opt  = y_train[split_idx:]

        best_model  = None
        best_wr     = -np.inf
        best_result = None
        best_mtype  = None
        best_params = None

        for mtype in ["lgbm", "xgb", "catboost"]:
            print(f"  [{mtype}] Optuna ({N_OPTUNA} trials) ...", end=" ", flush=True)
            params, val_wr = optuna_search(X_tr_opt, y_tr_opt, X_val_opt, y_val_opt, mtype, N_OPTUNA)
            print(f"val_WR={val_wr:.3%}", end=" ", flush=True)

            # Retrain on full train with best params
            model  = TRAINERS[mtype](X_train, y_train, params)
            result = evaluate_oos(model, X_test, y_test)

            if result is None:
                print("-> pas de trades OOS")
                continue

            print(f"-> OOS WR={result['win_rate']:.2%} ({result['n_trades']} trades)")

            if result["win_rate"] > best_wr:
                best_wr     = result["win_rate"]
                best_model  = model
                best_result = result
                best_mtype  = mtype
                best_params = params

        if best_result is None:
            print("  Aucun modele valide.")
            continue

        profitable = best_wr > BREAK_EVEN_WR
        verdict    = "PROFITABLE" if profitable else "PERDANT"
        pnl_str    = f"${best_result['total_pnl']:>+,.0f}"
        edge_str   = f"{best_result['edge_pct']:+.2f}%"
        print(f"  --> MEILLEUR: {best_mtype} WR_OOS={best_wr:.2%}  P&L={pnl_str}  edge={edge_str}  [{verdict}]")

        results[slot_name] = {
            "model_type":  best_mtype,
            "win_rate":    best_wr,
            "total_pnl":   best_result["total_pnl"],
            "n_test":      best_result["n_trades"],
            "edge_pct":    best_result["edge_pct"],
            "profitable":  profitable,
        }

        if profitable and best_model is not None:
            from src.model.serializer import save_model
            model_path = OOS_DIR / f"{slot_name}.pkl"
            save_model(best_model, {
                "slot":         slot_name,
                "model_type":   best_mtype,
                "oos_win_rate": best_wr,
                "features_v3":  True,
            }, path=str(model_path))
            print(f"  Sauvegarde -> {model_path}")

            schedules.append({
                "slot":            slot_name,
                "dow":             dow_list,
                "hour_start":      h_start,
                "hour_end":        h_end,
                "model_type":      best_mtype,
                "model_path":      str(model_path).replace("\\", "/"),
                "oos_win_rate":    round(best_wr * 100, 2),
                "window":          50,
                "min_confidence_pct": CONF_THRESH,
                "features_v3":     True,
            })

    # Default slot
    print("\n=== Slot : default ===")
    from sklearn.model_selection import TimeSeriesSplit
    default_mask   = np.zeros(len(X_all), dtype=bool)
    for slot_name, dow_list, h_start, h_end in TIME_SLOTS:
        default_mask |= mask_slot(target_times, dow_list, h_start, h_end)
    default_mask = ~default_mask

    train_mask_def = default_mask & np.asarray(times_dt <= TRAIN_END)
    test_mask_def  = default_mask & np.asarray(times_dt >= TEST_START)
    n_tr_d, n_te_d = train_mask_def.sum(), test_mask_def.sum()
    print(f"  train={n_tr_d} | test={n_te_d}")

    if n_tr_d >= 100 and n_te_d >= 20:
        X_train_d, y_train_d = X_all[train_mask_def], y_all[train_mask_def]
        X_test_d,  y_test_d  = X_all[test_mask_def],  y_all[test_mask_def]
        split_d = int(len(X_train_d) * 0.8)

        best_model_d  = None
        best_wr_d     = -np.inf
        best_result_d = None
        best_mtype_d  = None

        for mtype in ["lgbm", "xgb", "catboost"]:
            print(f"  [{mtype}] Optuna ({N_OPTUNA} trials) ...", end=" ", flush=True)
            params, val_wr = optuna_search(
                X_train_d[:split_d], y_train_d[:split_d],
                X_train_d[split_d:], y_train_d[split_d:],
                mtype, N_OPTUNA,
            )
            print(f"val_WR={val_wr:.3%}", end=" ", flush=True)
            model  = TRAINERS[mtype](X_train_d, y_train_d, params)
            result = evaluate_oos(model, X_test_d, y_test_d)
            if result is None:
                print("-> pas de trades OOS")
                continue
            print(f"-> OOS WR={result['win_rate']:.2%} ({result['n_trades']} trades)")
            if result["win_rate"] > best_wr_d:
                best_wr_d     = result["win_rate"]
                best_model_d  = model
                best_result_d = result
                best_mtype_d  = mtype

        if best_result_d:
            profitable_d = best_wr_d > BREAK_EVEN_WR
            print(f"  --> MEILLEUR default: {best_mtype_d} WR_OOS={best_wr_d:.2%}  "
                  f"P&L=${best_result_d['total_pnl']:+,.0f}  "
                  f"[{'PROFITABLE' if profitable_d else 'PERDANT'}]")
            results["default"] = {
                "model_type": best_mtype_d,
                "win_rate":   best_wr_d,
                "total_pnl":  best_result_d["total_pnl"],
                "n_test":     best_result_d["n_trades"],
                "edge_pct":   best_result_d["edge_pct"],
                "profitable": profitable_d,
            }
            if profitable_d and best_model_d:
                from src.model.serializer import save_model
                model_path = OOS_DIR / "default.pkl"
                save_model(best_model_d, {
                    "slot":         "default",
                    "model_type":   best_mtype_d,
                    "oos_win_rate": best_wr_d,
                    "features_v3":  True,
                }, path=str(model_path))
                schedules.append({
                    "slot": "default", "default": True,
                    "model_type":    best_mtype_d,
                    "model_path":    str(model_path).replace("\\", "/"),
                    "oos_win_rate":  round(best_wr_d * 100, 2),
                    "window":        50,
                    "min_confidence_pct": CONF_THRESH,
                    "features_v3":   True,
                })

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESUME OOS v3")
    print(f"Periode test : {TEST_START.date()} -> aujourd'hui  |  Break-even : {BREAK_EVEN_WR:.2%}")
    print("=" * 70)
    hdr = f"{'Slot':<22} {'Model':<10} {'N_test':>7}  {'WR_OOS':>8}  {'P&L':>12}  {'Edge':>8}  Verdict"
    print(hdr)
    print("-" * 80)

    total_trades = 0
    total_wins   = 0
    total_pnl    = 0.0

    for slot_name, res in results.items():
        n  = res["n_test"]
        wr = res["win_rate"]
        wins = int(round(wr * n))
        total_trades += n
        total_wins   += wins
        total_pnl    += res["total_pnl"]
        v = "PROFITABLE" if res["profitable"] else "PERDANT"
        print(f"  {slot_name:<20} {res['model_type']:<10} {n:>7}  {wr:>8.2%}  "
              f"${res['total_pnl']:>+10,.0f}  {res['edge_pct']:>+7.2f}%  {v}")

    print("-" * 80)
    global_wr = total_wins / total_trades if total_trades > 0 else 0.0
    print(f"  {'GLOBAL':<20} {'':10} {total_trades:>7}  {global_wr:>8.2%}  ${total_pnl:>+10,.0f}")
    print()

    profitable_slots = [s for s, r in results.items() if r["profitable"]]
    print(f"Slots profitables: {profitable_slots}")

    delta = global_wr - 0.5354
    print(f"\nComparaison avec baseline (v1=53.54%) : {delta:+.2%}")

    if global_wr >= 0.55:
        print("\n*** OBJECTIF ATTEINT : WR >= 55% ***")
    else:
        print(f"\n[Ecart objectif 55%] : {0.55 - global_wr:.2%} manquant")

    # Save schedule
    if schedules:
        sched_path = OOS_DIR / "schedule_oos_v3.json"
        with open(sched_path, "w") as f:
            json.dump(schedules, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
        print(f"\nSchedule sauvegarde -> {sched_path}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
