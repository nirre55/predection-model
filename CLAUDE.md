# CLAUDE.md — predection-model

## Projet
Modèle de prédiction de direction de bougies M5 BTC/USDT (LightGBM / XGBoost / CatBoost).
Options binaires : WIN = +90%, LOSS = -100%, break-even = 52.63%.

---

## Commandes essentielles

```bash
# Live trading
python cli.py live-v3              # V3 PRODUCTION (59.51% OOS) — UTILISER CELUI-CI
python cli.py live                 # V1 (53.54% OOS) — ancien

# Données
python fetch_extra_data.py         # Met a jour D1 OHLCV + Fear & Greed (quotidien)
python fetch_futures_data.py       # Met a jour taker vol, OI, funding (optionnel)

# Validation / backtests
python true_oos_backtest_v3.py     # Backtest complet v3 (long ~1h avec Optuna)
python true_oos_backtest.py        # Backtest baseline v1
python utils/analyze_predictions.py  # Analyse predictions CSV
```

---

## Architecture des modèles

### V3 — Production (commit cf1ec21)
- **Schedule** : `models/schedule_v3.json`
- **Modèles** : `models/oos_v3/*.pkl`
- **Predictions** : `models/predictions_v3.csv`
- **OOS global** : 59.51% sur 1109 trades (Nov 2025 - Avr 2026)

| Slot | Jours | Heures | Model | OOS WR |
|------|-------|--------|-------|--------|
| weekdays_day | Lun-Jeu | 7h-20h | LightGBM | 57.69% |
| weekdays_night | Lun-Jeu | 0h-7h | CatBoost | 65.48% |
| friday_all | Vendredi | 0h-24h | CatBoost | 62.32% |
| sunday_9_20 | Dimanche | 9h-20h | LightGBM | 68.66% |
| default | Reste | — | XGBoost | 55.22% |

### V1 — Ancien
- **Schedule** : `models/schedule.json`
- **OOS global** : 53.54%

---

## Features (725 au total pour v3)

```
OHLCV window 50 x 5          = 250
Indicateurs 9 x 50            = 450  (rsi, macd, atr, mfi, vdelta, body, streak, adx, stoch)
Time sin/cos                  =   5
Multi-timeframe 1h + 4h       =   5
Daily D1 context              =   7  (rsi, sma20/50, momentum, atr, vol, day_return)
Fear & Greed Index            =   4  (value, ma7, delta, regime)
Session flags                 =   4  (asian, london, ny, overlap)
TOTAL                         = 725
```

**Important** : le schedule_v3.json DOIT contenir `"indicators"` et `"include_time": true`
sinon le live générera le mauvais nombre de features (970 au lieu de 725).

---

## Données requises

| Fichier | Source | Fréquence MAJ |
|---------|--------|---------------|
| `data/raw/BTCUSDT_5m.parquet` | Binance (fetch_klines) | Auto au démarrage |
| `data/raw/BTCUSDT_1h.parquet` | Binance | Auto |
| `data/raw/BTCUSDT_4h.parquet` | Binance | Auto |
| `data/raw/BTCUSDT_1d.parquet` | Binance | `fetch_extra_data.py` |
| `data/raw/fear_greed.parquet` | alternative.me (gratuit) | `fetch_extra_data.py` |
| `data/raw/BTCUSDT_taker_5m.parquet` | Binance Futures | `fetch_futures_data.py` |

---

## Bugs critiques corrigés (ne pas reproduire)

1. **Target offset** : `target_idx = j+window+1` (pas +2). Erreur +1 candle = look-ahead.
2. **MTF look-ahead** : bougies 1h/4h → utiliser `searchsorted(times - ONE_HOUR_NS)` pas `searchsorted(times)`.
3. **D1 look-ahead** : utiliser la bougie D1 de J-1 (pas J). `day_start = (t_ns // ONE_DAY_NS) * ONE_DAY_NS` puis `searchsorted(day_start, side="left") - 1`.
4. **F&G look-ahead** : même logique que D1, utiliser J-1.
5. **Feature count mismatch** : toujours sauvegarder `"indicators"` et `"include_time"` dans le schedule JSON.

---

## Entraînement / validation

- **Train** : données ≤ 31 octobre 2025
- **Test OOS** : 1er novembre 2025 → aujourd'hui
- **Split Optuna** : 80% train_opt / 20% val_opt (temporel, pas aléatoire)
- **Walk-forward CV** : N_SPLITS=5, TimeSeriesSplit
- **min_move_pct** : 0.003 (filtre bougies plates à l'entraînement uniquement)

⚠️ Le test OOS a été utilisé plusieurs fois (v1, v2, v3) → risque de data snooping.
La vraie validation sera le live trading des prochaines semaines.

---

## Seuils et paramètres

- **Break-even WR** : 52.63% = 1 / (1 + 0.90)
- **Confidence threshold** : 12.5% (trades avec conf < 12.5% sont SKIP)
- **Window** : 50 bougies pour tous les slots v3
- **Validation statistique** : ~410 trades pour 95% de confiance à 53.5% WR
