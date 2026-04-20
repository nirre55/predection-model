# CLAUDE.md -- predection-model

## Projet
Outil d'analyse de bougies BTC/USDT (5m) pour identifier des points communs
entre des positions perdantes sur Polymarket.

---

## Commande principale

```bash
python analyze_candles.py <fichier.txt>
```

**Format du fichier d'entree** (une ligne par marche Polymarket) :
```
btc-updown-5m-1775571300
btc-updown-5m-1775574900
# commentaires ignores
```

**Sortie** :
- Rapport console : distributions de toutes les features
- `features.csv` : toutes les valeurs pour chaque bougie

**Mise a jour des donnees** :
```bash
python fetch_extra_data.py    # D1 OHLCV + Fear & Greed (quotidien)
python fetch_futures_data.py  # Taker vol (optionnel)
```

---

## Architecture

### Fichier principal
- `analyze_candles.py` -- outil d'analyse

### Donnees (data/raw/)
| Fichier | Colonnes | Periode |
|---------|----------|---------|
| `BTCUSDT_5m.parquet` | open_time, open, high, low, close, volume | 2021 -> present |
| `BTCUSDT_1d.parquet` | open_time, open, high, low, close, volume | 2022 -> present |
| `fear_greed.parquet` | date, fg_value | 2018 -> present |
| `BTCUSDT_taker_5m.parquet` | open_time, taker_buy_volume, total_volume | 2024 -> present |

**Important** : les parquets utilisent `datetime64[ms, UTC]` -> conversion int64 = millisecondes (pas ns).
Dans `load_df()` : `df["ts_ns"] = df[ts_col].astype("int64") * 1_000_000`

---

## Features analysees par bougie

| Categorie | Features |
|-----------|----------|
| Temps | hour_utc, day_of_week, session (asian/london/overlap/new_york/late) |
| Anatomie | direction, candle_type, body_pct, range_pct, upper/lower_wick_ratio |
| Momentum | rsi_14, stoch_k, atr_14, dist_sma20/50_pct, above_sma20/50 |
| Bollinger | bb_width_pct, bb_position |
| MACD | macd_hist, macd_trend |
| Volume | volume_ratio (vs moy 50 bougies) |
| Streak | prev_streak (nb bougies consecutives), prev_streak_dir |
| D1 context | d1_return_pct, d1_rsi, d1_above_sma20 (bougie J-1, pas look-ahead) |
| Fear & Greed | fg_value, fg_regime (bougie J-1) |
| Taker | taker_buy_ratio (si dispo) |

---

## Format timestamp Polymarket
`btc-updown-5m-1775571300` -> timestamp Unix en secondes
1775571300 -> Tue Apr 07 2026 14:15:00 UTC = bougie 14h15-14h20
