# predection-model

> Modele de prediction de direction de bougies M5 BTC/USDT avec LightGBM et XGBoost.
> Win rate actuel : **57-64% selon le slot horaire** (objectif 55%+ atteint) avec features multi-timeframe 1h/4h et target engineering.

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)

### Installation

```bash
git clone https://github.com/nirre55/predection-model.git
cd predection-model
uv sync
```

---

## Workflow complet

```
1. fetch             — telecharger les bougies Binance (5m + 1h + 4h) en cache local
2. ablation          — trouver la meilleure combinaison d'indicateurs
3. threshold_sweep   — trouver le seuil optimal de target engineering
4. ablation_schedule — entrainer un modele par slot horaire (multi-TF + target engineering)
5. recalibrate       — recalibrer les modeles sans re-entrainer
6. shap_analysis     — analyser l'importance des features (SHAP)
7. compare_periods   — comparer les performances sur 4 periodes
8. backtest_schedule — backtester le schedule sur donnees historiques
9. live              — lancer le predicteur en temps reel
```

---

## CLI principal (`cli.py`)

```bash
# Telecharger les bougies (cache local)
uv run python cli.py fetch

# Entrainer le modele global (toutes les donnees)
uv run python cli.py train

# Backtest rapide du modele global
uv run python cli.py backtest

# Lancer le predicteur en temps reel
uv run python cli.py live
```

---

## Scripts d'analyse

### `ablation.py` — Recherche de la meilleure combinaison d'indicateurs

Teste toutes les combinaisons possibles des indicateurs (rsi, macd, bb, atr, vol) avec LightGBM via walk-forward validation. Sauvegarde le meilleur modele calibre dans `models/model_calibrated.pkl`.

```bash
# Toutes les donnees en cache (defaut)
uv run python ablation.py

# Filtrer a partir d'une date
uv run python ablation.py --start "1 year ago UTC"
uv run python ablation.py --start "6 months ago UTC"
uv run python ablation.py --start "2025-01-01"
```

**Sorties :**
- `models/model_calibrated.pkl` — meilleur modele global
- `models/ablation_results.json` — classement de toutes les combinaisons

---

### `ablation_schedule.py` — Entrainement par slot horaire

Pour chaque slot horaire defini (dimanche 9h-20h, vendredi, samedi 7h-14h, jours de semaine, etc.), teste LGBM vs XGBoost avec des fenetres [20, 50, 100]. Integre les features multi-timeframe (1h/4h) et le target engineering (seuil `min_move_pct=0.001`). Sauvegarde le meilleur modele par slot et genere `models/schedule.json`.

```bash
# Sur 1 an (recommande — sweet spot donnees recentes vs historique)
uv run python ablation_schedule.py --start "1 year ago UTC"

# Sur 6 mois (retraining mensuel)
uv run python ablation_schedule.py --start "6 months ago UTC"

# Toutes les donnees
uv run python ablation_schedule.py
```

**Resultats actuels (1 an de donnees, seuil 0.1%, features multi-TF) :**

| Slot | Win Rate |
|------|---------|
| weekdays_day (lun-jeu 7h-20h) | 62.02% |
| friday_all | 59.98% |
| weekdays_night | 59.43% |
| sunday_9_20 | 59.03% |
| weekdays_eve | 58.01% |
| saturday_14_24 | 55.90% |
| sunday_20_24 | 58.43% |
| saturday_7_14 | 56.16% |
| saturday_0_7 | 53.21% |
| sunday_0_9 | 52.87% |

**Slots horaires :**

| Slot              | Jours               | Heures    |
|-------------------|---------------------|-----------|
| sunday_9_20       | Dimanche            | 9h - 20h  |
| sunday_0_9        | Dimanche            | 0h - 9h   |
| sunday_20_24      | Dimanche            | 20h - 24h |
| friday_all        | Vendredi            | 0h - 24h  |
| saturday_7_14     | Samedi              | 7h - 14h  |
| saturday_0_7      | Samedi              | 0h - 7h   |
| saturday_14_24    | Samedi              | 14h - 24h |
| weekdays_day      | Lun-Jeu             | 7h - 20h  |
| weekdays_night    | Lun-Jeu             | 0h - 7h   |
| weekdays_eve      | Lun-Jeu             | 20h - 24h |

**Sorties :**
- `models/schedule/` — un `.pkl` par slot (ex: `saturday_7_14.pkl`)
- `models/schedule.json` — config de routing utilisee par le predicteur live

---

### `threshold_sweep.py` — Sweep du seuil de target engineering

Mesure le win rate out-of-sample pour chaque valeur de seuil de filtrage. Les bougies avec un mouvement inferieur au seuil sont exclues de l'entrainement (trop aleatoires pour etre previsibles).

```bash
uv run python threshold_sweep.py --start "1 year ago UTC"
```

**Resultats actuels :**

| Seuil | Samples | Win Rate OOS |
|-------|---------|-------------|
| 0.00% (baseline) | 104 751 | 57.87% |
| 0.05% | 57 084 | 61.39% |
| 0.10% (configure) | 30 546 | 64.09% |
| 0.15% | 16 967 | 67.00% |
| 0.20% | 9 921 | 67.91% |

**Sorties :**
- `models/threshold_sweep_results.json`

---

### `shap_analysis.py` — Analyse SHAP des features

Calcule l'importance SHAP de chaque feature sur le modele actif. Permet de valider que les nouvelles features (multi-TF) contribuent vraiment et d'identifier les features redondantes.

```bash
uv run python shap_analysis.py --start "1 year ago UTC"
```

**Top features identifiees :**
1. `momentum_1h` — contexte de momentum 1h (feature multi-TF, impact dominant)
2. `rsi_t-1` — RSI 5m sur la derniere bougie
3. `rsi_1h` — RSI 1h (feature multi-TF, top 10)

**Sorties :**
- `models/shap_summary.png` — beeswarm plot des 25 features les plus importantes

---

### `recalibrate.py` — Recalibration rapide

Recalibre la couche de calibration (Platt scaling) de tous les modeles existants sans re-entrainer les modeles de base. Utile si les probabilites derivent ou si on veut tester une nouvelle periode de calibration.

```bash
uv run python recalibrate.py
```

**Modeles recalibres :**
- `models/model_calibrated.pkl`
- Tous les modeles dans `models/schedule/`

---

### `compare_periods.py` — Comparatif multi-periodes

Compare les performances sur 4 periodes d'entrainement (5 ans, 1 an, 6 mois, 3 mois) sans modifier les modeles en production. Utilise la combinaison optimale (rsi+macd+atr) trouvee par l'ablation.

```bash
uv run python compare_periods.py
```

**Deux sections :**
1. **Comparatif global** — window=50, LGBM vs XGBoost, les 4 periodes
2. **Comparatif par slot** — window=20, LGBM vs XGBoost, les 10 slots x 4 periodes

**Sorties :**
- `models/comparison_global.csv`
- `models/comparison_slots.csv`
- `models/comparison_report.json`

---

### `backtest_schedule.py` — Backtest historique du schedule

Simule le schedule de modeles sur les 20% de donnees les plus recentes (out-of-sample). Calcule le P&L reel avec les frais Binance (0.1% entree + 0.1% sortie = 0.2% aller-retour).

```bash
# Backtest standard
uv run python backtest_schedule.py

# Avec seuil de confiance (ne trader que si |proba - 50%| > X%)
uv run python backtest_schedule.py --confidence 5.0

# Avec position differente (defaut: 1000 USD)
uv run python backtest_schedule.py --position 500

# Depuis une date specifique
uv run python backtest_schedule.py --start "2025-01-01"

# Combinaison
uv run python backtest_schedule.py --confidence 5.0 --position 1000 --start "2025-06-01"
```

**Metriques calculees :**
- Win rate, nombre de trades
- P&L total, P&L par slot
- Profit factor
- Drawdown maximum
- Sharpe ratio annualise

**Sorties :**
- `models/backtest_equity.csv` — courbe d'equity trade par trade
- `models/backtest_report.json` — rapport complet par slot

**Note importante :** Les bougies M5 ont une amplitude moyenne de 0.05-0.10%. Les frais Binance spot sont de 0.2% aller-retour. Le backtest sera donc deficitaire sans seuil de confiance eleve ou sans utiliser les frais futures (maker: 0.02%).

---

## Predicteur live (`cli.py live`)

Lance le predicteur en temps reel. Il se declenche automatiquement a chaque nouvelle bougie M5 (toutes les 5 minutes).

```bash
uv run python cli.py live
```

**Comportement :**
- Detecte automatiquement `models/schedule.json` si present (routing par slot horaire)
- Sinon utilise `models/model_calibrated.pkl` (modele unique)
- Si les modeles ont `multitf_enabled: true`, fetche le contexte 1h/4h au demarrage et le rafraichit toutes les 60 min
- Affiche la direction predite (VERT/ROUGE), la probabilite et la confiance
- Compare la prediction avec la bougie Binance suivante (WIN/LOSS)
- Sauvegarde toutes les predictions dans `models/predictions.csv`

**Exemple de sortie :**
```
[14:00->14:05][weekdays_day] VERT 53.21% | ROUGE 46.79% | conf=6.4%
  --> Bougie fermee : VERT | WIN
```

**Format du CSV (`models/predictions.csv`) :**

| Colonne               | Description                          |
|-----------------------|--------------------------------------|
| predicted_at          | Heure de la prediction (UTC)         |
| candle_open           | Ouverture de la bougie predite       |
| candle_close          | Fermeture de la bougie predite       |
| slot                  | Slot horaire utilise                 |
| predicted_direction   | VERT ou ROUGE                        |
| probability_pct       | Probabilite de la direction predite  |
| other_direction       | Direction opposee                    |
| other_probability_pct | Probabilite de la direction opposee  |
| confidence_pct        | Confiance = |prob - 50%| x 2         |
| actual_direction      | Direction reelle de la bougie        |
| result                | WIN, LOSS ou PENDING                 |

Arreter avec `Ctrl+C` — la derniere prediction est sauvegardee comme PENDING.

---

## Structure du projet

```
predection-model/
├── cli.py                   # Entrypoint principal
├── ablation.py              # Recherche meilleure combinaison d'indicateurs
├── ablation_schedule.py     # Entrainement par slot horaire (multi-TF + target engineering)
├── threshold_sweep.py       # Sweep du seuil de target engineering
├── shap_analysis.py         # Analyse SHAP des features
├── recalibrate.py           # Recalibration rapide sans re-entrainement
├── compare_periods.py       # Comparatif multi-periodes
├── backtest_schedule.py     # Backtest historique avec frais
├── src/
│   ├── data/
│   │   ├── cache.py         # Cache local des bougies Binance (5m, 1h, 4h)
│   │   └── fetcher.py       # Telechargement Binance API
│   ├── features/
│   │   ├── builder.py       # Construction des features (OHLCV + indicateurs + multi-TF)
│   │   ├── indicators.py    # RSI, MACD, Bollinger, ATR, Volume
│   │   └── time_features.py # Encodage cyclique heure/jour (sin/cos)
│   ├── model/
│   │   ├── trainer.py       # Entrainement LGBM/XGBoost + calibration Platt
│   │   ├── serializer.py    # Sauvegarde/chargement .pkl
│   │   └── scheduler.py     # Routing temporel des modeles
│   ├── backtest/
│   │   └── walk_forward.py  # Validation walk-forward (TimeSeriesSplit)
│   └── live/
│       └── predictor.py     # Predicteur temps reel (avec refresh contexte 1h/4h)
└── models/
    ├── model_calibrated.pkl          # Modele global
    ├── schedule.json                 # Config routing temporel
    ├── schedule/                     # Modeles par slot (multitf_enabled: true)
    ├── predictions.csv               # Historique des predictions live
    ├── shap_summary.png              # Beeswarm plot importance des features
    └── threshold_sweep_results.json  # Resultats du sweep de seuil
```

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

## License

[MIT](LICENSE)
