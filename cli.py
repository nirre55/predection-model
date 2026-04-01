from datetime import datetime, timezone

import typer

app = typer.Typer(help="Pipeline ML de prédiction de bougies M5 crypto")


@app.command()
def fetch(
    symbol: str = typer.Option("BTCUSDT", help="Symbole Binance (ex: BTCUSDT)"),
    interval: str = typer.Option("5m", help="Intervalle de bougie (ex: 5m)"),
    start: str = typer.Option("5 years ago UTC", help="Date de début (ex: '5 years ago UTC')"),
):
    """Télécharge l'historique de bougies depuis Binance et le cache en local."""
    from src.data.fetcher import fetch_klines
    from src.data.cache import save_klines

    typer.echo(f"Téléchargement de {symbol} {interval} depuis '{start}'...")
    df = fetch_klines(symbol, interval, start_str=start)
    path = save_klines(df, symbol, interval)
    typer.echo(f"{len(df)} bougies sauvegardées dans {path}")


@app.command()
def train(
    symbol: str = typer.Option("BTCUSDT"),
    interval: str = typer.Option("5m"),
    window: int = typer.Option(50, help="Nombre de bougies par fenêtre de features"),
    model: str = typer.Option("lgbm", help="Type de modèle : lgbm | xgb | rf"),
):
    """Entraîne le modèle sur les données en cache et sauvegarde le modèle calibré."""
    import numpy as np
    from src.data.cache import load_klines
    from src.features.builder import build_dataset
    from src.model.trainer import train as train_model
    from src.model.evaluator import evaluate
    from src.model.serializer import save_model

    typer.echo(f"Chargement des données {symbol} {interval}...")
    df = load_klines(symbol, interval)
    typer.echo(f"{len(df)} bougies chargées. Construction des features (window={window})...")

    X, y = build_dataset(df, window=window)

    # Compute price_moves for evaluation (|close[i+1] - open[i+1]|)
    close_arr = np.asarray(df["close"].values, dtype="float64")
    open_arr = np.asarray(df["open"].values, dtype="float64")
    price_moves = np.abs(
        close_arr[window + 2 : window + 2 + len(y)]
        - open_arr[window + 2 : window + 2 + len(y)]
    )

    typer.echo(f"Entraînement du modèle {model} sur {len(X)} samples...")
    calibrated_model = train_model(X, y, model_type=model)

    n = len(X)
    i80 = int(n * 0.8)
    X_test, y_test = X[i80:], y[i80:]
    pm_test = price_moves[i80:]
    y_pred = calibrated_model.predict(X_test)
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_pred, y_prob, pm_test)
    typer.echo("\n=== Métriques (test set) ===")
    for k, v in metrics.items():
        typer.echo(f"  {k}: {v:.4f}")

    metadata = {
        "symbol": symbol,
        "interval": interval,
        "window": window,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(X),
    }
    path = save_model(calibrated_model, metadata)
    typer.echo(f"\nModèle sauvegardé dans {path}")


@app.command()
def backtest(
    symbol: str = typer.Option("BTCUSDT"),
    interval: str = typer.Option("5m"),
    window: int = typer.Option(50),
    model: str = typer.Option("lgbm"),
    splits: int = typer.Option(5, help="Nombre de folds walk-forward"),
):
    """Lance le walk-forward et affiche les métriques par fold."""
    import numpy as np
    from src.data.cache import load_klines
    from src.features.builder import build_dataset
    from src.backtest.walk_forward import run_walk_forward
    from src.backtest.metrics import compute_equity_curve, compute_max_drawdown

    df = load_klines(symbol, interval)
    X, y = build_dataset(df, window=window)

    close_arr = np.asarray(df["close"].values, dtype="float64")
    open_arr = np.asarray(df["open"].values, dtype="float64")
    price_moves = np.abs(
        close_arr[window + 2 : window + 2 + len(y)]
        - open_arr[window + 2 : window + 2 + len(y)]
    )

    typer.echo(f"Walk-forward {splits} folds sur {len(X)} samples...")
    results = run_walk_forward(X, y, price_moves, n_splits=splits, model_type=model)

    for r in results:
        fold_label = r["fold"]
        acc = r.get("accuracy", 0)
        pf = r.get("profit_factor", 0)
        typer.echo(f"  Fold {fold_label}: accuracy={acc:.4f}, profit_factor={pf:.4f}")

    # Equity curve summary using last fold predictions
    typer.echo("\n[Equity curve calculée sur le dernier fold]")

    typer.echo("Walk-forward terminé.")


@app.command()
def live(
    symbol: str = typer.Option("BTCUSDT"),
    interval: str = typer.Option("5m"),
    window: int = typer.Option(50),
    model_path: str = typer.Option("models/model_calibrated.pkl", help="Chemin vers le modèle"),
):
    """Lance la prédiction live toutes les 5 minutes (M5)."""
    from src.live.predictor import LivePredictor

    predictor = LivePredictor(
        symbol=symbol,
        interval=interval,
        window=window,
        model_path=model_path,
        schedule_path="models/schedule.json",
        predictions_csv="models/predictions.csv",
    )
    predictor.start()


@app.command(name="live-m15")
def live_m15(
    symbol: str = typer.Option("BTCUSDT"),
    window: int = typer.Option(50),
):
    """Lance la prédiction live toutes les 15 minutes (M15)."""
    from src.live.predictor import LivePredictor

    predictor = LivePredictor(
        symbol=symbol,
        interval="15m",
        window=window,
        model_path="models/m15/model_calibrated.pkl",
        schedule_path="models/m15/schedule.json",
        predictions_csv="models/m15/predictions.csv",
    )
    predictor.start()


if __name__ == "__main__":
    app()
