import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from src.model import trainer, evaluator


def run_walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    price_moves: np.ndarray,
    n_splits: int = 5,
    model_type: str = "lgbm",
) -> list[dict]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        pm_test = price_moves[test_idx]

        model = trainer.train(X_train, y_train, model_type=model_type)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = evaluator.evaluate(y_test, y_pred, y_prob, pm_test)
        metrics["fold"] = fold_idx + 1
        metrics["n_train"] = len(train_idx)
        metrics["n_test"] = len(test_idx)
        results.append(metrics)

    # Aggregate global metrics
    global_metrics: dict = {"fold": "global"}
    for key in ["accuracy", "precision_green", "recall_green", "profit_factor"]:
        values = [r[key] for r in results if r[key] != float("inf")]
        global_metrics[key] = float(np.mean(values)) if values else float("inf")

    results.append(global_metrics)
    return results
