import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    price_moves: np.ndarray,
) -> dict:
    correct_mask = y_pred == y_true
    incorrect_mask = ~correct_mask

    gains = price_moves[correct_mask].sum()
    losses = price_moves[incorrect_mask].sum()
    profit_factor = gains / losses if losses > 0 else float("inf")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_green": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_green": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_red": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_red": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "profit_factor": profit_factor,
    }
