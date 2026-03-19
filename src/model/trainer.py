import warnings
import numpy as np
from sklearn.isotonic import IsotonicRegression

# LightGBM stocke des feature names même pour les arrays numpy (comportement sklearn ≥1.6)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)


class _PrefitCalibratedClassifier:
    """Calibration isotonique sur un estimateur déjà entraîné (équivalent de cv='prefit')."""

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.classes_ = [0, 1]
        self._ir: IsotonicRegression = IsotonicRegression(out_of_bounds="clip")

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray) -> "_PrefitCalibratedClassifier":
        raw_proba = self.base_estimator.predict_proba(X_cal)[:, 1]
        self._ir.fit(raw_proba, y_cal)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_proba = self.base_estimator.predict_proba(X)[:, 1]
        cal_proba_1 = self._ir.predict(raw_proba)
        cal_proba_0 = 1.0 - cal_proba_1
        return np.column_stack([cal_proba_0, cal_proba_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train(
    X: np.ndarray, y: np.ndarray, model_type: str = "lgbm"
) -> _PrefitCalibratedClassifier:
    n = len(X)
    i60 = int(n * 0.6)
    i80 = int(n * 0.8)

    X_base, y_base = X[:i60], y[:i60]
    X_cal, y_cal = X[i60:i80], y[i60:i80]

    if model_type == "lgbm":
        from lightgbm import LGBMClassifier

        estimator = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
    elif model_type == "xgb":
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss",
        )
    else:
        from sklearn.ensemble import RandomForestClassifier

        estimator = RandomForestClassifier(n_estimators=200, random_state=42)

    estimator.fit(X_base, y_base)
    # Supprime les feature names stockés par sklearn pour éviter le warning
    # "X does not have valid feature names" lors des prédictions numpy
    try:
        object.__delattr__(estimator, "feature_names_in_")
    except AttributeError:
        pass

    calibrated = _PrefitCalibratedClassifier(estimator)
    calibrated.fit(X_cal, y_cal)

    return calibrated
