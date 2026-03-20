"""
Analyse SHAP des features du modèle entraîné.
Charge le modèle (schedule ou unique), reconstruit les features,
génère un beeswarm plot et sauvegarde models/shap_summary.png.

Usage:
    python shap_analysis.py
    python shap_analysis.py --start "1 year ago UTC"
"""
import argparse, warnings
from pathlib import Path
import numpy as np
import shap

warnings.filterwarnings("ignore")

from src.data.cache import load_klines
from src.data.fetcher import fetch_klines
from src.data.cache import save_klines
from src.model import serializer, scheduler as sched
from src.features.builder import build_dataset_with_target_times
from src.features.builder import ALL_INDICATORS

SYMBOL = "BTCUSDT"
OUTPUT_PATH = Path("models/shap_summary.png")

def _build_feature_names(window, indicators, include_time, multitf_enabled):
    names = []
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    for i in range(window):
        for col in ohlcv_cols:
            names.append(f"{col}_t-{window - i}")
    for ind in indicators:
        for i in range(window):
            names.append(f"{ind}_t-{window - i}")
    if include_time:
        names += ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
    if multitf_enabled:
        names += ["trend_1h", "trend_4h", "momentum_1h", "volatility_regime_1h", "rsi_1h"]
    return names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None)
    args = parser.parse_args()

    # Charger modèle
    if sched.has_schedule():
        sched_obj = sched.ModelScheduler()
        from datetime import datetime, timezone
        model, meta = sched_obj.get_model(datetime.now(timezone.utc))
    else:
        model, meta = serializer.load_model("models/model_calibrated.pkl")

    window = meta.get("window", 50)
    indicators = meta.get("indicators") or ALL_INDICATORS
    include_time = meta.get("include_time", False)
    multitf_enabled = meta.get("multitf_enabled", False)

    df = load_klines(SYMBOL, "5m")
    if args.start:
        import pandas as pd, re
        m = re.match(r"(\d+)\s+(year|month|day)s?\s+ago", args.start, re.I)
        if m:
            n, unit = int(m.group(1)), m.group(2).lower()
            offsets = {"year": pd.DateOffset(years=n), "month": pd.DateOffset(months=n), "day": pd.DateOffset(days=n)}
            start_ts = pd.Timestamp.now(tz="UTC") - offsets[unit]
        else:
            start_ts = pd.Timestamp(args.start).tz_localize("UTC") if "UTC" not in args.start else pd.Timestamp(args.start)
        df = df[df["open_time"] >= start_ts]

    df_1h, df_4h = None, None
    if multitf_enabled:
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

    X, y, _ = build_dataset_with_target_times(
        df, window=window, indicators=indicators,
        include_time=include_time, df_1h=df_1h, df_4h=df_4h,
    )

    # Utiliser les 5000 derniers samples pour la vitesse
    X_sample = X[-5000:]
    feature_names = _build_feature_names(window, indicators, include_time, multitf_enabled)

    print(f"Calcul SHAP sur {len(X_sample)} samples, {X_sample.shape[1]} features...")
    explainer = shap.TreeExplainer(model.base_estimator)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # classe 1 (VERT)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      show=False, max_display=25)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=150)
    print(f"Plot sauvegardé : {OUTPUT_PATH}")

    # Top 10 features
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:10]
    print("\nTop 10 features par importance SHAP :")
    for rank, idx in enumerate(top_idx, 1):
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        print(f"  {rank:>2}. {name:<35} {mean_abs[idx]:.6f}")

if __name__ == "__main__":
    main()
