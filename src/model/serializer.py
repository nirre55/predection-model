from pathlib import Path

import joblib


def save_model(
    model,
    metadata: dict,
    path: str = "models/model_calibrated.pkl",
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {"model": model, **metadata}
    joblib.dump(artifact, output_path)
    return output_path


def load_model(
    path: str = "models/model_calibrated.pkl",
) -> tuple:
    artifact = joblib.load(path)
    model = artifact["model"]
    metadata = {k: v for k, v in artifact.items() if k != "model"}
    return model, metadata
