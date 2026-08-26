"""Lightweight model registry for persistence and A/B comparison."""
from pathlib import Path

import joblib

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def save_model(model, name: str):
    """Persist a fitted estimator under models/<name>.joblib."""
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str):
    """Load a previously saved estimator."""
    path = MODEL_DIR / f"{name}.joblib"
    return joblib.load(path)


def list_models():
    """Return the names (without extension) of all saved models."""
    return [p.stem for p in MODEL_DIR.glob("*.joblib")]
