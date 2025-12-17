import os
import joblib
import pandas as pd
from typing import Dict

# Paths to saved models
MODEL_PATHS = {
    "tree": "models/tree_model.joblib",
    "pruned_tree": "models/pruned_tree_model.joblib",
    "bagging": "models/bagging_model.joblib",
}

# Load all models at startup
MODELS: Dict[str, object] = {}

for key, path in MODEL_PATHS.items():
    if os.path.exists(path):
        MODELS[key] = joblib.load(path)
    else:
        raise FileNotFoundError(f"Model not found: {path}")


def predict(model_name: str, X: pd.DataFrame) -> pd.Series:
    """
    Predict sales using the specified model
    """
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not loaded.")
    model = MODELS[model_name]
    preds = model.predict(X)
    return pd.Series(preds)
