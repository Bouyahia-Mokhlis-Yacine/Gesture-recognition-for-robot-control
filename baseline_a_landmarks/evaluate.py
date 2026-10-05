"""
Baseline A - Step 7: evaluate the trained MLP on the held-out test split.
Reports accuracy, precision, recall, F1, and a confusion matrix.

Usage:
    python baseline_a_landmarks/evaluate.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import PROJECT_ROOT, CLASSES, RESULTS_DIR
from common.metrics import evaluate_predictions

LANDMARK_DIR = PROJECT_ROOT / "data" / "landmarks"
MODEL_DIR = PROJECT_ROOT / "baseline_a_landmarks" / "model"


def main():
    df = pd.read_csv(LANDMARK_DIR / "test_landmarks.csv")
    feature_cols = [c for c in df.columns if c not in ("filepath", "label")]
    X_test = df[feature_cols].values.astype(np.float32)
    y_test = df["label"].values

    clf = joblib.load(MODEL_DIR / "mlp.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")

    y_pred = clf.predict(scaler.transform(X_test))

    evaluate_predictions(
        y_test, y_pred, CLASSES, RESULTS_DIR / "baseline_a", tag="baseline_a_test"
    )


if __name__ == "__main__":
    main()
