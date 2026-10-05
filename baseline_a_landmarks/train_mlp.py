"""
Baseline A - Steps 5/6: train the tiny MLP on landmark vectors and watch
train/val curves for overfitting.

Usage:
    python baseline_a_landmarks/train_mlp.py
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# we call fit() once per epoch ourselves (max_iter=1 + warm_start) to plot
# the train/val curve, so sklearn's "hasn't converged in 1 iteration" is expected.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import PROJECT_ROOT, CLASSES, SEED

LANDMARK_DIR = PROJECT_ROOT / "data" / "landmarks"
MODEL_DIR = PROJECT_ROOT / "baseline_a_landmarks" / "model"


def load_split(name):
    df = pd.read_csv(LANDMARK_DIR / f"{name}_landmarks.csv")
    feature_cols = [c for c in df.columns if c not in ("filepath", "label")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values
    return X, y


def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = MLPClassifier(
        # (128,) beat (64,32) and every other 2-layer architecture tested in
        # tune_mlp.py's sweep (94.7% vs 94.0% mean val acc over 5 seeds) --
        # single-hidden-layer nets consistently outperformed deeper/narrower
        # ones on this small (1.2k-row) landmark dataset.
        # See results/baseline_a/mlp_hparam_sweep.csv for the full sweep.
        hidden_layer_sizes=(128,),
        activation="relu",
        alpha=1e-3,
        max_iter=1,
        warm_start=True,
        random_state=SEED,
    )

    n_epochs = 60
    print(f"{'epoch':>6} {'train_acc':>10} {'val_acc':>10}")
    for epoch in range(1, n_epochs + 1):
        clf.fit(X_train_s, y_train)
        train_acc = clf.score(X_train_s, y_train)
        val_acc = clf.score(X_val_s, y_val)
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:6d} {train_acc:10.4f} {val_acc:10.4f}")

    final_train_acc = clf.score(X_train_s, y_train)
    final_val_acc = clf.score(X_val_s, y_val)
    print(f"\nFinal: train_acc={final_train_acc:.4f} val_acc={final_val_acc:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_DIR / "mlp.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(CLASSES, MODEL_DIR / "classes.joblib")
    print(f"Saved model to {MODEL_DIR}")


if __name__ == "__main__":
    main()
