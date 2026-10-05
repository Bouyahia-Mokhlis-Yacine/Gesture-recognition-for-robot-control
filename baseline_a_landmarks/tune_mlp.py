"""
Hyperparameter sweep for the Baseline A MLP -- the initial
(hidden_layer_sizes=(64,32), alpha=1e-3) was an untuned starting guess.
This checks whether that guess is leaving accuracy on the table.

Model selection uses the VAL split only (never the test split, to avoid
picking hyperparameters that happen to fit the test set). The winning
config is then retrained on train and scored once on test, for a clean
apples-to-apples comparison against results/baseline_a/baseline_a_test_metrics.json.

Usage:
    python baseline_a_landmarks/tune_mlp.py
"""
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import PROJECT_ROOT, RESULTS_DIR

LANDMARK_DIR = PROJECT_ROOT / "data" / "landmarks"

HIDDEN_LAYER_SIZES = [(32,), (64,), (128,), (32, 16), (64, 32), (128, 64), (64, 64), (100,)]
ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1]
SEEDS = [42, 43, 44, 45, 46]
MAX_ITER = 300  # generous cap; adam has its own convergence tolerance

BASELINE_CONFIG = {"hidden_layer_sizes": (64, 32), "alpha": 1e-3}


def load_split(name):
    df = pd.read_csv(LANDMARK_DIR / f"{name}_landmarks.csv")
    feature_cols = [c for c in df.columns if c not in ("filepath", "label")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values
    return X, y


def fit_and_score(X_train, y_train, X_eval, y_eval, hidden_layer_sizes, alpha, seed):
    scaler = StandardScaler().fit(X_train)
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        alpha=alpha,
        max_iter=MAX_ITER,
        random_state=seed,
    )
    clf.fit(scaler.transform(X_train), y_train)
    return clf.score(scaler.transform(X_eval), y_eval)


def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    rows = []
    configs = list(product(HIDDEN_LAYER_SIZES, ALPHAS))
    print(f"Sweeping {len(configs)} configs x {len(SEEDS)} seeds = {len(configs) * len(SEEDS)} fits "
          f"(model selection on val only)...")
    for hidden_layer_sizes, alpha in configs:
        val_accs = [
            fit_and_score(X_train, y_train, X_val, y_val, hidden_layer_sizes, alpha, seed)
            for seed in SEEDS
        ]
        n_params = _count_params(len(_feature_cols(X_train)), hidden_layer_sizes, 5)
        rows.append({
            "hidden_layer_sizes": str(hidden_layer_sizes),
            "alpha": alpha,
            "n_params": n_params,
            "val_acc_mean": np.mean(val_accs),
            "val_acc_std": np.std(val_accs),
        })
        print(f"  {str(hidden_layer_sizes):>14s}  alpha={alpha:<8g}  "
              f"val_acc={np.mean(val_accs):.4f} +/- {np.std(val_accs):.4f}")

    results = pd.DataFrame(rows).sort_values("val_acc_mean", ascending=False).reset_index(drop=True)

    out_dir = RESULTS_DIR / "baseline_a"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "mlp_hparam_sweep.csv", index=False)

    print("\n=== Top 10 configs by mean val accuracy ===")
    print(results.head(10).to_string(index=False))

    best = results.iloc[0]
    best_hls = eval(best["hidden_layer_sizes"])  # "(64, 32)" -> (64, 32)
    best_alpha = best["alpha"]

    baseline_val_accs = [
        fit_and_score(X_train, y_train, X_val, y_val, BASELINE_CONFIG["hidden_layer_sizes"],
                       BASELINE_CONFIG["alpha"], seed)
        for seed in SEEDS
    ]
    baseline_test_accs = [
        fit_and_score(X_train, y_train, X_test, y_test, BASELINE_CONFIG["hidden_layer_sizes"],
                       BASELINE_CONFIG["alpha"], seed)
        for seed in SEEDS
    ]
    best_test_accs = [
        fit_and_score(X_train, y_train, X_test, y_test, best_hls, best_alpha, seed)
        for seed in SEEDS
    ]

    print("\n=== Baseline vs. best-found config (5 seeds each) ===")
    print(f"{'':28s} {'hidden_layer_sizes':>20s} {'alpha':>8s} {'val_acc':>16s} {'test_acc':>16s}")
    print(f"{'current (train_mlp.py)':28s} {str(BASELINE_CONFIG['hidden_layer_sizes']):>20s} "
          f"{BASELINE_CONFIG['alpha']:>8g} "
          f"{np.mean(baseline_val_accs):>8.4f}+/-{np.std(baseline_val_accs):.4f} "
          f"{np.mean(baseline_test_accs):>8.4f}+/-{np.std(baseline_test_accs):.4f}")
    print(f"{'best found':28s} {str(best_hls):>20s} {best_alpha:>8g} "
          f"{best['val_acc_mean']:>8.4f}+/-{best['val_acc_std']:.4f} "
          f"{np.mean(best_test_accs):>8.4f}+/-{np.std(best_test_accs):.4f}")

    print(f"\nSaved full sweep to {out_dir / 'mlp_hparam_sweep.csv'}")


def _feature_cols(X):
    return range(X.shape[1])


def _count_params(input_dim, hidden_layer_sizes, output_dim):
    sizes = [input_dim] + list(hidden_layer_sizes) + [output_dim]
    return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))


if __name__ == "__main__":
    main()
