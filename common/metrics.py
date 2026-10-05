"""
Shared evaluation used by both baselines (Step 7 of the plan):
accuracy, per-class precision/recall/F1, and a confusion matrix
(printed + saved as CSV and PNG).
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def evaluate_predictions(y_true, y_pred, classes, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    print(f"\n=== {tag} evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, labels=classes, zero_division=0))
    print("Confusion matrix (rows=actual, cols=predicted):")
    header = "".join(f"{c[:6]:>8s}" for c in classes)
    print(" " * 10 + header)
    for i, cls in enumerate(classes):
        row = "".join(f"{v:8d}" for v in cm[i])
        print(f"{cls:10s}{row}")

    metrics = {
        "tag": tag,
        "accuracy": acc,
        "per_class": {
            cls: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, cls in enumerate(classes)
        },
        "macro_f1": float(np.mean(f1)),
    }

    with open(out_dir / f"{tag}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / f"{tag}_confusion_matrix.csv", "w") as f:
        f.write("," + ",".join(classes) + "\n")
        for i, cls in enumerate(classes):
            f.write(cls + "," + ",".join(str(v) for v in cm[i]) + "\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion matrix - {tag}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{tag}_confusion_matrix.png", dpi=150)
        plt.close()
    except ImportError:
        pass

    return metrics
