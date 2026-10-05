"""
Model E: evaluate the trained MobileNetV2 on the same test split used
by every other baseline, so results are directly comparable.

Usage:
    python baseline_e_cnn/evaluate.py
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import CLASSES, RESULTS_DIR, CROP_DIR
from common.metrics import evaluate_predictions
from common.cnn_training import EVAL_TFM
from common.cnn_backbones import build_model
from common.crop_dataset import GestureCropDataset

ARCH = "mobilenet_v2"
MODEL_DIR = Path(__file__).resolve().parent / "model"


def main():
    device = torch.device("cpu")
    test_ds = GestureCropDataset(CROP_DIR / "test_crops.csv", CLASSES, EVAL_TFM)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    model = build_model(ARCH, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_DIR / f"{ARCH}.pt", map_location=device))
    model.to(device).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            logits = model(X.to(device))
            preds = logits.argmax(1).cpu().numpy()
            y_pred.extend(CLASSES[p] for p in preds)
            y_true.extend(CLASSES[i] for i in y.numpy())

    evaluate_predictions(y_true, y_pred, CLASSES, RESULTS_DIR / "baseline_e", tag="model_e_mobilenet_v2_test")


if __name__ == "__main__":
    main()
