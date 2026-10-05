"""
Baseline B - Step 7: evaluate the trained MobileNetV3 on the same test
split used by Baseline A (Step 7 of PART 1), so the two are comparable.

Usage:
    python baseline_b_cnn/evaluate.py
    python baseline_b_cnn/evaluate.py --quantized   # evaluate the INT8 model instead
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import PROJECT_ROOT, CLASSES, RESULTS_DIR
from common.metrics import evaluate_predictions
from baseline_b_cnn.dataset import GestureCropDataset
from baseline_b_cnn.train_mobilenet import build_model, EVAL_TFM, CROP_DIR

MODEL_DIR = PROJECT_ROOT / "baseline_b_cnn" / "model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantized", action="store_true")
    ap.add_argument("--backend", choices=["qnnpack", "fbgemm"], default="qnnpack")
    args = ap.parse_args()

    if args.quantized:
        torch.backends.quantized.engine = args.backend

    device = torch.device("cpu")  # quantized INT8 models run on CPU
    test_ds = GestureCropDataset(CROP_DIR / "test_crops.csv", CLASSES, EVAL_TFM)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    if args.quantized:
        model = torch.jit.load(MODEL_DIR / "mobilenet_v3_small_int8.pt")
        tag = "baseline_b_int8_test"
    else:
        model = build_model(len(CLASSES))
        model.load_state_dict(torch.load(MODEL_DIR / "mobilenet_v3_small.pt", map_location=device))
        tag = "baseline_b_fp32_test"

    model.to(device).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            logits = model(X.to(device))
            preds = logits.argmax(1).cpu().numpy()
            y_pred.extend(CLASSES[p] for p in preds)
            y_true.extend(CLASSES[i] for i in y.numpy())

    evaluate_predictions(y_true, y_pred, CLASSES, RESULTS_DIR / "baseline_b", tag=tag)


if __name__ == "__main__":
    main()
