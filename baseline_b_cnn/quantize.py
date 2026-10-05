"""
Baseline B - extra experiment: post-training STATIC INT8 quantization of
the fine-tuned MobileNetV3 (dynamic quantization barely helps CNNs, since
it only speeds up Linear/LSTM weights -- static quantization is the
standard path for conv-heavy models and is what actually pays off on
ARM/Raspberry Pi CPUs via the qnnpack backend).

Uses FX graph-mode quantization rather than eager-mode QuantStub wrapping:
MobileNetV3's squeeze-and-excitation blocks do an elementwise `scale * input`
that eager-mode static quantization can't trace through (raises
"empty_strided not supported on quantized tensors"); FX mode's tracer
handles it correctly.

Run this on the SAME architecture you'll deploy to if possible: the qnnpack
backend below targets ARM (Raspberry Pi); swap to 'fbgemm' if you quantize
on an x86 dev machine and only care about x86 numbers.

CAVEAT (tested on this project's model): naive post-training static
quantization collapsed test accuracy to near-random (~23%, biased toward
one class) even after calibrating on the full training set and excluding
the squeeze-and-excitation blocks from quantization. This is a documented
sensitivity of MobileNetV3's hardswish/SE activations to PTQ, not a bug in
this script. If you need a usable INT8 model, the standard fixes are (a)
quantization-aware training (fine-tune with fake-quant ops inserted) rather
than post-training quantization, or (b) per-op qconfig tuning beyond what's
done here. Until then, treat "Baseline B INT8" results as a documented
negative finding for the report, not a deployable model -- run
evaluate.py --quantized yourself to see the current numbers before relying
on this model on the Pi.

Usage:
    python baseline_b_cnn/quantize.py --backend qnnpack
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.ao.quantization import QConfigMapping, get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import PROJECT_ROOT, CLASSES
from baseline_b_cnn.dataset import GestureCropDataset
from baseline_b_cnn.train_mobilenet import build_model, EVAL_TFM, CROP_DIR

MODEL_DIR = PROJECT_ROOT / "baseline_b_cnn" / "model"


def calibrate(model, csv_path, n_batches=10):
    ds = GestureCropDataset(csv_path, CLASSES, EVAL_TFM)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2)
    with torch.no_grad():
        for i, (X, _) in enumerate(loader):
            model(X)
            if i + 1 >= n_batches:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["qnnpack", "fbgemm"], default="qnnpack")
    args = ap.parse_args()

    torch.backends.quantized.engine = args.backend

    model = build_model(len(CLASSES))
    model.load_state_dict(torch.load(MODEL_DIR / "mobilenet_v3_small.pt", map_location="cpu"))
    model.eval()

    qconfig_mapping = QConfigMapping().set_global(get_default_qconfig(args.backend))
    example = torch.randn(1, 3, 224, 224)
    prepared = prepare_fx(model, qconfig_mapping, (example,))

    print("Calibrating with a few batches of training data...")
    calibrate(prepared, PROJECT_ROOT / "data" / "crops" / "train_crops.csv")

    quantized = convert_fx(prepared)
    quantized.eval()

    traced = torch.jit.trace(quantized, example)
    out_path = MODEL_DIR / "mobilenet_v3_small_int8.pt"
    traced.save(str(out_path))

    fp32_size = (MODEL_DIR / "mobilenet_v3_small.pt").stat().st_size / 1e6
    int8_size = out_path.stat().st_size / 1e6
    print(f"\nFP32 model size: {fp32_size:.2f} MB")
    print(f"INT8 model size: {int8_size:.2f} MB")
    print(f"Saved quantized model to {out_path}")
    print(f"(quantized with backend={args.backend} -- set torch.backends.quantized.engine "
          f"to the same backend before loading this model elsewhere)")


if __name__ == "__main__":
    main()
