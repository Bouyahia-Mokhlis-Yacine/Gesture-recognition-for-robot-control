"""
Model D: ResNet18 transfer learning on the same hand crops as
Baseline B (data/crops/, generated once by baseline_b_cnn/crop_hands.py --
not regenerated per model). Same two-stage frozen->fine-tune recipe as
Baseline B; only the backbone differs (common/cnn_backbones.py).

Usage:
    python baseline_d_cnn/train.py --frozen-epochs 15 --finetune-epochs 15 --unfreeze-blocks 6
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import CLASSES, SEED, CROP_DIR
from common.cnn_training import TRAIN_TFM, EVAL_TFM, BATCH_SIZE, run_epoch
from common.cnn_backbones import build_model, freeze_backbone, unfreeze_last_blocks
from common.crop_dataset import GestureCropDataset

ARCH = "resnet18"
MODEL_DIR = Path(__file__).resolve().parent / "model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-epochs", type=int, default=15)
    ap.add_argument("--finetune-epochs", type=int, default=15)
    ap.add_argument("--unfreeze-blocks", type=int, default=6)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = GestureCropDataset(CROP_DIR / "train_crops.csv", CLASSES, TRAIN_TFM)
    val_ds = GestureCropDataset(CROP_DIR / "val_crops.csv", CLASSES, EVAL_TFM)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model(ARCH, len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()

    print(f"\n=== [{ARCH}] Stage 1: frozen backbone, {args.frozen_epochs} epochs ===")
    freeze_backbone(model, ARCH)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    print(f"{'epoch':>6} {'train_loss':>11} {'train_acc':>10} {'val_loss':>10} {'val_acc':>9}")
    for epoch in range(1, args.frozen_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, criterion)
        print(f"{epoch:6d} {tr_loss:11.4f} {tr_acc:10.4f} {val_loss:10.4f} {val_acc:9.4f}  ({time.time()-t0:.1f}s)")

    print(f"\n=== [{ARCH}] Stage 2: fine-tune last {args.unfreeze_blocks} blocks, {args.finetune_epochs} epochs ===")
    unfreeze_last_blocks(model, ARCH, args.unfreeze_blocks)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5
    )
    print(f"{'epoch':>6} {'train_loss':>11} {'train_acc':>10} {'val_loss':>10} {'val_acc':>9}")
    for epoch in range(1, args.finetune_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, criterion)
        print(f"{epoch:6d} {tr_loss:11.4f} {tr_acc:10.4f} {val_loss:10.4f} {val_acc:9.4f}  ({time.time()-t0:.1f}s)")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_DIR / f"{ARCH}.pt"
    torch.save(model.state_dict(), out_path)
    print(f"\nSaved model to {out_path}")


if __name__ == "__main__":
    main()
