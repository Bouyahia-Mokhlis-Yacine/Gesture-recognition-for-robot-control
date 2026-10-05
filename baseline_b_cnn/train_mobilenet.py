"""
Baseline B - Steps 3-6: load pretrained MobileNetV3-Small, replace the
classifier head, train the head with the backbone frozen, then unfreeze
the last few blocks and fine-tune at a low learning rate.

Usage:
    python baseline_b_cnn/train_mobilenet.py --frozen-epochs 8 --finetune-epochs 6
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import PROJECT_ROOT, CLASSES, SEED, CROP_DIR
from common.cnn_training import IMG_SIZE, BATCH_SIZE, TRAIN_TFM, EVAL_TFM, run_epoch
from baseline_b_cnn.dataset import GestureCropDataset

MODEL_DIR = PROJECT_ROOT / "baseline_b_cnn" / "model"


def build_model(n_classes, pretrained: bool = True):
    """pretrained=False skips downloading/loading the ImageNet checkpoint --
    use it at inference time (about to load_state_dict() anyway) rather
    than training, to avoid a wasted download (can outright fail/hang on a
    Pi with no/flaky internet).
    """
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, n_classes)
    return model


def freeze_backbone(model):
    for p in model.features.parameters():
        p.requires_grad = False


def unfreeze_last_blocks(model, n_blocks=3):
    # model.features is a Sequential of inverted-residual blocks;
    # unfreeze only the last few so early low-level filters stay fixed.
    for block in list(model.features.children())[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-epochs", type=int, default=8)
    ap.add_argument("--finetune-epochs", type=int, default=6)
    ap.add_argument("--unfreeze-blocks", type=int, default=3)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = GestureCropDataset(CROP_DIR / "train_crops.csv", CLASSES, TRAIN_TFM)
    val_ds = GestureCropDataset(CROP_DIR / "val_crops.csv", CLASSES, EVAL_TFM)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model(len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()

    print(f"\n=== Stage 1: frozen backbone, {args.frozen_epochs} epochs ===")
    freeze_backbone(model)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    print(f"{'epoch':>6} {'train_loss':>11} {'train_acc':>10} {'val_loss':>10} {'val_acc':>9}")
    for epoch in range(1, args.frozen_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, criterion)
        print(f"{epoch:6d} {tr_loss:11.4f} {tr_acc:10.4f} {val_loss:10.4f} {val_acc:9.4f}  ({time.time()-t0:.1f}s)")

    print(f"\n=== Stage 2: fine-tune last {args.unfreeze_blocks} blocks, {args.finetune_epochs} epochs ===")
    unfreeze_last_blocks(model, args.unfreeze_blocks)
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
    torch.save(model.state_dict(), MODEL_DIR / "mobilenet_v3_small.pt")
    print(f"\nSaved model to {MODEL_DIR / 'mobilenet_v3_small.pt'}")


if __name__ == "__main__":
    main()
