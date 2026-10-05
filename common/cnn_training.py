"""
Shared training pieces for every crop+CNN baseline (B, C, D, E, F): image
size/batch size, augmentation, and the per-epoch train/eval loop. Backbone
construction is separate (common/cnn_backbones.py) since that differs per
architecture; everything here is identical regardless of backbone.
"""
import torch
from torchvision import transforms

IMG_SIZE = 224
BATCH_SIZE = 32

TRAIN_TFM = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    # No RandomHorizontalFlip: left/right are mirror-opposite classes here,
    # so flipping silently relabels a left-hand image to look like right
    # (still tagged "left") and vice versa -- it was teaching the model
    # contradictory signal for exactly those two classes.
    # Tried adding RandomRotation(15) + wider ColorJitter (incl.
    # saturation/hue) + occasional GaussianBlur to help live-camera
    # robustness -- on this ~1k-image dataset it made things worse (test
    # accuracy 93.7% -> 90.6%, left/right confusion got worse, not better)
    # rather than better. Reverted; more/more-diverse training data is the
    # real fix, not more aggressive augmentation on this little data.
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TFM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def run_epoch(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n_correct, n_total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X.size(0)
            n_correct += (logits.argmax(1) == y).sum().item()
            n_total += X.size(0)

    return total_loss / n_total, n_correct / n_total
