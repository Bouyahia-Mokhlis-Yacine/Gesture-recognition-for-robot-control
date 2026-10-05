"""
Registry of transfer-learning backbones for the crop-based CNN baselines
(Baseline B = mobilenet_v3_small, Models C/D/E/F = the variants below).
All share the same data pipeline (baseline_b_cnn/dataset.py, crops at
data/crops/) and the same two-stage frozen->fine-tune training shape
(baseline_b_cnn/train_mobilenet.py for B, train_backbone.py for the rest)
-- only backbone construction and which modules count as "the backbone"
for freezing differ per architecture family:

  - mobilenet_v2/v3, efficientnet_b0/b1 expose a `.features` Sequential
    (the conv backbone) and a `.classifier` Sequential ending in Linear.
  - resnet18 has no single backbone attribute -- the "backbone" is
    everything except `.fc`, and its freezable blocks are layer1..layer4.
"""
import torch.nn as nn
from torchvision.models import (
    mobilenet_v2, MobileNet_V2_Weights,
    resnet18, ResNet18_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b1, EfficientNet_B1_Weights,
)

ARCH_REGISTRY = {
    "mobilenet_v2": dict(ctor=mobilenet_v2, weights=MobileNet_V2_Weights.IMAGENET1K_V1, head="classifier"),
    "resnet18": dict(ctor=resnet18, weights=ResNet18_Weights.IMAGENET1K_V1, head="fc"),
    "efficientnet_b0": dict(ctor=efficientnet_b0, weights=EfficientNet_B0_Weights.IMAGENET1K_V1, head="classifier"),
    "efficientnet_b1": dict(ctor=efficientnet_b1, weights=EfficientNet_B1_Weights.IMAGENET1K_V1, head="classifier"),
}


def build_model(arch: str, n_classes: int, pretrained: bool = True):
    """pretrained=False skips downloading/loading the ImageNet checkpoint --
    use it whenever you're about to immediately overwrite the weights with
    load_state_dict() anyway (i.e. inference, not training). Downloading a
    full pretrained checkpoint just to discard it is wasteful, and on a Pi
    with no/flaky internet it can outright fail or hang.
    """
    cfg = ARCH_REGISTRY[arch]
    model = cfg["ctor"](weights=cfg["weights"] if pretrained else None)
    if cfg["head"] == "classifier":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, n_classes)
    else:  # "fc" (resnet18)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, n_classes)
    return model


def freeze_backbone(model, arch: str):
    if ARCH_REGISTRY[arch]["head"] == "fc":
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False
    else:
        for p in model.features.parameters():
            p.requires_grad = False


def unfreeze_last_blocks(model, arch: str, n_blocks: int):
    if ARCH_REGISTRY[arch]["head"] == "fc":
        blocks = [model.layer1, model.layer2, model.layer3, model.layer4]
    else:
        blocks = list(model.features.children())

    for block in blocks[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True
