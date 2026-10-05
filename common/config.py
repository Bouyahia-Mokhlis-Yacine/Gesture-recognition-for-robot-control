"""Shared paths and constants for both baselines."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Source data: frames extracted from the webcam recordings, one folder per class
RAW_FRAMES_DIR = PROJECT_ROOT / "extracted_frames"

# Gesture classes -- inferred from the folder names already on disk.
CLASSES = ["left", "right", "forward", "backward", "stop"]

# Model variants C-E: same crop dataset/pipeline as Baseline B, different
# pretrained backbone (common/cnn_backbones.py), each in its own
# baseline_<letter>_cnn/ folder (train.py + evaluate.py + model/), mirroring
# baseline_a_landmarks/ and baseline_b_cnn/.
MODEL_VARIANTS = {
    "c": {"arch": "efficientnet_b0", "label": "Model C -- EfficientNet-B0", "dir": "baseline_c_cnn"},
    "d": {"arch": "resnet18", "label": "Model D -- ResNet18", "dir": "baseline_d_cnn"},
    "e": {"arch": "mobilenet_v2", "label": "Model E -- MobileNetV2", "dir": "baseline_e_cnn"},
}

# Emitted instead of one of CLASSES when the top-class probability doesn't
# clear the model's confidence threshold -- i.e. the model sees a hand but
# doesn't recognize the sign as one of the 5 trained gestures. Set per
# model since Baseline A (MLP) and Baseline B (CNN) don't need the same bar.
UNKNOWN_LABEL = "unknown"
CONFIDENCE_THRESHOLD_A = 0.6  # Baseline A -- landmarks + MLP (lowered from 0.85: real
# gestures were scoring ~60% and getting reported as UNKNOWN_LABEL instead of recognized)
CONFIDENCE_THRESHOLD_B = 0.6   # Baseline B -- MobileNetV3

# Split files (list of image paths + label) live here so both baselines
# train/validate/test on the exact same images.
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
TRAIN_SPLIT = SPLITS_DIR / "train.csv"
VAL_SPLIT = SPLITS_DIR / "val.csv"
TEST_SPLIT = SPLITS_DIR / "test.csv"

RESULTS_DIR = PROJECT_ROOT / "results"

# Cropped-hand images (baseline_b_cnn/crop_hands.py output) -- shared input
# for every crop+CNN baseline (B, C, D, E, F).
CROP_DIR = PROJECT_ROOT / "data" / "crops"

SEED = 42
