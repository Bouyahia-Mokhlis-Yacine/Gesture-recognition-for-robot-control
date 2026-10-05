"""
Step 0 for BOTH baselines: build one stratified train/val/test split over the
raw extracted frames, so Baseline A (landmarks) and Baseline B (CNN) are
trained and evaluated on exactly the same images. Run this once.

Usage:
    python data/prepare_dataset.py
"""
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import RAW_FRAMES_DIR, CLASSES, SPLITS_DIR, SEED

TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def collect_images():
    per_class = {}
    for cls in CLASSES:
        cls_dir = RAW_FRAMES_DIR / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder not found: {cls_dir}")
        images = sorted(p for p in cls_dir.glob("*.jpg"))
        if not images:
            raise RuntimeError(f"No .jpg images found in {cls_dir}")
        per_class[cls] = images
    return per_class


def stratified_split(per_class):
    rng = random.Random(SEED)
    train_rows, val_rows, test_rows = [], [], []

    for cls, images in per_class.items():
        images = images[:]
        rng.shuffle(images)
        n = len(images)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)

        for p in images[:n_train]:
            train_rows.append((str(p), cls))
        for p in images[n_train:n_train + n_val]:
            val_rows.append((str(p), cls))
        for p in images[n_train + n_val:]:
            test_rows.append((str(p), cls))

    return train_rows, val_rows, test_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])
        writer.writerows(rows)


def main():
    per_class = collect_images()
    print("Images found per class:")
    for cls, images in per_class.items():
        print(f"  {cls:10s} {len(images)}")

    train_rows, val_rows, test_rows = stratified_split(per_class)

    write_csv(SPLITS_DIR / "train.csv", train_rows)
    write_csv(SPLITS_DIR / "val.csv", val_rows)
    write_csv(SPLITS_DIR / "test.csv", test_rows)

    print(f"\nWrote splits to {SPLITS_DIR}")
    print(f"  train: {len(train_rows)}")
    print(f"  val:   {len(val_rows)}")
    print(f"  test:  {len(test_rows)}")


if __name__ == "__main__":
    main()
