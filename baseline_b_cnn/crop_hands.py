"""
Baseline B - Step 2: detect the hand with MediaPipe (reusing the same
detector as Baseline A) and crop a square, padded box around it, resized
to 224x224. Runs once over the same train/val/test splits as Baseline A
and writes crops to data/crops/<split>/<label>/*.jpg, plus a CSV per split
pointing at the crops.

Usage:
    python baseline_b_cnn/crop_hands.py
"""
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, PROJECT_ROOT
from common.hand_detector import HandDetector

CROP_DIR = PROJECT_ROOT / "data" / "crops"
IMG_SIZE = 224
PAD_FRAC = 0.3  # extra margin around the tight hand bounding box


def bbox_from_landmarks(landmarks, img_w, img_h):
    xs = [p.x * img_w for p in landmarks]
    ys = [p.y * img_h for p in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    w, h = x_max - x_min, y_max - y_min
    side = max(w, h) * (1 + PAD_FRAC)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

    x0 = int(max(cx - side / 2, 0))
    y0 = int(max(cy - side / 2, 0))
    x1 = int(min(cx + side / 2, img_w))
    y1 = int(min(cy + side / 2, img_h))
    return x0, y0, x1, y1


def crop_split(split_name, csv_path, detector: HandDetector, out_rows):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    n_ok, n_fail = 0, 0
    for i, row in enumerate(rows):
        img_path, label = row["filepath"], row["label"]
        img = cv2.imread(img_path)
        if img is None:
            n_fail += 1
            continue

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmarks = detector.detect(rgb)
        if landmarks is None:
            n_fail += 1
            continue

        x0, y0, x1, y1 = bbox_from_landmarks(landmarks, w, h)
        if x1 <= x0 or y1 <= y0:
            n_fail += 1
            continue

        crop = img[y0:y1, x0:x1]
        crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        out_dir = CROP_DIR / split_name / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / Path(img_path).name
        cv2.imwrite(str(out_path), crop)

        out_rows.append([str(out_path), label])
        n_ok += 1

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(rows)} processed...")

    print(f"  -> {n_ok} crops written, {n_fail} frames skipped (no hand detected)")
    return out_rows


def main():
    with HandDetector(running_mode="IMAGE", min_detection_confidence=0.5) as detector:
        for split_name, split_csv in [
            ("train", TRAIN_SPLIT),
            ("val", VAL_SPLIT),
            ("test", TEST_SPLIT),
        ]:
            print(f"\nCropping hands for {split_name} split...")
            rows = []
            crop_split(split_name, split_csv, detector, rows)
            out_csv = CROP_DIR / f"{split_name}_crops.csv"
            with open(out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["filepath", "label"])
                w.writerows(rows)
            print(f"  Wrote {out_csv}")


if __name__ == "__main__":
    main()
