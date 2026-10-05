"""
Baseline A - Step 1/2: run MediaPipe Hands over every image in the splits
and dump the 21x3 landmark coordinates (relative to the wrist, so the
classifier is translation/scale invariant) to a CSV per split.

Usage:
    python baseline_a_landmarks/extract_landmarks.py
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, PROJECT_ROOT
from common.hand_detector import HandDetector

LANDMARK_DIR = PROJECT_ROOT / "data" / "landmarks"
N_LANDMARKS = 21


def normalize_landmarks(landmarks):
    """21 (x,y,z) points -> wrist-relative, scale-normalized flat vector.

    Makes the classifier robust to hand position/distance from the camera,
    which is exactly the kind of invariance a raw-pixel CNN has to learn
    from data instead of getting for free.
    """
    pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    wrist = pts[0]
    pts -= wrist
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()


def extract_split(csv_path: Path, detector: HandDetector, writer_rows):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n_ok, n_fail = 0, 0
    for i, row in enumerate(rows):
        img_path, label = row["filepath"], row["label"]
        img = cv2.imread(img_path)
        if img is None:
            n_fail += 1
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmarks = detector.detect(img_rgb)

        if landmarks is None:
            n_fail += 1
            continue

        vec = normalize_landmarks(landmarks)
        writer_rows.append([img_path, label] + vec.tolist())
        n_ok += 1

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(rows)} processed...")

    print(f"  -> {n_ok} landmarks extracted, {n_fail} frames skipped (no hand detected)")
    return writer_rows


def main():
    LANDMARK_DIR.mkdir(parents=True, exist_ok=True)

    header = ["filepath", "label"]
    for i in range(N_LANDMARKS):
        header += [f"x{i}", f"y{i}", f"z{i}"]

    with HandDetector(running_mode="IMAGE", min_detection_confidence=0.5) as detector:
        for name, split_csv in [
            ("train", TRAIN_SPLIT),
            ("val", VAL_SPLIT),
            ("test", TEST_SPLIT),
        ]:
            print(f"\nExtracting landmarks for {name} split...")
            rows = []
            extract_split(split_csv, detector, rows)
            out_path = LANDMARK_DIR / f"{name}_landmarks.csv"
            with open(out_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)
            print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
