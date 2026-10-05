"""
Rebuilds data/crops/{train,val,test}_crops.csv from whatever .jpg files are
actually present under data/crops/<split>/<label>/ right now.

Needed after manually deleting bad/mislabeled crops directly from those
folders (rather than re-running crop_hands.py, which would regenerate
crops from the original frames and undo the cleanup) -- the CSVs
crop_hands.py wrote are a manifest snapshot, so they go stale the moment
files are deleted on disk without regenerating them, and
GestureCropDataset has no existence check (it'll just crash on a missing
file at training time).

Usage:
    python baseline_b_cnn/sync_crops_csv.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import PROJECT_ROOT, CLASSES

CROP_DIR = PROJECT_ROOT / "data" / "crops"


def sync_split(split_name):
    rows = []
    for label in CLASSES:
        class_dir = CROP_DIR / split_name / label
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.glob("*.jpg")):
            rows.append([str(img_path), label])

    out_csv = CROP_DIR / f"{split_name}_crops.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label"])
        w.writerows(rows)

    counts = {label: sum(1 for r in rows if r[1] == label) for label in CLASSES}
    print(f"{split_name:6s}: {len(rows):4d} rows  {counts}")
    return len(rows)


def main():
    print("Rebuilding crop CSVs from files actually present on disk...")
    for split_name in ["train", "val", "test"]:
        sync_split(split_name)


if __name__ == "__main__":
    main()
