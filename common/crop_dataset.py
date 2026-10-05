"""Torch Dataset that reads the crop CSVs written by baseline_b_cnn/crop_hands.py.
Shared by every crop+CNN baseline (B, C, D, E, F) -- they all train on the
exact same crops, just with a different backbone architecture.
"""
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class GestureCropDataset(Dataset):
    def __init__(self, csv_path: Path, classes: list, transform=None):
        with open(csv_path, newline="") as f:
            self.rows = list(csv.DictReader(f))
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[row["label"]]
        return img, torch.tensor(label, dtype=torch.long)
