"""
Baseline B - Step 8: run the FULL pipeline (camera -> hand crop -> MobileNetV3
-> prediction) live on the Raspberry Pi and measure FPS, latency, CPU%, RAM,
and stability. Mirrors baseline_a_landmarks/benchmark_pi.py so the two are
directly comparable.

Usage:
    python baseline_b_cnn/benchmark_pi.py --seconds 60
    python baseline_b_cnn/benchmark_pi.py --seconds 60 --quantized --backend qnnpack
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import PROJECT_ROOT, RESULTS_DIR, CLASSES
from common.hand_detector import HandDetector
from common.camera import open_camera
from baseline_b_cnn.crop_hands import bbox_from_landmarks, IMG_SIZE
from baseline_b_cnn.train_mobilenet import build_model, EVAL_TFM

MODEL_DIR = PROJECT_ROOT / "baseline_b_cnn" / "model"


def load_model(quantized: bool, backend: str):
    if quantized:
        torch.backends.quantized.engine = backend
        model = torch.jit.load(MODEL_DIR / "mobilenet_v3_small_int8.pt")
    else:
        model = build_model(len(CLASSES))
        model.load_state_dict(torch.load(MODEL_DIR / "mobilenet_v3_small.pt", map_location="cpu"))
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--quantized", action="store_true")
    ap.add_argument("--backend", choices=["qnnpack", "fbgemm"], default="qnnpack")
    args = ap.parse_args()

    model = load_model(args.quantized, args.backend)
    tag = "baseline_b_int8" if args.quantized else "baseline_b_fp32"

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    detector = HandDetector(running_mode="IMAGE", min_detection_confidence=0.5)

    process = psutil.Process()
    process.cpu_percent()

    latencies_ms = []
    n_frames, n_hand_detected, n_errors = 0, 0, 0
    samples = []

    print(f"Running {tag} benchmark for {args.seconds:.0f}s on camera {args.camera}...")
    start = time.time()
    last_sample = start
    try:
        while time.time() - start < args.seconds:
            t0 = time.time()
            try:
                ok, frame = cap.read()
                if not ok:
                    n_errors += 1
                    continue
                # Match the training data's mirrored/selfie recording
                # convention (see robot/run_pipeline.py) so predictions
                # aren't left/right-swapped relative to training.
                frame = cv2.flip(frame, 1)

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks = detector.detect(rgb)

                if landmarks is not None:
                    n_hand_detected += 1
                    x0, y0, x1, y1 = bbox_from_landmarks(landmarks, w, h)
                    if x1 > x0 and y1 > y0:
                        crop = frame[y0:y1, x0:x1]
                        crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        tensor = EVAL_TFM(pil_img).unsqueeze(0)
                        with torch.no_grad():
                            logits = model(tensor)
                        _ = int(logits.argmax(1).item())

                n_frames += 1
            except Exception:
                n_errors += 1

            t1 = time.time()
            latencies_ms.append((t1 - t0) * 1000.0)

            if t1 - last_sample >= 1.0:
                samples.append((t1 - start, process.cpu_percent(), process.memory_info().rss / 1e6))
                last_sample = t1
    finally:
        cap.release()
        detector.close()

    elapsed = time.time() - start
    fps = n_frames / elapsed if elapsed > 0 else 0.0
    lat = np.array(latencies_ms)

    out_dir = RESULTS_DIR / "baseline_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"pi_benchmark_{tag}_samples.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_seconds", "cpu_percent", "rss_mb"])
        w.writerows(samples)

    print(f"\n=== {tag} - Raspberry Pi benchmark ===")
    print(f"Duration:            {elapsed:.1f} s")
    print(f"Frames processed:    {n_frames}")
    print(f"FPS:                 {fps:.2f}")
    print(f"Latency avg:         {lat.mean():.1f} ms")
    print(f"Latency p95:         {np.percentile(lat, 95):.1f} ms")
    print(f"Hand detected:       {n_hand_detected}/{n_frames} frames ({100 * n_hand_detected / max(n_frames,1):.1f}%)")
    print(f"Errors:              {n_errors}")
    if samples:
        cpu_vals = [s[1] for s in samples]
        rss_vals = [s[2] for s in samples]
        print(f"CPU avg/max:         {np.mean(cpu_vals):.1f}% / {np.max(cpu_vals):.1f}%")
        print(f"RAM avg/max:         {np.mean(rss_vals):.1f} MB / {np.max(rss_vals):.1f} MB")
    print(f"Stability:           {'OK, no crash' if n_errors == 0 else f'{n_errors} frame errors'}")
    print(f"\nSaved per-second CPU/RAM samples to {out_dir / f'pi_benchmark_{tag}_samples.csv'}")


if __name__ == "__main__":
    main()
