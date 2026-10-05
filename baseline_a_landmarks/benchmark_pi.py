"""
Baseline A - Step 8: run the FULL pipeline (camera -> hand landmarks -> MLP
-> prediction) live and measure FPS, per-frame latency, CPU%, RAM, and
stability. Run this ON the Raspberry Pi, not on a dev machine.

Usage:
    python baseline_a_landmarks/benchmark_pi.py --seconds 60
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import PROJECT_ROOT, RESULTS_DIR
from common.hand_detector import HandDetector
from common.camera import open_camera
from baseline_a_landmarks.extract_landmarks import normalize_landmarks

MODEL_DIR = PROJECT_ROOT / "baseline_a_landmarks" / "model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0, help="how long to run the benchmark")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    clf = joblib.load(MODEL_DIR / "mlp.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    detector = HandDetector(running_mode="IMAGE", min_detection_confidence=0.5)

    process = psutil.Process()
    process.cpu_percent()  # first call primes the internal counter

    latencies_ms = []
    n_frames = 0
    n_hand_detected = 0
    n_errors = 0
    samples = []  # (t, cpu%, rss_mb)

    print(f"Running benchmark for {args.seconds:.0f}s on camera {args.camera}...")
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

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks = detector.detect(rgb)

                if landmarks is not None:
                    n_hand_detected += 1
                    vec = normalize_landmarks(landmarks)
                    X = scaler.transform(vec.reshape(1, -1))
                    _ = clf.predict(X)[0]

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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS_DIR / "baseline_a"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "pi_benchmark_samples.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_seconds", "cpu_percent", "rss_mb"])
        w.writerows(samples)

    print("\n=== Baseline A - Raspberry Pi benchmark ===")
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
    print(f"\nSaved per-second CPU/RAM samples to {out_dir / 'pi_benchmark_samples.csv'}")


if __name__ == "__main__":
    main()
