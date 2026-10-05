# Technical notes

More detail than the README on the dataset, models, results and deployment.
Shared constants (classes, thresholds, paths, seed) are in `common/config.py`.

## 1. Dataset

- 24 videos of one person (robot USB webcam and laptop camera), recorded in
  mirrored "selfie" mode, about 10 minutes in total.
- 2,270 extracted frames in `extracted_frames/<class>/`: left 475, right 471,
  forward 441, backward 401, stop 482.
- One stratified, seeded (42) split shared by every model
  (`data/prepare_dataset.py` → `data/splits/{train,val,test}.csv`).

| Stage | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Extracted frames | 1,586 | 339 | 345 | 2,270 |
| Hand detected by MediaPipe (landmarks, Model A) | 1,253 | 280 | 261 | 1,794 |
| Hand crops after manual cleaning (Models B–E) | 1,068 | 234 | 223 | 1,525 |

Blurred or badly detected crops were removed by hand;
`baseline_b_cnn/sync_crops_csv.py` rebuilds the crop manifests from the files
on disk. The manifests store paths relative to the repository root, so scripts
must be run from there.

## 2. Preprocessing

All models use MediaPipe `HandLandmarker` (`models/hand_landmarker.task`,
one hand, confidence 0.5), wrapped in `common/hand_detector.py`.

- **Model A:** the 21 landmarks are made wrist-relative and divided by the hand's
  extent (largest wrist-to-landmark distance) → 63 values, then standardized
  with a `StandardScaler` fitted on the training set. They are deliberately not
  made rotation- or mirror-invariant, because orientation separates left, right
  and forward.
- **Models B–E:** square crop around the landmarks with a 30 % margin, resized
  to 224×224, ImageNet normalization. Training augmentation is limited to
  `RandomResizedCrop` (scale 0.8–1.0) and `ColorJitter` (brightness and
  contrast 0.2). Horizontal flips are not used (they turn a left gesture into a
  right one), and rotation augmentation lowered test accuracy from 93.7 % to
  90.6 %.

## 3. Models and training

| Model | Details |
|---|---|
| A | scikit-learn `MLPClassifier`, one hidden layer of 128 ReLU units (8,837 parameters), α = 1e-3, Adam 1e-3, 60 epochs, seed 42. Chosen by `tune_mlp.py` (8 layer configurations × 4 α values × 5 seeds; all between 93.1 % and 94.7 % validation accuracy). |
| B–E | torchvision backbones pretrained on ImageNet, new 5-class linear head. Stage 1: 15 epochs with the backbone frozen (Adam 1e-3). Stage 2: 15 epochs with the last 6 blocks unfrozen (Adam 1e-5). Batch 32, cross-entropy. Shared code in `common/cnn_training.py` and `common/cnn_backbones.py`. |
| B INT8 | `baseline_b_cnn/quantize.py`: static post-training quantization with PyTorch FX graph mode (`qnnpack` or `fbgemm`), calibrated on training crops. |

## 4. Results

Test sets: 261 landmark samples (A) and 223 cleaned crops (B–E). The INT8
model was evaluated on the 261 uncleaned crops, where its FP32 counterpart
scores 93.1 %. Metrics, confusion matrices and plots are in `results/`.

| Model | Accuracy | Macro-F1 | Parameters | File |
|---|---:|---:|---:|---:|
| A: landmarks + MLP | 91.57 % | 91.65 % | 8.8 k | 150 KB |
| B: MobileNetV3-Small | 93.72 % | 93.67 % | 1.52 M | 6.2 MB |
| B: INT8 | 29.50 % | 22.52 % | 1.52 M | 2.1 MB |
| C: EfficientNet-B0 | 90.13 % | 90.22 % | 4.01 M | 16.4 MB |
| D: ResNet18 | 95.07 % | 94.92 % | 11.18 M | 44.8 MB |
| E: MobileNetV2 | 91.93 % | 92.00 % | 2.23 M | 9.2 MB |

Per-class F1 (%):

| Model | left | right | forward | backward | stop |
|---|---:|---:|---:|---:|---:|
| A | 90.1 | 93.1 | 93.7 | 94.6 | 86.8 |
| B | 91.3 | 92.8 | 95.7 | 98.9 | 89.7 |
| C | 85.7 | 86.0 | 91.7 | 98.9 | 88.9 |
| D | 92.3 | 98.0 | 93.6 | 100.0 | 90.7 |
| E | 88.9 | 90.0 | 90.7 | 100.0 | 90.4 |

The INT8 model collapses towards one class (153 of 261 samples predicted
"forward", "right" never predicted). MobileNetV3's hard-swish activations and
squeeze-and-excitation layers are poorly represented by per-tensor 8-bit
ranges after post-training quantization, so the FP32 model is used.

**Additional landmark experiments** (5 seeds each, same split): bone vectors,
joint angles, an orientation-free canonical shape and pose-only features did
not beat the normalized landmarks, and removing orientation clearly hurt left
and right. Per-finger encoders beat one encoder shared by all fingers at the same
parameter count (+1.0 point), and a thumb-specific encoder recovered the loss
(91.65 ± 0.50 %), about the same as the deployed MLP.

## 5. Deployment

**Model choice.** Every model needs the MediaPipe detection first, which takes
about 124 ms per frame on the Pi 4 and uses all four cores. The MLP adds almost
nothing to that, so Model A is deployed. Model B remains selectable with
`--baseline b`.

**Performance on the Pi.** Capture at 1280×720 in MJPEG with a one-frame buffer
(`common/camera.py`). In the standalone Pi interface, detection runs on every
third frame. A stable power supply matters, because under-voltage throttling made
detection two to three times slower.

**Gesture → command.**

| Step | Setting |
|---|---|
| Mirroring | every live frame is flipped horizontally (the training videos are mirrored) |
| Confidence rejection | top probability < 0.6 → `unknown` (`common/prediction.py`) |
| Temporal smoothing | gesture confirmed when 3 of the last 5 predictions agree (5 of 8 in the web app), sent only when it changes (`robot/temporal_smoothing.py`) |
| Motion | forward/backward duty 0.65, pivot turns 0.4, per-side trims (`robot/motor_controller.py`) |

The robot stops when the hand leaves the frame, when an `unknown` gesture is
confirmed, when Stop is pressed in the interface, and when the program is
terminated (SIGINT/SIGTERM handlers stop the motors).

**Networked configurations** (`interfaces/1_laptop_only`,
`interfaces/3_laptop_with_robot_camera`): the laptop sends commands to the Pi
over HTTP (`POST /command`, port 5001) on each change and every 0.5 s while a
gesture is held. A watchdog on the Pi stops the motors after 1.5 s without
commands. The laptop finds the Pi by its MAC vendor with `nmap` (or use
`--pi-host`) and rediscovers it after 4 failed heartbeats. Configuration 3 also
streams the robot camera (`/video_feed`) to the laptop.

**Raspberry Pi installation notes** (`requirements-run.txt`):

- install `torch`/`torchvision` from the CPU-only index first, otherwise pip
  pulls large CUDA wheels;
- `mediapipe==0.10.18`: newer wheels crash on the Pi 4's Cortex-A72 (no AES
  instructions);
- then `pip install "numpy>=2"`, needed to load the saved `.joblib` models;
- `lgpio` provides the PWM-capable GPIO backend for `gpiozero`.

## 6. Limitations

- All data comes from a single person, so the accuracies say nothing about other
  users, hands or environments.
- The split is random at frame level, so near-identical frames of one video can
  appear in both training and test sets, which makes the test accuracy optimistic.
- "stop" and the thumb gestures account for most errors.
- On the Pi alone, detection limits recognition to a few frames per second.
