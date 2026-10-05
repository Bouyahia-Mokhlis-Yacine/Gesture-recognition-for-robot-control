# Real-Time Gesture Recognition for Raspberry Pi Robot Control

Control a four-wheel Raspberry Pi 4 robot with five static hand gestures shown
to a camera. A pretrained MediaPipe hand detector extracts 21 hand landmarks, a
small MLP classifies them, and the result is smoothed over time and turned into
motor commands. Several transfer-learned CNNs were also trained and compared;
the landmark classifier was deployed because it is almost as accurate at a
fraction of the computational cost on the Pi.

Developed during an internship at the Embedded Systems Laboratory, Research
Center in Industrial Technologies (CRTI).

- Demonstration video: [INSERT GESTURE RECOGNITION YOUTUBE LINK]
- Internship report: [INSERT INTERNSHIP REPORT LINK]

| Gesture | Hand shape | Robot action |
|---|---|---|
| forward | thumbs-up | drive forward |
| backward | open palm | drive backward |
| left | thumb pointing left | turn left on the spot |
| right | thumb pointing right | turn right on the spot |
| stop | closed fist | stop |

## Features

- Real-time recognition with MediaPipe landmarks and an 8.8 k-parameter MLP
  (91.6 % test accuracy).
- Comparison with four transfer-learned CNNs and an INT8-quantized model.
- Confidence rejection (`unknown` gesture) and majority-vote temporal smoothing,
  so a single wrong frame never moves the robot.
- Three deployment configurations: laptop only, Pi standalone, and laptop with
  the robot's camera.
- Fail-safe behaviour: the robot stops on hand loss, unknown gesture, manual
  stop, network loss (1.5 s watchdog) and program exit.
- A web application that compares the models live on the same camera feed.

## Pipeline

```
Camera
   ↓
Mirror frame (training data was recorded in selfie mode)
   ↓
MediaPipe hand detection → 21 hand landmarks
   ↓
Normalization (wrist-relative, scale-invariant) → MLP classifier
   ↓
Confidence threshold (0.6) + temporal smoothing (3 of last 5)
   ↓
Robot command (MOVE_FORWARD, TURN_LEFT, ...)
   ↓
Motor controller (GPIO PWM → 2 × L298N → 4 DC motors)
```

## Dataset

- 2,270 frames extracted from 24 webcam videos (`extracted_frames/<class>/`),
  roughly balanced over the 5 classes.
- One stratified 70 / 15 / 15 train / validation / test split (seed 42), shared
  by all models (`data/splits/`).
- MediaPipe detects a hand in 1,794 frames (landmark dataset, `data/landmarks/`);
  1,525 hand crops remain after manual cleaning (`data/crops/`).
- **Limitation:** all samples come from a single person, so the results describe
  recognition of new frames of a known user, not of new users.

## Model comparison

| Model | Approach | Test accuracy | Size |
|---|---|---:|---:|
| **A** | MediaPipe landmarks + MLP (**deployed**) | 91.6 % | 150 KB |
| B | MobileNetV3-Small, transfer learning | 93.7 % | 6.2 MB |
| B INT8 | MobileNetV3-Small, post-training INT8 | 29.5 % | 2.1 MB |
| C | EfficientNet-B0, transfer learning | 90.1 % | 16.4 MB |
| D | ResNet18, transfer learning | 95.1 % | 44.8 MB |
| E | MobileNetV2, transfer learning | 91.9 % | 9.2 MB |

Model A is tested on 261 landmark samples and the CNNs on 223 cleaned crops.
INT8 quantization made MobileNetV3's predictions collapse, so it was not used.
ResNet18 is the most accurate, but every model needs the MediaPipe detection
(about 124 ms per frame on the Pi 4), and the MLP adds almost nothing to that
cost. Details: `docs/technical.md`.

## Hardware

- Raspberry Pi 4 Model B
- USB webcam (or the laptop camera in the networked setups)
- 2 × L298N H-bridges, 4 DC gear motors (four-wheel skid-steer), separate motor
  battery
- Optional laptop for the networked configurations

Motor pins are defined in `robot/motor_controller.py` (BCM numbering).

## Installation

Python 3.11. Download the MediaPipe model bundle if `models/hand_landmarker.task`
is missing:

```bash
mkdir -p models
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

**Laptop (training, evaluation, demos):**

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision   # skip on a CUDA machine
pip install -r requirements.txt
```

**Raspberry Pi (running only):**

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements-run.txt
pip install "numpy>=2"
```

`requirements-run.txt` pins `mediapipe==0.10.18`, because newer versions crash
on the Pi 4.

## Running

Run all commands from the repository root.

**Robot (Pi standalone, or any machine with `--dry-run`):**

```bash
python robot/run_pipeline.py --baseline a              # camera → recognition → motors
python robot/run_pipeline.py --baseline a --dry-run    # print commands instead of driving motors
python interfaces/2_pi_standalone/pi_app.py            # same, with a web viewer on port 5000
```

**Networked setups (recognition on the laptop, motors on the Pi):**

```bash
# on the Pi
python interfaces/1_laptop_only/pi_server.py
# on the laptop (finds the Pi automatically, or use --pi-host <address>)
python interfaces/1_laptop_only/remote_app.py
```

Use `interfaces/3_laptop_with_robot_camera/` (same two commands) to also stream
the robot's camera to the laptop. The laptop interface is at
http://localhost:5000.

**Live model comparison:**

```bash
python web_app/app.py     # http://localhost:5000
```

**Reproducing the models:**

```bash
python data/prepare_dataset.py                       # split (already provided in data/splits/)
python baseline_a_landmarks/extract_landmarks.py     # landmarks (already provided)
python baseline_a_landmarks/train_mlp.py
python baseline_a_landmarks/evaluate.py              # → results/baseline_a/
python baseline_b_cnn/crop_hands.py                  # hand crops (already provided, cleaned)
python baseline_b_cnn/train_mobilenet.py --frozen-epochs 15 --finetune-epochs 15 --unfreeze-blocks 6
python baseline_b_cnn/evaluate.py                    # → results/baseline_b/
python baseline_b_cnn/quantize.py --backend qnnpack
python baseline_b_cnn/evaluate.py --quantized --backend qnnpack
python baseline_c_cnn/train.py && python baseline_c_cnn/evaluate.py   # same for d and e
```

Re-running `crop_hands.py` regenerates all crops, including the ones removed
during manual cleaning. The trained models are included in each `model/` folder.

## Repository structure

```
gesture-recognition/
├── extracted_frames/        dataset frames (one folder per gesture)
├── data/                    split, landmark and crop datasets
├── models/                  MediaPipe hand-landmarker bundle
├── common/                  shared config, detector, camera, metrics, training code
├── baseline_a_landmarks/    Model A: landmarks + MLP (deployed)
├── baseline_b_cnn/          Model B: MobileNetV3-Small (+ INT8)
├── baseline_c_cnn/ … baseline_e_cnn/   Models C–E
├── results/                 metrics and confusion matrices
├── robot/                   temporal smoothing, motor controller, robot pipeline
├── interfaces/              three deployment configurations
├── web_app/                 live model comparison
├── docs/technical.md        technical notes
├── Dockerfile               container for the comparison web app
└── requirements*.txt
```

## Status

This project is complete.
