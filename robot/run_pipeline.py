"""
Full deployed pipeline for whichever baseline wins Part 3's benchmark:

    Camera -> hand detection -> gesture recognition -> temporal smoothing
    -> gesture confirmed -> command -> robot/motor_controller.py -> wheels

By default this runs Baseline A (landmarks + MLP), since it's the fast/light
option most likely to win on a Pi 4. Pass --baseline b to run the CNN
instead, and --quantized/--backend if that CNN was quantized.

Wheel control is a differential-drive (skid-steer) 4-motor setup via
robot/motor_controller.py (gpiozero + L298N-style H-bridge -- see that
file's docstring for wiring). If gpiozero/GPIO hardware isn't available
(e.g. running this on a dev machine to test the vision pipeline only), pass
--dry-run and every command just prints instead of driving motors.

Safety: if the hand leaves the frame, or a gesture is confirmed as
UNKNOWN_LABEL (low confidence), the robot stops and the temporal smoother
resets -- a fresh confirmation is required before it moves again, so it
never keeps driving on a stale command once the camera loses the signal.

Usage:
    python robot/run_pipeline.py --baseline a
    python robot/run_pipeline.py --baseline a --dry-run   # no hardware, test on a dev machine
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import PROJECT_ROOT, CLASSES, UNKNOWN_LABEL, CONFIDENCE_THRESHOLD_A, CONFIDENCE_THRESHOLD_B
from common.hand_detector import HandDetector
from common.prediction import top_prediction
from common.camera import open_camera
from robot.temporal_smoothing import TemporalSmoother
from robot.motor_controller import MotorController

GESTURE_TO_COMMAND = {
    "left": "TURN_LEFT",
    "right": "TURN_RIGHT",
    "forward": "MOVE_FORWARD",
    "backward": "MOVE_BACKWARD",
    "stop": "STOP",
}

COMMAND_TO_ACTION = {
    "TURN_LEFT": MotorController.turn_left,
    "TURN_RIGHT": MotorController.turn_right,
    "MOVE_FORWARD": MotorController.forward,
    "MOVE_BACKWARD": MotorController.backward,
    "STOP": MotorController.stop,
}


def send_command(motors: MotorController, command: str):
    print(f">>> COMMAND: {command}")
    COMMAND_TO_ACTION[command](motors)


def build_baseline_a_predictor(threshold: float):
    import joblib

    from baseline_a_landmarks.extract_landmarks import normalize_landmarks

    model_dir = PROJECT_ROOT / "baseline_a_landmarks" / "model"
    clf = joblib.load(model_dir / "mlp.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")

    def predict(frame_rgb, landmarks):
        vec = normalize_landmarks(landmarks)
        X = scaler.transform(vec.reshape(1, -1))
        probs = clf.predict_proba(X)[0]
        label, _conf = top_prediction(probs, clf.classes_, threshold)
        return label

    return predict


def build_baseline_b_predictor(quantized: bool, backend: str, threshold: float):
    import torch
    from PIL import Image

    from baseline_b_cnn.crop_hands import bbox_from_landmarks, IMG_SIZE
    from baseline_b_cnn.train_mobilenet import build_model, EVAL_TFM

    model_dir = PROJECT_ROOT / "baseline_b_cnn" / "model"
    if quantized:
        torch.backends.quantized.engine = backend
        model = torch.jit.load(model_dir / "mobilenet_v3_small_int8.pt")
    else:
        model = build_model(len(CLASSES), pretrained=False)
        model.load_state_dict(torch.load(model_dir / "mobilenet_v3_small.pt", map_location="cpu"))
    model.eval()

    def predict(frame_bgr, landmarks):
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = bbox_from_landmarks(landmarks, w, h)
        if x1 <= x0 or y1 <= y0:
            return None
        crop = cv2.resize(frame_bgr[y0:y1, x0:x1], (IMG_SIZE, IMG_SIZE))
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = EVAL_TFM(pil_img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]
        label, _conf = top_prediction(probs, CLASSES, threshold)
        return label

    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["a", "b"], default="a")
    ap.add_argument("--quantized", action="store_true", help="baseline b only")
    ap.add_argument("--backend", choices=["qnnpack", "fbgemm"], default="qnnpack")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--window-size", type=int, default=5)
    ap.add_argument("--confirm-count", type=int, default=3)
    ap.add_argument("--confidence-threshold", type=float, default=None,
                     help="below this top-class probability, report UNKNOWN_LABEL instead of guessing. "
                          f"Defaults to CONFIDENCE_THRESHOLD_A ({CONFIDENCE_THRESHOLD_A}) for baseline a, "
                          f"CONFIDENCE_THRESHOLD_B ({CONFIDENCE_THRESHOLD_B}) for baseline b")
    ap.add_argument("--speed", type=float, default=0.65,
                     help="forward/backward motor speed 0.0-1.0")
    ap.add_argument("--turn-speed", type=float, default=0.4,
                     help="turn_left/turn_right motor speed 0.0-1.0 -- defaults lower than "
                          "--speed since pivoting in place is more prone to tipping/skidding "
                          "than driving straight")
    ap.add_argument("--left-trim", type=float, default=1.0,
                     help="left motor speed multiplier (0.0-1.0) -- trim the stronger side "
                          "down if the robot veers, e.g. --left-trim 0.8 if left is stronger")
    ap.add_argument("--right-trim", type=float, default=1.0,
                     help="right motor speed multiplier (0.0-1.0), same idea as --left-trim")
    ap.add_argument("--dry-run", action="store_true",
                     help="don't touch GPIO/motors -- print commands instead (for testing without a Pi/wheels)")
    args = ap.parse_args()

    if args.baseline == "a":
        threshold = args.confidence_threshold if args.confidence_threshold is not None else CONFIDENCE_THRESHOLD_A
        predict = build_baseline_a_predictor(threshold)
    else:
        threshold = args.confidence_threshold if args.confidence_threshold is not None else CONFIDENCE_THRESHOLD_B
        predict = build_baseline_b_predictor(args.quantized, args.backend, threshold)

    smoother = TemporalSmoother(window_size=args.window_size, confirm_count=args.confirm_count)
    motors = MotorController(speed=args.speed, turn_speed=args.turn_speed, dry_run=args.dry_run,
                              left_trim=args.left_trim, right_trim=args.right_trim)

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    detector = HandDetector(running_mode="IMAGE", min_detection_confidence=0.5)

    print(f"Running deployed pipeline (baseline {args.baseline}). Ctrl+C to stop.")
    hand_was_visible = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            # The training dataset was recorded in mirrored/selfie mode
            # (confirmed from a visible ID badge with reversed text in the
            # source footage), so live frames must be flipped the same way
            # -- otherwise left/right are swapped relative to training and
            # get systematically misclassified.
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = detector.detect(rgb)

            if landmarks is not None:
                hand_was_visible = True
                if args.baseline == "a":
                    raw_pred = predict(rgb, landmarks)
                else:
                    raw_pred = predict(frame, landmarks)

                if raw_pred is not None:
                    confirmed = smoother.push(raw_pred)
                    if confirmed is not None:
                        if confirmed == UNKNOWN_LABEL:
                            print("Gesture not recognized (low confidence) -- stopping")
                            send_command(motors, "STOP")
                        else:
                            send_command(motors, GESTURE_TO_COMMAND[confirmed])
            elif hand_was_visible:
                # Hand just left the frame -- stop rather than keep running
                # the last confirmed command, and require a fresh
                # confirmation before moving again.
                print("Hand lost -- stopping")
                send_command(motors, "STOP")
                smoother.reset()
                hand_was_visible = False
    except KeyboardInterrupt:
        pass
    finally:
        motors.close()
        cap.release()
        detector.close()


if __name__ == "__main__":
    main()
