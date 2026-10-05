"""
Interface 2 -- Pi standalone: fully self-contained pipeline + live web
viewer, running entirely on the Raspberry Pi. Camera capture, hand
detection, Baseline A (landmarks + MLP) gesture prediction, temporal
smoothing, and motor control (direct GPIO via robot/motor_controller.py,
no network hop) all happen in one process here. The Flask app is just a
viewer on top of that same loop -- open it from any browser on the network
to see the live camera feed with landmark overlay, the raw/confirmed
gesture, and the exact command being sent to the motors, in real time.

This is robot/run_pipeline.py's camera-in/motors-out logic with a viewer
UI grafted on. Unlike interfaces/1_laptop_only/ and
interfaces/3_laptop_with_robot_camera/, there's no HTTP command channel
to a separate pi_server.py process, so none of that script's network
discovery/heartbeat/watchdog machinery is needed here: a confirmed
gesture just calls the MotorController method directly, exactly like
run_pipeline.py does.

Usage:
    python interfaces/2_pi_standalone/pi_app.py
Then open http://<this Pi's IP>:5000 from any browser on the network.
"""
import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import joblib
from flask import Flask, Response, jsonify, render_template

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common.config import PROJECT_ROOT, CONFIDENCE_THRESHOLD_A, UNKNOWN_LABEL
from common.hand_detector import HandDetector
from common.prediction import top_prediction
from common.camera import open_camera
from baseline_a_landmarks.extract_landmarks import normalize_landmarks
from robot.temporal_smoothing import TemporalSmoother
from robot.motor_controller import MotorController

app = Flask(__name__)

MODEL_A_DIR = PROJECT_ROOT / "baseline_a_landmarks" / "model"

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

mlp = joblib.load(MODEL_A_DIR / "mlp.joblib")
scaler = joblib.load(MODEL_A_DIR / "scaler.joblib")

detector = HandDetector(running_mode="VIDEO", min_detection_confidence=0.5)

motors = None  # set in main() from CLI args
smoother = None  # set in main() from --window-size/--confirm-count
confidence_threshold = CONFIDENCE_THRESHOLD_A  # overridable via --confidence-threshold

state_lock = threading.Lock()
state = {
    "frame_jpeg": None,
    "frame_version": 0,
    "fps": 0.0,
    "hand_detected": False,
    "pred": None,
    "conf": 0.0,
    "confirmed": None,
    "sent_command": None,
    "dry_run": False,
}
reset_flag = threading.Event()
manual_stop_flag = threading.Event()


# MediaPipe hand detection dominates per-frame cost on a Pi 4 (~124ms with
# a proper power supply/full clock speed -- confirmed via vcgencmd on this
# Pi: throttled=0x0, full 1.8GHz, 54.5'C, so it's not power/thermal, just
# the model). Only re-running detection every Nth frame -- and reusing the
# last known landmarks/prediction on the frames in between -- trades
# gesture responsiveness (confirmation can lag by up to N-1 frames) for
# much higher overall throughput, since skip-frames are cheap (camera read
# + flip + JPEG encode only, no detection/classification at all). Same
# technique already used in web_app/app.py.
DETECT_EVERY_N_FRAMES = 3


def predict_a(landmarks):
    vec = normalize_landmarks(landmarks)
    X = scaler.transform(vec.reshape(1, -1))
    probs = mlp.predict_proba(X)[0]
    return top_prediction(probs, [str(c) for c in mlp.classes_], confidence_threshold)


def draw_overlay(frame, landmarks, pred, conf, confirmed):
    h, w = frame.shape[:2]
    if landmarks is not None:
        for lm in landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 220, 0), -1)

    def draw_label(y, text, color_bgr):
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 1, cv2.LINE_AA)

    raw_txt = f"raw: {pred} ({conf*100:.0f}%)" if pred else "raw: -"
    draw_label(30, raw_txt, (255, 200, 0))
    draw_label(56, f"confirmed: {confirmed or '-'}", (0, 220, 0))
    return frame


def capture_loop(camera_index):
    cap = open_camera(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    hand_was_visible = False
    last_sent_command = None
    frame_idx = 0

    # Cached across skip-frames -- reused as-is until the next detection frame.
    landmarks = None
    pred = conf = confirmed = None

    prev_t = time.time()
    fps_ema = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        # Training dataset was recorded in mirrored/selfie mode -- flip
        # live frames to match, or left/right get swapped (see common/camera.py).
        frame = cv2.flip(frame, 1)

        if reset_flag.is_set():
            smoother.reset()
            hand_was_visible = False
            reset_flag.clear()

        current_command = None

        if frame_idx % DETECT_EVERY_N_FRAMES == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = detector.detect(rgb)
            pred = conf = confirmed = None

            if landmarks is not None:
                hand_was_visible = True
                pred, conf = predict_a(landmarks)
                smoother.push(pred)
                confirmed = smoother.last_confirmed
                if confirmed is not None:
                    current_command = "STOP" if confirmed == UNKNOWN_LABEL else GESTURE_TO_COMMAND[confirmed]
            elif hand_was_visible:
                # Hand just left the frame -- stop immediately, require a
                # fresh confirmation before moving again.
                current_command = "STOP"
                smoother.reset()
                hand_was_visible = False
        frame_idx += 1

        if manual_stop_flag.is_set():
            current_command = "STOP"
            smoother.reset()
            hand_was_visible = False
            manual_stop_flag.clear()

        if current_command is not None and current_command != last_sent_command:
            COMMAND_TO_ACTION[current_command](motors)
            last_sent_command = current_command

        frame = draw_overlay(frame, landmarks, pred, conf, confirmed)

        now = time.time()
        instant_fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        fps_ema = instant_fps if fps_ema == 0.0 else 0.1 * instant_fps + 0.9 * fps_ema

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with state_lock:
            if ok:
                state["frame_jpeg"] = buf.tobytes()
                state["frame_version"] += 1
            state["fps"] = fps_ema
            state["hand_detected"] = landmarks is not None
            state["pred"] = pred
            state["conf"] = conf or 0.0
            state["confirmed"] = confirmed
            state["sent_command"] = last_sent_command


def gen_frames():
    last_version = -1
    while True:
        with state_lock:
            frame = state["frame_jpeg"]
            version = state["frame_version"]
        if frame is None or version == last_version:
            time.sleep(0.01)
            continue
        last_version = version
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


@app.route("/")
def index():
    return render_template("pi_index.html", dry_run=motors.dry_run)


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    with state_lock:
        return jsonify({
            "fps": round(state["fps"], 1),
            "hand_detected": state["hand_detected"],
            "pred": state["pred"],
            "conf": round(state["conf"], 2),
            "confirmed": state["confirmed"],
            "sent_command": state["sent_command"],
            "dry_run": state["dry_run"],
        })


@app.route("/reset", methods=["POST"])
def reset():
    reset_flag.set()
    return jsonify({"ok": True})


@app.route("/stop", methods=["POST"])
def manual_stop():
    manual_stop_flag.set()
    COMMAND_TO_ACTION["STOP"](motors)
    return jsonify({"ok": True})


def main():
    global motors, smoother, confidence_threshold
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--port", type=int, default=5000, help="port for the web interface")
    ap.add_argument("--window-size", type=int, default=5)
    ap.add_argument("--confirm-count", type=int, default=3)
    ap.add_argument("--confidence-threshold", type=float, default=None,
                     help=f"below this top-class probability, report UNKNOWN_LABEL instead of "
                          f"guessing. Defaults to CONFIDENCE_THRESHOLD_A ({CONFIDENCE_THRESHOLD_A})")
    ap.add_argument("--speed", type=float, default=0.65, help="forward/backward motor speed 0.0-1.0")
    ap.add_argument("--turn-speed", type=float, default=0.4, help="turn_left/turn_right motor speed 0.0-1.0")
    ap.add_argument("--left-trim", type=float, default=1.0)
    ap.add_argument("--right-trim", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true",
                     help="don't touch GPIO/motors -- print commands instead (for testing without wheels)")
    args = ap.parse_args()

    if args.confidence_threshold is not None:
        confidence_threshold = args.confidence_threshold

    smoother = TemporalSmoother(window_size=args.window_size, confirm_count=args.confirm_count)
    motors = MotorController(speed=args.speed, turn_speed=args.turn_speed, dry_run=args.dry_run,
                              left_trim=args.left_trim, right_trim=args.right_trim)
    state["dry_run"] = motors.dry_run

    # Explicit handlers for both signals rather than relying on Flask/
    # Python's default SIGINT->KeyboardInterrupt handling: the capture_loop
    # background thread is CPU-heavy (camera + MediaPipe + sklearn running
    # continuously), which can starve the main thread of the GIL time it
    # needs to notice a pending signal -- observed directly in the
    # interfaces/1_laptop_only/remote_app.py-style scripts, whose
    # similarly-heavy thread made Ctrl+C silently no-op (needed SIGTERM
    # instead) despite identical
    # try/except KeyboardInterrupt structure. Since this script drives real
    # motors, skipping the `motors.close()` cleanup on an unclean exit
    # would leave the robot's last command running -- so both signals get
    # an explicit handler that stops the motors and exits immediately,
    # instead of hoping a raised KeyboardInterrupt gets processed in time.
    def _shutdown(signum, frame):
        print(f"\n[pi_app] Received signal {signum} -- stopping motors and exiting")
        motors.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    threading.Thread(target=capture_loop, args=(args.camera,), daemon=True).start()
    print(f"[pi_app] Running (dry_run={motors.dry_run}). Open http://<this Pi's IP>:{args.port}")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
