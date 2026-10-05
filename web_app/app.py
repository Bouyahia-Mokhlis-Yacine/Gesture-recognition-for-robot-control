"""
Live demo: open the webcam and compare all available trained models --
Baseline A (landmarks + MLP), Baseline B (MobileNetV3-Small), and any of
Model C/D/E/F (EfficientNet-B0, ResNet18, MobileNetV2, EfficientNet-B1 --
common.config.MODEL_VARIANTS) that have been trained so far -- predicting
the SAME gestures in real time, side by side, in a browser. A model only
appears if its weight file exists on disk, so this works fine while C/D/E/F
are still being trained one at a time. Each model card has an on/off toggle
in the UI -- only "enabled" models are actually run per frame (not just
hidden), since running all 5 at once is too heavy on a slow machine. Only
A and B are enabled by default; toggle the rest on as needed. Reuses the
exact detection/prediction code from the baselines and
robot/temporal_smoothing.py -- this isn't a separate toy, it's the same
pipeline wired to a browser UI instead of benchmark_pi.py's stdout.

Usage:
    python web_app/app.py
Then open http://localhost:5000
"""
import sys
import threading
import time
from pathlib import Path

import cv2
import joblib
import torch
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import PROJECT_ROOT, CLASSES, CONFIDENCE_THRESHOLD_A, CONFIDENCE_THRESHOLD_B, MODEL_VARIANTS
from common.hand_detector import HandDetector
from common.prediction import top_prediction
from common.cnn_backbones import build_model as build_variant_model
from common.camera import open_camera
from baseline_a_landmarks.extract_landmarks import normalize_landmarks
from baseline_b_cnn.crop_hands import bbox_from_landmarks, IMG_SIZE
from baseline_b_cnn.train_mobilenet import build_model as build_mobilenet_v3, EVAL_TFM
from robot.temporal_smoothing import TemporalSmoother

app = Flask(__name__)

MODEL_A_DIR = PROJECT_ROOT / "baseline_a_landmarks" / "model"
MODEL_B_DIR = PROJECT_ROOT / "baseline_b_cnn" / "model"

CARD_COLOR = {
    "a": "#ffb020", "b": "#4fc3ff", "c": "#7ee787",
    "d": "#d2a8ff", "e": "#ff7b9c", "f": "#58e6d9",
}
CARD_LABEL = {"a": "Baseline A -- Landmarks + MLP", "b": "Baseline B -- MobileNetV3-Small"}
CARD_LABEL.update({letter: info["label"] for letter, info in MODEL_VARIANTS.items()})
CARD_DESC = {"a": "21 hand landmarks -> tiny MLP classifier", "b": "cropped hand image -> fine-tuned CNN"}
CARD_DESC.update({letter: f"cropped hand image -> fine-tuned {info['arch']}" for letter, info in MODEL_VARIANTS.items()})

mlp = joblib.load(MODEL_A_DIR / "mlp.joblib")
scaler = joblib.load(MODEL_A_DIR / "scaler.joblib")

# CNN-family models (b, c, d, e, f) -- only the ones actually trained so far
# get loaded, so the demo works while the rest are still training.
cnn_models = {}

b_path = MODEL_B_DIR / "mobilenet_v3_small.pt"
if b_path.exists():
    m = build_mobilenet_v3(len(CLASSES), pretrained=False)
    m.load_state_dict(torch.load(b_path, map_location="cpu"))
    m.eval()
    cnn_models["b"] = m

for letter, info in MODEL_VARIANTS.items():
    p = PROJECT_ROOT / info["dir"] / "model" / f"{info['arch']}.pt"
    if p.exists():
        m = build_variant_model(info["arch"], len(CLASSES), pretrained=False)
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m.eval()
        cnn_models[letter] = m

CNN_LETTERS = [letter for letter in ["b", "c", "d", "e", "f"] if letter in cnn_models]
ACTIVE_MODELS = ["a"] + CNN_LETTERS
print(f"[web_app] Loaded models: {ACTIVE_MODELS} "
      f"({len(CNN_LETTERS)} of {len(MODEL_VARIANTS) + 1} CNN-family models trained so far)")

detector = HandDetector(running_mode="VIDEO", min_detection_confidence=0.5)
smoothers = {letter: TemporalSmoother(window_size=8, confirm_count=5) for letter in ACTIVE_MODELS}

EMPTY_RESULT = {"pred": None, "conf": 0.0, "confirmed": None, "latency_ms": 0.0}

# MediaPipe hand detection dominates per-frame cost on a Pi 4 (~124ms with
# a proper power supply/full clock speed, was 250-370ms throttled at
# 900MHz on an undervolted supply -- measured directly on a Pi 4 with
# `top -H` confirming detection already uses all 4 cores via XNNPACK, so
# there's no free threading win left to take there). Only re-running
# detection every Nth frame -- and reusing the last known
# landmarks/predictions on the frames in between -- trades gesture
# responsiveness (updates less often) for a much smoother video feed and
# higher overall throughput, since skip-frames are cheap (camera read +
# flip + JPEG encode only, no detection/classification at all).
DETECT_EVERY_N_FRAMES = 3

# Running all loaded models every frame is heavy on a slow machine -- only
# "enabled" models actually get predicted on each frame; the rest are
# skipped entirely (not just hidden). Default to the two lightest so the
# demo isn't laggy out of the box; toggle the others on from the UI.
DEFAULT_ENABLED = {letter for letter in ("a", "b") if letter in ACTIVE_MODELS}
enabled_lock = threading.Lock()
enabled_models = set(DEFAULT_ENABLED)

state_lock = threading.Lock()
state = {
    "frame_jpeg": None,
    "frame_version": 0,  # bumped each time capture_loop writes a new JPEG,
    # so gen_frames only sends genuinely new frames instead of re-sending
    # the same bytes on every poll -- capture_loop only produces a new
    # frame every ~33ms (30fps) but was being polled every 10ms, so up to
    # ~2/3 of the video bandwidth was identical bytes retransmitted for no
    # reason. Wasteful on any link, actively harmful on flaky/congested
    # WiFi (this Pi's `/proc/net/wireless` shows real retry counts despite
    # a strong signal, i.e. contention/interference, not distance).
    "fps": 0.0,
    "hand_detected": False,
    "models": {letter: dict(EMPTY_RESULT) for letter in ACTIVE_MODELS},
    "enabled": sorted(enabled_models),
}
reset_flag = threading.Event()


def predict_a(landmarks):
    t0 = time.time()
    vec = normalize_landmarks(landmarks)
    X = scaler.transform(vec.reshape(1, -1))
    probs = mlp.predict_proba(X)[0]
    label, conf = top_prediction(probs, [str(c) for c in mlp.classes_], CONFIDENCE_THRESHOLD_A)
    return label, conf, (time.time() - t0) * 1000


def predict_cnn(model, frame_bgr, landmarks):
    t0 = time.time()
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = bbox_from_landmarks(landmarks, w, h)
    if x1 <= x0 or y1 <= y0:
        return None, 0.0, 0.0, None

    crop = cv2.resize(frame_bgr[y0:y1, x0:x1], (IMG_SIZE, IMG_SIZE))
    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = EVAL_TFM(pil_img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    label, conf = top_prediction(probs, CLASSES, CONFIDENCE_THRESHOLD_B)
    return label, conf, (time.time() - t0) * 1000, (x0, y0, x1, y1)


def draw_overlay(frame, landmarks, bbox, results, current_enabled):
    h, w = frame.shape[:2]
    if landmarks is not None:
        for lm in landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 220, 0), -1)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 120, 0), 2)

    def draw_label(y, text, color_bgr):
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 1, cv2.LINE_AA)

    i = 0
    for letter in ACTIVE_MODELS:
        if letter not in current_enabled:
            continue
        info = results[letter]
        txt = f"{letter.upper()}: {info['pred']} ({info['conf']*100:.0f}%)" if info["pred"] else f"{letter.upper()}: -"
        hex_color = CARD_COLOR[letter].lstrip("#")
        r, g, b = (int(hex_color[j:j + 2], 16) for j in (0, 2, 4))
        draw_label(30 + i * 24, txt, (b, g, r))  # BGR for OpenCV
        i += 1
    return frame


def capture_loop():
    cap = open_camera(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0)")

    # Cached across skip-frames -- reused as-is until the next detection frame.
    landmarks = None
    results = {letter: dict(EMPTY_RESULT) for letter in ACTIVE_MODELS}
    bbox = None
    current_enabled = set()

    prev_t = time.time()
    frame_idx = 0
    fps_ema = 0.0  # smoothed -- instantaneous 1/dt bounces wildly between
    # cheap skip-frames and expensive detect-frames (e.g. ~42fps vs ~7fps
    # at N=2), which isn't a meaningful "current speed" reading on its own.
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        if reset_flag.is_set():
            for s in smoothers.values():
                s.reset()
            reset_flag.clear()

        if frame_idx % DETECT_EVERY_N_FRAMES == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = detector.detect(rgb)

            with enabled_lock:
                current_enabled = set(enabled_models)

            results = {letter: dict(EMPTY_RESULT) for letter in ACTIVE_MODELS}
            bbox = None

            if landmarks is not None:
                if "a" in current_enabled:
                    pred, conf, lat = predict_a(landmarks)
                    smoothers["a"].push(pred)
                    results["a"] = {"pred": pred, "conf": conf, "confirmed": smoothers["a"].last_confirmed, "latency_ms": lat}

                for letter in CNN_LETTERS:
                    if letter not in current_enabled:
                        continue
                    c_pred, c_conf, c_lat, c_bbox = predict_cnn(cnn_models[letter], frame, landmarks)
                    if c_bbox is not None:
                        bbox = c_bbox
                    if c_pred is not None:
                        smoothers[letter].push(c_pred)
                        results[letter] = {"pred": c_pred, "conf": c_conf,
                                            "confirmed": smoothers[letter].last_confirmed, "latency_ms": c_lat}
        frame_idx += 1

        frame = draw_overlay(frame, landmarks, bbox, results, current_enabled)

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
            state["models"] = results
            state["enabled"] = sorted(current_enabled)


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
    with enabled_lock:
        current_enabled = set(enabled_models)
    models_meta = [
        {"id": letter, "label": CARD_LABEL[letter], "desc": CARD_DESC[letter], "color": CARD_COLOR[letter],
         "enabled": letter in current_enabled}
        for letter in ACTIVE_MODELS
    ]
    return render_template("index.html", classes=CLASSES, models=models_meta)


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    with state_lock:
        return jsonify({
            "fps": round(state["fps"], 1),
            "hand_detected": state["hand_detected"],
            "models": state["models"],
            "enabled": state["enabled"],
        })


@app.route("/reset", methods=["POST"])
def reset():
    reset_flag.set()
    return jsonify({"ok": True})


@app.route("/toggle", methods=["POST"])
def toggle():
    data = request.get_json(silent=True) or {}
    letter = data.get("id")
    on = bool(data.get("enabled"))
    if letter not in ACTIVE_MODELS:
        return jsonify({"ok": False, "error": "unknown model id"}), 400
    with enabled_lock:
        if on:
            enabled_models.add(letter)
        else:
            enabled_models.discard(letter)
        current = sorted(enabled_models)
    return jsonify({"ok": True, "enabled": current})


if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
