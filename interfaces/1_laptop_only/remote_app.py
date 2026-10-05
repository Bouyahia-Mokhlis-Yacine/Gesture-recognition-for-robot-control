"""
Interface 1 -- Laptop-only: run the camera + hand detection + Baseline A
(landmarks + MLP) gesture prediction HERE, on the laptop, and drive the
Pi's wheels over the network. Only commands cross the network, not video
-- the Pi in this interface has no camera of its own running (see
interfaces/3_laptop_with_robot_camera/ for that: same idea, plus the Pi
streams its own forward-facing camera back so you can see what's in front
of the robot too).

A laptop's CPU runs MediaPipe + the classifier far faster than a Pi 4, so
this is not a performance regression vs. robot/run_pipeline.py running on
the Pi -- it should be faster end to end.

The Pi side is pi_server.py (this same folder): a small Flask receiver
that turns POSTed commands into robot/motor_controller.py calls, with its
own watchdog that stops the motors if commands stop arriving. To make
that watchdog meaningful, this app resends the CURRENT command on a
heartbeat (HEARTBEAT_INTERVAL) while a gesture is held, not only when it
changes -- so "hand still showing forward" keeps the robot moving, and
losing the hand (or this process, or the network) reliably stops it.

By default the Pi is auto-discovered on the local /24 by matching its MAC
vendor ("Raspberry Pi Trading") via `nmap -sn`, instead of depending on a
fixed --pi-host. Both raspberrypi.local (mDNS) and a hardcoded IP have
proven unreliable on networks that reassign a new subnet on reconnect
(e.g. a mobile hotspot) -- MAC-based discovery doesn't care what the
subnet or hostname currently are. Requires `nmap` to be installed; pass
--pi-host explicitly to skip discovery (e.g. if nmap isn't available, or
there's more than one Pi on the network).

Usage:
    python interfaces/1_laptop_only/remote_app.py                       # auto-discover
    python interfaces/1_laptop_only/remote_app.py --pi-host 192.168.1.42 # explicit
Then open http://localhost:5000
"""
import argparse
import ipaddress
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import joblib
import requests
from flask import Flask, Response, jsonify, render_template

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common.config import PROJECT_ROOT, CONFIDENCE_THRESHOLD_A, UNKNOWN_LABEL
from common.hand_detector import HandDetector
from common.prediction import top_prediction
from common.camera import open_camera
from baseline_a_landmarks.extract_landmarks import normalize_landmarks
from robot.temporal_smoothing import TemporalSmoother

app = Flask(__name__)

MODEL_A_DIR = PROJECT_ROOT / "baseline_a_landmarks" / "model"

GESTURE_TO_COMMAND = {
    "left": "TURN_LEFT",
    "right": "TURN_RIGHT",
    "forward": "MOVE_FORWARD",
    "backward": "MOVE_BACKWARD",
    "stop": "STOP",
}

# Must stay comfortably below pi_server.py's --timeout (default 1.5s) so a
# held gesture never triggers the Pi's own-safety auto-stop.
HEARTBEAT_INTERVAL = 0.5
COMMAND_REQUEST_TIMEOUT = 0.5  # seconds -- a slow/dead link shouldn't stall the frame loop

mlp = joblib.load(MODEL_A_DIR / "mlp.joblib")
scaler = joblib.load(MODEL_A_DIR / "scaler.joblib")

detector = HandDetector(running_mode="VIDEO", min_detection_confidence=0.5)
smoother = TemporalSmoother(window_size=5, confirm_count=3)

pi_url = None  # set in main() from --pi-host/--pi-port, updated by _try_rediscover on failure
pi_port = None  # set in main(), reused by _try_rediscover to rebuild pi_url
_rediscovery_lock = threading.Lock()  # guards against overlapping rediscovery scans
_consecutive_failures = 0
REDISCOVER_AFTER_N_FAILURES = 4  # ~2s of failed heartbeats at HEARTBEAT_INTERVAL before rescanning

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
    "pi_connected": None,  # None = unknown yet, True/False after first attempt
    "pi_error": None,
}
reset_flag = threading.Event()
manual_stop_flag = threading.Event()


def predict_a(landmarks):
    vec = normalize_landmarks(landmarks)
    X = scaler.transform(vec.reshape(1, -1))
    probs = mlp.predict_proba(X)[0]
    return top_prediction(probs, [str(c) for c in mlp.classes_], CONFIDENCE_THRESHOLD_A)


def _local_subnet():
    """Best-effort /24 guess from this machine's own LAN IP. Opening a UDP
    "connection" to a public IP sends no packets (UDP is connectionless) --
    it just makes the OS pick which local interface/IP would be used, which
    is all that's needed to infer our own subnet."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    finally:
        s.close()
    return str(ipaddress.ip_network(f"{local_ip}/24", strict=False))


def discover_pi_ip(timeout=30, attempts=3):
    """Scan the local /24 for a host whose MAC vendor is Raspberry Pi
    Trading, via `nmap -sn`. Avoids depending on a fixed IP or on mDNS
    (raspberrypi.local) -- both break when the network itself reassigns a
    new subnet on reconnect (e.g. a mobile hotspot); MAC-based discovery
    doesn't care what the current subnet or hostname are.

    A single -sn pass relies on an ARP reply arriving in time, which this
    network has shown to occasionally drop even when the Pi is genuinely
    up and reachable by IP (observed directly: ssh succeeded the same
    second a scan reported nothing) -- so a miss is retried a few times
    before giving up, rather than treated as "Pi is down" on the first try.
    """
    subnet = _local_subnet()
    for attempt in range(1, attempts + 1):
        print(f"[remote_app] Auto-discovering Pi on {subnet} (nmap -sn, attempt {attempt}/{attempts})...")
        try:
            result = subprocess.run(
                ["nmap", "-sn", subnet],
                capture_output=True, text=True, timeout=timeout,
            )
        except FileNotFoundError:
            raise RuntimeError("nmap not found -- install it, or pass --pi-host explicitly")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"scanning {subnet} timed out -- pass --pi-host explicitly")

        ip = None
        for line in result.stdout.splitlines():
            m = re.match(r"Nmap scan report for (\d+\.\d+\.\d+\.\d+)", line)
            if m:
                ip = m.group(1)
            elif "Raspberry Pi" in line and ip:
                print(f"[remote_app] Found Pi at {ip}")
                return ip

    raise RuntimeError(
        f"No Raspberry Pi found on {subnet} after {attempts} scans -- is it powered on and connected? "
        f"Pass --pi-host explicitly to skip auto-discovery."
    )


def _try_rediscover():
    """Runs in its own thread -- rescans for the Pi and updates the global
    pi_url if found. Guarded so only one scan runs at a time (a scan takes
    a few seconds; the frame loop keeps sending heartbeats in the meantime,
    each of which would otherwise trigger its own overlapping scan)."""
    global pi_url, _consecutive_failures
    if not _rediscovery_lock.acquire(blocking=False):
        return
    try:
        print("[remote_app] Pi unreachable -- attempting auto-rediscovery...")
        new_ip = discover_pi_ip()
        try:
            requests.get(f"http://{new_ip}:{pi_port}/health", timeout=5)  # pay ARP cold-start cost here, not on the next heartbeat
        except requests.RequestException:
            pass
        pi_url = f"http://{new_ip}:{pi_port}/command"
        print(f"[remote_app] Rediscovered Pi -- now sending commands to {pi_url}")
        _consecutive_failures = 0
    except RuntimeError as e:
        print(f"[remote_app] Rediscovery failed: {e}")
    finally:
        _rediscovery_lock.release()


def send_command_async(command):
    def _send():
        global _consecutive_failures
        try:
            r = requests.post(pi_url, json={"command": command}, timeout=COMMAND_REQUEST_TIMEOUT)
            ok = r.ok
            err = None if ok else f"HTTP {r.status_code}"
        except requests.RequestException as e:
            ok, err = False, str(e)
        with state_lock:
            state["pi_connected"] = ok
            state["pi_error"] = err
        if ok:
            _consecutive_failures = 0
        else:
            _consecutive_failures += 1
            if _consecutive_failures >= REDISCOVER_AFTER_N_FAILURES:
                threading.Thread(target=_try_rediscover, daemon=True).start()

    threading.Thread(target=_send, daemon=True).start()


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


def capture_loop():
    cap = open_camera(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open laptop webcam (index 0)")

    hand_was_visible = False
    last_sent_command = None
    last_sent_time = 0.0

    prev_t = time.time()
    fps_ema = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        if reset_flag.is_set():
            smoother.reset()
            hand_was_visible = False
            reset_flag.clear()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = detector.detect(rgb)

        pred = conf = confirmed = None
        current_command = None

        if landmarks is not None:
            hand_was_visible = True
            pred, conf = predict_a(landmarks)
            smoother.push(pred)
            confirmed = smoother.last_confirmed
            if confirmed is not None:
                current_command = "STOP" if confirmed == UNKNOWN_LABEL else GESTURE_TO_COMMAND[confirmed]
        elif hand_was_visible:
            # Hand just left the frame -- stop immediately, require a fresh
            # confirmation before moving again (same as run_pipeline.py).
            current_command = "STOP"
            smoother.reset()
            hand_was_visible = False

        if manual_stop_flag.is_set():
            current_command = "STOP"
            smoother.reset()
            hand_was_visible = False
            manual_stop_flag.clear()

        now = time.time()
        if current_command is not None and (
            current_command != last_sent_command or now - last_sent_time > HEARTBEAT_INTERVAL
        ):
            send_command_async(current_command)
            last_sent_command = current_command
            last_sent_time = now

        frame = draw_overlay(frame, landmarks, pred, conf, confirmed)

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
    return render_template("remote_index.html", pi_url=pi_url)


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
            "pi_connected": state["pi_connected"],
            "pi_error": state["pi_error"],
            "pi_url": pi_url,
        })


@app.route("/reset", methods=["POST"])
def reset():
    reset_flag.set()
    return jsonify({"ok": True})


@app.route("/stop", methods=["POST"])
def manual_stop():
    manual_stop_flag.set()
    send_command_async("STOP")
    return jsonify({"ok": True})


def main():
    global pi_url, pi_port
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi-host", default=None,
                     help="Pi hostname/IP running pi_server.py (this folder) -- default: "
                          "auto-discover on the local network by MAC vendor (requires nmap)")
    ap.add_argument("--pi-port", type=int, default=5001)
    ap.add_argument("--port", type=int, default=5000, help="port for this laptop-side web interface")
    args = ap.parse_args()

    pi_port = args.pi_port
    pi_host = args.pi_host or discover_pi_ip()
    pi_url = f"http://{pi_host}:{pi_port}/command"
    print(f"[remote_app] Sending commands to {pi_url}")

    # Warm up the connection here, with a generous timeout, rather than
    # let the first real command eat COMMAND_REQUEST_TIMEOUT (0.5s) -- the
    # very first packet to a freshly-discovered IP pays for ARP resolution
    # (observed directly: first request ~500ms+, every one after ~30ms),
    # which was otherwise showing up as a spurious "Pi unreachable" the
    # moment the app started driving.
    try:
        requests.get(f"http://{pi_host}:{pi_port}/health", timeout=5)
        print("[remote_app] Warm-up connection to Pi succeeded")
    except requests.RequestException as e:
        print(f"[remote_app] Warm-up connection failed (will retry during normal operation): {e}")

    threading.Thread(target=capture_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
