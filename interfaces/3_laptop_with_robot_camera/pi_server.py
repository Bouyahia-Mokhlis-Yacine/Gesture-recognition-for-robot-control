"""
Interface 3 -- Laptop + robot camera: command receiver + robot's own
camera stream, runs on the Raspberry Pi.

Camera + hand detection + gesture prediction (for CONTROLLING the robot)
run on a laptop (see remote_app.py in this same folder) instead of on the
Pi, since a laptop's CPU handles MediaPipe + the classifier far faster
than a Pi 4 -- only confirmed gesture commands need to cross the network,
not video frames. This process receives those commands over HTTP and
drives the wheels via robot/motor_controller.py, same command set as
robot/run_pipeline.py.

Separately, this process ALSO streams the Pi's own camera -- what's
physically in front of the robot -- at GET /video_feed, so remote_app.py
can display it alongside the gesture-control camera feed ("what I'm
gesturing at" vs "what the robot sees"). This is a plain first-person
view, not fed through hand detection/prediction at all, and NOT flipped
(unlike the gesture-control feed, which mirrors to match the selfie-mode
training data) -- a forward-facing robot camera should show the scene
as-is. Pass --no-camera to skip this if nothing is plugged in (that's
effectively interfaces/1_laptop_only/pi_server.py's behavior -- use that
folder instead if you never want the camera code at all).

Safety: remote_app.py resends the current command on a heartbeat while a
gesture is held (not just on change), and a watchdog here stops the
motors if no command arrives within --timeout seconds. That protects
against the laptop crashing, losing wifi, or the client being killed --
the robot never keeps driving on a stale command once the network link
goes away, mirroring run_pipeline.py's "hand left the frame -> stop".

Usage:
    python interfaces/3_laptop_with_robot_camera/pi_server.py                  # real motors + camera stream
    python interfaces/3_laptop_with_robot_camera/pi_server.py --dry-run         # print only, no GPIO
    python interfaces/3_laptop_with_robot_camera/pi_server.py --no-camera       # motors only, no camera stream
    python interfaces/3_laptop_with_robot_camera/pi_server.py --width 320 --height 240  # lower-res if still laggy
Then point remote_app.py (this same folder) --pi-host at this machine's IP/hostname.

/video_feed defaults to 640x480 (lower than the laptop-side gesture camera's
1280x720) since this feed is just a forward-facing preview, not fed through
hand detection -- and a Pi 4 has to decode the camera's native MJPG then
re-encode every single frame to JPEG for streaming, so fewer pixels means
less of both per frame. Drop --width/--height further (e.g. 320x240) if it's
still laggy on your network/hardware.
"""
import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common.camera import open_camera
from robot.motor_controller import MotorController

app = Flask(__name__)

COMMAND_TO_ACTION = {
    "TURN_LEFT": MotorController.turn_left,
    "TURN_RIGHT": MotorController.turn_right,
    "MOVE_FORWARD": MotorController.forward,
    "MOVE_BACKWARD": MotorController.backward,
    "STOP": MotorController.stop,
}

motors = None
watchdog_timeout = 1.5
_last_command_time = time.time()
_lock = threading.Lock()

camera_state_lock = threading.Lock()
camera_state = {"frame_jpeg": None, "frame_version": 0, "fps": 0.0}


def _watchdog():
    while True:
        time.sleep(0.25)
        with _lock:
            idle = time.time() - _last_command_time
            if idle > watchdog_timeout:
                motors.stop()


def _camera_loop(camera_index, width, height, fps):
    cap = open_camera(camera_index, width=width, height=height, fps=fps)
    if not cap.isOpened():
        print(f"[pi_server] Could not open camera index {camera_index} -- /video_feed will stay empty")
        return

    prev_t = time.time()
    fps_ema = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        # Forward-facing "what's in front of the robot" view -- not
        # flipped, unlike the gesture-control feed on the laptop side
        # (which mirrors to match the selfie-mode training data).
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        now = time.time()
        instant_fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        fps_ema = instant_fps if fps_ema == 0.0 else 0.1 * instant_fps + 0.9 * fps_ema

        with camera_state_lock:
            if ok:
                camera_state["frame_jpeg"] = buf.tobytes()
                camera_state["frame_version"] += 1
            camera_state["fps"] = fps_ema


def _gen_camera_frames():
    last_version = -1
    while True:
        with camera_state_lock:
            frame = camera_state["frame_jpeg"]
            version = camera_state["frame_version"]
        if frame is None or version == last_version:
            time.sleep(0.01)
            continue
        last_version = version
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


@app.route("/video_feed")
def video_feed():
    return Response(_gen_camera_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/camera_stats")
def camera_stats():
    with camera_state_lock:
        return jsonify({"fps": round(camera_state["fps"], 1), "has_frame": camera_state["frame_jpeg"] is not None})


@app.route("/command", methods=["POST"])
def command():
    global _last_command_time
    data = request.get_json(force=True, silent=True) or {}
    cmd = data.get("command")
    if cmd not in COMMAND_TO_ACTION:
        return jsonify({"ok": False, "error": f"unknown command {cmd!r}"}), 400
    with _lock:
        _last_command_time = time.time()
        COMMAND_TO_ACTION[cmd](motors)
    return jsonify({"ok": True, "command": cmd})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "dry_run": motors.dry_run})


def main():
    global motors, watchdog_timeout
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5001)
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
    ap.add_argument("--timeout", type=float, default=1.5,
                     help="stop the motors if no command arrives within this many seconds")
    ap.add_argument("--dry-run", action="store_true",
                     help="don't touch GPIO/motors -- print commands instead")
    ap.add_argument("--camera", type=int, default=0, help="camera index for /video_feed")
    ap.add_argument("--no-camera", action="store_true",
                     help="skip opening a camera -- /video_feed stays empty (motors-only mode)")
    ap.add_argument("--width", type=int, default=640,
                     help="capture width for /video_feed -- kept lower than the laptop-side "
                          "gesture camera since this is just a forward-facing preview, not fed "
                          "through hand detection, and a Pi 4 has to decode the camera's native "
                          "MJPG then re-encode every frame to JPEG for streaming -- more pixels "
                          "means more of both per frame")
    ap.add_argument("--height", type=int, default=480, help="capture height for /video_feed")
    ap.add_argument("--fps", type=int, default=30, help="requested capture fps for /video_feed")
    args = ap.parse_args()

    watchdog_timeout = args.timeout
    motors = MotorController(speed=args.speed, turn_speed=args.turn_speed, dry_run=args.dry_run,
                              left_trim=args.left_trim, right_trim=args.right_trim)

    # Explicit handlers for both signals rather than relying on Flask/
    # Python's default SIGINT->KeyboardInterrupt handling: once the camera
    # thread is running, this process has the same CPU-heavy background
    # thread + Flask dev server shape as remote_app.py (this folder), whose
    # thread was observed to starve the main thread of GIL time to notice
    # a pending SIGINT (needed SIGTERM instead) despite an identical
    # try/except KeyboardInterrupt. Since this script drives real motors,
    # skipping `motors.close()` on an unclean exit would leave the robot's
    # last command running -- so both signals get an explicit handler.
    def _shutdown(signum, frame):
        print(f"\n[pi_server] Received signal {signum} -- stopping motors and exiting")
        with _lock:
            motors.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    threading.Thread(target=_watchdog, daemon=True).start()
    if not args.no_camera:
        threading.Thread(target=_camera_loop, args=(args.camera, args.width, args.height, args.fps),
                          daemon=True).start()
    print(f"Command server listening on {args.host}:{args.port} "
          f"(watchdog stops motors after {watchdog_timeout}s of silence). "
          f"Camera stream: {'disabled' if args.no_camera else f'/video_feed (index {args.camera})'}. "
          f"Ctrl+C to stop.")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
