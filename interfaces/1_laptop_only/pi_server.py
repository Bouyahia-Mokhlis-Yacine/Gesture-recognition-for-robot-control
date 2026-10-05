"""
Interface 1 -- Laptop-only: models + camera run on the laptop, this
process on the Pi ONLY receives confirmed gesture commands and drives the
wheels. No camera runs on the Pi in this interface -- for that, see
interfaces/3_laptop_with_robot_camera/ (same idea, plus the Pi streams its
own camera back so you can see what's in front of the robot too).

Camera + hand detection + gesture prediction run on a laptop (see the
remote_app.py in this same folder) instead of on the Pi, since a laptop's
CPU handles MediaPipe + the classifier far faster than a Pi 4 -- only
confirmed gesture commands need to cross the network, not video frames.
This process receives those commands over HTTP and drives the wheels via
robot/motor_controller.py, same command set as robot/run_pipeline.py.

Safety: remote_app.py resends the current command on a heartbeat while a
gesture is held (not just on change), and a watchdog here stops the
motors if no command arrives within --timeout seconds. That protects
against the laptop crashing, losing wifi, or the client being killed --
the robot never keeps driving on a stale command once the network link
goes away, mirroring run_pipeline.py's "hand left the frame -> stop".

Usage:
    python interfaces/1_laptop_only/pi_server.py                  # real motors
    python interfaces/1_laptop_only/pi_server.py --dry-run         # print only, no GPIO
Then point remote_app.py (this same folder) --pi-host at this machine's IP/hostname.
"""
import argparse
import signal
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
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


def _watchdog():
    while True:
        time.sleep(0.25)
        with _lock:
            idle = time.time() - _last_command_time
            if idle > watchdog_timeout:
                motors.stop()


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
    args = ap.parse_args()

    watchdog_timeout = args.timeout
    motors = MotorController(speed=args.speed, turn_speed=args.turn_speed, dry_run=args.dry_run,
                              left_trim=args.left_trim, right_trim=args.right_trim)

    # Explicit handlers for both signals rather than relying on Flask/
    # Python's default SIGINT->KeyboardInterrupt handling -- see
    # interfaces/2_pi_standalone/pi_app.py and interfaces/3.../pi_server.py
    # for why this matters when a CPU-heavy background thread is present.
    # This variant's only background thread (_watchdog) is a light sleep
    # loop, so plain KeyboardInterrupt has worked reliably here in
    # practice, but an explicit handler costs nothing and removes the risk
    # entirely for a script that drives real motors.
    def _shutdown(signum, frame):
        print(f"\n[pi_server] Received signal {signum} -- stopping motors and exiting")
        with _lock:
            motors.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    threading.Thread(target=_watchdog, daemon=True).start()
    print(f"Command server listening on {args.host}:{args.port} "
          f"(watchdog stops motors after {watchdog_timeout}s of silence). Ctrl+C to stop.")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
