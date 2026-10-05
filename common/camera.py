"""
Shared webcam-opening helper so every live-camera script requests the same
capture settings, instead of silently taking whatever low-res default the
driver falls back to (often 640x480) even when the camera supports better.
"""
import cv2

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
# Was 1920x1080, dropped to 720p: on a Pi 4, MediaPipe hand detection
# doesn't need 1080p input to work well, but does pay for it -- measured
# ~366ms/frame detection at 1080p vs ~284ms at 720p. See the profiling
# writeup in this session for the fuller breakdown (detection itself, not
# resolution, is the dominant bottleneck on this hardware either way).


def open_camera(index=0, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, fps=DEFAULT_FPS, max_probe=5):
    """Opens camera `index` and requests width/height/fps -- these are
    requests, not guarantees, cv2 silently falls back to the closest mode
    the hardware actually supports, so check cap.get(...) after if you need
    to know what was actually granted.

    Falls back to probing index..index+max_probe-1 if the requested index
    doesn't open or doesn't yield a real frame: a USB webcam that drops and
    reconnects (e.g. from a power/undervoltage event -- check
    `vcgencmd get_throttled` on a Pi if this keeps happening) often
    re-enumerates to a different /dev/videoN, which would otherwise silently
    break anything that hardcodes index 0.
    """
    for i in range(index, index + max_probe):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        # MJPEG lets many USB webcams reach their full resolution+framerate;
        # uncompressed YUYV (often the driver default) can't fit high-res/
        # high-fps combos over USB bandwidth and silently drops fps instead
        # (e.g. one tested cam: 1080p verified 30fps over MJPEG, only 5fps
        # over YUYV -- check `v4l2-ctl -d /dev/videoN --list-formats-ext` if
        # a camera still comes up slow after this).
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        # Keep the driver's internal queue at 1 frame: without this, if a
        # reader ever falls a frame behind (e.g. a slow JPEG re-encode on a
        # loaded Pi), OpenCV queues up multiple stale frames instead of
        # dropping them, which shows up as growing lag (old frames) rather
        # than just lower fps. Not all backends honor this, but it's a
        # no-op where unsupported rather than an error.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if ok and frame is not None:
            if i != index:
                print(f"[camera] index {index} unavailable, using {i} instead")
            return cap
        cap.release()
    return cv2.VideoCapture(index)  # exhausted the probe range -- return a
    # (guaranteed-unopened) capture on the originally requested index so
    # existing `if not cap.isOpened(): raise ...` callers still work as-is.
