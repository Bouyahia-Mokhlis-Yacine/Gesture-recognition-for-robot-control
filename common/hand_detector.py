"""
Thin wrapper around MediaPipe's Tasks HandLandmarker (mediapipe>=1.0 dropped
the old `mp.solutions.hands` Solutions API in favor of this). Used by both
baselines so there's one place that knows about the model file / API shape.
"""
import time
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


class HandDetector:
    def __init__(self, running_mode="IMAGE", min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self._mode = running_mode
        mode = {
            "IMAGE": vision.RunningMode.IMAGE,
            "VIDEO": vision.RunningMode.VIDEO,
        }[running_mode]
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mode,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, frame_rgb):
        """frame_rgb: HxWx3 uint8 RGB array. Returns list of 21 NormalizedLandmark
        for the first detected hand, or None if no hand was found.

        In VIDEO mode this feeds MediaPipe's inter-frame tracking (only re-runs
        the heavy palm detector when tracking is lost), which is both faster
        and much less jittery than re-detecting from scratch on every frame
        (IMAGE mode) -- so a live webcam loop should use running_mode="VIDEO".
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        if self._mode == "VIDEO":
            timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
            self._last_timestamp_ms = timestamp_ms
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None
        return result.hand_landmarks[0]

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
