"""Face landmarks via MediaPipe Tasks (the legacy mp.solutions API was removed in recent releases)."""
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .config import FACE_MODEL_PATH, FACE_MODEL_URL


def ensure_model(path: Path = FACE_MODEL_PATH) -> Path:
    """Download the face landmarker model on first run (~4 MB)."""
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    tmp = path.with_suffix(".part")
    try:
        print(f"[hydra] downloading face model to {path} ...")
        urllib.request.urlretrieve(FACE_MODEL_URL, tmp)
        if tmp.stat().st_size < 1_000_000:
            raise RuntimeError("downloaded file is too small")
        tmp.replace(path)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"Could not download the face model ({exc}). Download it manually from\n"
            f"{FACE_MODEL_URL}\nand save it as {path}"
        ) from exc
    return path


class FaceTracker:
    """process(frame_bgr) -> (N, 2) float32 array of normalised landmarks, or None if no face."""

    def __init__(self, static: bool = False):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        self.static = static
        self._last_ts = 0
        model_bytes = ensure_model().read_bytes()  # buffer avoids Windows non-ASCII path issues
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.IMAGE if static else vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray):
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self.static:
            result = self._landmarker.detect(image)
        else:
            ts = max(int(time.monotonic() * 1000), self._last_ts + 1)  # must strictly increase
            self._last_ts = ts
            result = self._landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            return None
        return np.array([[p.x, p.y] for p in result.face_landmarks[0]], dtype=np.float32)

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass
