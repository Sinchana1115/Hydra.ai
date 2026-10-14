"""Background webcam thread: tracks the face, draws AR overlays, serves JPEG frames for the MJPEG stream."""
import sys
import threading
import time

import cv2
import numpy as np

from .config import CAMERA_INDEX
from .tracker import FaceTracker
from .tryon import apply_blush, apply_lipstick


def _placeholder(lines: list[str]) -> np.ndarray:
    img = np.zeros((480, 640, 3), np.uint8)
    img[:] = (30, 20, 40)
    y = 200
    for line in lines:
        cv2.putText(img, line[:60], (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28
    return img


def _wrap(text: str, width: int = 58) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > width:
            lines.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}".strip()
    if cur:
        lines.append(cur)
    return lines[:8]


class CameraService:
    def __init__(self):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._jpeg: bytes | None = None
        self._clean = None
        self._landmarks = None
        self.style = {"lipstick": None, "blush": None}
        self.error: str | None = None
        self.running = False

    # ---- lifecycle -------------------------------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.error = None
        self._thread = threading.Thread(target=self._run, name="hydra-camera", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.running = False

    def restart(self):
        self.stop()
        self.start()

    def _fail(self, message: str):
        self.error = message
        self.running = False
        print(f"[hydra] camera error: {message}")
        ok, buf = cv2.imencode(".jpg", _placeholder(["Camera not available:"] + _wrap(message)))
        with self._lock:
            self._jpeg = buf.tobytes()

    # ---- main loop -------------------------------------------------
    def _run(self):
        try:
            tracker = FaceTracker(static=False)
        except Exception as exc:  # model download or mediapipe problem
            self._fail(f"Face tracking failed to start: {exc}")
            return

        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(CAMERA_INDEX, backend)
        if not cap.isOpened():
            tracker.close()
            self._fail(
                f"Could not open camera index {CAMERA_INDEX}. Close other apps using it, "
                "allow camera access, or set HYDRA_CAMERA=1 in .env."
            )
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        failures = 0

        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    failures += 1
                    if failures > 100:
                        self._fail("The camera stopped returning frames.")
                        return
                    time.sleep(0.05)
                    continue
                failures = 0

                frame = cv2.flip(frame, 1)  # mirror view
                landmarks = tracker.process(frame)
                out = frame.copy()
                with self._lock:
                    style = dict(self.style)
                if landmarks is not None:
                    if style["lipstick"]:
                        out = apply_lipstick(out, landmarks, style["lipstick"])
                    if style["blush"]:
                        out = apply_blush(out, landmarks, style["blush"])
                else:
                    cv2.putText(out, "No face detected", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 2, cv2.LINE_AA)

                ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self._lock:
                        self._jpeg = buf.tobytes()
                        self._clean = frame
                        self._landmarks = landmarks
        finally:
            cap.release()
            tracker.close()
            self.running = False

    # ---- accessors -------------------------------------------------
    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def snapshot(self):
        """Latest un-overlaid frame and its landmarks, or None if no face is visible."""
        with self._lock:
            if self._clean is None or self._landmarks is None:
                return None
            return self._clean.copy(), self._landmarks.copy()

    def set_style(self, lipstick=..., blush=...):
        with self._lock:
            if lipstick is not ...:
                self.style["lipstick"] = lipstick
            if blush is not ...:
                self.style["blush"] = blush
