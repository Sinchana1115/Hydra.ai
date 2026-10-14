"""AR overlays. Blends in LAB space so natural lip texture and highlights survive."""
import cv2
import numpy as np

from .skin import hex_to_lab_cv

OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LEFT_CHEEK, RIGHT_CHEEK = 205, 425
LEFT_EDGE, RIGHT_EDGE = 234, 454  # face width reference


def _pts(landmarks: np.ndarray, idx: list[int], w: int, h: int) -> np.ndarray:
    return (landmarks[idx] * np.array([w, h], dtype=np.float32)).astype(np.int32)


def _blend_lab(frame: np.ndarray, mask01: np.ndarray, hex_color: str, strength: float, l_weight: float) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = hex_to_lab_cv(hex_color)
    alpha = mask01 * strength
    lab[..., 1] = lab[..., 1] * (1 - alpha) + target[1] * alpha
    lab[..., 2] = lab[..., 2] * (1 - alpha) + target[2] * alpha
    lab[..., 0] = lab[..., 0] * (1 - alpha * l_weight) + target[0] * alpha * l_weight
    blended = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    out = frame.copy()
    m = mask01 > 0.01
    out[m] = blended[m]  # leave every pixel outside the mask exactly as captured
    return out


def apply_lipstick(frame: np.ndarray, landmarks: np.ndarray, hex_color: str, opacity: float = 0.65) -> np.ndarray:
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [_pts(landmarks, OUTER_LIPS, w, h)], 255)
    cv2.fillPoly(mask, [_pts(landmarks, INNER_LIPS, w, h)], 0)  # keep an open mouth clear
    mask01 = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32) / 255.0
    return _blend_lab(frame, mask01, hex_color, opacity, l_weight=0.35)


def apply_blush(frame: np.ndarray, landmarks: np.ndarray, hex_color: str, opacity: float = 0.30) -> np.ndarray:
    h, w = frame.shape[:2]
    face_w = abs(float(landmarks[RIGHT_EDGE][0] - landmarks[LEFT_EDGE][0])) * w
    radius = max(int(face_w * 0.11), 4)
    mask = np.zeros((h, w), np.uint8)
    for i in (LEFT_CHEEK, RIGHT_CHEEK):
        cv2.circle(mask, (int(landmarks[i][0] * w), int(landmarks[i][1] * h)), radius, 255, -1)
    k = radius * 2 + 1
    mask01 = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0
    return _blend_lab(frame, mask01, hex_color, opacity, l_weight=0.15)
