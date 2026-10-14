"""Skin tone analysis and shade matching. Landmarks are a numpy array (N, 2) of normalised x, y."""
import math

import cv2
import numpy as np

# FaceMesh indices on both cheeks.
CHEEK_LANDMARKS = [50, 101, 205, 187, 280, 330, 425, 411]


def hex_to_lab_cv(hex_color: str) -> np.ndarray:
    """Hex -> LAB in OpenCV 8-bit scale (L 0-255, a/b offset by 128)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    px = np.uint8([[[b, g, r]]])
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def cv_lab_to_real(lab_cv) -> tuple[float, float, float]:
    L, a, b = (float(v) for v in lab_cv)
    return L * 100.0 / 255.0, a - 128.0, b - 128.0


def _tone_from_ita(ita: float) -> tuple[str, str]:
    """Individual Typology Angle -> (tone label, depth bucket)."""
    if ita > 55:
        return "very light", "light"
    if ita > 41:
        return "light", "light"
    if ita > 28:
        return "intermediate", "medium"
    if ita > 10:
        return "tan", "medium"
    if ita > -30:
        return "brown", "deep"
    return "dark", "deep"


def analyze_skin(frame_bgr: np.ndarray, landmarks: np.ndarray) -> dict:
    """Estimate skin tone from small cheek patches.

    This is a heuristic and is sensitive to lighting and white balance.
    """
    h, w = frame_bgr.shape[:2]
    patches = []
    for i in CHEEK_LANDMARKS:
        x, y = int(landmarks[i][0] * w), int(landmarks[i][1] * h)
        patch = frame_bgr[max(y - 4, 0):y + 5, max(x - 4, 0):x + 5]
        if patch.size:
            patches.append(patch.reshape(-1, 3))
    if not patches:
        raise ValueError("Could not sample skin patches. Move your face fully into the frame.")

    px = np.vstack(patches).reshape(-1, 1, 3)
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    L, a, b = cv_lab_to_real(np.median(lab, axis=0))
    spread = float(np.std(lab[:, 0]) * 100.0 / 255.0)

    ita = math.degrees(math.atan2(L - 50.0, b if abs(b) > 1e-6 else 1e-6))
    tone, depth = _tone_from_ita(ita)

    hue = math.degrees(math.atan2(b, a))  # lower = pinker (cool), higher = yellower (warm)
    undertone = "cool" if hue < 52 else "neutral" if hue < 60 else "warm"

    warnings = []
    if spread > 10:
        warnings.append("Lighting looks uneven across your cheeks. Face a window or soft even light and rescan.")
    if L < 20:
        warnings.append("The image is very dark, so the result may be off. Add more light.")
    if L > 92:
        warnings.append("The image is very bright or overexposed, so the result may be off.")

    return {
        "lab": [round(L, 1), round(a, 1), round(b, 1)],
        "ita": round(ita, 1),
        "tone": tone,
        "depth": depth,
        "undertone": undertone,
        "warnings": warnings,
    }


def match_foundations(skin: dict, foundations: list[dict], k: int = 3) -> list[dict]:
    """Closest foundation shades by LAB distance (delta E 76). Lower is a closer match."""
    target = np.array(skin["lab"], dtype=np.float32)
    scored = []
    for item in foundations:
        shade = np.array(cv_lab_to_real(hex_to_lab_cv(item["hex"])), dtype=np.float32)
        scored.append({**item, "delta_e": round(float(np.linalg.norm(target - shade)), 1)})
    return sorted(scored, key=lambda s: s["delta_e"])[:k]


def recommend_by_fit(skin: dict, items: list[dict], k: int = 3) -> list[dict]:
    """Rule-based fit for lipstick and blush: undertone plus depth."""
    scored = []
    for item in items:
        score = 0
        if item["undertone"] == skin["undertone"]:
            score += 2
        elif "neutral" in (item["undertone"], skin["undertone"]):
            score += 1
        if skin["depth"] in item.get("depths", []):
            score += 2
        scored.append({**item, "score": score})
    return sorted(scored, key=lambda s: -s["score"])[:k]
