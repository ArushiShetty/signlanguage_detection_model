"""Hand crop for alphabet images. Single source of truth for training, collection, and live inference."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from src.config import HAND_CROP_PADDING


class HandCropper:
    """MediaPipe Hands cropper. Use static_image_mode=True for datasets, False for video."""

    def __init__(self, static_image_mode: bool = True, max_hands: int = 1):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def crop(self, bgr_image) -> tuple[np.ndarray | None, dict | None]:
        """
        Returns (crop_rgb_uint8_square, info) or (None, None) if no hand found.
        Steps: BGR->RGB, MediaPipe Hands, landmark min/max -> bbox,
        pad by HAND_CROP_PADDING, make it SQUARE around the center, clamp to image,
        return the crop (NOT yet resized).
        info = {"bbox": (x,y,w,h), "landmarks": [(x_norm,y_norm), ...21]}
        """
        if bgr_image is None or bgr_image.size == 0:
            return None, None

        h, w = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None, None

        lms = results.multi_hand_landmarks[0]
        xs = np.array([p.x * w for p in lms.landmark], dtype=np.float32)
        ys = np.array([p.y * h for p in lms.landmark], dtype=np.float32)
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        x1 -= bw * HAND_CROP_PADDING
        x2 += bw * HAND_CROP_PADDING
        y1 -= bh * HAND_CROP_PADDING
        y2 += bh * HAND_CROP_PADDING

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        side = max(x2 - x1, y2 - y1)
        half = side / 2.0
        x1, y1 = cx - half, cy - half
        x2, y2 = cx + half, cy + half

        x1i = int(np.floor(max(0, x1)))
        y1i = int(np.floor(max(0, y1)))
        x2i = int(np.ceil(min(w, x2)))
        y2i = int(np.ceil(min(h, y2)))
        if x2i <= x1i or y2i <= y1i:
            return None, None

        crop_rgb = rgb[y1i:y2i, x1i:x2i]
        ch, cw = crop_rgb.shape[:2]
        side_px = max(ch, cw)
        square = np.zeros((side_px, side_px, 3), dtype=np.uint8)
        oy = (side_px - ch) // 2
        ox = (side_px - cw) // 2
        square[oy : oy + ch, ox : ox + cw] = crop_rgb

        info = {
            "bbox": (x1i, y1i, x2i - x1i, y2i - y1i),
            "landmarks": [(float(p.x), float(p.y), float(p.z)) for p in lms.landmark],
        }
        return square, info

    def close(self) -> None:
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def resize_for_model(crop_rgb, size: int) -> np.ndarray:
    """cv2.resize to (size,size), float32, values stay in 0..255. Models normalize internally."""
    resized = cv2.resize(crop_rgb, (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)
