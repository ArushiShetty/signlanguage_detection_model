"""Live webcam: full frame with skeleton, second window with the square hand crop."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import HandCropper

CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS


def draw_hand(frame_bgr, landmarks, bbox):
    h, w = frame_bgr.shape[:2]
    if bbox:
        x, y, bw, bh = bbox
        cv2.rectangle(frame_bgr, (x, y), (x + bw, y + bh), (34, 211, 238), 2)
    pts = [(int(x * w), int(y * h)) for x, y in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], (139, 92, 246), 2)
    for p in pts:
        cv2.circle(frame_bgr, p, 3, (34, 211, 238), -1)


def main():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam (index 0).")
    except Exception as exc:
        print(f"Camera error: {exc}")
        return 1

    cropper = HandCropper(static_image_mode=False, max_hands=1)
    print("test_crop: webcam + crop windows. Press q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame from the webcam.")
                break
            crop, info = cropper.crop(frame)
            vis = frame.copy()
            if info:
                draw_hand(vis, info["landmarks"], info["bbox"])
                cv2.putText(vis, "hand detected", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (52, 211, 153), 2)
            else:
                cv2.putText(vis, "no hand", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (113, 113, 248), 2)
            cv2.imshow("webcam", vis)
            if crop is not None:
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                cv2.imshow("crop", crop_bgr)
            else:
                blank = vis * 0
                cv2.putText(blank, "no crop", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                cv2.imshow("crop", blank)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cropper.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
