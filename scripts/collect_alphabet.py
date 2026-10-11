"""Capture your own 224x224 hand crops per alphabet class."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ALPHABET_LABELS, DATA_DIR
from src.preprocess import HandCropper, resize_for_model

OUT = DATA_DIR / "custom" / "alphabet"


def _key_to_label(key: int) -> str | None:
    if key in (8, 127):
        return "del"
    if key == ord(" "):
        return "space"
    if ord("a") <= key <= ord("z"):
        return chr(key).upper()
    if ord("A") <= key <= ord("Z"):
        return chr(key)
    return None


def _counts() -> Counter:
    c: Counter = Counter()
    for label in ALPHABET_LABELS:
        folder = OUT / label
        if folder.exists():
            c[label] = len(list(folder.glob("*.jpg")))
    return c


def _save(crop_rgb, label: str) -> None:
    folder = OUT / label
    folder.mkdir(parents=True, exist_ok=True)
    resized = resize_for_model(crop_rgb, 224).astype("uint8")
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    idx = len(list(folder.glob("*.jpg")))
    cv2.imwrite(str(folder / f"{idx:05d}.jpg"), bgr)


def main():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam (index 0).")
    except Exception as exc:
        print(f"Camera error: {exc}")
        return 1

    cropper = HandCropper(static_image_mode=False)
    counts = _counts()
    print("A-Z save that letter. Space=space, Backspace=del. Hold a key (OS key-repeat) to burst. q quits.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame from the webcam.")
                break
            crop, info = cropper.crop(frame)
            vis = frame.copy()
            y = 28
            for label in ALPHABET_LABELS:
                cv2.putText(
                    vis, f"{label}:{counts[label]}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (34, 211, 238), 1,
                )
                y += 16
                if y > vis.shape[0] - 20:
                    break
            if info:
                x, yy, bw, bh = info["bbox"]
                cv2.rectangle(vis, (x, yy), (x + bw, yy + bh), (52, 211, 153), 2)
            cv2.imshow("collect_alphabet", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            label = _key_to_label(key)
            if label and crop is not None:
                _save(crop, label)
                counts[label] += 1
    finally:
        cropper.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
