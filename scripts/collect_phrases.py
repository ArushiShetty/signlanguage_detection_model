"""Record landmark sequences per phrase. CLI: python scripts/collect_phrases.py --label thank_you --samples 80"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, PHRASE_LABELS, SEQ_LEN
from src.landmarks import LandmarkExtractor

HAND_CONNS = list(mp.solutions.hands.HAND_CONNECTIONS)


def _draw_landmarks(frame, results):
    if results is None:
        return
    mp.solutions.drawing_utils.draw_landmarks(
        frame, results.left_hand_landmarks, HAND_CONNS
    )
    mp.solutions.drawing_utils.draw_landmarks(
        frame, results.right_hand_landmarks, HAND_CONNS
    )
    mp.solutions.drawing_utils.draw_landmarks(
        frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
    )


def _session_id() -> str:
    return datetime.now(timezone.utc).strftime("s%Y%m%d")


def _next_path(folder: Path, session: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%H%M%S_%f")
    return folder / f"{session}_{ts}.npy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, choices=PHRASE_LABELS)
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--session", default=_session_id(), help="Filename prefix so val can split by session")
    args = parser.parse_args()

    out_dir = DATA_DIR / "custom" / "phrases" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam (index 0).")
    except Exception as exc:
        print(f"Camera error: {exc}")
        return 1

    extractor = LandmarkExtractor(static_image_mode=False)
    saved: list[Path] = []
    collected = 0
    print("Get ready (1.5s) then 30 frames are recorded. q quit, r discard last sample.")

    try:
        while collected < args.samples:
            # live preview until we start a sample automatically in a loop
            ready_until = time.time() + 1.5
            recording = False
            buf: list[np.ndarray] = []
            phase = "ready"

            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read a frame from the webcam.")
                    return 1
                feats, results = extractor.process(frame)
                vis = frame.copy()
                _draw_landmarks(vis, results)
                now = time.time()
                if phase == "ready":
                    cv2.putText(vis, "Get ready", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (251, 191, 36), 2)
                    if now >= ready_until:
                        phase = "record"
                        recording = True
                        buf = []
                elif phase == "record":
                    buf.append(feats)
                    cv2.putText(
                        vis, f"Recording {len(buf)}/{SEQ_LEN}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (52, 211, 153), 2,
                    )
                    if len(buf) >= SEQ_LEN:
                        arr = np.stack(buf[:SEQ_LEN], axis=0).astype(np.float32)
                        path = _next_path(out_dir, args.session)
                        np.save(path, arr)
                        saved.append(path)
                        collected += 1
                        phase = "pause"
                        pause_until = time.time() + 0.6
                else:
                    cv2.putText(
                        vis, f"Saved {collected}/{args.samples}  (r=undo)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (34, 211, 238), 2,
                    )
                    if now >= pause_until:
                        break

                cv2.putText(vis, args.label, (20, vis.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("collect_phrases", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    return 0
                if key == ord("r") and saved:
                    last = saved.pop()
                    try:
                        last.unlink()
                        collected = max(0, collected - 1)
                        print(f"Discarded {last.name}")
                    except OSError as exc:
                        print(f"Could not delete last sample: {exc}")
    finally:
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()
    print(f"Done. {collected} samples in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
