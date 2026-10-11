"""OpenCV desktop demo. Keys: 1=alphabet, 2=phrases, m=switch model, c=clear, q=quit."""

from __future__ import annotations

import sys

import cv2
import mediapipe as mp

from src.predictor import SignPredictor

HAND_CONNS = list(mp.solutions.hands.HAND_CONNECTIONS)


def _draw_hands(frame, hands):
    h, w = frame.shape[:2]
    for landmarks in hands or []:
        pts = [(int(x * w), int(y * h)) for x, y in landmarks]
        if len(pts) >= 21:
            for a, b in HAND_CONNS:
                cv2.line(frame, pts[a], pts[b], (139, 92, 246), 2)
            for p in pts:
                cv2.circle(frame, p, 3, (34, 211, 238), -1)


def main():
    mode = "alphabet"
    model_name = "cnn"
    try:
        predictor = SignPredictor()
    except Exception as exc:
        print(f"Could not start predictor: {exc}")
        return 1
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam (index 0).")
    except Exception as exc:
        print(f"Camera error: {exc}")
        return 1

    print("1 alphabet  2 phrases  m CNN/MobileNet  c clear  q quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame from the webcam.")
                break
            result = predictor.process_frame(frame, mode, model_name)
            vis = frame.copy()
            _draw_hands(vis, result.get("hands"))
            bbox = result.get("bbox")
            if bbox:
                x, y, bw, bh = [int(v) for v in bbox]
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), (34, 211, 238), 2)
            pred = result.get("prediction") or {}
            label = pred.get("display") or pred.get("label") or "-"
            conf = pred.get("confidence")
            conf_s = f"{conf:.2f}" if conf is not None else "-"
            status = "hand" if result.get("hand_detected") else "no hand"
            header = f"{mode} | {model_name} | {status} | {label} {conf_s} | {result.get('latency_ms')}ms"
            cv2.rectangle(vis, (0, vis.shape[0] - 50), (vis.shape[1], vis.shape[0]), (10, 12, 18), -1)
            cv2.putText(vis, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 211, 238), 2)
            cv2.putText(
                vis, predictor.sentence[-80:] or "(empty sentence)",
                (10, vis.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
            cv2.imshow("SignSpeak demo", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("1"):
                mode = "alphabet"
                predictor.reset()
            if key == ord("2"):
                mode = "phrase"
                predictor.reset()
            if key == ord("m"):
                model_name = "mobilenet" if model_name == "cnn" else "cnn"
            if key == ord("c"):
                predictor.reset()
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
