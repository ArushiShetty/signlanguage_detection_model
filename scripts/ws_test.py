"""Send one JPEG over the SignSpeak WebSocket and print the reply."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8000/ws")
    parser.add_argument("--image", help="Optional path to a BGR/JPEG image. Otherwise a blank 320x240 frame.")
    parser.add_argument("--mode", default="alphabet")
    parser.add_argument("--model", default="cnn")
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not read image: {args.image}")
            return 1
        frame = cv2.resize(frame, (320, 240))
    else:
        frame = np.zeros((240, 320, 3), dtype="uint8")

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        print("JPEG encode failed")
        return 1
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    try:
        import websockets
    except ImportError as exc:
        print(f"websockets is required: {exc}")
        return 1

    async with websockets.connect(args.url) as sock:
        payload = json.dumps({"type": "frame", "image": b64, "mode": args.mode, "model": args.model})
        await sock.send(payload)
        reply = await sock.recv()
        print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
