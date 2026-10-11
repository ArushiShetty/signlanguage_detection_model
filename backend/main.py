"""FastAPI + WebSocket inference server."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ALPHABET_LABELS, PHRASE_DISPLAY, PHRASE_LABELS
from src.predictor import SignPredictor, load_labels


def _decode_jpeg(image_b64: str) -> np.ndarray | None:
    raw = image_b64.split(",", 1)[-1]
    try:
        buf = base64.b64decode(raw)
        arr = np.frombuffer(buf, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.shared = SignPredictor()
        app.state.load_error = None
    except Exception as exc:
        app.state.shared = None
        app.state.load_error = str(exc)
        print(f"SignPredictor failed to load: {exc}")
    yield


app = FastAPI(title="SignSpeak", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "ok": True,
        "models_loaded": app.state.shared is not None,
        "error": app.state.load_error,
    }


@app.get("/reload_models")
@app.post("/reload_models")
def reload_models():
    import importlib
    import src.asl_geometry
    import src.landmarks
    import src.preprocess
    import src.predictor
    importlib.reload(src.asl_geometry)
    importlib.reload(src.landmarks)
    importlib.reload(src.preprocess)
    importlib.reload(src.predictor)
    from src.predictor import SignPredictor
    app.state.shared = SignPredictor()
    return {"ok": True, "reloaded": True}


@app.get("/labels")
def labels():
    try:
        return load_labels()
    except Exception:
        return {
            "alphabet": ALPHABET_LABELS,
            "phrases": PHRASE_LABELS,
            "display": PHRASE_DISPLAY,
        }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    if app.state.shared is None:
        await ws.send_json({"error": app.state.load_error or "models not loaded"})
        await ws.close()
        return
    predictor = app.state.shared.spawn()
    try:
        while True:
            message = await ws.receive_text()
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await ws.send_json({"error": "invalid JSON"})
                continue
            msg_type = data.get("type")
            if msg_type == "reset":
                predictor.reset()
                await ws.send_json({"ok": True, "sentence": predictor.sentence})
                continue
            if msg_type == "backspace":
                predictor.backspace()
                await ws.send_json({"ok": True, "sentence": predictor.sentence})
                continue
            if msg_type != "frame":
                await ws.send_json({"error": f"unknown type {msg_type}"})
                continue
            frame = _decode_jpeg(data.get("image") or "")
            if frame is None:
                await ws.send_json({"error": "could not decode JPEG"})
                continue
            mode = data.get("mode") or "alphabet"
            model_name = data.get("model") or "cnn"

            def _run():
                return predictor.process_frame(frame, mode, model_name)

            result = await asyncio.to_thread(_run)
            await ws.send_json(result)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await ws.send_json({"error": str(exc)})
        except Exception:
            return
