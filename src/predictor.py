"""Loads models, smoothing, and sentence logic for live inference."""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from src.config import (
    ALPHA_COOLDOWN_S,
    ALPHA_MIN_CONF,
    ALPHA_MIN_VOTES,
    ALPHA_WINDOW,
    ALPHABET_LABELS,
    CNN_IMG_SIZE,
    MOBILENET_IMG_SIZE,
    MODELS_DIR,
    N_FEATURES,
    PHRASE_CONSECUTIVE,
    PHRASE_COOLDOWN_S,
    PHRASE_DISPLAY,
    PHRASE_LABELS,
    PHRASE_MIN_CONF,
    PHRASE_STRIDE,
    SEQ_LEN,
)
from src.asl_geometry import classify_alphabet_landmarks, classify_phrase_sequence
from src.landmarks import LandmarkExtractor
from src.preprocess import HandCropper, resize_for_model


def load_labels(path: Path | None = None) -> dict:
    labels_path = path or (MODELS_DIR / "labels.json")
    with open(labels_path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("alphabet") != ALPHABET_LABELS:
        raise ValueError("models/labels.json alphabet order must match src.config.ALPHABET_LABELS")
    if data.get("phrases") != PHRASE_LABELS:
        raise ValueError("models/labels.json phrases order must match src.config.PHRASE_LABELS")
    return data


def _topk(probs: np.ndarray, labels: list[str], k: int = 3) -> list[dict]:
    idx = np.argsort(probs)[::-1][:k]
    return [{"label": labels[i], "confidence": float(probs[i])} for i in idx]


class SignPredictor:
    def __init__(self, *, shared: SignPredictor | None = None):
        self.cnn = None
        self.mobilenet = None
        self.phrases = None
        self.alphabet_labels = list(ALPHABET_LABELS)
        self.phrase_labels = list(PHRASE_LABELS)
        self._last_alpha: dict[str, float] = {}
        self._last_commit_t = 0.0
        self._last_phrase_t = 0.0
        self._phrase_last_label: str | None = None
        self._phrase_run = 0
        self._phrase_frame = 0
        self._hand_hist: deque[bool] = deque(maxlen=10)
        self.votes: deque = deque(maxlen=ALPHA_WINDOW)
        self.seq: deque = deque(maxlen=SEQ_LEN)
        self.sentence = ""
        self.cropper: HandCropper | None = None
        self.extractor: LandmarkExtractor | None = None

        if shared is not None:
            self.cnn = shared.cnn
            self.mobilenet = shared.mobilenet
            self.phrases = shared.phrases
            self.alphabet_labels = shared.alphabet_labels
            self.phrase_labels = shared.phrase_labels
            self.cropper = HandCropper(static_image_mode=False)
            self.extractor = LandmarkExtractor(static_image_mode=False)
            return

        try:
            labels = load_labels()
            self.alphabet_labels = labels["alphabet"]
            self.phrase_labels = labels["phrases"]
        except Exception:
            self.alphabet_labels = list(ALPHABET_LABELS)
            self.phrase_labels = list(PHRASE_LABELS)

        self.cropper = HandCropper(static_image_mode=False)
        self.extractor = LandmarkExtractor(static_image_mode=False)

    def spawn(self) -> SignPredictor:
        """Per-connection buffers; shared model weights."""
        return SignPredictor(shared=self)

    def reset(self) -> None:
        self.sentence = ""
        self.votes.clear()
        self.seq.clear()
        self._hand_hist.clear()
        self._last_alpha.clear()
        self._last_commit_t = 0.0
        self._last_phrase_t = 0.0
        self._phrase_last_label = None
        self._phrase_run = 0
        self._phrase_frame = 0

    def backspace(self) -> None:
        if self.sentence:
            self.sentence = self.sentence[:-1]
        self.votes.clear()
        self.seq.clear()
        self._hand_hist.clear()
        self._last_alpha.clear()
        self._last_commit_t = 0.0
        self._last_phrase_t = 0.0
        self._phrase_last_label = None
        self._phrase_run = 0
        self._phrase_frame = 0

    def process_frame(self, bgr_frame, mode: str, model_name: str) -> dict:
        t0 = time.perf_counter()
        if mode == "phrase":
            out = self._phrase_frame_fn(bgr_frame)
        else:
            out = self._alphabet_frame(bgr_frame, model_name)
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        out["sentence"] = self.sentence
        return out

    def _empty(self, extra: dict | None = None) -> dict:
        base = {
            "hand_detected": False,
            "hands": [],
            "bbox": None,
            "prediction": None,
            "top3": [],
            "committed": None,
            "sentence": self.sentence,
            "hold_progress": 0.0,
            "pose": [],
        }
        if extra:
            base.update(extra)
        return base

    def _alphabet_frame(self, bgr_frame, model_name: str) -> dict:
        crop, info = self.cropper.crop(bgr_frame)
        if crop is None or info is None:
            self.votes.clear()
            return self._empty()

        hands = [info["landmarks"]]
        bbox = list(info["bbox"])
        probs = classify_alphabet_landmarks(info["landmarks"])

        top3 = _topk(probs, self.alphabet_labels)
        label = top3[0]["label"]
        conf = top3[0]["confidence"]

        now = time.time()
        on_cd = (now - self._last_commit_t) < ALPHA_COOLDOWN_S
        display = " " if label == "space" else ("⌫" if label == "del" else label)

        if on_cd:
            self.votes.clear()
            return {
                "hand_detected": True,
                "hands": hands,
                "bbox": bbox,
                "prediction": {"label": label, "display": display, "confidence": conf},
                "top3": top3,
                "committed": None,
                "sentence": self.sentence,
                "hold_progress": 0.0,
                "pose": [],
            }

        self.votes.append((label, conf))
        votes_for = sum(1 for lab, _ in self.votes if lab == label)
        mean_conf = float(np.mean([c for lab, c in self.votes if lab == label])) if votes_for else 0.0
        hold = min(1.0, votes_for / max(ALPHA_MIN_VOTES, 1))
        committed = None

        if (
            votes_for >= ALPHA_MIN_VOTES
            and mean_conf >= ALPHA_MIN_CONF
        ):
            committed = {"token": label}
            self._apply_letter(label)
            self._last_commit_t = now
            self.votes.clear()
            hold = 0.0

        return {
            "hand_detected": True,
            "hands": hands,
            "bbox": bbox,
            "prediction": {"label": label, "display": display, "confidence": conf},
            "top3": top3,
            "committed": committed,
            "sentence": self.sentence,
            "hold_progress": hold,
            "pose": [],
        }

    def _apply_letter(self, label: str) -> None:
        if label == "space":
            self.sentence += " "
        elif label == "del":
            self.sentence = self.sentence[:-1]
        else:
            self.sentence += label

    def _phrase_frame_fn(self, bgr_frame) -> dict:
        feats, results = self.extractor.process(bgr_frame)
        overlay = self.extractor.overlay_points(results)
        visible = self.extractor.hands_visible(results)
        self._hand_hist.append(visible)
        self.seq.append(feats)
        self._phrase_frame += 1

        out = {
            "hand_detected": visible,
            "hands": overlay["hands"],
            "bbox": None,
            "prediction": None,
            "top3": [],
            "committed": None,
            "sentence": self.sentence,
            "hold_progress": min(1.0, len(self.seq) / SEQ_LEN),
            "pose": overlay["pose"],
        }
        if len(self.seq) < 8:
            return out
        recent_hand = any(self._hand_hist)
        if not recent_hand:
            return out
        if self._phrase_frame % PHRASE_STRIDE != 0:
            return out

        raw_list = list(self.seq)
        if len(raw_list) < SEQ_LEN:
            idx = np.round(np.linspace(0, len(raw_list) - 1, SEQ_LEN)).astype(int)
            raw_seq = np.array([raw_list[i] for i in idx], dtype=np.float32)
        else:
            raw_seq = np.stack(raw_list, axis=0)

        probs = classify_phrase_sequence(raw_seq)

        top3 = _topk(probs, self.phrase_labels)
        label = top3[0]["label"]
        conf = top3[0]["confidence"]
        display = PHRASE_DISPLAY.get(label, label)
        out["prediction"] = {"label": label, "display": display, "confidence": conf}
        out["top3"] = top3

        now = time.time()
        on_phrase_cd = (now - self._last_phrase_t) < PHRASE_COOLDOWN_S
        if on_phrase_cd:
            self.seq.clear()
            self._phrase_run = 0
            self._phrase_last_label = None
            out["prediction"] = {"label": "idle", "display": "Waiting...", "confidence": 0.9}
            out["hold_progress"] = 0.0
            return out

        if label == self._phrase_last_label and label != "idle":
            self._phrase_run += 1
        else:
            self._phrase_last_label = label
            self._phrase_run = 1 if label != "idle" else 0

        hold_phrase = min(1.0, self._phrase_run / max(PHRASE_CONSECUTIVE, 1))
        out["hold_progress"] = hold_phrase

        if (
            label != "idle"
            and conf >= PHRASE_MIN_CONF
            and self._phrase_run >= PHRASE_CONSECUTIVE
        ):
            token = PHRASE_DISPLAY.get(label, label)
            if self.sentence and not self.sentence.endswith(" "):
                self.sentence += " "
            self.sentence += token + " "
            out["committed"] = {"token": token, "kind": "phrase"}
            self._last_phrase_t = now
            self.seq.clear()
            self._phrase_run = 0
            self._phrase_last_label = None
            out["hold_progress"] = 0.0
        return out
