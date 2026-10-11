from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ClientFrame(BaseModel):
    type: Literal["frame"] = "frame"
    image: str
    mode: Literal["alphabet", "phrase"] = "alphabet"
    model: Literal["cnn", "mobilenet"] = "cnn"


class ClientReset(BaseModel):
    type: Literal["reset"] = "reset"


class Prediction(BaseModel):
    label: str
    display: str
    confidence: float


class TopItem(BaseModel):
    label: str
    confidence: float


class Committed(BaseModel):
    token: str
    kind: str | None = None


class InferenceResult(BaseModel):
    hand_detected: bool
    hands: list[Any] = Field(default_factory=list)
    bbox: list[int] | None = None
    prediction: Prediction | None = None
    top3: list[TopItem] = Field(default_factory=list)
    committed: Committed | None = None
    sentence: str = ""
    latency_ms: float = 0.0
    hold_progress: float = 0.0
    pose: list[Any] = Field(default_factory=list)
    error: str | None = None
