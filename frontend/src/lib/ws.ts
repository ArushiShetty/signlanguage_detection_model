export type Mode = "alphabet" | "phrase";
export type ModelName = "cnn" | "mobilenet";

export type Prediction = {
  label: string;
  display: string;
  confidence: number;
};

export type FrameResult = {
  hand_detected?: boolean;
  hands?: number[][][];
  bbox?: number[] | null;
  prediction?: Prediction | null;
  top3?: { label: string; confidence: number }[];
  committed?: { token: string; kind?: string } | null;
  sentence?: string;
  latency_ms?: number;
  hold_progress?: number;
  pose?: number[][];
  error?: string;
  ok?: boolean;
};

export function wsUrl(): string {
  return import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws";
}

export function connectSignSocket(): WebSocket {
  return new WebSocket(wsUrl());
}
