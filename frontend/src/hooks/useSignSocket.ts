import { useCallback, useEffect, useRef, useState } from "react";
import { connectSignSocket, type FrameResult, type Mode, type ModelName } from "../lib/ws";

type Options = {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  enabled: boolean;
  mode: Mode;
  model: ModelName;
};

export function useSignSocket({ videoRef, enabled, mode, model }: Options) {
  const [prediction, setPrediction] = useState<FrameResult | null>(null);
  const [connected, setConnected] = useState(false);
  const [latency, setLatency] = useState(0);
  const [fps, setFps] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const hiddenRef = useRef<HTMLCanvasElement | null>(null);
  const backoffRef = useRef(500);
  const fpsWindow = useRef<number[]>([]);
  const modeRef = useRef(mode);
  const modelRef = useRef(model);
  modeRef.current = mode;
  modelRef.current = model;

  const sendReset = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "reset" }));
    }
  }, []);

  const sendBackspace = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "backspace" }));
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;
    let stopped = false;
    let loopOn = false;

    const canvas = document.createElement("canvas");
    canvas.width = 320;
    canvas.height = 240;
    hiddenRef.current = canvas;

    const grabJpeg = () => {
      const video = videoRef.current;
      const ctx = canvas.getContext("2d");
      if (!video || !ctx || video.readyState < 2) return null;
      ctx.drawImage(video, 0, 0, 320, 240);
      return canvas.toDataURL("image/jpeg", 0.7);
    };

    const tick = async () => {
      const ws = wsRef.current;
      if (stopped || !ws || ws.readyState !== WebSocket.OPEN) {
        loopOn = false;
        return;
      }
      const image = grabJpeg();
      if (image) {
        const t0 = performance.now();
        ws.send(
          JSON.stringify({
            type: "frame",
            image,
            mode: modeRef.current,
            model: modelRef.current,
          }),
        );
        await new Promise<void>((resolve) => {
          const onMsg = (ev: MessageEvent) => {
            ws.removeEventListener("message", onMsg);
            try {
              const data = JSON.parse(String(ev.data)) as FrameResult;
              setPrediction(data);
              if (typeof data.latency_ms === "number") setLatency(data.latency_ms);
            } catch {
              /* ignore malformed */
            }
            const now = performance.now();
            fpsWindow.current.push(now);
            fpsWindow.current = fpsWindow.current.filter((t) => now - t < 1000);
            setFps(fpsWindow.current.length);
            void t0;
            resolve();
          };
          ws.addEventListener("message", onMsg);
        });
      }
      if (!stopped) requestAnimationFrame(() => void tick());
    };

    const connect = () => {
      if (stopped) return;
      const ws = connectSignSocket();
      wsRef.current = ws;
      ws.onopen = () => {
        setConnected(true);
        backoffRef.current = 500;
        if (!loopOn) {
          loopOn = true;
          void tick();
        }
      };
      ws.onclose = () => {
        setConnected(false);
        loopOn = false;
        if (stopped) return;
        const wait = backoffRef.current;
        backoffRef.current = Math.min(wait * 2, 8000);
        window.setTimeout(connect, wait);
      };
      ws.onerror = () => {
        ws.close();
      };
    };

    connect();
    return () => {
      stopped = true;
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [enabled, videoRef]);

  useEffect(() => {
    sendReset();
  }, [mode, sendReset]);

  return {
    prediction,
    hands: prediction?.hands ?? [],
    committed: prediction?.committed ?? null,
    sentence: prediction?.sentence ?? "",
    latency,
    fps,
    connected,
    sendReset,
    sendBackspace,
    pose: prediction?.pose ?? [],
    bbox: prediction?.bbox ?? null,
    holdProgress: prediction?.hold_progress ?? 0,
  };
}
