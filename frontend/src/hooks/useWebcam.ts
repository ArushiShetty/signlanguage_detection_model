import { useCallback, useEffect, useRef, useState } from "react";

export function useWebcam() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const start = useCallback(async () => {
    setError(null);
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Webcam API is not available in this browser environment.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      setReady(true);
    } catch (exc: any) {
      console.error("Camera access error:", exc);
      let msg = "Could not access webcam.";
      if (exc?.name === "NotAllowedError" || exc?.name === "PermissionDeniedError") {
        msg = "Camera permission was denied. Click the lock/tune icon in your address bar -> Site settings -> Camera -> Allow, then click Enable Camera again.";
      } else if (exc?.name === "NotFoundError" || exc?.name === "DevicesNotFoundError") {
        msg = "No webcam device was detected. Please connect a webcam and try again.";
      } else if (exc?.name === "NotReadableError" || exc?.name === "TrackStartError") {
        msg = "Your webcam is currently being used by another application (e.g. Zoom, Teams, or another tab). Close other camera apps and try again.";
      } else if (exc?.message) {
        msg = exc.message;
      }
      setError(msg);
      setReady(false);
    }
  }, []);

  // Attach stream as soon as videoRef mounts in the DOM
  useEffect(() => {
    if (ready && streamRef.current && videoRef.current) {
      const video = videoRef.current;
      video.srcObject = streamRef.current;
      video.play().catch((err) => {
        console.warn("Video play error:", err);
      });
    }
  }, [ready]);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
    };
  }, []);

  return { videoRef, ready, error, start };
}
