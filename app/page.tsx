"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type CameraState = "idle" | "requesting" | "ready" | "streaming" | "error";

type RuntimeStatus = {
  label: string;
  state: "idle" | "loading" | "ready" | "error";
  detail: string;
};

type TrackRow = {
  id: number;
  label: string;
  confidence: number;
  ageMs: number;
};

const emptyTracks: TrackRow[] = [];

const runtimeDefaults = {
  yoloModelUrl: process.env.NEXT_PUBLIC_YOLO_MODEL_URL ?? "/models/yolo/yolo11n.onnx",
  llmModelId: process.env.NEXT_PUBLIC_LLM_MODEL_ID ?? "google/gemma-4-E2B-it",
  llmRuntime: process.env.NEXT_PUBLIC_LLM_RUNTIME ?? "webgpu",
};

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [cameraError, setCameraError] = useState<string>("");
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [fps, setFps] = useState(0);
  const [yoloMs] = useState(0);
  const [trackMs] = useState(0);
  const [chatInput, setChatInput] = useState("");
  const [includeFrame, setIncludeFrame] = useState(true);

  const llmStatus: RuntimeStatus = useMemo(
    () => ({
      label: "Gemma4-E2B",
      state: "idle",
      detail: `${runtimeDefaults.llmRuntime} / ${runtimeDefaults.llmModelId}`,
    }),
    [],
  );

  const yoloStatus: RuntimeStatus = useMemo(
    () => ({
      label: "YOLO11n",
      state: "idle",
      detail: runtimeDefaults.yoloModelUrl,
    }),
    [],
  );

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraState((state) => (state === "error" ? "error" : "ready"));
    setFps(0);
  }, []);

  const refreshDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) {
      setCameraError("Browser does not expose media device enumeration.");
      setCameraState("error");
      return;
    }

    const mediaDevices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = mediaDevices.filter((device) => device.kind === "videoinput");
    setDevices(videoInputs);
    setSelectedDeviceId((current) => current || videoInputs[0]?.deviceId || "");
  }, []);

  const startCamera = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("Camera access requires a modern browser with getUserMedia support.");
      setCameraState("error");
      return;
    }

    setCameraError("");
    setCameraState("requesting");

    try {
      stopCamera();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: selectedDeviceId
          ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
          : { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      await refreshDevices();
      setCameraState("streaming");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown camera error.";
      setCameraError(message);
      setCameraState("error");
    }
  }, [refreshDevices, selectedDeviceId, stopCamera]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void refreshDevices();
    }, 0);

    return () => {
      window.clearTimeout(timer);
      stopCamera();
    };
  }, [refreshDevices, stopCamera]);

  useEffect(() => {
    let animationFrame = 0;
    let lastPaint = performance.now();
    let frameCount = 0;

    const paint = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas?.getContext("2d");

      if (video && canvas && context && video.videoWidth > 0 && video.videoHeight > 0) {
        const rect = video.getBoundingClientRect();
        const pixelRatio = window.devicePixelRatio || 1;
        const nextWidth = Math.round(rect.width * pixelRatio);
        const nextHeight = Math.round(rect.height * pixelRatio);

        if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
          canvas.width = nextWidth;
          canvas.height = nextHeight;
        }

        context.clearRect(0, 0, canvas.width, canvas.height);
        context.save();
        context.scale(pixelRatio, pixelRatio);
        context.strokeStyle = "rgba(125, 211, 252, 0.85)";
        context.lineWidth = 2;
        context.strokeRect(16, 16, Math.min(220, rect.width - 32), Math.min(120, rect.height - 32));
        context.fillStyle = "rgba(8, 47, 73, 0.86)";
        context.fillRect(16, 16, 126, 26);
        context.fillStyle = "#e0f2fe";
        context.font = "12px ui-sans-serif, system-ui";
        context.fillText("overlay ready", 26, 34);
        context.restore();

        frameCount += 1;
        const now = performance.now();
        if (now - lastPaint >= 1000) {
          setFps(Math.round((frameCount * 1000) / (now - lastPaint)));
          frameCount = 0;
          lastPaint = now;
        }
      }

      animationFrame = requestAnimationFrame(paint);
    };

    animationFrame = requestAnimationFrame(paint);
    return () => cancelAnimationFrame(animationFrame);
  }, []);

  return (
    <main className="app-shell">
      <header className="top-menu">
        <div className="brand-block">
          <span className="brand-mark">V7</span>
          <div>
            <h1>V7RC WebYOLO ByteTrack</h1>
            <p>Chrome-local camera, detection, tracking, and Gemma4-E2B chat</p>
          </div>
        </div>

        <div className="control-strip">
          <label className="field">
            <span>Camera</span>
            <select
              value={selectedDeviceId}
              onChange={(event) => setSelectedDeviceId(event.target.value)}
              disabled={cameraState === "requesting"}
            >
              {devices.length === 0 ? (
                <option value="">No camera listed</option>
              ) : (
                devices.map((device, index) => (
                  <option key={device.deviceId || index} value={device.deviceId}>
                    {device.label || `Camera ${index + 1}`}
                  </option>
                ))
              )}
            </select>
          </label>

          <button
            className="primary-button"
            type="button"
            onClick={cameraState === "streaming" ? stopCamera : startCamera}
            disabled={cameraState === "requesting"}
          >
            {cameraState === "streaming" ? "Stop" : "Start"}
          </button>
        </div>

        <div className="metric-strip">
          <Metric label="FPS" value={fps.toString()} />
          <Metric label="YOLO" value={`${yoloMs.toFixed(1)} ms`} />
          <Metric label="ByteTrack" value={`${trackMs.toFixed(1)} ms`} />
        </div>
      </header>

      <section className="workspace">
        <div className="camera-panel">
          <div className="video-stage">
            <video ref={videoRef} muted playsInline />
            <canvas ref={canvasRef} aria-hidden="true" />
            {cameraState !== "streaming" ? (
              <div className="stage-empty">
                <strong>{cameraState === "requesting" ? "Requesting camera" : "Camera idle"}</strong>
                <span>{cameraError || "Select a camera and start the local pipeline."}</span>
              </div>
            ) : null}
          </div>
        </div>

        <aside className="side-panel">
          <StatusCard status={yoloStatus} />
          <StatusCard status={llmStatus} />
          <div className="object-list">
            <div className="panel-heading">
              <h2>Tracked Objects</h2>
              <span>{emptyTracks.length}</span>
            </div>
            <div className="table-head">
              <span>ID</span>
              <span>Object</span>
              <span>Conf.</span>
            </div>
            {emptyTracks.length === 0 ? (
              <p className="empty-copy">No active tracks yet.</p>
            ) : (
              emptyTracks.map((track) => (
                <div className="track-row" key={track.id}>
                  <span>#{track.id}</span>
                  <span>{track.label}</span>
                  <span>{track.confidence.toFixed(2)}</span>
                </div>
              ))
            )}
          </div>
        </aside>
      </section>

      <section className="chat-panel">
        <div className="chat-log">
          <div className="message system-message">
            Browser-local Gemma4-E2B runtime is planned for the next implementation phase.
          </div>
        </div>
        <form
          className="chat-form"
          onSubmit={(event) => {
            event.preventDefault();
            setChatInput("");
          }}
        >
          <label className="frame-toggle">
            <input
              type="checkbox"
              checked={includeFrame}
              onChange={(event) => setIncludeFrame(event.target.checked)}
            />
            <span>Include current frame</span>
          </label>
          <input
            value={chatInput}
            onChange={(event) => setChatInput(event.target.value)}
            placeholder="Ask about the scene..."
          />
          <button type="submit" disabled={!chatInput.trim()}>
            Send
          </button>
        </form>
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StatusCard({ status }: { status: RuntimeStatus }) {
  return (
    <div className="status-card">
      <div className="status-title">
        <span className={`status-dot ${status.state}`} />
        <h2>{status.label}</h2>
      </div>
      <p>{status.detail}</p>
    </div>
  );
}
