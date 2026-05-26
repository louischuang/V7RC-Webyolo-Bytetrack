"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ByteTracker, type Track } from "./lib/bytetrack";
import { type Detection, YoloDetector } from "./lib/yolo";

type CameraState = "idle" | "requesting" | "ready" | "streaming" | "error";

type RuntimeStatus = {
  label: string;
  state: "idle" | "loading" | "ready" | "error";
  detail: string;
};

type TrackRow = {
  id: string;
  label: string;
  confidence: number;
  ageMs: number;
};

const runtimeDefaults = {
  yoloModelUrl: process.env.NEXT_PUBLIC_YOLO_MODEL_URL ?? "/models/yolo/yolo11n.onnx",
  yoloInputSize: Number(process.env.NEXT_PUBLIC_YOLO_INPUT_SIZE ?? 640),
  yoloConfidenceThreshold: Number(process.env.NEXT_PUBLIC_YOLO_CONF_THRESHOLD ?? 0.25),
  yoloIouThreshold: Number(process.env.NEXT_PUBLIC_YOLO_IOU_THRESHOLD ?? 0.45),
  yoloFrameInterval: Number(process.env.NEXT_PUBLIC_YOLO_FRAME_INTERVAL ?? 3),
  llmModelId: process.env.NEXT_PUBLIC_LLM_MODEL_ID ?? "google/gemma-4-E2B-it",
  llmRuntime: process.env.NEXT_PUBLIC_LLM_RUNTIME ?? "webgpu",
};

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const detectorRef = useRef<YoloDetector | null>(null);
  const trackerRef = useRef<ByteTracker>(
    new ByteTracker({
      highThreshold: Number(process.env.NEXT_PUBLIC_TRACK_HIGH_THRESH ?? 0.6),
      lowThreshold: Number(process.env.NEXT_PUBLIC_TRACK_LOW_THRESH ?? 0.1),
      matchThreshold: Number(process.env.NEXT_PUBLIC_TRACK_MATCH_THRESH ?? 0.8),
      bufferFrames: Number(process.env.NEXT_PUBLIC_TRACK_BUFFER_FRAMES ?? 30),
    }),
  );
  const tracksRef = useRef<Track[]>([]);
  const detectLoopRef = useRef<number | null>(null);
  const detectionFrameRef = useRef(0);
  const detectingRef = useRef(false);
  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [cameraError, setCameraError] = useState<string>("");
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [tracks, setTracks] = useState<Track[]>([]);
  const [fps, setFps] = useState(0);
  const [yoloMs, setYoloMs] = useState(0);
  const [trackMs, setTrackMs] = useState(0);
  const [yoloStatus, setYoloStatus] = useState<RuntimeStatus>({
    label: "YOLO11n",
    state: "loading",
    detail: `Loading ${runtimeDefaults.yoloModelUrl}`,
  });
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

  const trackRows: TrackRow[] = useMemo(
    () =>
      tracks.map((track) => ({
        id: track.id,
        label: track.label,
        confidence: track.confidence,
        ageMs: track.missed,
      })),
    [tracks],
  );

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraState((state) => (state === "error" ? "error" : "ready"));
    trackerRef.current.reset();
    setTracks([]);
    tracksRef.current = [];
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
    let cancelled = false;

    const loadDetector = async () => {
      setYoloStatus({
        label: "YOLO11n",
        state: "loading",
        detail: `Loading ${runtimeDefaults.yoloModelUrl}`,
      });

      try {
        const detector = await YoloDetector.create({
          modelUrl: runtimeDefaults.yoloModelUrl,
          inputSize: runtimeDefaults.yoloInputSize,
          confidenceThreshold: runtimeDefaults.yoloConfidenceThreshold,
          iouThreshold: runtimeDefaults.yoloIouThreshold,
        });

        if (cancelled) {
          return;
        }

        detectorRef.current = detector;
        setYoloStatus({
          label: "YOLO11n",
          state: "ready",
          detail: `${detector.backend.toUpperCase()} / ${runtimeDefaults.yoloModelUrl}`,
        });
      } catch (error) {
        if (cancelled) {
          return;
        }

        const message = error instanceof Error ? error.message : "Unknown YOLO load error.";
        setYoloStatus({
          label: "YOLO11n",
          state: "error",
          detail: message,
        });
      }
    };

    void loadDetector();

    return () => {
      cancelled = true;
      detectorRef.current = null;
    };
  }, []);

  useEffect(() => {
    const detect = async () => {
      const video = videoRef.current;
      const detector = detectorRef.current;

      if (
        cameraState === "streaming" &&
        video &&
        detector &&
        video.videoWidth > 0 &&
        video.videoHeight > 0 &&
        !detectingRef.current
      ) {
        detectionFrameRef.current += 1;

        if (detectionFrameRef.current % Math.max(1, runtimeDefaults.yoloFrameInterval) === 0) {
          detectingRef.current = true;
          const startedAt = performance.now();

          try {
            const nextDetections = await detector.detect(video);
            setYoloMs(performance.now() - startedAt);

            const trackStartedAt = performance.now();
            const nextTracks = trackerRef.current.update(nextDetections);
            tracksRef.current = nextTracks;
            setTracks(nextTracks);
            setTrackMs(performance.now() - trackStartedAt);
          } catch (error) {
            const message = error instanceof Error ? error.message : "Unknown YOLO inference error.";
            setYoloStatus((status) => ({
              ...status,
              state: "error",
              detail: message,
            }));
          } finally {
            detectingRef.current = false;
          }
        }
      }

      detectLoopRef.current = window.requestAnimationFrame(detect);
    };

    detectLoopRef.current = window.requestAnimationFrame(detect);

    return () => {
      if (detectLoopRef.current !== null) {
        window.cancelAnimationFrame(detectLoopRef.current);
      }
    };
  }, [cameraState]);

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
        drawDetections(context, tracksRef.current, video, rect.width, rect.height);
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
              <span>{trackRows.length}</span>
            </div>
            <div className="table-head">
              <span>ID</span>
              <span>Object</span>
              <span>Conf.</span>
            </div>
            {trackRows.length === 0 ? (
              <p className="empty-copy">No active tracks yet.</p>
            ) : (
              trackRows.map((track) => (
                <div className="track-row" key={track.id}>
                  <span>{track.id}</span>
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

function drawDetections(
  context: CanvasRenderingContext2D,
  detections: Detection[],
  video: HTMLVideoElement,
  stageWidth: number,
  stageHeight: number,
) {
  if (detections.length === 0) {
    context.strokeStyle = "rgba(125, 211, 252, 0.4)";
    context.lineWidth = 1.5;
    context.strokeRect(16, 16, Math.min(220, stageWidth - 32), Math.min(120, stageHeight - 32));
    context.fillStyle = "rgba(8, 47, 73, 0.78)";
    context.fillRect(16, 16, 126, 26);
    context.fillStyle = "#e0f2fe";
    context.font = "12px ui-sans-serif, system-ui";
    context.fillText("overlay ready", 26, 34);
    return;
  }

  const videoRatio = video.videoWidth / video.videoHeight;
  const stageRatio = stageWidth / stageHeight;
  const drawWidth = stageRatio > videoRatio ? stageHeight * videoRatio : stageWidth;
  const drawHeight = stageRatio > videoRatio ? stageHeight : stageWidth / videoRatio;
  const offsetX = (stageWidth - drawWidth) / 2;
  const offsetY = (stageHeight - drawHeight) / 2;
  const scaleX = drawWidth / video.videoWidth;
  const scaleY = drawHeight / video.videoHeight;

  for (const detection of detections) {
    const x = offsetX + detection.box.x * scaleX;
    const y = offsetY + detection.box.y * scaleY;
    const width = detection.box.width * scaleX;
    const height = detection.box.height * scaleY;
    const label = `${detection.label} ${detection.id} ${detection.confidence.toFixed(2)}`;
    const labelWidth = Math.max(92, context.measureText(label).width + 16);
    const labelY = y > 30 ? y - 28 : y + 4;

    context.strokeStyle = "#2dd4bf";
    context.lineWidth = 2;
    context.strokeRect(x, y, width, height);

    context.fillStyle = "rgba(15, 23, 42, 0.92)";
    context.fillRect(x, labelY, labelWidth, 24);
    context.fillStyle = "#ccfbf1";
    context.font = "12px ui-sans-serif, system-ui";
    context.fillText(label, x + 8, labelY + 16);
  }
}
