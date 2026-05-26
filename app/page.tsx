"use client";

import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ByteTracker, type Track } from "./lib/bytetrack";
import { BrowserLlm, type BrowserLlmMessage, type BrowserLlmStatus, getWebGpuStatus } from "./lib/browser-llm";
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

const defaultLlmRuntime = process.env.NEXT_PUBLIC_LLM_RUNTIME ?? "transformers";
const defaultLlmModelId =
  defaultLlmRuntime === "transformers" ? "onnx-community/gemma-4-E2B-it-ONNX" : "gemma-4-E2B-it-q4f16_1-MLC";

const runtimeDefaults = {
  yoloModelUrl: process.env.NEXT_PUBLIC_YOLO_MODEL_URL ?? "/models/yolo/yolo11n.onnx",
  yoloInputSize: Number(process.env.NEXT_PUBLIC_YOLO_INPUT_SIZE ?? 640),
  yoloConfidenceThreshold: Number(process.env.NEXT_PUBLIC_YOLO_CONF_THRESHOLD ?? 0.25),
  yoloIouThreshold: Number(process.env.NEXT_PUBLIC_YOLO_IOU_THRESHOLD ?? 0.45),
  yoloFrameInterval: Number(process.env.NEXT_PUBLIC_YOLO_FRAME_INTERVAL ?? 3),
  llmModelId: process.env.NEXT_PUBLIC_LLM_MODEL_ID ?? defaultLlmModelId,
  llmModelUrl:
    process.env.NEXT_PUBLIC_LLM_MODEL_URL ??
    (defaultLlmRuntime === "transformers" ? "onnx-community/gemma-4-E2B-it-ONNX" : "/models/gemma4-e2b-it"),
  llmModelLibUrl:
    process.env.NEXT_PUBLIC_LLM_MODEL_LIB_URL ??
    "/models/gemma4-e2b-it/libs/gemma-4-E2B-it-q4f16_1-MLC-webgpu.wasm",
  llmMaxNewTokens: Number(process.env.NEXT_PUBLIC_LLM_MAX_NEW_TOKENS ?? 512),
  llmTemperature: Number(process.env.NEXT_PUBLIC_LLM_TEMPERATURE ?? 0.2),
  llmRuntime: defaultLlmRuntime,
};

const defaultSystemPrompt =
  "You are the perception and planning module for a mobile robot. Use the camera image, YOLO11n detections, and ByteTrack IDs to understand the world. Reply with concise observations and action-oriented guidance for future motor control, but do not claim that you have directly controlled any motors yet.";

const defaultFixedPrompt =
  "Focus on navigable space, nearby people or obstacles, tracked object IDs, and any motion-relevant risk.";

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const detectorRef = useRef<YoloDetector | null>(null);
  const llmRef = useRef<BrowserLlm | null>(null);
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
  const inferenceLoopRef = useRef(false);
  const inferenceRunningRef = useRef(false);
  const inferenceRoundRef = useRef(0);
  const runInferenceRoundRef = useRef<() => Promise<void>>(async () => {});
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
  const [mirrorPreview, setMirrorPreview] = useState(true);
  const [systemPrompt, setSystemPrompt] = useState(defaultSystemPrompt);
  const [fixedPrompt, setFixedPrompt] = useState(defaultFixedPrompt);
  const [chatMessages, setChatMessages] = useState<BrowserLlmMessage[]>([]);
  const [llmState, setLlmState] = useState<BrowserLlmStatus>("checking");
  const [llmDetail, setLlmDetail] = useState("Checking WebGPU support...");
  const [llmProgress, setLlmProgress] = useState<number | null>(null);
  const [includeFrame, setIncludeFrame] = useState(true);
  const [loopRunning, setLoopRunning] = useState(false);
  const [lastInferenceMs, setLastInferenceMs] = useState<number | null>(null);

  const llmStatus: RuntimeStatus = useMemo(
    () => ({
      label: "Gemma4-E2B",
      state: llmState === "checking" || llmState === "loading" || llmState === "generating" ? "loading" : llmState,
      detail: llmDetail,
    }),
    [llmDetail, llmState],
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
  const selectedCamera = useMemo(
    () => devices.find((device) => device.deviceId === selectedDeviceId) ?? null,
    [devices, selectedDeviceId],
  );
  const cameraDetail = useMemo(() => {
    if (devices.length === 0) {
      return "No camera detected";
    }

    if (!selectedCamera) {
      return `${devices.length} camera${devices.length > 1 ? "s" : ""} available`;
    }

    return isIphoneCamera(selectedCamera)
      ? "iPhone camera selected"
      : `${devices.length} camera${devices.length > 1 ? "s" : ""} available`;
  }, [devices, selectedCamera]);

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
    setSelectedDeviceId((current) => chooseCameraDeviceId(videoInputs, current));
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
    const handleDeviceChange = () => {
      void refreshDevices();
    };

    navigator.mediaDevices?.addEventListener?.("devicechange", handleDeviceChange);

    return () => {
      window.clearTimeout(timer);
      navigator.mediaDevices?.removeEventListener?.("devicechange", handleDeviceChange);
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
    let cancelled = false;

    const checkGpu = async () => {
      setLlmState("checking");
      const status = await getWebGpuStatus();
      if (cancelled) {
        return;
      }

      if (status.ok) {
        setLlmState("idle");
      } else {
        setLlmState("error");
      }
      setLlmDetail(status.detail);
    };

    void checkGpu();

    return () => {
      cancelled = true;
    };
  }, []);

  const loadGemma = useCallback(async () => {
    setLlmState("loading");
    setLlmProgress(0);
    setLlmDetail(`Loading ${runtimeDefaults.llmModelId}`);

    try {
      const llm = new BrowserLlm({
        runtime: runtimeDefaults.llmRuntime === "webllm" ? "webllm" : "transformers",
        modelId: runtimeDefaults.llmModelId,
        modelUrl: runtimeDefaults.llmModelUrl,
        modelLibUrl: runtimeDefaults.llmModelLibUrl,
        maxNewTokens: runtimeDefaults.llmMaxNewTokens,
        temperature: runtimeDefaults.llmTemperature,
      });

      await llm.load((progress) => {
        setLlmDetail(progress.text);
        setLlmProgress(typeof progress.progress === "number" ? progress.progress : null);
      });

      llmRef.current = llm;
      setLlmState("ready");
      setLlmProgress(null);
      setLlmDetail(`${runtimeDefaults.llmRuntime} ready / ${runtimeDefaults.llmModelId}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown Gemma load error.";
      setLlmState("error");
      setLlmProgress(null);
      setLlmDetail(message);
    }
  }, []);

  const runInferenceRound = useCallback(async () => {
    if (inferenceRunningRef.current) {
      return;
    }

    if (!llmRef.current) {
      setChatMessages((messages) => [
        ...messages,
        { role: "assistant", content: "Load Gemma4-E2B first, then start the perception loop." },
      ]);
      inferenceLoopRef.current = false;
      setLoopRunning(false);
      return;
    }

    const prompt = fixedPrompt.trim() || defaultFixedPrompt;
    const sceneSummary = includeFrame ? buildSceneSummary(tracksRef.current) : "";
    const imageDataUrl = includeFrame ? captureVideoFrame(videoRef.current) : undefined;
    const userMessage = buildGemmaUserPrompt(prompt, sceneSummary, true);
    const nextMessages: BrowserLlmMessage[] = [
      ...(systemPrompt.trim() ? [{ role: "system" as const, content: systemPrompt.trim() }] : []),
      { role: "user", content: userMessage },
    ];
    const round = inferenceRoundRef.current + 1;
    const startedAt = performance.now();

    inferenceRunningRef.current = true;
    inferenceRoundRef.current = round;
    setChatMessages((messages) => [
      ...messages.slice(-24),
      { role: "user", content: `[Round ${round}] ${prompt}` },
    ]);
    setLlmState("generating");
    setLlmDetail(`Generating local loop round ${round}...`);

    try {
      const generation = await llmRef.current.generate(nextMessages, imageDataUrl);
      const elapsedMs = performance.now() - startedAt;
      const emptyResponse = !generation.text.trim() || isDegenerateLlmResponse(generation.text);
      const fallbackResponse = emptyResponse ? buildLocalFallbackResponse(sceneSummary, generation.diagnostics) : "";
      setLastInferenceMs(elapsedMs);
      setChatMessages((messages) => [
        ...messages.slice(-24),
        {
          role: "assistant",
          content: `${emptyResponse ? fallbackResponse : generation.text}\n\nInference time: ${formatDuration(elapsedMs)}`,
        },
      ]);
      setLlmState("ready");
      setLlmDetail(`${runtimeDefaults.llmRuntime} ready / ${runtimeDefaults.llmModelId}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown Gemma generation error.";
      setChatMessages((messages) => [...messages, { role: "assistant", content: message }]);
      setLlmState("error");
      setLlmDetail(message);
      inferenceLoopRef.current = false;
      setLoopRunning(false);
    } finally {
      inferenceRunningRef.current = false;
      if (inferenceLoopRef.current) {
        window.setTimeout(() => {
          void runInferenceRoundRef.current();
        }, 100);
      }
    }
  }, [fixedPrompt, includeFrame, systemPrompt]);

  useEffect(() => {
    runInferenceRoundRef.current = runInferenceRound;
  }, [runInferenceRound]);

  const toggleInferenceLoop = useCallback(() => {
    if (inferenceLoopRef.current) {
      inferenceLoopRef.current = false;
      setLoopRunning(false);
      setLlmDetail(
        inferenceRunningRef.current
          ? "Stopping after the current local inference finishes..."
          : llmRef.current
            ? `${runtimeDefaults.llmRuntime} ready / ${runtimeDefaults.llmModelId}`
            : llmDetail,
      );
      return;
    }

    inferenceLoopRef.current = true;
    setLoopRunning(true);
    void runInferenceRoundRef.current();
  }, [llmDetail]);

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
        drawDetections(context, tracksRef.current, video, rect.width, rect.height, mirrorPreview);
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
  }, [mirrorPreview]);

  return (
    <main className="app-shell">
      <header className="top-menu">
        <div className="brand-block">
          <span className="brand-mark">V7</span>
          <div>
            <h1>V7RC WebYOLO ByteTrack</h1>
            <p>Robot perception loop with camera, detection, tracking, and Gemma4-E2B</p>
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
                    {formatCameraLabel(device, index)}
                  </option>
                ))
              )}
            </select>
            <small>{cameraDetail}</small>
          </label>
          <button
            className="icon-button"
            type="button"
            onClick={() => void refreshDevices()}
            disabled={cameraState === "requesting"}
            title="Refresh cameras"
          >
            Refresh
          </button>

          <button
            className="primary-button"
            type="button"
            onClick={cameraState === "streaming" ? stopCamera : startCamera}
            disabled={cameraState === "requesting"}
          >
            {cameraState === "streaming" ? "Stop" : "Start"}
          </button>
          <label className="inline-toggle">
            <input
              type="checkbox"
              checked={mirrorPreview}
              onChange={(event) => setMirrorPreview(event.target.checked)}
            />
            <span>Mirror</span>
          </label>
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
            <video className={mirrorPreview ? "mirrored" : undefined} ref={videoRef} muted playsInline />
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
          <StatusCard
            status={llmStatus}
            action={
              <button
                className="secondary-button compact-button"
                type="button"
                onClick={loadGemma}
                disabled={llmState === "loading" || llmState === "generating" || llmState === "ready"}
              >
                {llmState === "ready" ? "Loaded" : "Load"}
              </button>
            }
            progress={llmProgress}
          />
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
        <div className="prompt-panel">
          <label className="prompt-field">
            <span>System Prompt</span>
            <textarea value={systemPrompt} onChange={(event) => setSystemPrompt(event.target.value)} rows={5} />
          </label>
          <label className="prompt-field">
            <span>Fixed Prompt</span>
            <textarea value={fixedPrompt} onChange={(event) => setFixedPrompt(event.target.value)} rows={3} />
          </label>
        </div>

        <div className="conversation-panel">
          <div className="chat-log">
            {chatMessages.map((message, index) => (
              <div className={`message ${message.role}-message`} key={`${message.role}-${index}`}>
                {message.content}
              </div>
            ))}
          </div>
          <div className="loop-controls">
            <label className="frame-toggle">
              <input
                type="checkbox"
                checked={includeFrame}
                onChange={(event) => setIncludeFrame(event.target.checked)}
              />
              <span>Include current frame</span>
            </label>
            {lastInferenceMs !== null && <span className="loop-timing">Last run {formatDuration(lastInferenceMs)}</span>}
            <button type="button" onClick={toggleInferenceLoop} disabled={llmState === "loading"}>
              {loopRunning ? "停止" : "開始"}
            </button>
          </div>
        </div>
      </section>
    </main>
  );
}

function buildSceneSummary(tracks: Track[]) {
  if (tracks.length === 0) {
    return "No active tracked objects.";
  }

  return tracks
    .slice(0, 12)
    .map((track) => `${track.id}: ${track.label}, confidence ${track.confidence.toFixed(2)}`)
    .join("\n");
}

function chooseCameraDeviceId(devices: MediaDeviceInfo[], currentDeviceId: string) {
  if (currentDeviceId && devices.some((device) => device.deviceId === currentDeviceId)) {
    return currentDeviceId;
  }

  return devices.find(isIphoneCamera)?.deviceId || devices[0]?.deviceId || "";
}

function isIphoneCamera(device: MediaDeviceInfo) {
  const label = device.label.toLowerCase();
  return (
    label.includes("iphone") ||
    label.includes("continuity") ||
    label.includes("desk view") ||
    label.includes("接續互通") ||
    label.includes("連續互通")
  );
}

function formatCameraLabel(device: MediaDeviceInfo, index: number) {
  const label = device.label || `Camera ${index + 1}`;
  return isIphoneCamera(device) ? `${label} (iPhone)` : label;
}

function buildGemmaUserPrompt(prompt: string, sceneSummary: string, includeInstructions: boolean) {
  const instructions =
    "You are a concise local vision assistant. Use the attached image when provided, and use the tracked scene summary as supporting metadata.";
  const basePrompt = includeInstructions ? `${instructions}\n\n${prompt}` : prompt;

  if (!sceneSummary) {
    return basePrompt;
  }

  return `${basePrompt}\n\nCurrent tracked scene:\n${sceneSummary}`;
}

function captureVideoFrame(video: HTMLVideoElement | null) {
  if (!video || video.videoWidth <= 0 || video.videoHeight <= 0) {
    return undefined;
  }

  const maxSide = 768;
  const scale = Math.min(1, maxSide / Math.max(video.videoWidth, video.videoHeight));
  const width = Math.max(1, Math.round(video.videoWidth * scale));
  const height = Math.max(1, Math.round(video.videoHeight * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) {
    return undefined;
  }

  context.drawImage(video, 0, 0, width, height);
  return canvas.toDataURL("image/jpeg", 0.85);
}

function buildLocalFallbackResponse(sceneSummary: string, diagnostics: string[]) {
  const sceneLine = sceneSummary
    ? `Local scene summary: ${sceneSummary.replace(/\n/gu, " ")}`
    : "Local scene summary: no scene summary was attached to this prompt.";

  return `Gemma generated only control tokens for this request. ${sceneLine}\n\nDiagnostics:\n${diagnostics.join("\n")}`;
}

function isDegenerateLlmResponse(response: string) {
  const text = response.trim();
  return text.length > 0 && text.length < 8 && !/[a-z0-9\u4e00-\u9fff]/iu.test(text);
}

function formatDuration(milliseconds: number) {
  if (milliseconds < 1000) {
    return `${Math.round(milliseconds)} ms`;
  }

  return `${(milliseconds / 1000).toFixed(2)} s`;
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StatusCard({ status, action, progress }: { status: RuntimeStatus; action?: ReactNode; progress?: number | null }) {
  return (
    <div className="status-card">
      <div className="status-title">
        <div className="status-heading">
          <span className={`status-dot ${status.state}`} />
          <h2>{status.label}</h2>
        </div>
        {action}
      </div>
      <p>{status.detail}</p>
      {progress !== null && progress !== undefined ? (
        <div className="progress-track status-progress" aria-label={`${status.label} load progress`}>
          <span style={{ width: `${Math.round(progress * 100)}%` }} />
        </div>
      ) : null}
    </div>
  );
}

function drawDetections(
  context: CanvasRenderingContext2D,
  detections: Detection[],
  video: HTMLVideoElement,
  stageWidth: number,
  stageHeight: number,
  mirrorPreview: boolean,
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
    const boxColor = "#2dd4bf";
    const unmirroredX = offsetX + detection.box.x * scaleX;
    const y = offsetY + detection.box.y * scaleY;
    const width = detection.box.width * scaleX;
    const height = detection.box.height * scaleY;
    const x = mirrorPreview ? stageWidth - unmirroredX - width : unmirroredX;
    const label = `${detection.label} ${detection.id} ${detection.confidence.toFixed(2)}`;
    const labelY = y > 30 ? y - 28 : y + 4;

    context.strokeStyle = boxColor;
    context.lineWidth = 2;
    context.strokeRect(x, y, width, height);

    context.fillStyle = boxColor;
    context.fillRect(x, labelY, width, 24);
    context.fillStyle = "#020617";
    context.font = "12px ui-sans-serif, system-ui";
    context.save();
    context.beginPath();
    context.rect(x, labelY, width, 24);
    context.clip();
    context.fillText(label, x + 8, labelY + 16);
    context.restore();
  }
}
