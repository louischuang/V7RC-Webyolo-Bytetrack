"use client";

import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type Hls from "hls.js";
import { ByteTracker, type Track } from "./lib/bytetrack";
import { BrowserLlm, type BrowserLlmMessage, type BrowserLlmStatus, getWebGpuStatus } from "./lib/browser-llm";
import {
  createNeutralIntent,
  encodeSrtFrame,
  frameToDebugString,
  intentToSrtPwm,
  previewSrtChannels,
  type V7rcDriveMode,
  type V7rcRobotIntent,
} from "./lib/v7rc-protocol";
import {
  canUseWebBluetooth,
  createMockV7rcTransport,
  createWebBluetoothV7rcTransport,
  type V7rcTransport,
  type V7rcTransportStatus,
} from "./lib/v7rc-transport";
import { type DetectableSource, type Detection, YoloDetector } from "./lib/yolo";

type CameraState = "idle" | "requesting" | "ready" | "streaming" | "error";
type SourceMode = "camera" | "mjpg" | "rtsp" | "youtube";
type SourceSurface = "video" | "image";
type GatewayOutput = "mjpg" | "hls" | "mp4";
type YoutubeOutput = "hls" | "mp4";

type GatewayStreamResponse = {
  id: string;
  output: GatewayOutput;
  status: string;
  url: string;
};

type GatewaySessionStatus = {
  id: string;
  output: GatewayOutput;
  status: string;
  clients: number;
  lastError?: string;
  logs?: Array<{ at: string; message: string }>;
};

type YoutubeResolveResult = {
  ok: boolean;
  protocol: string;
  mediaHost: string;
  durationMs: number;
  format: string;
};

type GatewayStatus = "idle" | "checking" | "ready" | "connecting" | "streaming" | "error";

type RuntimeStatus = {
  label: string;
  state: "idle" | "loading" | "ready" | "generating" | "error";
  detail: string;
};

type TrackRow = {
  id: string;
  label: string;
  confidence: number;
  ageMs: number;
};

type RobotMode = "suggestion" | "armed";
type RobotTaskMode = "autopilot" | "mission";
type RobotTaskStatus = "idle" | "running" | "complete" | "blocked" | "unsafe" | "error";
type LlmDevice = "wasm" | "webgpu";
type NormalizedPoint = { x: number; y: number };
type RoadCalibration = {
  bottomLeftX: number;
  bottomRightX: number;
  bottomY: number;
  topLeftX: number;
  topRightX: number;
  topY: number;
};
type LaneBand = {
  birdLeft: NormalizedPoint[];
  birdRight: NormalizedPoint[];
  sourceLeft: NormalizedPoint[];
  sourceRight: NormalizedPoint[];
};
type LaneDetection = {
  birdPaths: NormalizedPoint[][];
  laneBands: LaneBand[];
  left: NormalizedPoint[][];
  confidence: number;
  roiConfidence: number;
  road: RoadCalibration;
  right: NormalizedPoint[][];
  sourcePaths: NormalizedPoint[][];
  updatedAt: number;
};
type LaneDetectionOptions = {
  joinGapY: number;
  minPixelScore: number;
};

const defaultLlmRuntime = process.env.NEXT_PUBLIC_LLM_RUNTIME ?? "transformers";
const defaultLlmDevice: LlmDevice = process.env.NEXT_PUBLIC_LLM_DEVICE === "wasm" ? "wasm" : "webgpu";
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
  llmMaxNewTokens: Number(process.env.NEXT_PUBLIC_LLM_MAX_NEW_TOKENS ?? 160),
  llmTemperature: Number(process.env.NEXT_PUBLIC_LLM_TEMPERATURE ?? 0.2),
  llmRuntime: defaultLlmRuntime,
  llmDevice: defaultLlmDevice,
  llmLoopDelayMs: Number(process.env.NEXT_PUBLIC_LLM_LOOP_DELAY_MS ?? 1200),
  llmFrameMaxSide: Number(process.env.NEXT_PUBLIC_LLM_FRAME_MAX_SIDE ?? 448),
  llmFrameJpegQuality: Number(process.env.NEXT_PUBLIC_LLM_FRAME_JPEG_QUALITY ?? 0.72),
  streamGatewayUrl: process.env.NEXT_PUBLIC_STREAM_GATEWAY_URL ?? "http://localhost:3010",
};

const legacyDefaultSystemPrompt =
  "You are the perception and planning module for a mobile robot. Use the camera image, YOLO11n detections, and ByteTrack IDs to understand the world. Reply with concise observations and action-oriented guidance for future motor control, but do not claim that you have directly controlled any motors yet.";

const legacyDefaultFixedPrompt =
  "Focus on navigable space, nearby people or obstacles, tracked object IDs, and any motion-relevant risk.";

const defaultSystemPrompt =
  "你是行動機器人的感知與規劃模組。請使用目前的攝影機影像、YOLO11n 偵測結果，以及 ByteTrack 追蹤 ID 來理解周圍世界。回覆時請保持精簡，描述重要觀察，並提供可用於後續馬達控制的行動建議；但不要宣稱你已經直接控制任何馬達。";

const defaultFixedPrompt =
  "請專注於可通行空間、附近的人或障礙物、正在追蹤的物件 ID，以及任何與移動安全相關的風險。";

const gemmaSettingsStorageKey = "v7rc.gemma4-e2b.settings.v1";
const birdViewSettingsStorageKey = "v7rc.bird-view.settings.v1";
const sourceModes: Array<{ id: SourceMode; label: string }> = [
  { id: "camera", label: "Camera" },
  { id: "mjpg", label: "MJPG" },
  { id: "rtsp", label: "RTSP" },
  { id: "youtube", label: "YouTube" },
];
const robotDriveModes: Array<{ id: V7rcDriveMode; label: string }> = [
  { id: "vehicle", label: "車輛" },
  { id: "mecanum", label: "麥克納姆輪" },
  { id: "tank", label: "坦克" },
];
const robotTaskModes: Array<{ id: RobotTaskMode; label: string }> = [
  { id: "autopilot", label: "自動駕駛" },
  { id: "mission", label: "解任務" },
];
const robotCommandIntervalMs = 30;
const laneDetectionIntervalMs = 140;
const birdViewDefaultTopY = 0.65;
const birdViewDefaultTopCenterX = 0.5;
const birdViewDefaultTopWidth = 0.16;
const birdViewDefaultBottomY = 0.93;
const birdViewDefaultBottomWidth = 0.9;
const birdViewDefaultHeightScale = 1.5;
const birdViewBaseWidth = 320;
const birdViewBaseHeight = 220;
const laneDetectionDefaultJoinGapY = 0.16;
const laneDetectionDefaultMinPixelScore = 18;
const laneDetectionDefaultSmoothing = 0.62;

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const birdViewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const laneDetectionCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const chatLogRef = useRef<HTMLDivElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const hlsRef = useRef<Hls | null>(null);
  const detectorRef = useRef<YoloDetector | null>(null);
  const llmRef = useRef<BrowserLlm | null>(null);
  const robotTransportRef = useRef<V7rcTransport | null>(null);
  const robotIntentRef = useRef<V7rcRobotIntent>(createNeutralIntent());
  const robotDriveModeRef = useRef<V7rcDriveMode>("vehicle");
  const robotWriteInFlightRef = useRef(false);
  const robotLastStatusUpdateRef = useRef(0);
  const laneDetectionRef = useRef<LaneDetection | null>(null);
  const roadCalibrationRef = useRef<RoadCalibration>(
    createManualRoadCalibration(
      birdViewDefaultTopY,
      birdViewDefaultTopCenterX,
      birdViewDefaultTopWidth,
      birdViewDefaultBottomY,
      birdViewDefaultBottomWidth,
    ),
  );
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
  const gatewayStreamIdRef = useRef<string | null>(null);
  const inferenceLoopRef = useRef(false);
  const inferenceRunningRef = useRef(false);
  const inferenceRoundRef = useRef(0);
  const runInferenceRoundRef = useRef<() => Promise<void>>(async () => {});
  const sourceDeepLinkAppliedRef = useRef(false);
  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [cameraError, setCameraError] = useState<string>("");
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [sourceMode, setSourceMode] = useState<SourceMode>("camera");
  const [sourceSurface, setSourceSurface] = useState<SourceSurface>("video");
  const [youtubeOutput, setYoutubeOutput] = useState<YoutubeOutput>("mp4");
  const [streamUrls, setStreamUrls] = useState<Record<Exclude<SourceMode, "camera">, string>>({
    mjpg: "",
    rtsp: "",
    youtube: "",
  });
  const [gatewayStreamId, setGatewayStreamId] = useState<string>("");
  const [gatewayStatus, setGatewayStatus] = useState<GatewayStatus>("idle");
  const [gatewayDetail, setGatewayDetail] = useState("");
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
  const [responseCount, setResponseCount] = useState(0);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [cameraSettingsOpen, setCameraSettingsOpen] = useState(false);
  const [settingsHydrated, setSettingsHydrated] = useState(false);
  const [birdViewSettingsHydrated, setBirdViewSettingsHydrated] = useState(false);
  const [robotMode, setRobotMode] = useState<RobotMode>("suggestion");
  const [robotStatus, setRobotStatus] = useState<V7rcTransportStatus>({
    connected: false,
    deviceName: "V7RC Robot",
    lastMessage: "Robot transport is idle.",
    lastPacket: "",
    mode: "mock",
  });
  const [robotError, setRobotError] = useState("");
  const [robotIntent, setRobotIntent] = useState<V7rcRobotIntent>(() => createNeutralIntent());
  const [robotDriveMode, setRobotDriveMode] = useState<V7rcDriveMode>("vehicle");
  const [robotTaskMode, setRobotTaskMode] = useState<RobotTaskMode>("autopilot");
  const [robotTaskStatus, setRobotTaskStatus] = useState<RobotTaskStatus>("idle");
  const [robotTaskMessage, setRobotTaskMessage] = useState("選擇任務模式後按 Start。");
  const [laneConfidence, setLaneConfidence] = useState(0);
  const [roiTopY, setRoiTopY] = useState(birdViewDefaultTopY);
  const [roiTopCenterX, setRoiTopCenterX] = useState(birdViewDefaultTopCenterX);
  const [roiTopWidth, setRoiTopWidth] = useState(birdViewDefaultTopWidth);
  const [roiBottomY, setRoiBottomY] = useState(birdViewDefaultBottomY);
  const [roiBottomWidth, setRoiBottomWidth] = useState(birdViewDefaultBottomWidth);
  const [birdViewHeightScale, setBirdViewHeightScale] = useState(birdViewDefaultHeightScale);
  const [laneJoinGapY, setLaneJoinGapY] = useState(laneDetectionDefaultJoinGapY);
  const [laneMinPixelScore, setLaneMinPixelScore] = useState(laneDetectionDefaultMinPixelScore);
  const [laneSmoothing, setLaneSmoothing] = useState(laneDetectionDefaultSmoothing);
  const [roiConfidence, setRoiConfidence] = useState(0);

  const llmStatus: RuntimeStatus = useMemo(
    () => ({
      label: "Gemma4-E2B",
      state: llmState === "checking" || llmState === "loading" ? "loading" : llmState,
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

  const robotRuntimeStatus: RuntimeStatus = useMemo(
    () => ({
      detail: robotError || `${robotStatus.deviceName} / ${robotStatus.lastMessage}`,
      label: "Robot / V7RC",
      state: robotError ? "error" : robotStatus.connected ? "ready" : "idle",
    }),
    [robotError, robotStatus],
  );

  const robotChannelPreview = useMemo(() => previewSrtChannels(robotIntent, robotDriveMode).slice(0, 4), [robotDriveMode, robotIntent]);

  const robotPacketPreview = useMemo(
    () => frameToDebugString(encodeSrtFrame(intentToSrtPwm(robotIntent, robotDriveMode))),
    [robotDriveMode, robotIntent],
  );
  const robotTaskRunning = robotTaskStatus === "running";
  const robotTaskDetail = useMemo(() => {
    if (robotTaskMode === "autopilot") {
      return `Lane candidate detector active (${Math.round(laneConfidence * 100)}%). Manual ROI confidence ${Math.round(roiConfidence * 100)}%.`;
    }

    return "Gemma JSON mission planner pending; controller will expand plans into 30ms SRT frames.";
  }, [laneConfidence, robotTaskMode, roiConfidence]);
  const selectedCamera = useMemo(
    () => devices.find((device) => device.deviceId === selectedDeviceId) ?? null,
    [devices, selectedDeviceId],
  );
  const cameraDetail = useMemo(() => {
    if (sourceMode !== "camera") {
      return `${sourceMode.toUpperCase()} stream URL`;
    }

    if (devices.length === 0) {
      return "No camera detected";
    }

    if (!selectedCamera) {
      return `${devices.length} camera${devices.length > 1 ? "s" : ""} available`;
    }

    return isIphoneCamera(selectedCamera)
      ? "iPhone camera selected"
      : `${devices.length} camera${devices.length > 1 ? "s" : ""} available`;
  }, [devices, selectedCamera, sourceMode]);
  const cameraStatus: RuntimeStatus = useMemo(
    () => ({
      detail: cameraError || cameraDetail,
      label: "Camera",
      state: cameraState === "error" ? "error" : cameraState === "requesting" ? "loading" : cameraState === "streaming" ? "ready" : "idle",
    }),
    [cameraDetail, cameraError, cameraState],
  );
  const sourceSummary = useMemo(() => {
    if (sourceMode === "camera") {
      return selectedCamera ? formatCameraLabel(selectedCamera, devices.indexOf(selectedCamera)) : "Camera source";
    }

    const url = streamUrls[sourceMode].trim();
    return url ? `${sourceMode.toUpperCase()} / ${url}` : `${sourceMode.toUpperCase()} source`;
  }, [devices, selectedCamera, sourceMode, streamUrls]);

  const resetBirdViewSettings = useCallback(() => {
    setBirdViewHeightScale(birdViewDefaultHeightScale);
    setLaneJoinGapY(laneDetectionDefaultJoinGapY);
    setLaneMinPixelScore(laneDetectionDefaultMinPixelScore);
    setLaneSmoothing(laneDetectionDefaultSmoothing);
    setRoiBottomWidth(birdViewDefaultBottomWidth);
    setRoiBottomY(birdViewDefaultBottomY);
    setRoiConfidence(0);
    setRoiTopCenterX(birdViewDefaultTopCenterX);
    setRoiTopWidth(birdViewDefaultTopWidth);
    setRoiTopY(birdViewDefaultTopY);
  }, []);

  useEffect(() => {
    let cachedSettings: Partial<{
      includeFrame: boolean;
      systemPrompt: string;
      fixedPrompt: string;
    }> | null = null;

    try {
      const cached = window.localStorage.getItem(gemmaSettingsStorageKey);
      if (!cached) {
        return;
      }

      cachedSettings = JSON.parse(cached) as Partial<{
        includeFrame: boolean;
        systemPrompt: string;
        fixedPrompt: string;
      }>;
    } catch {
      window.localStorage.removeItem(gemmaSettingsStorageKey);
    } finally {
      queueMicrotask(() => {
        if (typeof cachedSettings?.includeFrame === "boolean") {
          setIncludeFrame(cachedSettings.includeFrame);
        }
        if (typeof cachedSettings?.systemPrompt === "string") {
          setSystemPrompt(
            cachedSettings.systemPrompt === legacyDefaultSystemPrompt
              ? defaultSystemPrompt
              : cachedSettings.systemPrompt,
          );
        }
        if (typeof cachedSettings?.fixedPrompt === "string") {
          setFixedPrompt(
            cachedSettings.fixedPrompt === legacyDefaultFixedPrompt ? defaultFixedPrompt : cachedSettings.fixedPrompt,
          );
        }
        setSettingsHydrated(true);
      });
    }
  }, []);

  useEffect(() => {
    if (!settingsHydrated) {
      return;
    }

    window.localStorage.setItem(
      gemmaSettingsStorageKey,
      JSON.stringify({
        includeFrame,
        systemPrompt,
        fixedPrompt,
      }),
    );
  }, [fixedPrompt, includeFrame, settingsHydrated, systemPrompt]);

  useEffect(() => {
    let cachedSettings: Partial<{
      birdViewHeightScale: number;
      laneJoinGapY: number;
      laneMinPixelScore: number;
      laneSmoothing: number;
      roiBottomWidth: number;
      roiBottomY: number;
      roiTopCenterX: number;
      roiTopWidth: number;
      roiTopY: number;
    }> | null = null;

    try {
      const cached = window.localStorage.getItem(birdViewSettingsStorageKey);
      if (cached) {
        cachedSettings = JSON.parse(cached) as typeof cachedSettings;
      }
    } catch {
      window.localStorage.removeItem(birdViewSettingsStorageKey);
    } finally {
      queueMicrotask(() => {
        if (typeof cachedSettings?.birdViewHeightScale === "number") {
          setBirdViewHeightScale(clamp(cachedSettings.birdViewHeightScale, 1, 2));
        }
        if (typeof cachedSettings?.laneJoinGapY === "number") {
          setLaneJoinGapY(clamp(cachedSettings.laneJoinGapY, 0.04, 0.25));
        }
        if (typeof cachedSettings?.laneMinPixelScore === "number") {
          setLaneMinPixelScore(clamp(cachedSettings.laneMinPixelScore, 8, 42));
        }
        if (typeof cachedSettings?.laneSmoothing === "number") {
          setLaneSmoothing(clamp(cachedSettings.laneSmoothing, 0, 0.9));
        }
        if (typeof cachedSettings?.roiBottomWidth === "number") {
          setRoiBottomWidth(clamp(cachedSettings.roiBottomWidth, 0.72, 0.98));
        }
        if (typeof cachedSettings?.roiBottomY === "number") {
          setRoiBottomY(clamp(cachedSettings.roiBottomY, 0.78, 0.99));
        }
        if (typeof cachedSettings?.roiTopCenterX === "number") {
          setRoiTopCenterX(clamp(cachedSettings.roiTopCenterX, 0.35, 0.65));
        }
        if (typeof cachedSettings?.roiTopWidth === "number") {
          setRoiTopWidth(clamp(cachedSettings.roiTopWidth, 0.08, 0.42));
        }
        if (typeof cachedSettings?.roiTopY === "number") {
          setRoiTopY(clamp(cachedSettings.roiTopY, 0.58, 0.72));
        }
        setBirdViewSettingsHydrated(true);
      });
    }
  }, []);

  useEffect(() => {
    if (!birdViewSettingsHydrated) {
      return;
    }

    window.localStorage.setItem(
      birdViewSettingsStorageKey,
      JSON.stringify({
        birdViewHeightScale,
        laneJoinGapY,
        laneMinPixelScore,
        laneSmoothing,
        roiBottomWidth,
        roiBottomY,
        roiTopCenterX,
        roiTopWidth,
        roiTopY,
      }),
    );
  }, [
    birdViewHeightScale,
    birdViewSettingsHydrated,
    laneJoinGapY,
    laneMinPixelScore,
    laneSmoothing,
    roiBottomWidth,
    roiBottomY,
    roiTopCenterX,
    roiTopWidth,
    roiTopY,
  ]);

  useEffect(() => {
    const manualCalibration = createManualRoadCalibration(roiTopY, roiTopCenterX, roiTopWidth, roiBottomY, roiBottomWidth);
    roadCalibrationRef.current = manualCalibration;
    laneDetectionRef.current = laneDetectionRef.current
      ? { ...laneDetectionRef.current, road: manualCalibration, roiConfidence: 0 }
      : null;
  }, [roiBottomWidth, roiBottomY, roiTopCenterX, roiTopWidth, roiTopY]);

  const stopCamera = useCallback(() => {
    if (gatewayStreamIdRef.current) {
      void stopGatewayStream(gatewayStreamIdRef.current);
      gatewayStreamIdRef.current = null;
    }
    hlsRef.current?.destroy();
    hlsRef.current = null;
    setGatewayStreamId("");
    setGatewayStatus("idle");
    setGatewayDetail("");
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
      videoRef.current.removeAttribute("src");
      videoRef.current.load();
    }
    if (imageRef.current) {
      imageRef.current.removeAttribute("src");
    }
    setSourceSurface("video");
    setCameraState((state) => (state === "error" ? "error" : "ready"));
    trackerRef.current.reset();
    setTracks([]);
    tracksRef.current = [];
    setFps(0);
  }, []);

  const startUrlSourceWith = useCallback(async (
    mode: Exclude<SourceMode, "camera">,
    sourceUrl: string,
    outputOverride?: YoutubeOutput,
  ) => {
    const url = sourceUrl.trim();
    if (!url) {
      setCameraError(`${mode.toUpperCase()} URL is required.`);
      setCameraState("error");
      return;
    }

    setCameraError("");
    setCameraState("requesting");
    stopCamera();

    try {
      if (mode === "mjpg") {
        if (!imageRef.current) {
          throw new Error("MJPG image surface is not ready.");
        }

        imageRef.current.crossOrigin = "anonymous";
        imageRef.current.src = url;
        setSourceSurface("image");
        setCameraState("streaming");
        return;
      }

      if (mode === "rtsp" || mode === "youtube") {
        setGatewayStatus("connecting");
        setGatewayDetail("Requesting stream gateway conversion...");
        const gatewayOutput: GatewayOutput = mode === "youtube" ? outputOverride ?? youtubeOutput : "hls";
        const gatewayStream = await createGatewayStream(mode, url, gatewayOutput);
        gatewayStreamIdRef.current = gatewayStream.id;
        setGatewayStreamId(gatewayStream.id);
        setGatewayStatus("streaming");
        setGatewayDetail(`Gateway ${gatewayStream.output.toUpperCase()} stream ${gatewayStream.id}`);
        if (gatewayStream.output === "mjpg") {
          if (!imageRef.current) {
            throw new Error("Gateway image surface is not ready.");
          }
          imageRef.current.crossOrigin = "anonymous";
          imageRef.current.src = gatewayStream.url;
          setSourceSurface("image");
          setCameraState("streaming");
          return;
        }

        if (gatewayStream.output === "hls") {
          await waitForHlsManifest(gatewayStream.url);
          await startHlsVideoUrl(gatewayStream.url, videoRef.current, (hls) => {
            hlsRef.current = hls;
          });
        } else {
          await startVideoUrl(gatewayStream.url, videoRef.current);
        }
        setSourceSurface("video");
        setCameraState("streaming");
        return;
      }

      await startVideoUrl(url, videoRef.current);
      setSourceSurface("video");
      setCameraState("streaming");
    } catch (error) {
      const message = error instanceof Error ? error.message : `Could not start ${mode.toUpperCase()} stream.`;
      setCameraError(message);
      if (mode === "rtsp" || mode === "youtube") {
        setGatewayStatus("error");
        setGatewayDetail(message);
      }
      setCameraState("error");
    }
  }, [stopCamera, youtubeOutput]);

  const startUrlSource = useCallback(async () => {
    if (sourceMode === "camera") {
      return;
    }

    await startUrlSourceWith(sourceMode, streamUrls[sourceMode]);
  }, [sourceMode, startUrlSourceWith, streamUrls]);

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
      setSourceSurface("video");
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
        device: runtimeDefaults.llmDevice,
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
      setLlmDetail(`${runtimeDefaults.llmRuntime} ${runtimeDefaults.llmDevice} ready / ${runtimeDefaults.llmModelId}`);
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
    const imageDataUrl = includeFrame
      ? captureSourceFrame(
          getActiveSource(sourceSurface, videoRef.current, imageRef.current),
          runtimeDefaults.llmFrameMaxSide,
          runtimeDefaults.llmFrameJpegQuality,
        )
      : undefined;
    const userMessage = buildGemmaUserPrompt(prompt, sceneSummary, true);
    const nextMessages: BrowserLlmMessage[] = [
      ...(systemPrompt.trim() ? [{ role: "system" as const, content: systemPrompt.trim() }] : []),
      { role: "user", content: userMessage },
    ];
    const startedAt = performance.now();

    inferenceRunningRef.current = true;
    inferenceRoundRef.current += 1;
    setLlmState("generating");
    setLlmDetail(`Generating local loop round ${inferenceRoundRef.current}...`);

    try {
      await waitForNextAnimationFrame();
      const generation = await llmRef.current.generate(nextMessages, imageDataUrl);
      const elapsedMs = performance.now() - startedAt;
      const emptyResponse = !generation.text.trim() || isDegenerateLlmResponse(generation.text);
      const fallbackResponse = emptyResponse ? buildLocalFallbackResponse(sceneSummary, generation.diagnostics) : "";
      setLastInferenceMs(elapsedMs);
      setResponseCount((count) => count + 1);
      setChatMessages((messages) => [
        ...messages.slice(-24),
        {
          role: "assistant",
          content: emptyResponse ? fallbackResponse : generation.text,
        },
      ]);
      setLlmState("ready");
      setLlmDetail(`${runtimeDefaults.llmRuntime} ${runtimeDefaults.llmDevice} ready / ${runtimeDefaults.llmModelId}`);
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
        }, runtimeDefaults.llmLoopDelayMs);
      }
    }
  }, [fixedPrompt, includeFrame, sourceSurface, systemPrompt]);

  useEffect(() => {
    runInferenceRoundRef.current = runInferenceRound;
  }, [runInferenceRound]);

  useEffect(() => {
    const chatLog = chatLogRef.current;
    if (!chatLog) {
      return;
    }

    chatLog.scrollTo({
      top: chatLog.scrollHeight,
      behavior: "smooth",
    });
  }, [chatMessages]);

  const toggleInferenceLoop = useCallback(() => {
    if (inferenceLoopRef.current) {
      inferenceLoopRef.current = false;
      setLoopRunning(false);
      setLlmDetail(
        inferenceRunningRef.current
          ? "Stopping after the current local inference finishes..."
          : llmRef.current
            ? `${runtimeDefaults.llmRuntime} ${runtimeDefaults.llmDevice} ready / ${runtimeDefaults.llmModelId}`
            : llmDetail,
      );
      return;
    }

    inferenceLoopRef.current = true;
    setLoopRunning(true);
    void runInferenceRoundRef.current();
  }, [llmDetail]);

  const selectSourceMode = useCallback(
    (nextMode: SourceMode) => {
      if (nextMode === sourceMode) {
        return;
      }

      stopCamera();
      setCameraError("");
      setSourceMode(nextMode);
    },
    [sourceMode, stopCamera],
  );

  const setStreamUrl = useCallback((mode: Exclude<SourceMode, "camera">, url: string) => {
    setStreamUrls((current) => ({
      ...current,
      [mode]: url,
    }));
  }, []);

  useEffect(() => {
    if (sourceDeepLinkAppliedRef.current) {
      return;
    }

    const params = new URLSearchParams(window.location.search);
    const deepLinkSource = parseSourceMode(params.get("source"));
    if (!deepLinkSource || deepLinkSource === "camera") {
      sourceDeepLinkAppliedRef.current = true;
      return;
    }

    const deepLinkUrl = params.get("url") || "";
    const deepLinkOutput = parseYoutubeOutput(params.get("output"));
    sourceDeepLinkAppliedRef.current = true;

    queueMicrotask(() => {
      setSourceMode(deepLinkSource);
      if (deepLinkSource === "youtube" && deepLinkOutput) {
        setYoutubeOutput(deepLinkOutput);
      }
      if (deepLinkUrl) {
        setStreamUrl(deepLinkSource, deepLinkUrl);
      }

      if (deepLinkUrl && params.get("autostart") === "1") {
        void startUrlSourceWith(deepLinkSource, deepLinkUrl, deepLinkOutput ?? undefined);
      }
    });
  }, [setStreamUrl, startUrlSourceWith]);

  const checkGatewayHealth = useCallback(async () => {
    setGatewayStatus("checking");
    setGatewayDetail(`Checking ${runtimeDefaults.streamGatewayUrl}`);

    try {
      const health = await getGatewayHealth();
      setGatewayStatus("ready");
      setGatewayDetail(`Gateway ready, active sessions ${health.sessions}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Stream gateway is not available.";
      setGatewayStatus("error");
      setGatewayDetail(message);
    }
  }, []);

  const checkYoutubeSource = useCallback(async () => {
    const url = streamUrls.youtube.trim();
    if (!url) {
      setGatewayStatus("error");
      setGatewayDetail("YouTube URL is required.");
      return;
    }

    setGatewayStatus("connecting");
    setGatewayDetail("Resolving YouTube URL with yt-dlp...");

    try {
      const result = await resolveYoutubeSource(url);
      setGatewayStatus("ready");
      setGatewayDetail(`YouTube resolved via ${result.mediaHost} in ${formatDuration(result.durationMs)}.`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not resolve YouTube URL.";
      setGatewayStatus("error");
      setGatewayDetail(message);
    }
  }, [streamUrls.youtube]);

  useEffect(() => {
    robotIntentRef.current = robotIntent;
  }, [robotIntent]);

  useEffect(() => {
    robotDriveModeRef.current = robotDriveMode;
  }, [robotDriveMode]);

  const connectMockRobot = useCallback(async () => {
    setRobotError("");
    const transport = createMockV7rcTransport();
    robotTransportRef.current = transport;
    setRobotStatus(await transport.connect());
  }, []);

  const connectBluetoothRobot = useCallback(async () => {
    setRobotError("");
    if (!canUseWebBluetooth()) {
      setRobotError("Web Bluetooth is not available. Use Chrome on localhost or HTTPS.");
      return;
    }

    const transport = createWebBluetoothV7rcTransport((message) => {
      setRobotStatus((current) => ({ ...current, lastMessage: message }));
    });
    robotTransportRef.current = transport;

    try {
      setRobotStatus(await transport.connect());
    } catch (error) {
      robotTransportRef.current = null;
      setRobotError(error instanceof Error ? error.message : "Could not connect to V7RC robot.");
    }
  }, []);

  const disconnectRobot = useCallback(async () => {
    setRobotError("");
    const transport = robotTransportRef.current;
    if (!transport) {
      setRobotStatus((current) => ({ ...current, connected: false, lastMessage: "Robot transport is idle." }));
      return;
    }

    try {
      if (transport.getStatus().connected) {
        const neutralFrame = encodeSrtFrame(intentToSrtPwm(createNeutralIntent(), robotDriveModeRef.current));
        await transport.write(neutralFrame);
      }
      setRobotStatus(await transport.disconnect());
      robotTransportRef.current = null;
      setRobotMode("suggestion");
      setRobotIntent(createNeutralIntent());
    } catch (error) {
      setRobotError(error instanceof Error ? error.message : "Could not disconnect robot cleanly.");
    }
  }, []);

  const sendRobotIntent = useCallback((intent: V7rcRobotIntent) => {
    setRobotError("");
    setRobotIntent(intent);
  }, []);

  const sendRobotNeutral = useCallback(() => {
    sendRobotIntent(createNeutralIntent());
    setRobotMode("suggestion");
  }, [sendRobotIntent]);

  const sendRobotEmergencyStop = useCallback(() => {
    sendRobotIntent({
      ...createNeutralIntent(),
      emergencyStop: true,
      neutral: true,
    });
    setRobotMode("suggestion");
  }, [sendRobotIntent]);

  const toggleRobotTask = useCallback(() => {
    if (robotTaskStatus === "running") {
      setRobotTaskStatus("idle");
      setRobotTaskMessage("任務已停止，等待下一次 Start。");
      sendRobotIntent(createNeutralIntent());
      return;
    }

    setRobotTaskStatus("running");
    if (robotTaskMode === "autopilot") {
      setRobotTaskMessage("自動駕駛啟動：等待 OpenCV 車道線與 YOLO 安全狀態。");
      return;
    }

    setRobotTaskMessage("解任務啟動：等待 Gemma 產生短 JSON 動作計畫。");
  }, [robotTaskMode, robotTaskStatus, sendRobotIntent]);

  useEffect(() => {
    if (!robotStatus.connected) {
      return;
    }

    const writeCurrentCommand = async () => {
      const transport = robotTransportRef.current;
      if (!transport?.getStatus().connected || robotWriteInFlightRef.current) {
        return;
      }

      robotWriteInFlightRef.current = true;
      try {
        const frame = encodeSrtFrame(intentToSrtPwm(robotIntentRef.current, robotDriveModeRef.current));
        const status = await transport.write(frame);
        const now = performance.now();
        if (now - robotLastStatusUpdateRef.current > 250) {
          robotLastStatusUpdateRef.current = now;
          setRobotStatus(status);
        }
      } catch (error) {
        setRobotError(error instanceof Error ? error.message : "Could not send V7RC SRT command.");
      } finally {
        robotWriteInFlightRef.current = false;
      }
    };

    void writeCurrentCommand();
    const interval = window.setInterval(() => {
      void writeCurrentCommand();
    }, robotCommandIntervalMs);

    return () => {
      window.clearInterval(interval);
    };
  }, [robotStatus.connected]);

  useEffect(() => {
    if (sourceMode === "rtsp" || sourceMode === "youtube") {
      queueMicrotask(() => {
        void checkGatewayHealth();
      });
      return;
    }

    queueMicrotask(() => {
      setGatewayStatus("idle");
      setGatewayDetail("");
    });
  }, [checkGatewayHealth, sourceMode]);

  useEffect(() => {
    if (!gatewayStreamId || (sourceMode !== "rtsp" && sourceMode !== "youtube")) {
      return;
    }

    let cancelled = false;
    const updateGatewaySession = async () => {
      try {
        const session = await getGatewayStreamStatus(gatewayStreamId);
        if (cancelled) {
          return;
        }

        const lastLog =
          session.status === "running" || session.status === "ready"
            ? ""
            : session.lastError || session.logs?.at(-1)?.message || "";
        setGatewayStatus(session.status === "running" || session.status === "ready" ? "streaming" : "connecting");
        setGatewayDetail(
          lastLog
            ? `Gateway ${session.output.toUpperCase()} ${session.status}, clients ${session.clients}: ${lastLog}`
            : `Gateway ${session.output.toUpperCase()} ${session.status}, clients ${session.clients}`,
        );
      } catch (error) {
        if (cancelled) {
          return;
        }

        const message = error instanceof Error ? error.message : "Could not read gateway stream status.";
        setGatewayStatus("error");
        setGatewayDetail(message);
      }
    };

    const interval = window.setInterval(() => {
      void updateGatewaySession();
    }, 2000);
    void updateGatewaySession();

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [gatewayStreamId, sourceMode]);

  useEffect(() => {
    const detect = async () => {
      const source = getActiveSource(sourceSurface, videoRef.current, imageRef.current);
      const detector = detectorRef.current;

      if (
        cameraState === "streaming" &&
        source &&
        detector &&
        isSourceReady(source) &&
        !detectingRef.current
      ) {
        detectionFrameRef.current += 1;

        if (detectionFrameRef.current % Math.max(1, runtimeDefaults.yoloFrameInterval) === 0) {
          detectingRef.current = true;
          const startedAt = performance.now();

          try {
            const nextDetections = await detector.detect(source);
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
  }, [cameraState, sourceSurface]);

  useEffect(() => {
    let animationFrame = 0;
    let lastPaint = performance.now();
    let frameCount = 0;

    const paint = () => {
      const source = getActiveSource(sourceSurface, videoRef.current, imageRef.current);
      const canvas = canvasRef.current;
      const context = canvas?.getContext("2d");

      if (source && canvas && context && isSourceReady(source)) {
        const rect = source.getBoundingClientRect();
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
        drawCameraLaneGuide(
          context,
          source,
          rect.width,
          rect.height,
          mirrorPreview && sourceMode === "camera",
          roadCalibrationRef.current,
        );
        drawDetectedLaneLines(
          context,
          source,
          rect.width,
          rect.height,
          mirrorPreview && sourceMode === "camera",
          laneDetectionRef.current,
        );
        drawDetections(context, tracksRef.current, source, rect.width, rect.height, mirrorPreview && sourceMode === "camera");
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
  }, [mirrorPreview, sourceMode, sourceSurface]);

  useEffect(() => {
    let animationFrame = 0;

    const paintBirdView = () => {
      const canvas = birdViewCanvasRef.current;
      const context = canvas?.getContext("2d");
      const source = getActiveSource(sourceSurface, videoRef.current, imageRef.current);
      const sourceDimensions = source && isSourceReady(source) ? getSourceDimensions(source) : null;
      if (canvas && context) {
        drawBirdsEyeView(
          context,
          canvas,
          tracksRef.current,
          sourceDimensions,
          laneDetectionRef.current,
          roadCalibrationRef.current,
          robotTaskMode,
          robotTaskStatus,
          source && isSourceReady(source) ? source : null,
        );
      }

      animationFrame = requestAnimationFrame(paintBirdView);
    };

    animationFrame = requestAnimationFrame(paintBirdView);
    return () => cancelAnimationFrame(animationFrame);
  }, [robotTaskMode, robotTaskStatus, sourceSurface]);

  useEffect(() => {
    let animationFrame = 0;
    let lastDetectionAt = 0;
    let lastStateUpdateAt = 0;

    const detectLaneCandidates = () => {
      const now = performance.now();
      if (now - lastDetectionAt >= laneDetectionIntervalMs) {
        const source = getActiveSource(sourceSurface, videoRef.current, imageRef.current);
        if (source && isSourceReady(source)) {
          laneDetectionCanvasRef.current ??= document.createElement("canvas");
          const detection = detectLaneLinesFromSource(source, laneDetectionCanvasRef.current, roadCalibrationRef.current, {
            joinGapY: laneJoinGapY,
            minPixelScore: laneMinPixelScore,
          });
          const nextDetection = smoothLaneDetection(detection, laneDetectionRef.current, roadCalibrationRef.current, laneSmoothing);
          laneDetectionRef.current = nextDetection;

          if (now - lastStateUpdateAt >= 500) {
            lastStateUpdateAt = now;
            setLaneConfidence(nextDetection.confidence);
            setRoiConfidence(nextDetection.roiConfidence);
          }
        } else {
          laneDetectionRef.current = null;
          if (now - lastStateUpdateAt >= 500) {
            lastStateUpdateAt = now;
            setLaneConfidence(0);
            setRoiConfidence(0);
          }
        }

        lastDetectionAt = now;
      }

      animationFrame = requestAnimationFrame(detectLaneCandidates);
    };

    animationFrame = requestAnimationFrame(detectLaneCandidates);
    return () => cancelAnimationFrame(animationFrame);
  }, [laneJoinGapY, laneMinPixelScore, laneSmoothing, roiBottomWidth, roiBottomY, roiTopCenterX, roiTopWidth, roiTopY, sourceSurface]);

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

        <div className="metric-strip">
          <Metric label="FPS" value={fps.toString()} />
          <Metric label="YOLO" value={`${yoloMs.toFixed(1)} ms`} />
          <Metric label="ByteTrack" value={`${trackMs.toFixed(1)} ms`} />
        </div>
      </header>

      <section className="workspace">
        <div className="main-panel">
          <div className="camera-panel">
            <div className="video-stage">
              <video
                className={sourceSurface === "image" ? "hidden-source" : sourceMode === "camera" && mirrorPreview ? "mirrored" : undefined}
                ref={videoRef}
                muted
                playsInline
              />
              {/* eslint-disable-next-line @next/next/no-img-element -- MJPG streams need a raw img surface. */}
              <img
                alt=""
                className={sourceSurface === "image" ? undefined : "hidden-source"}
                ref={imageRef}
                onError={() => {
                  if (sourceSurface === "image") {
                    setCameraError("Could not load image stream.");
                    setCameraState("error");
                  }
                }}
              />
              <canvas ref={canvasRef} aria-hidden="true" />
              {cameraState !== "streaming" ? (
                <div className="stage-empty">
                  <strong>{cameraState === "requesting" ? `Requesting ${sourceMode}` : `${sourceMode.toUpperCase()} idle`}</strong>
                  <span>{cameraError || "Select a source and start the local pipeline."}</span>
                </div>
              ) : null}
            </div>
          </div>

          <section className="chat-panel">
            <div className="conversation-panel">
              <div className="chat-log" ref={chatLogRef}>
                {chatMessages.map((message, index) => (
                  <div className={`message ${message.role}-message`} key={`${message.role}-${index}`}>
                    {message.content}
                  </div>
                ))}
              </div>
              <div className="conversation-footer">
                <span>回覆 {responseCount} 次</span>
                {lastInferenceMs !== null ? <span>最後推論 {formatDuration(lastInferenceMs)}</span> : null}
              </div>
            </div>
          </section>
        </div>

        <aside className="task-panel">
          <section className="task-card">
            <div className="task-card-title">
              <div>
                <h2>Robot Task</h2>
                <p>{robotTaskMessage}</p>
              </div>
              <button
                className={`icon-button task-start-button ${robotTaskRunning ? "active" : ""}`}
                type="button"
                onClick={toggleRobotTask}
                title={robotTaskRunning ? "Stop task" : "Start task"}
                aria-label={robotTaskRunning ? "Stop task" : "Start task"}
              >
                {robotTaskRunning ? (
                  <svg aria-hidden="true" viewBox="0 0 24 24">
                    <path d="M8 8h8v8H8z" />
                  </svg>
                ) : (
                  <svg aria-hidden="true" viewBox="0 0 24 24">
                    <path d="m8 5 11 7-11 7V5z" />
                  </svg>
                )}
              </button>
            </div>
            <div className="segmented-control task-mode-control">
              {robotTaskModes.map((mode) => (
                <button
                  className={robotTaskMode === mode.id ? "active" : ""}
                  key={mode.id}
                  type="button"
                  onClick={() => {
                    setRobotTaskMode(mode.id);
                    setRobotTaskStatus("idle");
                    setRobotTaskMessage(mode.id === "autopilot" ? "自動駕駛待命。" : "解任務待命。");
                  }}
                >
                  {mode.label}
                </button>
              ))}
            </div>
            <div className="task-state-grid">
              <div>
                <span>Status</span>
                <strong>{robotTaskStatus.toUpperCase()}</strong>
              </div>
              <div>
                <span>Mode</span>
                <strong>{robotTaskMode === "autopilot" ? "Autopilot" : "Mission"}</strong>
              </div>
              <div>
                <span>ROI Confidence</span>
                <strong>{Math.round(roiConfidence * 100)}%</strong>
              </div>
            </div>
            <p className="task-detail">{robotTaskDetail}</p>
          </section>

          <section className="birdview-card">
            <div className="panel-heading">
              <h2>Bird&apos;s-Eye View</h2>
              <span>{tracksRef.current.length}</span>
            </div>
            <div className="birdview-stage">
              <canvas
                ref={birdViewCanvasRef}
                width={birdViewBaseWidth}
                height={Math.round(birdViewBaseHeight * birdViewHeightScale)}
                style={{ aspectRatio: `${birdViewBaseWidth} / ${Math.round(birdViewBaseHeight * birdViewHeightScale)}` }}
                aria-label="Bird's-eye view"
              />
            </div>
            <div className="roi-controls" aria-label="Bird's-eye ROI controls">
              <label>
                <span>View Height</span>
                <input
                  type="range"
                  min="1"
                  max="2"
                  step="0.05"
                  value={birdViewHeightScale}
                  onChange={(event) => setBirdViewHeightScale(Number(event.target.value))}
                />
                <strong>{birdViewHeightScale.toFixed(2)}</strong>
              </label>
              <label>
                <span>Top Y</span>
                <input
                  type="range"
                  min="0.58"
                  max="0.72"
                  step="0.01"
                  value={roiTopY}
                  onChange={(event) => {
                    setRoiConfidence(0);
                    setRoiTopY(Number(event.target.value));
                  }}
                />
                <strong>{roiTopY.toFixed(2)}</strong>
              </label>
              <label>
                <span>Top Center</span>
                <input
                  type="range"
                  min="0.35"
                  max="0.65"
                  step="0.01"
                  value={roiTopCenterX}
                  onChange={(event) => {
                    setRoiConfidence(0);
                    setRoiTopCenterX(Number(event.target.value));
                  }}
                />
                <strong>{roiTopCenterX.toFixed(2)}</strong>
              </label>
              <label>
                <span>Top Width</span>
                <input
                  type="range"
                  min="0.08"
                  max="0.42"
                  step="0.01"
                  value={roiTopWidth}
                  onChange={(event) => {
                    setRoiConfidence(0);
                    setRoiTopWidth(Number(event.target.value));
                  }}
                />
                <strong>{roiTopWidth.toFixed(2)}</strong>
              </label>
              <label>
                <span>Bottom Width</span>
                <input
                  type="range"
                  min="0.72"
                  max="0.98"
                  step="0.01"
                  value={roiBottomWidth}
                  onChange={(event) => {
                    setRoiBottomWidth(Number(event.target.value));
                    setRoiConfidence(0);
                  }}
                />
                <strong>{roiBottomWidth.toFixed(2)}</strong>
              </label>
              <label>
                <span>Bottom Y</span>
                <input
                  type="range"
                  min="0.78"
                  max="0.99"
                  step="0.01"
                  value={roiBottomY}
                  onChange={(event) => {
                    setRoiBottomY(Number(event.target.value));
                    setRoiConfidence(0);
                  }}
                />
                <strong>{roiBottomY.toFixed(2)}</strong>
              </label>
              <label>
                <span>Lane Threshold</span>
                <input
                  type="range"
                  min="8"
                  max="42"
                  step="1"
                  value={laneMinPixelScore}
                  onChange={(event) => {
                    setLaneMinPixelScore(Number(event.target.value));
                    setRoiConfidence(0);
                  }}
                />
                <strong>{Math.round(laneMinPixelScore)}</strong>
              </label>
              <label>
                <span>Join Gap</span>
                <input
                  type="range"
                  min="0.04"
                  max="0.25"
                  step="0.01"
                  value={laneJoinGapY}
                  onChange={(event) => {
                    setLaneJoinGapY(Number(event.target.value));
                    setRoiConfidence(0);
                  }}
                />
                <strong>{laneJoinGapY.toFixed(2)}</strong>
              </label>
              <label>
                <span>Lane Smooth</span>
                <input
                  type="range"
                  min="0"
                  max="0.9"
                  step="0.05"
                  value={laneSmoothing}
                  onChange={(event) => setLaneSmoothing(Number(event.target.value))}
                />
                <strong>{laneSmoothing.toFixed(2)}</strong>
              </label>
              <button className="secondary-button compact-button" type="button" onClick={resetBirdViewSettings}>
                Reset Bird View
              </button>
            </div>
            <p>YOLO/ByteTrack objects use box bottom-center projection. Manual ROI controls the bird-view road transform.</p>
          </section>
        </aside>

        <aside className="side-panel">
          <div className="camera-source-card">
            <StatusCard
              status={cameraStatus}
              action={
                <div className="status-actions">
                  <button
                    className="primary-button compact-button"
                    type="button"
                    onClick={cameraState === "streaming" ? stopCamera : sourceMode === "camera" ? startCamera : startUrlSource}
                    disabled={cameraState === "requesting"}
                  >
                    {cameraState === "streaming" ? "Stop" : "Start"}
                  </button>
                  <button
                    className="icon-button compact-icon-button"
                    type="button"
                    onClick={() => setCameraSettingsOpen(true)}
                    title="Camera settings"
                    aria-label="Camera settings"
                  >
                    <svg aria-hidden="true" viewBox="0 0 24 24">
                      <path d="M12 8.4a3.6 3.6 0 1 0 0 7.2 3.6 3.6 0 0 0 0-7.2Z" />
                      <path d="M19.4 13.5a7.8 7.8 0 0 0 0-3l2-1.5-2-3.5-2.4 1a7.7 7.7 0 0 0-2.6-1.5L14 2.4h-4L9.6 5a7.7 7.7 0 0 0-2.6 1.5l-2.4-1-2 3.5 2 1.5a7.8 7.8 0 0 0 0 3l-2 1.5 2 3.5 2.4-1a7.7 7.7 0 0 0 2.6 1.5l.4 2.6h4l.4-2.6a7.7 7.7 0 0 0 2.6-1.5l2.4 1 2-3.5-2-1.5Z" />
                    </svg>
                  </button>
                </div>
              }
            />
            <div className="camera-source-summary">
              <span>Source</span>
              <strong>{sourceMode.toUpperCase()}</strong>
              <small>{sourceSummary}</small>
              {sourceMode === "rtsp" || sourceMode === "youtube" ? (
                <small className={`gateway-detail ${gatewayStatus}`}>
                  {gatewayDetail || `Gateway ${runtimeDefaults.streamGatewayUrl}`}
                </small>
              ) : null}
            </div>
          </div>
          <StatusCard status={yoloStatus} />
          <StatusCard
            status={llmStatus}
            action={
              <div className="status-actions">
                <button
                  className="secondary-button compact-button"
                  type="button"
                  onClick={loadGemma}
                  disabled={llmState === "loading" || llmState === "generating" || llmState === "ready"}
                >
                  {llmState === "ready" || llmState === "generating" ? "Loaded" : "Load"}
                </button>
                <button
                  className="icon-button compact-icon-button"
                  type="button"
                  onClick={() => setSettingsOpen(true)}
                  title="Gemma settings"
                  aria-label="Gemma settings"
                >
                  <svg aria-hidden="true" viewBox="0 0 24 24">
                    <path d="M12 8.4a3.6 3.6 0 1 0 0 7.2 3.6 3.6 0 0 0 0-7.2Z" />
                    <path d="M19.4 13.5a7.8 7.8 0 0 0 0-3l2-1.5-2-3.5-2.4 1a7.7 7.7 0 0 0-2.6-1.5L14 2.4h-4L9.6 5a7.7 7.7 0 0 0-2.6 1.5l-2.4-1-2 3.5 2 1.5a7.8 7.8 0 0 0 0 3l-2 1.5 2 3.5 2.4-1a7.7 7.7 0 0 0 2.6 1.5l.4 2.6h4l.4-2.6a7.7 7.7 0 0 0 2.6-1.5l2.4 1 2-3.5-2-1.5Z" />
                  </svg>
                </button>
                <button
                  className="icon-button compact-icon-button"
                  type="button"
                  onClick={toggleInferenceLoop}
                  disabled={llmState === "loading"}
                  title={loopRunning ? "Stop Gemma loop" : "Start Gemma loop"}
                  aria-label={loopRunning ? "Stop Gemma loop" : "Start Gemma loop"}
                >
                  {loopRunning ? (
                    <svg aria-hidden="true" viewBox="0 0 24 24">
                      <path d="M8 8h8v8H8z" />
                    </svg>
                  ) : (
                    <svg aria-hidden="true" viewBox="0 0 24 24">
                      <path d="m8 5 11 7-11 7V5z" />
                    </svg>
                  )}
                </button>
              </div>
            }
            progress={llmProgress}
          />
          <div className="robot-card">
            <StatusCard
              status={robotRuntimeStatus}
              action={
                <div className="status-actions">
                  <button
                    className="secondary-button compact-button"
                    type="button"
                    onClick={connectMockRobot}
                    disabled={robotStatus.connected}
                  >
                    Mock
                  </button>
                  <button
                    className="secondary-button compact-button"
                    type="button"
                    onClick={connectBluetoothRobot}
                    disabled={robotStatus.connected}
                  >
                    BLE
                  </button>
                </div>
              }
            />
            <div className="robot-controls">
              <label className="inline-toggle">
                <input
                  type="checkbox"
                  checked={robotMode === "armed"}
                  onChange={(event) => {
                    const nextArmed = event.target.checked;
                    setRobotMode(nextArmed ? "armed" : "suggestion");
                    setRobotIntent((current) => ({
                      ...current,
                      autonomy: nextArmed,
                      emergencyStop: false,
                      neutral: !nextArmed,
                      speedScale: nextArmed ? Math.max(current.speedScale, 0.25) : 0,
                    }));
                  }}
                  disabled={!robotStatus.connected}
                />
                <span>{robotMode === "armed" ? "Armed" : "Suggestion"}</span>
              </label>
              <button
                className="secondary-button compact-button"
                type="button"
                onClick={() => sendRobotNeutral()}
                disabled={!robotStatus.connected}
              >
                Neutral
              </button>
              <button
                className="danger-button compact-button"
                type="button"
                onClick={() => sendRobotEmergencyStop()}
                disabled={!robotStatus.connected}
              >
                E-stop
              </button>
              <button
                className="secondary-button compact-button"
                type="button"
                onClick={() => void disconnectRobot()}
                disabled={!robotStatus.connected}
              >
                Disconnect
              </button>
            </div>
            <div className="robot-drive-mode">
              <span>Robot 模式</span>
              <div className="segmented-control">
                {robotDriveModes.map((mode) => (
                  <button
                    className={robotDriveMode === mode.id ? "active" : ""}
                    key={mode.id}
                    type="button"
                    onClick={() => setRobotDriveMode(mode.id)}
                  >
                    {mode.label}
                  </button>
                ))}
              </div>
              <small>BLE connected 後每 {robotCommandIntervalMs}ms 以 SRT 格式送出目前 Channel 狀態。</small>
            </div>
            <div className="robot-packet">
              <span>SRT preview</span>
              <code>{robotStatus.lastPacket || robotPacketPreview}</code>
            </div>
            <div className="robot-channel-table">
              {robotChannelPreview.map((channel) => (
                <div className="robot-channel-row" key={channel.index}>
                  <span>CH{channel.index}</span>
                  <span>{channel.logical}</span>
                  <span>{channel.pwmUs} us</span>
                </div>
              ))}
            </div>
          </div>
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

      {cameraSettingsOpen ? (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setCameraSettingsOpen(false)}>
          <div
            aria-labelledby="camera-settings-title"
            className="settings-modal"
            role="dialog"
            aria-modal="true"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="modal-title">
              <h2 id="camera-settings-title">Camera Settings</h2>
              <button
                className="icon-button compact-icon-button"
                type="button"
                onClick={() => setCameraSettingsOpen(false)}
                title="Close camera settings"
                aria-label="Close camera settings"
              >
                <svg aria-hidden="true" viewBox="0 0 24 24">
                  <path d="m6 6 12 12M18 6 6 18" />
                </svg>
              </button>
            </div>
            <div className="source-switch settings-source-switch" aria-label="Source">
              <span>Source</span>
              {sourceModes.map((mode) => (
                <button
                  className={sourceMode === mode.id ? "active" : undefined}
                  type="button"
                  key={mode.id}
                  onClick={() => selectSourceMode(mode.id)}
                  disabled={cameraState === "requesting"}
                >
                  {mode.label}
                </button>
              ))}
            </div>
            {sourceMode === "camera" ? (
              <div className="settings-grid">
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
                <div className="settings-row">
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => void refreshDevices()}
                    disabled={cameraState === "requesting"}
                  >
                    Refresh cameras
                  </button>
                  <label className="inline-toggle">
                    <input
                      type="checkbox"
                      checked={mirrorPreview}
                      onChange={(event) => setMirrorPreview(event.target.checked)}
                    />
                    <span>Mirror preview</span>
                  </label>
                </div>
              </div>
            ) : (
              <div className="settings-grid">
                <label className="field stream-url-field">
                  <span>{sourceMode.toUpperCase()} URL</span>
                  <input
                    value={streamUrls[sourceMode]}
                    onChange={(event) => setStreamUrl(sourceMode, event.target.value)}
                    disabled={cameraState === "requesting"}
                    placeholder={getSourcePlaceholder(sourceMode)}
                  />
                  <small>{cameraDetail}</small>
                  {sourceMode === "rtsp" || sourceMode === "youtube" ? (
                    <small className={`gateway-detail ${gatewayStatus}`}>
                      {gatewayDetail || `Gateway ${runtimeDefaults.streamGatewayUrl}`}
                    </small>
                  ) : null}
                </label>
                {sourceMode === "youtube" ? (
                  <div className="settings-row">
                    <div className="output-switch" aria-label="YouTube output">
                      <span>Output</span>
                      {(["mp4", "hls"] as const).map((output) => (
                        <button
                          className={youtubeOutput === output ? "active" : undefined}
                          disabled={cameraState === "requesting" || cameraState === "streaming"}
                          key={output}
                          onClick={() => setYoutubeOutput(output)}
                          type="button"
                        >
                          {output.toUpperCase()}
                        </button>
                      ))}
                    </div>
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={() => void checkYoutubeSource()}
                      disabled={cameraState === "requesting" || gatewayStatus === "connecting"}
                    >
                      Check source
                    </button>
                  </div>
                ) : null}
                {sourceMode === "rtsp" ? (
                  <div className="settings-row">
                    <div className="output-switch" aria-label="RTSP output">
                      <span>Output</span>
                      <button className="active" disabled type="button">
                        HLS
                      </button>
                    </div>
                  </div>
                ) : null}
              </div>
            )}
          </div>
        </div>
      ) : null}

      {settingsOpen ? (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setSettingsOpen(false)}>
          <div
            aria-labelledby="gemma-settings-title"
            className="settings-modal"
            role="dialog"
            aria-modal="true"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="modal-title">
              <h2 id="gemma-settings-title">Gemma4-E2B Settings</h2>
              <button
                className="icon-button compact-icon-button"
                type="button"
                onClick={() => setSettingsOpen(false)}
                title="Close settings"
                aria-label="Close settings"
              >
                <svg aria-hidden="true" viewBox="0 0 24 24">
                  <path d="m6 6 12 12M18 6 6 18" />
                </svg>
              </button>
            </div>
            <label className="frame-toggle settings-toggle">
              <input
                type="checkbox"
                checked={includeFrame}
                onChange={(event) => setIncludeFrame(event.target.checked)}
              />
              <span>Include current frame</span>
            </label>
            <label className="prompt-field">
              <span>System Prompt</span>
              <textarea value={systemPrompt} onChange={(event) => setSystemPrompt(event.target.value)} rows={6} />
            </label>
            <label className="prompt-field">
              <span>Fixed Prompt</span>
              <textarea value={fixedPrompt} onChange={(event) => setFixedPrompt(event.target.value)} rows={4} />
            </label>
            <div className="modal-actions">
              <button className="primary-button" type="button" onClick={() => setSettingsOpen(false)}>
                Done
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}

function buildSceneSummary(tracks: Track[]) {
  if (tracks.length === 0) {
    return "目前沒有正在追蹤的物件。";
  }

  return tracks
    .slice(0, 12)
    .map((track) => `${track.id}: ${track.label}，信心值 ${track.confidence.toFixed(2)}`)
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
    "你是一個在本機瀏覽器執行的精簡視覺助理。若有附上影像，請以影像為主要依據，並將追蹤場景摘要作為輔助資訊。";
  const basePrompt = includeInstructions ? `${instructions}\n\n${prompt}` : prompt;

  if (!sceneSummary) {
    return basePrompt;
  }

  return `${basePrompt}\n\n目前追蹤場景：\n${sceneSummary}`;
}

function getActiveSource(
  sourceSurface: SourceSurface,
  video: HTMLVideoElement | null,
  image: HTMLImageElement | null,
): DetectableSource | null {
  return sourceSurface === "image" ? image : video;
}

function isSourceReady(source: DetectableSource) {
  const dimensions = getSourceDimensions(source);
  return dimensions.width > 0 && dimensions.height > 0;
}

function getSourceDimensions(source: DetectableSource) {
  if (source instanceof HTMLVideoElement) {
    return {
      width: source.videoWidth,
      height: source.videoHeight,
    };
  }

  return {
    width: source.naturalWidth,
    height: source.naturalHeight,
  };
}

function getSourcePlaceholder(sourceMode: Exclude<SourceMode, "camera">) {
  if (sourceMode === "mjpg") {
    return "http://robot.local:8080/video.mjpg";
  }

  if (sourceMode === "rtsp") {
    return "https://host/stream.m3u8 or converted RTSP stream URL";
  }

  return "https://www.youtube.com/watch?v=...";
}

function parseSourceMode(value: string | null): SourceMode | null {
  if (value === "camera" || value === "mjpg" || value === "rtsp" || value === "youtube") {
    return value;
  }

  return null;
}

function parseYoutubeOutput(value: string | null): YoutubeOutput | null {
  if (value === "hls" || value === "mp4") {
    return value;
  }

  return null;
}

function parseGatewayOutput(value: unknown): GatewayOutput {
  if (value === "hls" || value === "mp4") {
    return value;
  }

  return "mjpg";
}

async function getGatewayHealth(): Promise<{ ok: boolean; sessions: number }> {
  const response = await fetch(`${runtimeDefaults.streamGatewayUrl.replace(/\/$/u, "")}/health`, {
    cache: "no-store",
  });
  const payload = (await response.json().catch(() => null)) as { ok?: boolean; sessions?: number; error?: string } | null;

  if (!response.ok || !payload?.ok) {
    throw new Error(payload?.error || `Stream gateway health returned ${response.status}.`);
  }

  return {
    ok: true,
    sessions: typeof payload.sessions === "number" ? payload.sessions : 0,
  };
}

async function createGatewayStream(
  sourceMode: "rtsp" | "youtube",
  url: string,
  output: GatewayOutput,
): Promise<GatewayStreamResponse> {
  const response = await fetch(`${runtimeDefaults.streamGatewayUrl.replace(/\/$/u, "")}/api/streams`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      sourceType: sourceMode,
      url,
      output,
    }),
  });

  const payload = (await response.json().catch(() => null)) as Partial<GatewayStreamResponse> & { error?: string } | null;
  if (!response.ok || !payload?.url || !payload.id) {
    throw new Error(payload?.error || `Stream gateway returned ${response.status}.`);
  }

  return {
    id: payload.id,
    output: parseGatewayOutput(payload.output),
    status: payload.status || "ready",
    url: payload.url,
  };
}

async function resolveYoutubeSource(url: string): Promise<YoutubeResolveResult> {
  const response = await fetch(`${runtimeDefaults.streamGatewayUrl.replace(/\/$/u, "")}/api/youtube/resolve`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ url }),
  });
  const payload = (await response.json().catch(() => null)) as Partial<YoutubeResolveResult> & { error?: string } | null;

  if (!response.ok || !payload?.ok) {
    throw new Error(payload?.error || `YouTube resolver returned ${response.status}.`);
  }

  return {
    ok: true,
    protocol: payload.protocol || "",
    mediaHost: payload.mediaHost || "unknown host",
    durationMs: typeof payload.durationMs === "number" ? payload.durationMs : 0,
    format: payload.format || "",
  };
}

async function getGatewayStreamStatus(id: string): Promise<GatewaySessionStatus> {
  const response = await fetch(
    `${runtimeDefaults.streamGatewayUrl.replace(/\/$/u, "")}/api/streams/${encodeURIComponent(id)}`,
    {
      cache: "no-store",
    },
  );
  const payload = (await response.json().catch(() => null)) as Partial<GatewaySessionStatus> & { error?: string } | null;

  if (!response.ok || !payload?.id) {
    throw new Error(payload?.error || `Stream gateway status returned ${response.status}.`);
  }

  return {
    id: payload.id,
    output: parseGatewayOutput(payload.output),
    status: payload.status || "unknown",
    clients: typeof payload.clients === "number" ? payload.clients : 0,
    lastError: payload.lastError,
    logs: payload.logs,
  };
}

async function stopGatewayStream(id: string) {
  await fetch(`${runtimeDefaults.streamGatewayUrl.replace(/\/$/u, "")}/api/streams/${encodeURIComponent(id)}`, {
    method: "DELETE",
  }).catch(() => undefined);
}

async function startVideoUrl(url: string, video: HTMLVideoElement | null) {
  if (!video) {
    throw new Error("Video surface is not ready.");
  }

  video.crossOrigin = "anonymous";
  video.src = url;
  video.muted = true;
  await video.play();
}

async function waitForHlsManifest(url: string) {
  const startedAt = performance.now();
  let lastError = "HLS manifest is not ready.";

  while (performance.now() - startedAt < 15000) {
    try {
      const response = await fetch(url, { cache: "no-store" });
      const body = await response.text();
      if (response.ok && body.includes("#EXTM3U")) {
        return;
      }
      lastError = `HLS manifest returned ${response.status}.`;
    } catch (error) {
      lastError = error instanceof Error ? error.message : "Could not read HLS manifest.";
    }

    await new Promise((resolve) => {
      window.setTimeout(resolve, 500);
    });
  }

  throw new Error(lastError);
}

async function startHlsVideoUrl(url: string, video: HTMLVideoElement | null, setController: (hls: Hls) => void) {
  if (!video) {
    throw new Error("Video surface is not ready.");
  }

  video.crossOrigin = "anonymous";
  video.muted = true;

  const { default: HlsRuntime } = await import("hls.js");
  if (!HlsRuntime.isSupported()) {
    if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = url;
      await video.play();
      return;
    }

    throw new Error("This browser cannot play HLS streams.");
  }

  const hls = new HlsRuntime({
    backBufferLength: 30,
    liveSyncDurationCount: 2,
    lowLatencyMode: true,
  });
  setController(hls);

  await new Promise<void>((resolve, reject) => {
    let settled = false;
    const resolveOnce = () => {
      if (settled) {
        return;
      }
      settled = true;
      resolve();
    };
    const rejectOnce = (_event: unknown, data: { details?: string; fatal?: boolean; type?: string }) => {
      if (!data.fatal || settled) {
        return;
      }
      settled = true;
      hls.destroy();
      reject(new Error(`HLS playback failed: ${data.details || data.type || "unknown error"}`));
    };

    hls.on(HlsRuntime.Events.MANIFEST_PARSED, resolveOnce);
    hls.on(HlsRuntime.Events.ERROR, rejectOnce);
    hls.attachMedia(video);
    hls.loadSource(url);
  });

  await video.play();
}

function captureSourceFrame(source: DetectableSource | null, maxSide: number, jpegQuality: number) {
  if (!source || !isSourceReady(source)) {
    return undefined;
  }

  const dimensions = getSourceDimensions(source);
  const safeMaxSide = Math.max(64, maxSide);
  const safeJpegQuality = Math.min(0.95, Math.max(0.4, jpegQuality));
  const scale = Math.min(1, safeMaxSide / Math.max(dimensions.width, dimensions.height));
  const width = Math.max(1, Math.round(dimensions.width * scale));
  const height = Math.max(1, Math.round(dimensions.height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) {
    return undefined;
  }

  context.drawImage(source, 0, 0, width, height);
  return canvas.toDataURL("image/jpeg", safeJpegQuality);
}

function waitForNextAnimationFrame() {
  return new Promise<void>((resolve) => {
    window.requestAnimationFrame(() => {
      resolve();
    });
  });
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

function drawBirdsEyeView(
  context: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  tracks: Track[],
  sourceDimensions: { width: number; height: number } | null,
  laneDetection: LaneDetection | null,
  roadCalibration: RoadCalibration,
  taskMode: RobotTaskMode,
  taskStatus: RobotTaskStatus,
  source: DetectableSource | null,
) {
  const { width, height } = canvas;
  context.clearRect(0, 0, width, height);

  const gradient = context.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, "#18202b");
  gradient.addColorStop(1, "#06080d");
  context.fillStyle = gradient;
  context.fillRect(0, 0, width, height);

  const roadTopWidth = width * 0.82;
  const roadBottomWidth = width * 0.82;
  const roadTopY = height * 0.12;
  const roadBottomY = height * 0.94;
  const destinationRoad = {
    bottomWidth: roadBottomWidth,
    bottomY: roadBottomY,
    topWidth: roadTopWidth,
    topY: roadTopY,
  };
  context.beginPath();
  context.moveTo((width - roadTopWidth) / 2, roadTopY);
  context.lineTo((width + roadTopWidth) / 2, roadTopY);
  context.lineTo((width + roadBottomWidth) / 2, roadBottomY);
  context.lineTo((width - roadBottomWidth) / 2, roadBottomY);
  context.closePath();
  if (source && isSourceReady(source)) {
    drawSourceAsBirdView(context, source, roadCalibration, destinationRoad, width, height);
  } else {
    context.fillStyle = "rgba(30, 41, 59, 0.92)";
    context.fill();
  }
  context.strokeStyle = "rgba(148, 163, 184, 0.28)";
  context.lineWidth = 1;
  context.stroke();

  context.strokeStyle = "#facc15";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo((width - roadTopWidth) / 2, roadTopY);
  context.lineTo((width - roadBottomWidth) / 2, roadBottomY);
  context.moveTo((width + roadTopWidth) / 2, roadTopY);
  context.lineTo((width + roadBottomWidth) / 2, roadBottomY);
  context.stroke();

  context.setLineDash([10, 10]);
  context.strokeStyle = "rgba(226, 232, 240, 0.5)";
  context.beginPath();
  context.moveTo(width / 2, roadTopY + 10);
  context.lineTo(width / 2, roadBottomY - 10);
  context.stroke();
  context.setLineDash([]);

  drawBirdViewDetectedLane(context, width, height, laneDetection, destinationRoad);

  context.fillStyle = taskStatus === "running" ? "#99f6e4" : "#94a3b8";
  context.font = "12px sans-serif";
  context.fillText(taskMode === "autopilot" ? "Autopilot lane view" : "Mission target view", 12, 22);

  for (const track of tracks.slice(0, 16)) {
    const projected = projectTrackToBirdView(track, width, height, sourceDimensions, {
      bottomWidth: roadBottomWidth,
      bottomY: roadBottomY,
      topWidth: roadTopWidth,
      topY: roadTopY,
    }, roadCalibration);
    const color = "#2dd4bf";
    context.fillStyle = color;
    context.strokeStyle = "#020617";
    context.lineWidth = 2;
    context.beginPath();
    context.arc(projected.x, projected.y, projected.radius, 0, Math.PI * 2);
    context.fill();
    context.stroke();

    const label = `${track.id} ${track.label}`;
    context.font = "11px sans-serif";
    const labelWidth = context.measureText(label).width + 8;
    context.fillStyle = color;
    context.fillRect(projected.x + 8, projected.y - 9, labelWidth, 16);
    context.fillStyle = "#020617";
    context.fillText(label, projected.x + 12, projected.y + 3);
  }

  if (tracks.length === 0) {
    context.fillStyle = "#94a3b8";
    context.font = "12px sans-serif";
    context.fillText("No projected tracks", 12, height - 16);
  }
}

function drawBirdViewDetectedLane(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  laneDetection: LaneDetection | null,
  destinationRoad: { topWidth: number; bottomWidth: number; topY: number; bottomY: number },
) {
  if (!laneDetection || laneDetection.confidence <= 0) {
    return;
  }

  context.save();
  drawLaneBands(context, laneDetection.laneBands, "bird", (point) =>
    projectBirdPointToBirdView(point, width, destinationRoad),
  );

  context.strokeStyle = "#fb923c";
  context.lineWidth = 3;
  context.lineCap = "round";
  context.lineJoin = "round";

  for (const path of laneDetection.birdPaths) {
    drawProjectedLanePath(context, path, (point) =>
      projectBirdPointToBirdView(point, width, destinationRoad),
    );
  }

  context.restore();
}

function drawSourceAsBirdView(
  context: CanvasRenderingContext2D,
  source: DetectableSource,
  roadCalibration: RoadCalibration,
  destinationRoad: { topWidth: number; bottomWidth: number; topY: number; bottomY: number },
  stageWidth: number,
  stageHeight: number,
) {
  const outputWidth = Math.max(1, Math.round(destinationRoad.bottomWidth));
  const outputHeight = Math.max(1, Math.round(destinationRoad.bottomY - destinationRoad.topY));
  const birdFrame = renderBirdViewFrame(source, outputWidth, outputHeight, roadCalibration);
  const destinationLeft = (stageWidth - destinationRoad.bottomWidth) / 2;

  context.save();
  context.clip();
  context.drawImage(birdFrame, destinationLeft, destinationRoad.topY, destinationRoad.bottomWidth, outputHeight);
  context.fillStyle = "rgba(2, 6, 23, 0.28)";
  context.fillRect(destinationLeft, destinationRoad.topY, destinationRoad.bottomWidth, outputHeight);
  context.restore();
}

function renderBirdViewFrame(
  source: DetectableSource,
  outputWidth: number,
  outputHeight: number,
  roadCalibration: RoadCalibration,
) {
  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = outputWidth;
  sourceCanvas.height = outputHeight;
  const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
  const outputCanvas = document.createElement("canvas");
  outputCanvas.width = outputWidth;
  outputCanvas.height = outputHeight;
  const outputContext = outputCanvas.getContext("2d", { willReadFrequently: true });
  if (!sourceContext || !outputContext) {
    return outputCanvas;
  }

  sourceContext.drawImage(source, 0, 0, outputWidth, outputHeight);
  const sourceImage = sourceContext.getImageData(0, 0, outputWidth, outputHeight);
  const outputImage = outputContext.createImageData(outputWidth, outputHeight);

  for (let y = 0; y < outputHeight; y += 1) {
    const depth = y / Math.max(1, outputHeight - 1);
    const sourceY = roadCalibration.topY + (roadCalibration.bottomY - roadCalibration.topY) * depth;
    const sourceRoad = sourceRoadAtY(sourceY, roadCalibration);

    for (let x = 0; x < outputWidth; x += 1) {
      const lateral = x / Math.max(1, outputWidth - 1);
      const sourceX = sourceRoad.left + (sourceRoad.right - sourceRoad.left) * lateral;
      const sampleX = clamp(Math.round(sourceX * (outputWidth - 1)), 0, outputWidth - 1);
      const sampleY = clamp(Math.round(sourceY * (outputHeight - 1)), 0, outputHeight - 1);
      const sourceOffset = (sampleY * outputWidth + sampleX) * 4;
      const outputOffset = (y * outputWidth + x) * 4;
      outputImage.data[outputOffset] = sourceImage.data[sourceOffset];
      outputImage.data[outputOffset + 1] = sourceImage.data[sourceOffset + 1];
      outputImage.data[outputOffset + 2] = sourceImage.data[sourceOffset + 2];
      outputImage.data[outputOffset + 3] = 255;
    }
  }

  outputContext.putImageData(outputImage, 0, 0);
  return outputCanvas;
}

function drawCameraLaneGuide(
  context: CanvasRenderingContext2D,
  source: DetectableSource,
  stageWidth: number,
  stageHeight: number,
  mirrorPreview: boolean,
  roadCalibration: RoadCalibration,
) {
  const points = sourceCalibrationToStagePoints(source, stageWidth, stageHeight, mirrorPreview, roadCalibration);

  context.save();
  context.fillStyle = "rgba(250, 204, 21, 0.09)";
  context.strokeStyle = "rgba(250, 204, 21, 0.72)";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(points.topLeft.x, points.topLeft.y);
  context.lineTo(points.topRight.x, points.topRight.y);
  context.lineTo(points.bottomRight.x, points.bottomRight.y);
  context.lineTo(points.bottomLeft.x, points.bottomLeft.y);
  context.closePath();
  context.fill();
  context.stroke();

  context.strokeStyle = "rgba(45, 212, 191, 0.82)";
  context.setLineDash([10, 8]);
  context.beginPath();
  context.moveTo((points.topLeft.x + points.topRight.x) / 2, (points.topLeft.y + points.topRight.y) / 2);
  context.lineTo((points.bottomLeft.x + points.bottomRight.x) / 2, (points.bottomLeft.y + points.bottomRight.y) / 2);
  context.stroke();
  context.setLineDash([]);

  context.fillStyle = "rgba(250, 204, 21, 0.9)";
  context.font = "12px ui-sans-serif, system-ui";
  context.fillText("lane guide", points.topLeft.x + 8, points.topLeft.y - 8);
  context.restore();
}

function drawDetectedLaneLines(
  context: CanvasRenderingContext2D,
  source: DetectableSource,
  stageWidth: number,
  stageHeight: number,
  mirrorPreview: boolean,
  laneDetection: LaneDetection | null,
) {
  if (!laneDetection || laneDetection.confidence <= 0) {
    return;
  }

  context.save();
  drawLaneBands(context, laneDetection.laneBands, "source", (point) =>
    sourcePointToStagePoint(source, stageWidth, stageHeight, mirrorPreview, point.x, point.y),
  );

  context.strokeStyle = "#fb923c";
  context.lineWidth = 3;
  context.lineCap = "round";
  context.lineJoin = "round";

  for (const path of laneDetection.sourcePaths) {
    drawProjectedLanePath(context, path, (point) =>
      sourcePointToStagePoint(source, stageWidth, stageHeight, mirrorPreview, point.x, point.y),
    );
  }

  context.fillStyle = "#fb923c";
  context.font = "12px ui-sans-serif, system-ui";
  context.fillText(`lane candidate ${Math.round(laneDetection.confidence * 100)}%`, 12, stageHeight - 16);
  context.restore();
}

function drawProjectedLanePath(
  context: CanvasRenderingContext2D,
  path: NormalizedPoint[],
  project: (point: NormalizedPoint) => { x: number; y: number },
) {
  if (path.length < 2) {
    return;
  }

  const projected = path.map(project);
  context.beginPath();
  context.moveTo(projected[0].x, projected[0].y);

  for (let index = 1; index < projected.length - 1; index += 1) {
    const current = projected[index];
    const next = projected[index + 1];
    context.quadraticCurveTo(current.x, current.y, (current.x + next.x) / 2, (current.y + next.y) / 2);
  }

  const last = projected[projected.length - 1];
  context.lineTo(last.x, last.y);
  context.stroke();
}

function drawLaneBands(
  context: CanvasRenderingContext2D,
  laneBands: LaneBand[],
  space: "bird" | "source",
  project: (point: NormalizedPoint) => { x: number; y: number },
) {
  laneBands.forEach((band, index) => {
    const leftPath = space === "bird" ? band.birdLeft : band.sourceLeft;
    const rightPath = space === "bird" ? band.birdRight : band.sourceRight;
    if (leftPath.length < 2 || rightPath.length < 2) {
      return;
    }

    const left = leftPath.map(project);
    const right = rightPath.map(project);
    const color = index % 2 === 0 ? "rgba(45, 212, 191, 0.18)" : "rgba(59, 130, 246, 0.16)";

    context.save();
    context.fillStyle = color;
    context.strokeStyle = index % 2 === 0 ? "rgba(45, 212, 191, 0.45)" : "rgba(147, 197, 253, 0.42)";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(left[0].x, left[0].y);
    for (const point of left.slice(1)) {
      context.lineTo(point.x, point.y);
    }
    for (const point of [...right].reverse()) {
      context.lineTo(point.x, point.y);
    }
    context.closePath();
    context.fill();
    context.stroke();

    const labelAnchor = left[Math.max(0, Math.floor(left.length * 0.72))];
    const labelPair = right[Math.max(0, Math.floor(right.length * 0.72))];
    context.fillStyle = index % 2 === 0 ? "#99f6e4" : "#bfdbfe";
    context.font = "11px ui-sans-serif, system-ui";
    context.fillText(`lane ${index + 1}`, (labelAnchor.x + labelPair.x) / 2 - 16, (labelAnchor.y + labelPair.y) / 2);
    context.restore();
  });
}

function detectLaneLinesFromSource(
  source: DetectableSource,
  canvas: HTMLCanvasElement,
  roadCalibration: RoadCalibration,
  options: LaneDetectionOptions,
): LaneDetection {
  const width = 260;
  const height = 180;
  const birdFrame = renderBirdViewFrame(source, width, height, roadCalibration);
  canvas.width = birdFrame.width;
  canvas.height = birdFrame.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    return emptyLaneDetection(roadCalibration);
  }

  context.drawImage(birdFrame, 0, 0);
  const image = context.getImageData(0, 0, width, height);
  const birdPointsByColumn = new Map<number, NormalizedPoint[]>();
  const scanStartY = Math.floor(height * 0.04);
  const scanEndY = Math.floor(height * 0.98);
  const columnCount = 12;

  for (let y = scanStartY; y <= scanEndY; y += 3) {
    for (let column = 0; column < columnCount; column += 1) {
      const startX = Math.floor((column / columnCount) * width);
      const endX = Math.min(width - 1, Math.ceil(((column + 1) / columnCount) * width));
      const candidate = findLanePixelCandidate(image.data, width, y, startX, endX, options.minPixelScore);

      if (!candidate) {
        continue;
      }

      const bucket = Math.max(0, Math.min(columnCount - 1, Math.floor((candidate.x / width) * columnCount)));
      const points = birdPointsByColumn.get(bucket) ?? [];
      points.push({ x: candidate.x / width, y: y / height });
      birdPointsByColumn.set(bucket, points);
    }
  }

  const detectedBirdPaths = mergeNearbyBirdLanePaths(
    [...birdPointsByColumn.values()].flatMap((points) => fitLanePaths(points, options)),
  );
  const birdPaths = extendBirdLanePaths(detectedBirdPaths);
  const pointCount = [...birdPointsByColumn.values()].reduce((total, points) => total + points.length, 0);
  const rowCount = Math.max(1, Math.floor((scanEndY - scanStartY) / 3));
  const confidence = clamp01(
    Math.min(pointCount, rowCount * 2.4) / (rowCount * 2.4),
  );

  return createLaneDetectionFromBirdPaths(birdPaths, roadCalibration, confidence);
}

function findLanePixelCandidate(
  data: Uint8ClampedArray,
  width: number,
  y: number,
  startX: number,
  endX: number,
  minPixelScore: number,
) {
  if (endX <= startX) {
    return null;
  }

  let bestScore = 0;
  let bestX = 0;

  for (let x = startX; x <= endX; x += 1) {
    const offset = (y * width + x) * 4;
    const r = data[offset];
    const g = data[offset + 1];
    const b = data[offset + 2];
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
    const whiteScore = luminance > 150 && max - min < 95 ? luminance - 145 : 0;
    const yellowScore = r > 125 && g > 115 && b < 145 ? (r + g) / 2 - b : 0;
    const score = Math.max(whiteScore, yellowScore);

    if (score > bestScore) {
      bestScore = score;
      bestX = x;
    }
  }

  return bestScore > minPixelScore ? { score: bestScore, x: bestX } : null;
}

function createLaneDetectionFromBirdPaths(
  birdPaths: NormalizedPoint[][],
  roadCalibration: RoadCalibration,
  confidence: number,
): LaneDetection {
  const sourcePaths = birdPaths.map((path) => path.map((point) => birdPointToSourcePoint(point, roadCalibration)));
  const laneBands = createLaneBands(birdPaths, roadCalibration);

  return {
    birdPaths,
    confidence,
    laneBands,
    left: sourcePaths.filter((path) => pathAverageX(path) < 0.5),
    road: roadCalibration,
    right: sourcePaths.filter((path) => pathAverageX(path) >= 0.5),
    roiConfidence: confidence,
    sourcePaths,
    updatedAt: performance.now(),
  };
}

function smoothLaneDetection(
  current: LaneDetection,
  previous: LaneDetection | null,
  roadCalibration: RoadCalibration,
  smoothing: number,
) {
  if (!previous || previous.birdPaths.length === 0 || current.birdPaths.length === 0 || smoothing <= 0) {
    return createLaneDetectionFromBirdPaths(current.birdPaths, roadCalibration, current.confidence);
  }

  const safeSmoothing = clamp(smoothing, 0, 0.9);
  const smoothedPaths = current.birdPaths.map((path) => {
    const previousPath = findNearestLanePath(path, previous.birdPaths);
    if (!previousPath) {
      return path;
    }

    return smoothBirdLanePath(path, previousPath, safeSmoothing);
  });
  const confidence = current.confidence * (1 - safeSmoothing * 0.25) + previous.confidence * safeSmoothing * 0.25;

  return createLaneDetectionFromBirdPaths(smoothedPaths, roadCalibration, confidence);
}

function findNearestLanePath(path: NormalizedPoint[], candidates: NormalizedPoint[][]) {
  const center = pathAverageX(path);
  let nearest: { distance: number; path: NormalizedPoint[] } | null = null;

  for (const candidate of candidates) {
    const distance = Math.abs(pathAverageX(candidate) - center);
    if (!nearest || distance < nearest.distance) {
      nearest = { distance, path: candidate };
    }
  }

  return nearest && nearest.distance < 0.14 ? nearest.path : null;
}

function smoothBirdLanePath(current: NormalizedPoint[], previous: NormalizedPoint[], smoothing: number) {
  const currentRange = pathYRange(current);
  const previousRange = pathYRange(previous);
  const startY = Math.max(0.02, Math.min(currentRange.min, previousRange.min));
  const endY = Math.min(0.98, Math.max(currentRange.max, previousRange.max));
  const sampleCount = Math.max(8, Math.min(16, current.length + 2));
  const points: NormalizedPoint[] = [];

  for (let sample = 0; sample < sampleCount; sample += 1) {
    const y = startY + ((endY - startY) * sample) / Math.max(1, sampleCount - 1);
    const currentX = pathXAtY(current, y);
    const previousX = pathXAtY(previous, y);
    points.push({
      x: clamp01(previousX * smoothing + currentX * (1 - smoothing)),
      y,
    });
  }

  return points;
}

function fitLanePaths(points: NormalizedPoint[], options: LaneDetectionOptions): NormalizedPoint[][] {
  if (points.length < 6) {
    return [];
  }

  const sorted = [...points].sort((a, b) => a.y - b.y);
  const rawSegments: NormalizedPoint[][] = [];
  let currentSegment: NormalizedPoint[] = [];
  const maxGapY = options.joinGapY;
  const maxGapX = 0.09;

  for (const point of sorted) {
    const previous = currentSegment.at(-1);
    if (previous && (point.y - previous.y > maxGapY || Math.abs(point.x - previous.x) > maxGapX)) {
      rawSegments.push(currentSegment);
      currentSegment = [];
    }
    currentSegment.push(point);
  }
  if (currentSegment.length > 0) {
    rawSegments.push(currentSegment);
  }

  const bucketSize = 4;
  const smoothedSegments: NormalizedPoint[][] = [];

  for (const segment of rawSegments) {
    if (segment.length < 6) {
      continue;
    }

    const bucketed: NormalizedPoint[] = [];
    for (let index = 0; index < segment.length; index += bucketSize) {
      const bucket = segment.slice(index, index + bucketSize);
      const average = bucket.reduce(
        (total, point) => ({ x: total.x + point.x, y: total.y + point.y }),
        { x: 0, y: 0 },
      );
      bucketed.push({
        x: clamp01(average.x / bucket.length),
        y: clamp01(average.y / bucket.length),
      });
    }

    if (bucketed.length < 2) {
      continue;
    }

    smoothedSegments.push(
      bucketed.map((point, index) => {
        const previous = bucketed[Math.max(0, index - 1)];
        const next = bucketed[Math.min(bucketed.length - 1, index + 1)];

        return {
          x: clamp01((previous.x + point.x * 2 + next.x) / 4),
          y: point.y,
        };
      }),
    );
  }

  return smoothedSegments;
}

function mergeNearbyBirdLanePaths(paths: NormalizedPoint[][]) {
  return paths
    .filter((path) => path.length >= 2)
    .sort((a, b) => pathAverageX(a) - pathAverageX(b));
}

function extendBirdLanePaths(paths: NormalizedPoint[][]) {
  return paths.map(extendBirdLanePath);
}

function extendBirdLanePath(path: NormalizedPoint[]) {
  const sorted = [...path].sort((a, b) => a.y - b.y);
  if (sorted.length < 2) {
    return sorted;
  }

  const range = pathYRange(sorted);
  if (range.max - range.min < 0.08) {
    return sorted;
  }

  const targetTopY = 0.02;
  const targetBottomY = 0.98;
  const extended = [...sorted];

  if (range.min > targetTopY) {
    extended.unshift({
      x: extrapolatePathXAtY(sorted[0], sorted[Math.min(1, sorted.length - 1)], targetTopY),
      y: targetTopY,
    });
  }

  if (range.max < targetBottomY) {
    extended.push({
      x: extrapolatePathXAtY(sorted[Math.max(0, sorted.length - 2)], sorted[sorted.length - 1], targetBottomY),
      y: targetBottomY,
    });
  }

  return extended;
}

function extrapolatePathXAtY(start: NormalizedPoint, end: NormalizedPoint, y: number) {
  const slope = (end.x - start.x) / Math.max(0.001, end.y - start.y);
  return clamp01(start.x + slope * (y - start.y));
}

function createLaneBands(birdPaths: NormalizedPoint[][], roadCalibration: RoadCalibration): LaneBand[] {
  const sortedPaths = mergeNearbyBirdLanePaths(birdPaths);
  const laneBands: LaneBand[] = [];

  for (let index = 0; index < sortedPaths.length - 1; index += 1) {
    const leftBoundary = sortedPaths[index];
    const rightBoundary = sortedPaths[index + 1];
    const leftRange = pathYRange(leftBoundary);
    const rightRange = pathYRange(rightBoundary);
    const startY = Math.max(leftRange.min, rightRange.min, 0.02);
    const endY = Math.min(leftRange.max, rightRange.max, 0.98);

    if (endY - startY < 0.12) {
      continue;
    }

    const birdLeft: NormalizedPoint[] = [];
    const birdRight: NormalizedPoint[] = [];
    const sampleCount = 10;

    for (let sample = 0; sample < sampleCount; sample += 1) {
      const y = startY + ((endY - startY) * sample) / (sampleCount - 1);
      const leftX = pathXAtY(leftBoundary, y);
      const rightX = pathXAtY(rightBoundary, y);
      const laneWidth = rightX - leftX;

      if (laneWidth < 0.045 || laneWidth > 0.45) {
        continue;
      }

      birdLeft.push({ x: leftX, y });
      birdRight.push({ x: rightX, y });
    }

    if (birdLeft.length < 4 || birdRight.length < 4) {
      continue;
    }

    laneBands.push({
      birdLeft,
      birdRight,
      sourceLeft: birdLeft.map((point) => birdPointToSourcePoint(point, roadCalibration)),
      sourceRight: birdRight.map((point) => birdPointToSourcePoint(point, roadCalibration)),
    });
  }

  return laneBands;
}

function birdPointToSourcePoint(point: NormalizedPoint, roadCalibration: RoadCalibration): NormalizedPoint {
  const sourceY = roadCalibration.topY + (roadCalibration.bottomY - roadCalibration.topY) * clamp01(point.y);
  const sourceRoad = sourceRoadAtY(sourceY, roadCalibration);

  return {
    x: clamp01(sourceRoad.left + (sourceRoad.right - sourceRoad.left) * clamp01(point.x)),
    y: clamp01(sourceY),
  };
}

function pathAverageX(path: NormalizedPoint[]) {
  if (path.length === 0) {
    return 0.5;
  }

  return path.reduce((total, point) => total + point.x, 0) / path.length;
}

function emptyLaneDetection(roadCalibration: RoadCalibration): LaneDetection {
  return {
    birdPaths: [],
    confidence: 0,
    laneBands: [],
    left: [],
    road: roadCalibration,
    right: [],
    roiConfidence: 0,
    sourcePaths: [],
    updatedAt: performance.now(),
  };
}

function projectTrackToBirdView(
  track: Track,
  width: number,
  height: number,
  sourceDimensions: { width: number; height: number } | null,
  destinationRoad: { topWidth: number; bottomWidth: number; topY: number; bottomY: number },
  roadCalibration: RoadCalibration,
) {
  const bottomCenterX = track.box.x + track.box.width / 2;
  const bottomY = track.box.y + track.box.height;
  const sourceWidth = sourceDimensions?.width ?? runtimeDefaults.yoloInputSize;
  const sourceHeight = sourceDimensions?.height ?? runtimeDefaults.yoloInputSize;
  return projectNormalizedPointToBirdView(
    {
      x: clamp01(bottomCenterX / Math.max(1, sourceWidth)),
      y: clamp01(bottomY / Math.max(1, sourceHeight)),
    },
    width,
    height,
    destinationRoad,
    roadCalibration,
  );
}

function projectNormalizedPointToBirdView(
  point: NormalizedPoint,
  width: number,
  height: number,
  destinationRoad: { topWidth: number; bottomWidth: number; topY: number; bottomY: number },
  roadCalibration: RoadCalibration,
) {
  const sourceRoad = sourceRoadAtY(point.y, roadCalibration);
  const lateral = clamp((point.x - sourceRoad.left) / Math.max(0.01, sourceRoad.right - sourceRoad.left), -0.35, 1.35);
  const depth = clamp01((point.y - roadCalibration.topY) / (roadCalibration.bottomY - roadCalibration.topY));
  const easedDepth = depth * depth * (3 - 2 * depth);
  const destinationWidth = destinationRoad.topWidth + (destinationRoad.bottomWidth - destinationRoad.topWidth) * easedDepth;
  const destinationLeft = (width - destinationWidth) / 2;
  const destinationY = destinationRoad.topY + (destinationRoad.bottomY - destinationRoad.topY) * easedDepth;

  return {
    x: destinationLeft + lateral * destinationWidth,
    y: destinationY,
    radius: Math.max(5, Math.min(13, 7 + 8 * easedDepth)),
  };
}

function projectBirdPointToBirdView(
  point: NormalizedPoint,
  width: number,
  destinationRoad: { topWidth: number; bottomWidth: number; topY: number; bottomY: number },
) {
  const destinationWidth = destinationRoad.bottomWidth;
  const destinationLeft = (width - destinationWidth) / 2;

  return {
    x: destinationLeft + clamp01(point.x) * destinationWidth,
    y: destinationRoad.topY + clamp01(point.y) * (destinationRoad.bottomY - destinationRoad.topY),
  };
}

function createManualRoadCalibration(
  topY: number,
  topCenterX: number,
  topWidth: number,
  bottomY: number,
  bottomWidth: number,
): RoadCalibration {
  const safeTopY = clamp(topY, 0.5, 0.8);
  const safeTopCenterX = clamp(topCenterX, 0.25, 0.75);
  const safeTopWidth = clamp(topWidth, 0.06, 0.48);
  const safeBottomY = clamp(bottomY, safeTopY + 0.08, 0.99);
  const safeBottomWidth = clamp(bottomWidth, 0.62, 0.98);

  return {
    bottomLeftX: (1 - safeBottomWidth) / 2,
    bottomRightX: (1 + safeBottomWidth) / 2,
    bottomY: safeBottomY,
    topLeftX: clamp(safeTopCenterX - safeTopWidth / 2, 0.02, 0.92),
    topRightX: clamp(safeTopCenterX + safeTopWidth / 2, 0.08, 0.98),
    topY: safeTopY,
  };
}

function pathXAtY(path: NormalizedPoint[], y: number) {
  const sorted = [...path].sort((a, b) => a.y - b.y);
  if (y <= sorted[0].y) {
    return sorted[0].x;
  }

  for (let index = 1; index < sorted.length; index += 1) {
    const previous = sorted[index - 1];
    const current = sorted[index];
    if (y <= current.y) {
      const ratio = clamp01((y - previous.y) / Math.max(0.001, current.y - previous.y));
      return clamp01(previous.x + (current.x - previous.x) * ratio);
    }
  }

  return sorted[sorted.length - 1].x;
}

function pathYRange(path: NormalizedPoint[]) {
  return path.reduce(
    (range, point) => ({
      max: Math.max(range.max, point.y),
      min: Math.min(range.min, point.y),
    }),
    { max: 0, min: 1 },
  );
}

function sourceRoadAtY(normalizedY: number, roadCalibration: RoadCalibration) {
  const depth = clamp01(
    (normalizedY - roadCalibration.topY) /
      (roadCalibration.bottomY - roadCalibration.topY),
  );

  return {
    left: roadCalibration.topLeftX + (roadCalibration.bottomLeftX - roadCalibration.topLeftX) * depth,
    right:
      roadCalibration.topRightX +
      (roadCalibration.bottomRightX - roadCalibration.topRightX) * depth,
  };
}

function sourceCalibrationToStagePoints(
  source: DetectableSource,
  stageWidth: number,
  stageHeight: number,
  mirrorPreview: boolean,
  roadCalibration: RoadCalibration,
) {
  const dimensions = getSourceDimensions(source);
  const videoRatio = dimensions.width / dimensions.height;
  const stageRatio = stageWidth / stageHeight;
  const drawWidth = stageRatio > videoRatio ? stageHeight * videoRatio : stageWidth;
  const drawHeight = stageRatio > videoRatio ? stageHeight : stageWidth / videoRatio;
  const offsetX = (stageWidth - drawWidth) / 2;
  const offsetY = (stageHeight - drawHeight) / 2;

  const toStagePoint = (x: number, y: number) => {
    const stageX = offsetX + x * drawWidth;
    return {
      x: mirrorPreview ? stageWidth - stageX : stageX,
      y: offsetY + y * drawHeight,
    };
  };

  return {
    bottomLeft: toStagePoint(roadCalibration.bottomLeftX, roadCalibration.bottomY),
    bottomRight: toStagePoint(roadCalibration.bottomRightX, roadCalibration.bottomY),
    topLeft: toStagePoint(roadCalibration.topLeftX, roadCalibration.topY),
    topRight: toStagePoint(roadCalibration.topRightX, roadCalibration.topY),
  };
}

function sourcePointToStagePoint(
  source: DetectableSource,
  stageWidth: number,
  stageHeight: number,
  mirrorPreview: boolean,
  normalizedX: number,
  normalizedY: number,
) {
  const dimensions = getSourceDimensions(source);
  const videoRatio = dimensions.width / dimensions.height;
  const stageRatio = stageWidth / stageHeight;
  const drawWidth = stageRatio > videoRatio ? stageHeight * videoRatio : stageWidth;
  const drawHeight = stageRatio > videoRatio ? stageHeight : stageWidth / videoRatio;
  const offsetX = (stageWidth - drawWidth) / 2;
  const offsetY = (stageHeight - drawHeight) / 2;
  const stageX = offsetX + normalizedX * drawWidth;

  return {
    x: mirrorPreview ? stageWidth - stageX : stageX,
    y: offsetY + normalizedY * drawHeight,
  };
}

function clamp01(value: number) {
  if (!Number.isFinite(value)) {
    return 0;
  }

  return Math.max(0, Math.min(1, value));
}

function clamp(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) {
    return min;
  }

  return Math.max(min, Math.min(max, value));
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
  source: DetectableSource,
  stageWidth: number,
  stageHeight: number,
  mirrorPreview: boolean,
) {
  if (detections.length === 0) {
    return;
  }

  const dimensions = getSourceDimensions(source);
  const videoRatio = dimensions.width / dimensions.height;
  const stageRatio = stageWidth / stageHeight;
  const drawWidth = stageRatio > videoRatio ? stageHeight * videoRatio : stageWidth;
  const drawHeight = stageRatio > videoRatio ? stageHeight : stageWidth / videoRatio;
  const offsetX = (stageWidth - drawWidth) / 2;
  const offsetY = (stageHeight - drawHeight) / 2;
  const scaleX = drawWidth / dimensions.width;
  const scaleY = drawHeight / dimensions.height;

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
