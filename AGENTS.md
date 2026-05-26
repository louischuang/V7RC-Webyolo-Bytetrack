# AGENTS.md

## Project Mission

Build a local-first Chrome web application for real-time camera/stream object detection, object tracking, and multimodal chat.

The MVP runs fully in Chrome for camera or browser-compatible stream playback, YOLO inference, ByteTrack tracking, Gemma4-E2B inference, overlays, chat, and UI rendering. Next.js serves the app and static model artifacts only; it must not proxy chat to a server-side LLM API. An optional stream-gateway service may convert RTSP or YouTube sources into browser-compatible streams, but it must not run YOLO, ByteTrack, or LLM inference.

## Primary User Goals

- Open the app in Chrome.
- Select a source: Camera, MJPG, RTSP, or YouTube.
- Select a webcam, including a Mac built-in camera or iPhone Continuity Camera when macOS exposes it as a camera device.
- Enter browser-compatible stream URLs or gateway-backed RTSP/YouTube URLs.
- See the live camera or stream source.
- Run YOLO object detection on active source frames.
- Run ByteTrack to assign stable IDs to detected objects.
- Draw bounding boxes, class labels, confidence, and track IDs over the live camera.
- Show a live object list with object name, ID, and confidence.
- Show FPS, average YOLO inference time, and average ByteTrack processing time.
- Chat with an in-browser Gemma4-E2B model using text plus optional current-frame image context.
- Package the app as a Docker container for production deployment.
- Keep large model weights outside the container image and mount or download them into host-managed volumes.

## Working Language

- User-facing planning and docs may be written in Traditional Chinese.
- Code, file names, identifiers, comments, and commit messages should prefer English unless a user explicitly asks otherwise.

## Expected Tech Stack

- App framework: Next.js with TypeScript.
- Browser inference: ONNX Runtime Web with WebGPU preferred, WASM fallback.
- Detector: Ultralytics YOLO11n detection model exported to ONNX for MVP; YOLO11s is the first quality upgrade if browser performance is acceptable.
- Tracker: ByteTrack implemented in TypeScript and run in the browser main thread first; move to Web Worker if UI latency requires it.
- Camera: Browser `navigator.mediaDevices.getUserMedia()` and `enumerateDevices()`.
- Stream gateway: optional Docker service using ffmpeg, GStreamer, or MediaMTX to convert RTSP/YouTube into MJPG, HLS, or later WebRTC.
- Overlay: HTML canvas layered over the video element.
- Browser LLM runtime: WebGPU-first runtime such as Transformers.js, WebLLM, MLC WebLLM, or another proven browser inference library.
- LLM artifact: `google/gemma-4-E2B-it` or a browser-ready quantized conversion derived from it.
- Packaging: Docker multi-stage build for Next.js standalone output.

## Important Model Note

The user requested `Gemma4:E2B` running directly in the web page. Hugging Face currently lists Google official repositories for `google/gemma-4-E2B` and `google/gemma-4-E2B-it`, including multimodal image-text usage through Transformers. Use the instruction-tuned model as the preferred source model for chat.

Do not design this as an Ollama, Python, or server API flow. The browser should download/cache the model artifact and execute inference locally with WebGPU when possible.

Do not assume the raw Hugging Face safetensors checkpoint is the best browser artifact. Prefer a browser-ready quantized package when available, and keep the source model ID and runtime model URL configurable:

- `NEXT_PUBLIC_LLM_RUNTIME`
- `NEXT_PUBLIC_LLM_MODEL_ID`
- `NEXT_PUBLIC_LLM_MODEL_URL`
- `NEXT_PUBLIC_LLM_MAX_NEW_TOKENS`

## Architecture Principles

- Keep camera frames local. Do not upload video frames to remote services.
- RTSP and YouTube gateway conversion is transport normalization only; keep inference in Chrome unless the user explicitly changes architecture.
- Keep object detection and tracking in the browser whenever feasible.
- Keep large LLM weights outside the Docker image unless the user explicitly chooses an offline bundled image.
- Keep the Docker image reproducible and small.
- Prefer explicit performance instrumentation from the start.
- Use stable track IDs from ByteTrack rather than generating ad hoc IDs from detections.
- Design for model/runtime replacement: YOLO model path, label map, detection thresholds, tracker parameters, and browser LLM artifact URL should be configurable.
- Prefer ONNX Runtime Web for YOLO in the MVP. Consider TFLite/LiteRT only as a benchmark branch or if a target device/browser shows clearly better results.

## Security and Privacy

- Camera permission must be requested only after an explicit user action.
- The app must work on `localhost`; production camera access requires HTTPS.
- Never send camera frames to external APIs by default.
- Multimodal chat should keep selected snapshots or frame representations inside the browser runtime.
- Treat model downloads as explicit browser-side cached assets or a controlled host setup step that places model artifacts in a static model directory.
- Make first-load model download size and progress visible to the user.

## UI Layout Contract

The first usable screen should be the actual app, not a marketing landing page.

Layout:

1. Top menu/status bar.
   - Source selector: Camera, MJPG, RTSP, YouTube.
   - Start/stop active source.
   - Camera selector or stream URL input depending on source.
   - Detector status.
   - Tracker status.
   - Browser LLM status.
   - FPS.
   - Average YOLO inference time.
   - Average ByteTrack time.
2. Main upper area.
   - Left: live webcam video with canvas overlay.
   - Right: tracked object list with object name, track ID, and confidence.
3. Bottom area.
   - Gemma response transcript.
   - Response count and last inference time.
   - Prompt and include-current-frame settings live in the Gemma settings modal.

## Development Rules for Future Agents

- Read existing files before changing architecture.
- Preserve user changes in a dirty worktree.
- Keep implementation scoped to the current milestone.
- Add tests around pure logic such as detection decoding, IoU, Kalman matching, and ByteTrack state transitions.
- Verify browser behavior in Chrome or Chromium when camera and WebGPU features are involved.
- Use real performance measurements rather than assumptions.
- Avoid adding cloud dependencies unless the user asks.

## Definition of Done for MVP

- Docker build succeeds.
- App runs in Chrome.
- Camera selector lists available cameras.
- Source selector switches Camera, MJPG, RTSP, and YouTube modes.
- Camera or stream source renders.
- YOLO inference runs on sampled frames.
- ByteTrack assigns stable IDs across frames.
- Bounding boxes and IDs draw correctly on top of the video.
- Object list updates live.
- FPS and timing metrics are visible.
- In-browser Gemma4-E2B chat can load and return responses without calling a chat API.
- Model storage is externalized from the container image.
