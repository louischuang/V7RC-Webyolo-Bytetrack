# MVP.md

## MVP Goal

Create a Docker-packaged Next.js web app that runs in Chrome, opens a webcam or browser-compatible stream source, performs YOLO object detection in the browser, tracks objects with ByteTrack, overlays bounding boxes and IDs on the video, lists tracked objects, and runs Gemma4-E2B chat directly inside the web page without a server-side LLM API.

## MVP Scope

### In Scope

- Next.js TypeScript app.
- Chrome-first webcam access.
- Source selection for Camera, MJPG, RTSP, and YouTube.
- Camera selection from browser-exposed devices.
- macOS iPhone Continuity Camera support when Chrome lists the iPhone as a `videoinput`.
- Direct browser playback for camera, MJPG, and browser-compatible video URLs.
- Stream gateway plan for RTSP and YouTube URLs that Chrome cannot play directly.
- Live video display.
- Canvas overlay for object bounding boxes, class labels, confidence, and ByteTrack ID.
- Browser-side YOLO ONNX inference.
- Browser-side ByteTrack.
- Object list panel.
- Top menu with app status and metrics.
- Bottom chat panel.
- Browser-side Gemma4-E2B inference using WebGPU-first runtime support.
- Browser-side model download, progress display, and cache-aware loading.
- Docker production build.
- Host-mounted model/data volumes.
- Optional stream-gateway container for RTSP/YouTube conversion.
- Setup scripts or documentation for downloading local model assets outside the container.

### Out of Scope for MVP

- Multi-camera simultaneous detection.
- Cloud inference.
- User accounts.
- Persistent chat history database.
- Recording video.
- Fine-tuning YOLO or Gemma models.
- Complex alert rules.
- Mobile browser support beyond basic responsive layout.
- Server-side LLM APIs for chat.
- Ollama, Python, or llama.cpp server dependency for MVP chat.
- GPU acceleration inside the Docker container, because YOLO and Gemma inference are planned in Chrome.

## Proposed System Architecture

```text
Chrome
  ├─ Source selection: Camera / MJPG / RTSP / YouTube
  ├─ Camera capture: getUserMedia()
  ├─ Browser-compatible streams: video or MJPG image surface
  ├─ Canvas overlay
  ├─ YOLO ONNX inference: ONNX Runtime Web
  ├─ ByteTrack tracker: TypeScript
  ├─ Gemma4-E2B inference: WebGPU browser runtime
  ├─ Browser cache: model artifacts
  ├─ Metrics collector
  └─ UI state: React

Next.js server
  ├─ Static app serving
  ├─ YOLO model artifact serving
  └─ optional Gemma4 browser artifact serving

Stream gateway
  ├─ RTSP input
  ├─ YouTube/watch URL input
  ├─ ffmpeg/GStreamer/MediaMTX conversion
  └─ Browser-compatible output: MJPG, HLS, MP4 fragment, or WebRTC

Host storage
  └─ Optional mounted model artifacts outside the Docker image
```

## Stream Source Strategy

Chrome can read camera devices through `getUserMedia()`, MJPG streams through a raw image surface, and common browser video formats through a `<video>` element. Chrome cannot directly play native `rtsp://` URLs or YouTube watch pages as canvas-readable video sources.

MVP behavior:

- `Camera`: use `getUserMedia()` and the selected Chrome `videoinput`.
- `MJPG`: accept a direct MJPG URL and render it through an `<img>` stream surface.
- `RTSP`: accept a URL field and use the stream gateway to expose a browser-readable HLS URL.
- `YouTube`: accept a watch/share URL and use the stream gateway to expose selectable MP4 or HLS output, not the regular watch page.

Recommended stream-gateway MVP:

- Add a `stream-gateway` Docker service next to `web`.
- Use ffmpeg or MediaMTX first because they are proven and simple to deploy.
- Convert RTSP to HLS for browser playback now, with WebRTC reserved for lower-latency robot control.
- Convert YouTube through `yt-dlp` plus ffmpeg into MP4 or HLS when allowed by the source and deployment policy.
- Expose generated streams under stable local URLs such as `/streams/{id}.mjpg`, `/streams/{id}/index.m3u8`, or `/streams/{id}.mp4`.
- Keep YOLO, ByteTrack, and Gemma inference in Chrome; the gateway only normalizes video transport.

Long-term low-latency option:

- RTSP to WebRTC is the best fit for robot closed-loop control when latency matters.
- WebRTC requires signaling, ICE/TURN planning, lifecycle cleanup, and browser autoplay/error handling, so it is a later milestone after MJPG/HLS gateway validation.

## Runtime Model Strategy

### YOLO

Use Ultralytics YOLO11n detection exported to ONNX for MVP.

Rationale:

- `YOLO11n` is the best first target for browser webcam inference because it is the smallest YOLO11 detection model while still providing COCO pretrained classes.
- Ultralytics reports `YOLO11n` at 2.6M parameters, 6.5B FLOPs, 39.5 mAP val 50-95, and 56.1 ms CPU ONNX speed at 640 pixels.
- `YOLO11s` is the first upgrade candidate when quality matters more than latency; it improves mAP but is materially heavier.
- Older `YOLOv8n` remains a fallback if YOLO11 export/runtime compatibility causes issues, because YOLOv8 browser examples are common.

Configurable values:

- `NEXT_PUBLIC_YOLO_MODEL_URL`
- `NEXT_PUBLIC_YOLO_LABELS_URL`
- `NEXT_PUBLIC_YOLO_INPUT_SIZE`
- `NEXT_PUBLIC_YOLO_CONF_THRESHOLD`
- `NEXT_PUBLIC_YOLO_IOU_THRESHOLD`
- `NEXT_PUBLIC_YOLO_FRAME_INTERVAL`

The model can be served from the Next.js public folder for development and from a mounted model directory or CDN-like static path in production.

Recommended export shape:

```bash
yolo export model=yolo11n.pt format=onnx imgsz=640 dynamic=false simplify=true opset=17 nms=false
```

Keep NMS in TypeScript for MVP so the app can expose thresholds, debug decoded boxes, and feed ByteTrack consistently. Revisit `nms=true` only after the pipeline is stable.

### ONNX Runtime Web vs TFLite/LiteRT

Use ONNX Runtime Web for the MVP.

Why ONNX Runtime Web is the default:

- It is a direct, well-supported path from Ultralytics YOLO exports.
- It has a browser package, `onnxruntime-web`, with WebGPU import support.
- It supports Chrome and Edge on macOS with WebGPU, plus WebAssembly fallback.
- It keeps preprocessing, decoding, NMS, and ByteTrack in one TypeScript pipeline.
- It avoids extra TensorFlow conversion steps and conversion-specific output shape surprises.

Why not TFLite/LiteRT first:

- LiteRT/TFLite is excellent for mobile and edge deployment, but browser arbitrary-model support is less straightforward for this Next.js Chrome-first MVP.
- TFLite conversion adds another validation surface before we even prove the camera and tracker loop.
- Browser-side TFLite/LiteRT should still be tested later because quantized models may reduce download size and memory pressure.

Decision:

- MVP primary: `YOLO11n.onnx` + ONNX Runtime Web WebGPU.
- MVP fallback: ONNX Runtime Web WASM.
- Benchmark later: `YOLO11n.tflite`/LiteRT Web and possibly TF.js export.
- Quality upgrade: `YOLO11s.onnx` if FPS and thermals remain acceptable.

### ByteTrack

Implement the tracker in TypeScript with:

- Detection input: bounding box, score, class ID, class label.
- IoU matching.
- High-score and low-score association phases.
- Track lifecycle: tracked, lost, removed.
- Configurable thresholds.

Configurable values:

- `NEXT_PUBLIC_TRACK_HIGH_THRESH`
- `NEXT_PUBLIC_TRACK_LOW_THRESH`
- `NEXT_PUBLIC_TRACK_MATCH_THRESH`
- `NEXT_PUBLIC_TRACK_BUFFER_FRAMES`

### Browser-Local Multimodal LLM

The user requested `Gemma4:E2B` running directly inside the web page. Hugging Face currently lists Google official repositories for `google/gemma-4-E2B` and the instruction-tuned `google/gemma-4-E2B-it`. The MVP should use a browser runtime such as Transformers.js, WebLLM, MLC WebLLM, or another WebGPU-first runtime that can run a compatible Gemma4-E2B browser artifact.

Recommended environment variables:

- `NEXT_PUBLIC_LLM_RUNTIME=webgpu`
- `NEXT_PUBLIC_LLM_MODEL_ID=google/gemma-4-E2B-it`
- `NEXT_PUBLIC_LLM_MODEL_URL=/models/gemma4-e2b-it`
- `NEXT_PUBLIC_LLM_MAX_NEW_TOKENS=512`
- `NEXT_PUBLIC_LLM_TEMPERATURE=0.7`

If the selected browser runtime requires a converted or quantized artifact, `NEXT_PUBLIC_LLM_MODEL_URL` should point to that browser-ready package. The raw Hugging Face safetensors checkpoint is a source model, not necessarily the production browser artifact.

Model weights should be stored outside the app container, for example:

```text
./models/yolo:/app/public/models/yolo
./models/gemma4-e2b-it:/app/public/models/gemma4-e2b-it
```

## Docker Strategy

Use a multi-stage Dockerfile:

1. Install dependencies.
2. Build Next.js standalone output.
3. Copy standalone server, static files, and public assets into a slim runtime image.

Use Docker Compose for local production-like deployment:

- `web`: Next.js app.
- `stream-gateway`: optional RTSP/YouTube to browser stream converter.
- Named or bind-mounted volume for model weights.
- Optional bind-mounted volume for gateway temporary HLS segments.

Do not bake large YOLO or LLM model files into the image.

## UX Requirements

### Top Menu

Must show:

- App name.
- Source selector: Camera, MJPG, RTSP, YouTube.
- Source-specific controls: camera selector or stream URL input.
- Start/stop active source.
- Detection enabled toggle.
- Tracking enabled toggle.
- Browser LLM status.
- FPS.
- Average YOLO inference time.
- Average ByteTrack time.

### Camera and Overlay

- Preserve video aspect ratio.
- Overlay canvas must match the displayed video dimensions.
- Bounding boxes must align with the displayed video after resize.
- Labels must include object class, confidence, and track ID.

Example label:

```text
person #12 0.87
```

### Object List

Columns:

- ID.
- Object name.
- Confidence.
- Last seen timestamp or age.

Sort by most recently updated track first.

### Chat

Features:

- Include current frame option.
- Fixed robot perception prompt.
- Start/stop Gemma perception loop.
- Response count and last inference time.
- Loading state.
- Model download progress.
- Error state when WebGPU is unavailable, model artifacts are missing, or the browser runs out of memory.

## Performance Targets

Initial MVP targets on a modern MacBook in Chrome:

- UI remains responsive.
- Camera preview at 30 FPS where available.
- Detection loop can run independently from display refresh.
- YOLO inference target: under 100 ms per sampled frame for a small model.
- ByteTrack target: under 5 ms per detection frame for common object counts.
- Metrics use rolling averages over recent frames.

## Risks and Mitigations

- Browser WebGPU availability varies.
  - Use ONNX Runtime Web WASM fallback.
- YOLO output format differs by model export.
  - Isolate decoding in a dedicated adapter.
- Camera labels are hidden until permission is granted.
  - Request permission before expecting friendly camera names.
- iPhone camera availability depends on macOS Continuity Camera and Chrome device enumeration.
  - Treat it as a normal `videoinput` if present.
- Browser Gemma4-E2B support depends on WebGPU, available memory, and browser-ready model artifacts.
  - Show capability diagnostics and provide clear errors.
- Large model downloads make Docker images slow and brittle.
  - Keep models in host volumes or browser cache and document explicit pre-download commands.

## MVP Acceptance Checklist

- `npm run dev` starts the app locally.
- Chrome can grant camera permission.
- Camera selector works.
- Live video is visible.
- YOLO model loads successfully.
- Detections are visible as boxes.
- ByteTrack IDs persist across consecutive frames.
- Object list matches overlay IDs.
- FPS and average timing metrics update.
- Chat runs Gemma4-E2B in Chrome without a chat API call.
- Docker image builds.
- Docker container serves the app.
- External model volumes are documented and usable.
