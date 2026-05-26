# V7RC WebYOLO ByteTrack

Chrome-first local web app for webcam YOLO detection, ByteTrack object IDs, and browser-local Gemma4-E2B chat.

## Development

```bash
npm install
npm run dev
```

Open `http://localhost:3000` in Chrome.

## Model Storage

Large model artifacts should live outside the Docker image. The app serves model artifacts as static files, but inference runs in Chrome.

```text
public/models/              # local development path
  yolo/
    yolo11n.onnx
  gemma4-e2b-it-onnx/
    ...

models/                     # production host-mounted path
  yolo/
    yolo11n.onnx
  gemma4-e2b-it-onnx/
    ...
```

Default browser model URLs:

```env
NEXT_PUBLIC_YOLO_MODEL_URL=/models/yolo/yolo11n.onnx
NEXT_PUBLIC_LLM_RUNTIME=transformers
NEXT_PUBLIC_LLM_MODEL_ID=gemma-4-E2B-it-ONNX
NEXT_PUBLIC_LLM_MODEL_URL=/models/gemma4-e2b-it-onnx
```

More details: [docs/models.md](docs/models.md).

Gemma model downloads are cached by WebLLM in browser-managed storage such as IndexedDB, not JavaScript `localStorage`. Clearing Chrome site data will remove the browser-side cached model.

## Video Sources and Stream Gateway

The app can switch between `Camera`, `MJPG`, `RTSP`, and `YouTube` source modes. Camera uses Chrome `getUserMedia()`. MJPG can be read directly through a raw image stream surface. RTSP and YouTube usually need a stream gateway because Chrome cannot directly play native `rtsp://` URLs or YouTube watch pages as canvas-readable media.

Implemented gateway MVP:

- `stream-gateway` Docker service runs on `${STREAM_GATEWAY_PORT:-3001}`.
- RTSP and YouTube modes call the gateway API and receive a browser-readable stream URL.
- The default gateway output is MJPG because it plugs into the existing image-frame YOLO path.
- YouTube support uses `yt-dlp` plus ffmpeg when allowed by the source and deployment policy.
- HLS output is available at the gateway API level for later frontend/player work.
- Later: RTSP -> WebRTC for lower-latency robot closed-loop control.

The gateway runs next to the Next.js app in Docker Compose and exposes browser-compatible URLs such as:

```text
http://localhost:3001/streams/robot-front.mjpg
http://localhost:3001/streams/robot-front/index.m3u8
```

More details: [docs/stream-gateway.md](docs/stream-gateway.md).

## Prepare YOLO11n

The MVP expects `public/models/yolo/yolo11n.onnx` during local development.

```bash
bash scripts/prepare-yolo11n.sh
```

This creates a local Python virtual environment, installs Ultralytics, exports `yolo11n.pt` to ONNX, and copies it into the app's public model directory. The Docker setup can instead mount `./models/yolo/yolo11n.onnx` to `/app/public/models/yolo/yolo11n.onnx`.

Generated files are intentionally ignored by git:

```text
.venv-yolo/
.venv-models/
.model-export/
public/models/**/*.onnx
public/models/gemma4-e2b-it
public/models/gemma4-e2b-it-onnx
models/
```

## Prepare Gemma4-E2B ONNX

Download the browser-ready Gemma4-E2B artifact to the host-side model volume:

```bash
bash scripts/prepare-gemma4-e2b-onnx.sh
```

This downloads the Transformers.js-compatible ONNX q4f16 files from `onnx-community/gemma-4-E2B-it-ONNX` to:

```text
models/gemma4-e2b-it-onnx/
```

Docker Compose mounts `./models` into `/app/public/models`, so production can serve:

```env
NEXT_PUBLIC_LLM_RUNTIME=transformers
NEXT_PUBLIC_LLM_MODEL_ID=gemma-4-E2B-it-ONNX
NEXT_PUBLIC_LLM_MODEL_URL=/models/gemma4-e2b-it-onnx
```

## Optional Gemma4-E2B WebLLM

The earlier MLC/WebLLM artifact remains available for comparison:

```bash
bash scripts/prepare-gemma4-e2b-webllm.sh
```

Use it only when explicitly selecting the WebLLM runtime:

```env
NEXT_PUBLIC_LLM_RUNTIME=webllm
NEXT_PUBLIC_LLM_MODEL_ID=gemma-4-E2B-it-q4f16_1-MLC
NEXT_PUBLIC_LLM_MODEL_URL=/models/gemma4-e2b-it
NEXT_PUBLIC_LLM_MODEL_LIB_URL=/models/gemma4-e2b-it/libs/gemma-4-E2B-it-q4f16_1-MLC-webgpu.wasm
```

## Docker

```bash
docker compose up --build
```

Chrome camera access works on `localhost`. For production hosts, serve over HTTPS.

## iPhone Camera on Mac

On macOS, iPhone camera support uses Apple's Continuity Camera. Keep the iPhone nearby, signed into the same Apple ID, with Wi-Fi and Bluetooth enabled. In Chrome, grant camera permission, then use `Refresh` in the Camera control if the iPhone does not appear immediately. The app marks iPhone/Continuity/Desk View devices in the camera picker and prefers them when no camera has been selected yet.

## Runtime Notes

- YOLO inference runs in Chrome with ONNX Runtime Web.
- ByteTrack runs in Chrome with TypeScript IoU association and stable `T1`, `T2`, ... IDs.
- Gemma4-E2B runs through browser-local Transformers.js ONNX/WebGPU generation in a Web Worker.
- When `Include current frame` is enabled, the app captures the active source frame as an image and sends it to Gemma4 together with the YOLO/ByteTrack track summary.
- Gemma settings live behind the Gemma4-E2B settings button and are cached in browser `localStorage`.
- The Gemma perception loop is controlled from the Gemma4-E2B card and keeps the current MVP observation-only.
