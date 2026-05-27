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

- `stream-gateway` Docker service runs on `${STREAM_GATEWAY_PORT:-3010}`.
- RTSP and YouTube modes call the gateway API and receive a browser-readable stream URL.
- RTSP currently uses HLS in the frontend.
- YouTube can be started as either MP4 or HLS from the output selector.
- MP4 is the fastest YouTube path because the gateway resolves the media URL and streams fragmented MP4 to Chrome without transcoding when possible.
- HLS is the compatibility path for longer-running streams and browser-friendly HTTP playback.
- YouTube support uses `yt-dlp` plus ffmpeg when allowed by the source and deployment policy.
- YouTube mode includes a `Check` action that asks the gateway to resolve the watch URL before starting the stream.
- `YTDLP_FORMAT`, `YTDLP_TIMEOUT_MS`, `YTDLP_COOKIES_FILE`, and `YTDLP_USER_AGENT` can tune YouTube resolution in Docker Compose.
- Later: RTSP -> WebRTC for lower-latency robot closed-loop control.

The gateway runs next to the Next.js app in Docker Compose and exposes browser-compatible URLs such as:

```text
http://localhost:3010/streams/robot-front.mjpg
http://localhost:3010/streams/robot-front/index.m3u8
http://localhost:3010/streams/youtube-demo.mp4
```

More details: [docs/stream-gateway.md](docs/stream-gateway.md).

## Prepare YOLO11n

The MVP expects `public/models/yolo/yolo11n.onnx` during local development.

```bash
bash scripts/prepare-yolo11n.sh
```

This creates a local Python virtual environment, installs Ultralytics, exports `yolo11n.pt` to ONNX, and copies it into the app's public model directory. The Docker setup can instead mount `./models/yolo/yolo11n.onnx` to `/app/public/models/yolo/yolo11n.onnx`.

For Docker testing or production, keep a copy under the host model volume:

```bash
mkdir -p models/yolo
cp public/models/yolo/yolo11n.onnx models/yolo/yolo11n.onnx
```

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

Source deep links are supported for unattended browser tests and robot launch flows:

```text
http://localhost:3000/?source=youtube&autostart=1&url=https%3A%2F%2Fyoutu.be%2F...
```

## iPhone Camera on Mac

On macOS, iPhone camera support uses Apple's Continuity Camera. Keep the iPhone nearby, signed into the same Apple ID, with Wi-Fi and Bluetooth enabled. In Chrome, grant camera permission, then use `Refresh` in the Camera control if the iPhone does not appear immediately. The app marks iPhone/Continuity/Desk View devices in the camera picker and prefers them when no camera has been selected yet.

## Runtime Notes

- YOLO inference runs in Chrome with ONNX Runtime Web.
- ByteTrack runs in Chrome with TypeScript IoU association and stable `T1`, `T2`, ... IDs.
- Gemma4-E2B runs through browser-local Transformers.js ONNX/WebGPU generation in a Web Worker.
- When `Include current frame` is enabled, the app captures the active source frame as an image and sends it to Gemma4 together with the YOLO/ByteTrack track summary.
- Gemma settings live behind the Gemma4-E2B settings button and are cached in browser `localStorage`.
- The Gemma perception loop is controlled from the Gemma4-E2B card and keeps the current MVP observation-only.
