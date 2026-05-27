# Model Artifacts

This app is designed around browser-side inference. Next.js and Docker serve the UI and static model files; Chrome performs YOLO detection, ByteTrack tracking, and Gemma4-E2B chat locally.

## Model Layout

Use this layout for local development:

```text
public/models/
  yolo/
    yolo11n.onnx
  gemma4-e2b-it-onnx/
    ...
```

Use this layout for production host storage:

```text
models/
  yolo/
    yolo11n.onnx
  gemma4-e2b-it-onnx/
    ...
```

Docker Compose mounts production model storage here:

```text
./models:/app/public/models:ro
```

## YOLO11n

MVP detector:

- Source model: Ultralytics `yolo11n.pt`
- Browser artifact: `yolo11n.onnx`
- Runtime: `onnxruntime-web`
- Preferred execution provider: WebGPU
- Fallback execution provider: WASM
- Input size: `640`
- NMS: implemented in TypeScript, not embedded in ONNX

Default URL:

```env
NEXT_PUBLIC_YOLO_MODEL_URL=/models/yolo/yolo11n.onnx
```

Local export command:

```bash
bash scripts/prepare-yolo11n.sh
```

The script does this:

1. Creates `.venv-yolo/`.
2. Installs Ultralytics, ONNX, ONNX Slim, and ONNX Runtime.
3. Downloads `yolo11n.pt`.
4. Exports fixed-shape ONNX with 640 input.
5. Copies the artifact to `public/models/yolo/yolo11n.onnx`.

Equivalent export command:

```bash
yolo export model=yolo11n.pt format=onnx imgsz=640 dynamic=false simplify=true opset=17 nms=false
```

Generated local artifacts are ignored by git:

```text
.venv-yolo/
.model-export/
public/models/**/*.onnx
```

For production, copy the exported file to:

```text
models/yolo/yolo11n.onnx
```

The Docker Compose web service mounts `./models` to `/app/public/models:ro`, so `models/yolo/yolo11n.onnx` must exist for containerized YOLO inference. Local development can still use `public/models/yolo/yolo11n.onnx`.

## Gemma4-E2B

MVP chat design:

- Source model ID: `google/gemma-4-E2B-it`
- Execution target: browser, not a server API
- Runtime: Transformers.js / ONNX worker
- Default LLM device: WASM, so YOLO keeps WebGPU priority for safety perception
- Current artifact: `onnx-community/gemma-4-E2B-it-ONNX`
- Current scope: text generation with YOLO/ByteTrack track summaries

Default config:

```env
NEXT_PUBLIC_LLM_RUNTIME=transformers
NEXT_PUBLIC_LLM_DEVICE=wasm
NEXT_PUBLIC_LLM_MODEL_ID=gemma-4-E2B-it-ONNX
NEXT_PUBLIC_LLM_MODEL_URL=/models/gemma4-e2b-it-onnx
NEXT_PUBLIC_LLM_MAX_NEW_TOKENS=160
NEXT_PUBLIC_LLM_TEMPERATURE=0.2
```

Important deployment note:

The raw Hugging Face safetensors checkpoint is the source model, not necessarily the final browser artifact. For browser deployment, prefer a runtime-specific package such as an MLC/WebLLM or Transformers.js-compatible quantized artifact.

The current integrated runtime uses Transformers.js with the ONNX community artifact `onnx-community/gemma-4-E2B-it-ONNX`. The model loads in a Web Worker. For robot safety, the default LLM device is `wasm`, even though this can make Gemma slower, because YOLO/ByteTrack must remain the real-time accident and obstacle perception path. Set `NEXT_PUBLIC_LLM_DEVICE=webgpu` only when benchmarking LLM speed or testing without safety-critical vision.

First load downloads model shards into browser-managed cache storage. For offline production, mirror the q4f16 ONNX artifact under:

```text
models/gemma4-e2b-it-onnx/
```

Then mount `./models` into the container so the app can serve it at:

```text
/models/gemma4-e2b-it-onnx
```

The current chat implementation sends text plus an optional summary of active YOLO/ByteTrack tracks. With the default Transformers.js ONNX runtime, `Include current frame` also captures the current webcam frame as a JPEG image and sends it as a Gemma4 multimodal image input.

Download helper:

```bash
bash scripts/prepare-gemma4-e2b-onnx.sh
```

The helper downloads `onnx-community/gemma-4-E2B-it-ONNX` q4f16 files with `huggingface_hub` into `models/gemma4-e2b-it-onnx/` and creates a local development symlink at `public/models/gemma4-e2b-it-onnx`.

### Optional WebLLM Artifact

The MLC/WebLLM artifact `welcoma/gemma-4-E2B-it-q4f16_1-MLC` is still supported behind `NEXT_PUBLIC_LLM_RUNTIME=webllm`, but testing in Chrome showed repeated control-token-only responses with WebLLM `0.2.83`. Keep it as an experiment path, not the default.

Prepare it with:

```bash
bash scripts/prepare-gemma4-e2b-webllm.sh
```

The helper creates `models/gemma4-e2b-it/resolve/main` as a real directory with file symlinks. Do not replace it with a symlink back to `..`; that creates a recursive path that prevents Next.js production servers from starting.

### Browser Cache Storage

Do not use JavaScript `localStorage` for Gemma model files. `localStorage` is too small and string-only, so it is suitable for settings but not model weights.

Transformers.js and WebLLM should cache model artifacts in browser storage, typically Cache API, IndexedDB, and related browser-managed storage.

Practical behavior:

- First `Load Gemma` downloads model shards and initializes WebGPU sessions.
- Chrome stores those artifacts under this site's browser data.
- Later loads should reuse the local browser cache instead of re-downloading everything.
- Clearing site data, using a different browser profile, or changing host/origin can remove or bypass that cache.
- For production, keep a host-side copy under `models/gemma4-e2b-it-onnx/` as the deployable source, then let each browser cache it locally after first use.

## Docker Deployment

Expected host tree:

```text
project/
  docker-compose.yml
  models/
    yolo/
      yolo11n.onnx
    gemma4-e2b-it-onnx/
      ...
```

Start:

```bash
docker compose up --build
```

Chrome downloads the model files from the web app and runs inference locally. Camera access works on `localhost`; production hostnames must use HTTPS for browser camera permissions.

## Camera and Stream Sources

The camera selector lists Chrome `videoinput` devices. On macOS, iPhone camera support comes through Apple's Continuity Camera and appears as another video input after Chrome has camera permission.

Behavior:

- iPhone, Continuity Camera, and Desk View labels are marked as iPhone options in the selector.
- If no camera is selected yet, the app prefers an iPhone/Continuity camera when one is available.
- The `Refresh` camera button reruns device enumeration after plugging in, waking, or selecting the iPhone camera from macOS.
- The `devicechange` browser event also refreshes the list automatically when macOS reports a camera change.

The app also exposes source modes for MJPG, RTSP, and YouTube:

- `MJPG` can point directly at an MJPG stream URL when CORS allows canvas reads.
- `RTSP` should point to a stream-gateway output URL, not a native `rtsp://` URL.
- `YouTube` should point to a stream-gateway output URL, not a regular YouTube watch page.

Chrome playback limitations:

- Native `rtsp://` is not supported by the browser video element.
- YouTube watch pages are iframe/player pages, not canvas-readable media streams.
- Cross-origin streams must allow canvas access if the app needs to capture frames for YOLO and Gemma.
- HLS support in Chrome may require a JavaScript HLS player unless the target platform provides native support.

The planned stream gateway is documented in [stream-gateway.md](stream-gateway.md).

## Tracking Runtime

ByteTrack runs in the browser after YOLO inference. The current implementation keeps the tracker in TypeScript so it can share the same frame coordinates as the overlay and object list.

Config:

```env
NEXT_PUBLIC_TRACK_HIGH_THRESH=0.6
NEXT_PUBLIC_TRACK_LOW_THRESH=0.1
NEXT_PUBLIC_TRACK_MATCH_THRESH=0.8
NEXT_PUBLIC_TRACK_BUFFER_FRAMES=30
```

Behavior:

- High-confidence detections start and update tracks.
- Low-confidence detections can recover unmatched active tracks.
- Matching uses class-aware IoU association with ByteTrack-style cost threshold semantics.
- Matching uses a lightweight constant-velocity prediction and center-distance score to reduce ID switches when objects cross.
- Tracks stay alive for `NEXT_PUBLIC_TRACK_BUFFER_FRAMES` missed detection frames.
- Lost tracks inside the buffer can be matched again instead of immediately creating a new ID.
- UI track IDs use `T1`, `T2`, etc.

## Verification

Check model availability:

```bash
curl -I http://localhost:3000/models/yolo/yolo11n.onnx
curl -I http://localhost:3000/models/gemma4-e2b-it-onnx/config.json
```

Expected YOLO response:

```text
HTTP/1.1 200 OK
Content-Type: application/octet-stream
```

If YOLO status is not ready in the UI:

- Confirm `NEXT_PUBLIC_YOLO_MODEL_URL`.
- Confirm the ONNX file is mounted or present under `public/models/yolo/`.
- Check browser console for WebGPU or ONNX Runtime errors.
- Confirm WASM fallback files from `onnxruntime-web` are bundled and served correctly.
