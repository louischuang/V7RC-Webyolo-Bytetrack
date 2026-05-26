# Model Artifacts

This app is designed around browser-side inference. Next.js and Docker serve the UI and static model files; Chrome performs YOLO detection, ByteTrack tracking, and Gemma4-E2B chat locally.

## Model Layout

Use this layout for local development:

```text
public/models/
  yolo/
    yolo11n.onnx
  gemma4-e2b-it/
    ...
```

Use this layout for production host storage:

```text
models/
  yolo/
    yolo11n.onnx
  gemma4-e2b-it/
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

## Gemma4-E2B

MVP chat design:

- Source model ID: `google/gemma-4-E2B-it`
- Source model ID: `google/gemma-4-E2B-it`
- Execution target: browser, not a server API
- Runtime: WebLLM / MLC WebGPU
- Current artifact: `welcoma/gemma-4-E2B-it-q4f16_1-MLC`
- Current scope: text generation path validated by the artifact publisher

Default config:

```env
NEXT_PUBLIC_LLM_RUNTIME=webllm
NEXT_PUBLIC_LLM_MODEL_ID=gemma-4-E2B-it-q4f16_1-MLC
NEXT_PUBLIC_LLM_MODEL_URL=/models/gemma4-e2b-it
NEXT_PUBLIC_LLM_MODEL_LIB_URL=/models/gemma4-e2b-it/libs/gemma-4-E2B-it-q4f16_1-MLC-webgpu.wasm
NEXT_PUBLIC_LLM_MAX_NEW_TOKENS=512
NEXT_PUBLIC_LLM_TEMPERATURE=0.7
```

Important deployment note:

The raw Hugging Face safetensors checkpoint is the source model, not necessarily the final browser artifact. For browser deployment, prefer a runtime-specific package such as an MLC/WebLLM or Transformers.js-compatible quantized artifact.

The first integrated runtime uses the public WebLLM package documented by MLC and the community MLC artifact `welcoma/gemma-4-E2B-it-q4f16_1-MLC`. First load downloads model shards into the browser cache. For offline production, mirror the artifact under:

```text
models/gemma4-e2b-it/
```

Then mount `./models` into the container so the app can serve it at:

```text
/models/gemma4-e2b-it
```

The current chat implementation sends text plus an optional summary of active YOLO/ByteTrack tracks. Raw image-frame multimodal input is intentionally left for the next milestone because the current WebLLM artifact documents a validated text path.

Download helper:

```bash
bash scripts/prepare-gemma4-e2b-webllm.sh
```

The helper downloads `welcoma/gemma-4-E2B-it-q4f16_1-MLC` with `huggingface_hub` into `models/gemma4-e2b-it/` and creates a local development symlink at `public/models/gemma4-e2b-it`.

WebLLM expects Hugging Face style URLs such as `/resolve/main/mlc-chat-config.json`, so the helper also creates:

```text
models/gemma4-e2b-it/resolve/main -> models/gemma4-e2b-it
```

### Browser Cache Storage

Do not use JavaScript `localStorage` for Gemma model files. `localStorage` is too small and string-only, so it is suitable for settings but not model weights.

WebLLM should cache model artifacts in browser storage, typically IndexedDB and related browser-managed cache storage. The current adapter enables:

```ts
useIndexedDBCache: true
```

Practical behavior:

- First `Load Gemma` downloads model shards and the WebGPU WASM library.
- Chrome stores those artifacts under this site's browser data.
- Later loads should reuse the local browser cache instead of re-downloading everything.
- Clearing site data, using a different browser profile, or changing host/origin can remove or bypass that cache.
- For production, keep a host-side copy under `models/gemma4-e2b-it/` as the deployable source, then let each browser cache it locally after first use.

## Docker Deployment

Expected host tree:

```text
project/
  docker-compose.yml
  models/
    yolo/
      yolo11n.onnx
    gemma4-e2b-it/
      ...
```

Start:

```bash
docker compose up --build
```

Chrome downloads the model files from the web app and runs inference locally. Camera access works on `localhost`; production hostnames must use HTTPS for browser camera permissions.

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
curl -I http://localhost:3000/models/gemma4-e2b-it/
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
