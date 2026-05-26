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
  gemma4-e2b-it/
    ...

models/                     # production host-mounted path
  yolo/
    yolo11n.onnx
  gemma4-e2b-it/
    ...
```

Default browser model URLs:

```env
NEXT_PUBLIC_YOLO_MODEL_URL=/models/yolo/yolo11n.onnx
NEXT_PUBLIC_LLM_RUNTIME=webllm
NEXT_PUBLIC_LLM_MODEL_ID=gemma-4-E2B-it-q4f16_1-MLC
NEXT_PUBLIC_LLM_MODEL_URL=https://huggingface.co/welcoma/gemma-4-E2B-it-q4f16_1-MLC
NEXT_PUBLIC_LLM_MODEL_LIB_URL=https://huggingface.co/welcoma/gemma-4-E2B-it-q4f16_1-MLC/resolve/main/libs/gemma-4-E2B-it-q4f16_1-MLC-webgpu.wasm
```

More details: [docs/models.md](docs/models.md).

## Prepare YOLO11n

The MVP expects `public/models/yolo/yolo11n.onnx` during local development.

```bash
bash scripts/prepare-yolo11n.sh
```

This creates a local Python virtual environment, installs Ultralytics, exports `yolo11n.pt` to ONNX, and copies it into the app's public model directory. The Docker setup can instead mount `./models/yolo/yolo11n.onnx` to `/app/public/models/yolo/yolo11n.onnx`.

Generated files are intentionally ignored by git:

```text
.venv-yolo/
.model-export/
public/models/**/*.onnx
models/
```

## Docker

```bash
docker compose up --build
```

Chrome camera access works on `localhost`. For production hosts, serve over HTTPS.

## Runtime Notes

- YOLO inference runs in Chrome with ONNX Runtime Web.
- ByteTrack runs in Chrome with TypeScript IoU association and stable `T1`, `T2`, ... IDs.
- Gemma4-E2B runs through browser-local WebLLM/WebGPU text generation. Current frame context is summarized from active tracks; raw image multimodal input is the next milestone.
