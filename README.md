# V7RC WebYOLO ByteTrack

Chrome-first local web app for webcam YOLO detection, ByteTrack object IDs, and browser-local Gemma4-E2B chat.

## Development

```bash
npm install
npm run dev
```

Open `http://localhost:3000` in Chrome.

## Model Storage

Large model artifacts should live outside the Docker image:

```text
models/
  yolo/
    yolo11n.onnx
    coco.names
  gemma4-e2b-it/
    ...
```

The web app serves mounted model files as static assets and runs inference in Chrome.

## Docker

```bash
docker compose up --build
```

Chrome camera access works on `localhost`. For production hosts, serve over HTTPS.
