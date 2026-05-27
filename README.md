# V7RC WebYOLO ByteTrack

Chrome-first local web app for webcam/stream YOLO detection, ByteTrack object IDs, browser-local Gemma4-E2B perception, and the next robot closed loop over Bluetooth through the V7RC protocol.

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
NEXT_PUBLIC_LLM_DEVICE=wasm
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
NEXT_PUBLIC_LLM_DEVICE=wasm
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

## Robot Control Roadmap

The current app is still observation-first: camera/stream frames go through YOLO11n, ByteTrack, and Gemma4-E2B in Chrome. The next development stage adds a cautious robot control loop:

```text
Vision source -> YOLO/ByteTrack -> Gemma4-E2B action JSON -> safety controller -> V7RC channel frame -> Web Bluetooth robot link
```

Planned browser-side modules:

- `RobotGoal`: stores the active mission, such as "find a red box".
- `PerceptionState`: current frame, detections, tracks, color hints, and recent command state.
- `GemmaAction`: structured LLM output with observation, goal status, target, motion intent, arm intent, and stop reason.
- `SafetyController`: clamps speed/servo values, applies neutral timeout, validates confidence, and blocks motion unless autonomy is enabled.
- `V7rcProtocol`: converts normalized intent values into V7RC command packets such as `HEX`, `DEG`, `SRV`, `SR2`, `SRT`, or `CMD`. Initial module is implemented at `app/lib/v7rc-protocol.ts`.
- `BluetoothTransport`: connects to the robot with Chrome Web Bluetooth and writes command frames to the configured BLE characteristic. Initial mock/Web Bluetooth transport is implemented at `app/lib/v7rc-transport.ts`.

V7RC protocol notes:

- Each BLE command packet is 20 bytes or less.
- The first 3 characters are the command code.
- Every packet ends with `#`.
- `HEX` is the preferred 16-channel PWM command: `HEX + 16 raw bytes + #`.
- In `HEX`, payload byte 0 maps to channel 0 and byte 15 maps to channel 15.
- `HEX` PWM conversion is `pwm_us = value * 10`, so byte values `100`, `150`, and `200` map to `1000 us`, `1500 us`, and `2000 us`.

Reference: [V7RC IO Command Protocol](https://github.com/v7rc/V7RC-Protocol/blob/main/protocol.en.md).

V7RC BLE UART-style UUIDs:

| Direction | UUID | Property |
| --- | --- | --- |
| Service | `6E400001-B5A3-F393-E0A9-E50E24DCCA9E` | BLE service |
| RX | `6E400002-B5A3-F393-E0A9-E50E24DCCA9E` | Write / Write Without Response |
| TX | `6E400003-B5A3-F393-E0A9-E50E24DCCA9E` | Notify |

Initial logical channel semantics on top of `HEX` channel indices are planned as:

| Channel | Meaning |
| --- | --- |
| `0` | Drive throttle |
| `1` | Steering / yaw |
| `2` | Strafe / lateral, optional |
| `3` | Speed scale or mode |
| `4` | Arm base yaw |
| `5` | Arm shoulder |
| `6` | Arm elbow |
| `7` | Wrist / gripper |
| `8` | Tool / auxiliary |
| `9` | Autonomy enable, if firmware supports it |
| `10` | Neutral / brake, if firmware supports it |
| `11` | Emergency stop, if firmware supports it |
| `12..15` | Reserved / neutral |

The implementation should use the service UUID for Web Bluetooth device filtering, write V7RC command packets to RX, and subscribe to TX notifications for acknowledgements or telemetry if the firmware emits them. A mock transport is still needed for UI and Gemma loop testing without hardware.

Current UI state:

- The right panel includes a `Robot / V7RC` card.
- `Mock` connects a local mock transport for packet testing without hardware.
- `BLE` opens Chrome Web Bluetooth pairing using the V7RC service UUID.
- `Neutral` and `E-stop` update the current control state.
- The card supports Vehicle, Mecanum, and Tank modes, sends 4-channel `SRT` PWM frames every 30ms while connected, and shows the last packet plus up to four channel previews.

Safety rules for the first control MVP:

- Bluetooth pairing must be user initiated.
- Autonomy must be explicitly enabled before any motor command is sent.
- The robot sends neutral on disconnect, tab close, stopped loop, invalid Gemma output, low confidence, target lost, or timeout.
- Gemma proposes structured intent; it does not write raw motor bytes directly.
- The first hardware tests should use low speed limits and suggestion mode before live motion.

More details: [docs/robot-control.md](docs/robot-control.md).

## iPhone Camera on Mac

On macOS, iPhone camera support uses Apple's Continuity Camera. Keep the iPhone nearby, signed into the same Apple ID, with Wi-Fi and Bluetooth enabled. In Chrome, grant camera permission, then use `Refresh` in the Camera control if the iPhone does not appear immediately. The app marks iPhone/Continuity/Desk View devices in the camera picker and prefers them when no camera has been selected yet.

## Runtime Notes

- YOLO inference runs in Chrome with ONNX Runtime Web.
- ByteTrack runs in Chrome with TypeScript IoU association and stable `T1`, `T2`, ... IDs.
- Gemma4-E2B runs through browser-local Transformers.js ONNX generation in a Web Worker. The default device is `wasm` so the slow LLM layer does not compete with YOLO for WebGPU; set `NEXT_PUBLIC_LLM_DEVICE=webgpu` only for benchmark or non-safety tests.
- When `Include current frame` is enabled, the app captures the active source frame as an image and sends it to Gemma4 together with the YOLO/ByteTrack track summary.
- Gemma settings live behind the Gemma4-E2B settings button and are cached in browser `localStorage`.
- The Gemma perception loop is controlled from the Gemma4-E2B card. Current builds are observation-only; the next stage adds suggestion mode, then gated Bluetooth control.
