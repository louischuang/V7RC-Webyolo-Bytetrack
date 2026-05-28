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
NEXT_PUBLIC_LLM_DEVICE=webgpu
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
NEXT_PUBLIC_LLM_DEVICE=webgpu
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

## Lane Robustness Technical Evaluation

Bad weather, dusk, tunnels, shadows, glare, and worn lane markings can make plain white/yellow thresholding unstable. The lane-following roadmap should be evaluated as a layered browser pipeline instead of a single detector.

### Layer 1: Classical Image Enhancement

Run before lane candidate extraction:

- CLAHE/local contrast enhancement for dusk, tunnel, and low-contrast scenes.
- Automatic gamma correction to recover dark road texture without overexposing bright lane paint.
- HSV/HLS white/yellow masks instead of RGB-only brightness checks.
- Sobel gradient and Canny edge evidence for faded lane boundaries.
- Morphological close/dilate to connect broken lane markings before path fitting.
- Evidence debug overlay in Bird's-Eye View: color mask, edge mask, temporal mask, and final lane.

### Layer 2: Geometry And Temporal Fallback

Use when frame-by-frame lane evidence is incomplete:

- Smooth lane paths across frames with EMA or a lightweight Kalman-style model.
- Continue a recent valid ego lane for a short timeout when current confidence drops.
- Infer a missing lane boundary from one detected side plus expected lane width.
- Track vanishing point / heading consistency to reject wall lines, tunnel lights, and dashboard overlays.
- Use bird-view ROI and lane-width constraints before converting a candidate into a command.
- Label fallback output as `predicted lane` or `inferred lane` so testing can separate detection from estimation.

### Layer 3: Switchable Neural Segmentation

Keep YOLO11n object detection as the current safety/object baseline, then evaluate alternate road/lane perception modules behind a model switch:

| Mode | Purpose | Notes |
| --- | --- | --- |
| `YOLO11n + classical lane` | Current baseline | Fastest path; object detection remains independent. |
| `YOLOP / YOLOPv2 style` | Joint object, drivable area, and lane line perception | Evaluate browser ONNX compatibility, FPS, memory, and output decoding complexity. |
| `ONNX road/lane segmentation` | Dedicated road mask or lane mask fallback | Useful when lane markings are weak but road region is visible. |

The technical evaluation should measure FPS, YOLO time, lane processing time, total UI responsiveness, memory growth, and correctness on clear highway, tunnel, dusk, rainy/low-contrast, and worn-lane clips.

Current implementation status:

- The Bird's-Eye View card now exposes a `Perception` selector for `Classical`, `YOLOP`, and `ONNX Seg`.
- `Classical` keeps the current YOLO11n object detector plus browser classical lane pipeline.
- `YOLOP` and `ONNX Seg` route through the Layer 3 segmentation adapter benchmark path. If `NEXT_PUBLIC_LANE_MODEL_URL` points to a browser-compatible ONNX segmentation artifact, the browser loads it with ONNX Runtime Web and decodes its mask into bird-view lane paths. Without a model URL, the adapter uses the same bird-view input and a segmentation-style mask fallback so latency, confidence, drop count, and UI wiring can be compared without disrupting YOLO11n safety detection.
- Benchmark snapshots include the active perception mode plus lane and segmentation timing fields.

Configurable lane model environment fields:

- `NEXT_PUBLIC_LANE_MODEL_URL`
- `NEXT_PUBLIC_LANE_MODEL_INPUT_SIZE`
- `NEXT_PUBLIC_LANE_MODEL_PROVIDER`
- `NEXT_PUBLIC_LANE_MODEL_FRAME_INTERVAL`
- `NEXT_PUBLIC_LANE_MODEL_THRESHOLD`

## Next MVP: Task And Autopilot Console

The next UI milestone changes the main workspace from two columns into three columns:

```text
Top status bar
└─ Main workspace
   ├─ Left: 16:9 camera/video stage with overlay, reduced width
   ├─ Middle: mission/autopilot cards, same width as the right cards
   └─ Right: Camera, YOLO11n, Gemma4-E2B, Robot/V7RC, and tracked objects
```

The new middle column should contain:

- `Robot Task` card with two segmented modes:
  - `Autopilot`: deterministic lane-following.
  - `Mission`: LLM-assisted goal solving.
- Start/Stop icon button beside the selected task mode.
- `Bird's-Eye View` card that renders a top-down driving view from the current source frame.
- YOLO11n/ByteTrack objects projected into the bird's-eye view by transforming the bottom-center point of each box through the same perspective transform used for the lane view.

### Autopilot Mode

Autopilot should be deterministic and safety-first. It does not wait for Gemma.

Planned browser pipeline:

```text
Camera frame
  -> OpenCV.js worker
  -> ROI crop
  -> perspective transform / bird's-eye view
  -> color threshold and/or grayscale threshold
  -> blur + Canny
  -> HoughLinesP or contour/sliding-window lane extraction
  -> lane center offset
  -> safety controller
  -> V7RC SRT command stream every 30ms
```

OpenCV.js is the planned browser CV runtime. Prefer a pinned self-hosted OpenCV.js/WebAssembly artifact under `public/vendor/opencv/` so production does not depend on a CDN. Official OpenCV.js documentation shows browser use through an `opencv.js` script and the global `cv` object; it can be loaded asynchronously and should release `cv.Mat` objects after each frame to avoid heap growth.

Autopilot control rule for the first pass:

- If lane detection is valid and no obstacle hazard is present, drive forward at 50% throttle.
- Use lane center offset to generate steering.
- If a person, obstacle, missing lane, stale frame, or command timeout is detected, send stop/neutral first.
- YOLO/ByteTrack safety rules always override lane-follow commands.

### Mission Mode

Mission mode asks Gemma4-E2B to produce short command plans, then the browser controller expands those plans into the 30ms V7RC `SRT` command loop. Gemma does not write raw SRT frames.

Recommended mission command payload:

```json
{
  "version": 1,
  "message": "移動搜尋物體",
  "missionStatus": "running",
  "planDurationMs": 2000,
  "actions": [
    { "move": "forward", "ms": 300, "power": 0.35 },
    { "move": "turn_right", "ms": 500, "power": 0.25 },
    { "move": "forward", "ms": 1200, "power": 0.35 },
    { "move": "stop", "ms": 300 }
  ]
}
```

Payload rules:

- `message` is shown in the task card. Use `任務完成` or `mission complete` when the goal is complete.
- `missionStatus` should be one of `running`, `complete`, `blocked`, `unsafe`, or `failed`.
- Total action time should stay near `planDurationMs`, defaulting to 2000ms.
- `move` should use a controlled enum: `forward`, `backward`, `turn_left`, `turn_right`, `strafe_left`, `strafe_right`, or `stop`.
- `ms` is duration in milliseconds, not seconds.
- `power` is optional, clamped by the safety controller, and defaults to a low safe value.
- The browser controller converts each action into repeated 30ms `SRT` frames.
- The safety controller may interrupt any action sequence and send stop/neutral.

Mission prompt direction:

```text
你是機器人的任務規劃器。請根據目前影像、YOLO11n 物件、ByteTrack ID、鳥瞰圖與任務目標，只輸出一個 JSON，不要輸出 Markdown。
每次輸出最多約 2 秒的短動作計畫。若任務完成，message 必須包含「任務完成」或「mission complete」。
若看到目標，請在 message 中提到目標與 track ID。若不安全或無法完成，請用 message 說明原因，並讓 actions 以 stop 結尾。
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
- `HEX` is the full 16-channel PWM command: `HEX + 16 raw bytes + #`.
- In `HEX`, payload byte 0 maps to channel 0 and byte 15 maps to channel 15.
- `HEX` PWM conversion is `pwm_us = value * 10`, so byte values `100`, `150`, and `200` map to `1000 us`, `1500 us`, and `2000 us`.

Reference: [V7RC IO Command Protocol](https://github.com/v7rc/V7RC-Protocol/blob/main/protocol.en.md).

V7RC BLE UART-style UUIDs:

| Direction | UUID | Property |
| --- | --- | --- |
| Service | `6E400001-B5A3-F393-E0A9-E50E24DCCA9E` | BLE service |
| RX | `6E400002-B5A3-F393-E0A9-E50E24DCCA9E` | Write / Write Without Response |
| TX | `6E400003-B5A3-F393-E0A9-E50E24DCCA9E` | Notify |

Current drivetrain control uses 4-channel text `SRT` frames:

| Robot mode | Channel mapping |
| --- | --- |
| Vehicle | `CH0` steering wheel, `CH1` throttle, `CH2-CH3` neutral |
| Tank | `CH0` turn, `CH1` throttle, `CH2-CH3` neutral |
| Mecanum | `CH0` strafe, `CH1` throttle, `CH2` turn, `CH3` neutral |

`SRT` command examples look like `SRT1500150015001500#`. The browser sends the current 4-channel `SRT` state every 30ms after Mock or BLE transport is connected.

Older full-channel logical semantics on top of `HEX` channel indices are kept as a later expansion path:

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
- Gemma4-E2B runs through browser-local Transformers.js ONNX generation in a Web Worker. The current ONNX q4f16 Gemma artifact requires `NEXT_PUBLIC_LLM_DEVICE=webgpu`; WASM may fail with missing quantized operator support such as `GatherBlockQuantized`. Keep YOLO active and protect the vision loop through scheduling, shorter prompts, and lower LLM frequency rather than forcing this artifact onto WASM.
- When `Include current frame` is enabled, the app captures the active source frame as an image and sends it to Gemma4 together with the YOLO/ByteTrack track summary.
- Gemma settings live behind the Gemma4-E2B settings button and are cached in browser `localStorage`.
- The Gemma perception loop is controlled from the Gemma4-E2B card. Current builds are observation-only; the next stage adds suggestion mode, then gated Bluetooth control.
