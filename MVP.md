# MVP.md

## MVP Goal

Create a Docker-packaged Next.js web app that runs in Chrome, opens a webcam or browser-compatible stream source, performs YOLO object detection in the browser, tracks objects with ByteTrack, overlays bounding boxes and IDs on the video, lists tracked objects, runs Gemma4-E2B directly inside the web page without a server-side LLM API, and prepares the robot closed loop that can command a V7RC robot over Bluetooth through the V7RC protocol.

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
- Web Bluetooth connection planning for a V7RC robot device.
- V7RC protocol mapping from normalized robot intents to motor/servo channels.
- Goal-driven perception loop planning, starting with targets such as "find a colored box".
- Safety gating for robot motion commands before autonomous control is enabled.

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
- Direct motor control without an explicit user-enabled autonomy mode.
- Unbounded autonomous motion without command rate limits, neutral timeout, and emergency stop.
- Full manipulation planning or inverse kinematics for the arm.
- SLAM, mapping, or persistent world modeling.

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
  ├─ Goal loop: perception summary + target state + action JSON
  ├─ Web Bluetooth transport: V7RC robot connection
  ├─ V7RC protocol encoder: channel frame + safety envelope
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

## Robot Control Roadmap

The next MVP layer turns the current perception loop into a cautious robot exploration loop:

```text
Camera / stream frame
  ├─ YOLO11n detections
  ├─ ByteTrack object IDs
  ├─ Goal state: target object/color/task
  └─ Recent command state

Gemma4-E2B browser loop
  ├─ Reads frame + detections + tracked IDs + goal
  ├─ Produces structured action JSON
  └─ Explains confidence, hazards, and stop conditions

Safety controller
  ├─ Validates action JSON
  ├─ Clamps speed/servo ranges
  ├─ Requires autonomy enabled
  ├─ Applies timeout/heartbeat/e-stop rules
  └─ Converts intent to V7RC channel values

Web Bluetooth
  └─ Sends V7RC protocol frames to the robot
```

### Bluetooth Connection

Use Chrome Web Bluetooth for MVP because the app is Chrome-first and robot control should remain local to the browser session.

Planned UI/state:

- Robot card in the side panel or top menu.
- `Connect`, `Disconnect`, `Autonomy`, `Neutral`, and `E-stop` controls.
- Device status: disconnected, connecting, connected, armed, autonomy active, error.
- BLE service/characteristic configuration once the V7RC firmware UUIDs are known.
- Command log with last command packet, write timestamp, and acknowledgement/error if the firmware exposes notifications.

Important constraints:

- Web Bluetooth requires Chrome and a secure context. `localhost` is allowed for development; production should use HTTPS.
- Browser pairing is user initiated; the app cannot silently connect without a user gesture.
- The robot must go neutral when the tab closes, Bluetooth disconnects, the loop stops, or commands time out.

### V7RC Protocol and Channel Semantics

The V7RC IO Command Protocol is a compact command-string protocol designed for BLE packet limits. Commands are 20 bytes or less, the first 3 characters are the command code, and every packet ends with `#`. Unused fields are padded with `0`, while `CMD` uses spaces for padding.

Primary commands for this MVP:

- `HEX`: recommended 16-channel PWM control. Format: `HEX + 16 raw bytes + #`. Payload byte 0 maps to channel 0 and byte 15 maps to channel 15. PWM conversion is `pwm_us = value * 10`, so `100` means `1000 us`, `150` means `1500 us`, and `200` means `2000 us`.
- `DEG`: 16-channel angle control. Format: `DEG + 16 raw bytes + #`. Angle conversion is `degree = value - 127`, so `37` means `-90 deg`, `127` means `0 deg`, and `217` means `90 deg`.
- `SRV`: basic 4-channel PWM command such as `SRV1500100018002000#`.
- `SR2`: second PWM group for C5 to C8 on boards that expose more PWM channels.
- `SRT`: tank mode based on basic PWM, where CH1 and CH2 are converted by firmware into tank-control signals.
- `CMD`: pass-through device command with up to 16 characters, padded with spaces.

Proposed normalized intent range:

- Motion channels: `-1.0` to `1.0`.
- Servo channels: `-1.0` to `1.0`, mapped to calibrated PWM/position limits.
- Binary/mode channels: `0` or `1`.
- All outputs pass through clamping, deadband, slew-rate limits, and neutral fallback.

Initial logical channel map on top of `HEX` channel indices:

| V7RC channel | Logical meaning | Range | Notes |
| --- | --- | --- | --- |
| `0` | Drive throttle | `-1..1` | Map to PWM around calibrated neutral, usually 1500 us. |
| `1` | Steering / yaw | `-1..1` | Differential turn or steering, depending on chassis firmware. |
| `2` | Strafe / lateral | `-1..1` | Optional for mecanum/omni chassis; neutral for two-wheel chassis. |
| `3` | Speed scale / mode | `0..1` | If firmware has no speed channel, apply this only in the browser safety controller. |
| `4` | Arm base yaw | `-1..1` | Optional manipulator channel. |
| `5` | Arm shoulder | `-1..1` | Optional manipulator channel. |
| `6` | Arm elbow | `-1..1` | Optional manipulator channel. |
| `7` | Wrist / gripper | `-1..1` | Split later if hardware exposes separate wrist and gripper channels. |
| `8` | Tool / auxiliary | `0..1` | Optional actuator. |
| `9` | Autonomy enable | `0/1` | Only if firmware implements this as a channel. |
| `10` | Neutral / brake | `0/1` | Prefer browser-side neutral frame unless firmware provides a brake channel. |
| `11` | Emergency stop | `0/1` | Prefer a dedicated firmware stop path if available. |
| `12..15` | Reserved | neutral | Keep neutral until hardware-specific mappings are confirmed. |

Protocol adapter tasks:

- Define the actual BLE service UUID, command characteristic UUID, notification characteristic UUID, and write mode used by the robot firmware.
- Implement `HEX`, `DEG`, `SRV`, `SR2`, `SRT`, and `CMD` encoders with exact byte-length validation.
- Keep `HEX` as the first full-channel command path because it controls 16 PWM channels in one 20-byte frame.
- Confirm whether the target firmware expects `HEX` raw bytes, text commands such as `SRV`, or tank mode `SRT` for drivetrain control.
- Implement a TypeScript encoder/decoder behind a small `V7rcTransport` interface.
- Add a simulator/mock transport so the perception loop can be tested without a robot.

### Gemma-Controlled Robot Loop

Gemma must not directly write raw motor values. It should produce structured intent, then a safety controller translates that intent into V7RC channels.

Recommended Gemma output schema:

```json
{
  "goal_status": "searching | approaching | aligned | complete | blocked | unsafe",
  "observation": "short scene summary",
  "target": {
    "label": "box",
    "color": "red",
    "track_id": "T12",
    "confidence": 0.82
  },
  "intent": {
    "linear": 0.15,
    "turn": -0.2,
    "strafe": 0,
    "speed_scale": 0.25,
    "arm": {
      "base": 0,
      "shoulder": 0,
      "elbow": 0,
      "wrist": 0,
      "gripper": 0
    }
  },
  "safety": {
    "stop": false,
    "reason": ""
  }
}
```

The first implementation should run in "suggestion mode": Gemma proposes actions, the UI displays the translated channel values, but Bluetooth transmission stays disabled until the user enables autonomy. This lets us validate perception and command quality before moving hardware.

### Goal Definition and Closed Loop

The MVP goal system should start with a compact goal editor:

- Goal type: search, approach, align, inspect, stop.
- Target object: free text plus optional YOLO class filter.
- Target color: red, blue, green, yellow, black, white, unknown.
- Success condition: object found, centered in frame, close enough by box size, or user stop.
- Safety constraints: max speed, max turn, minimum confidence, stop if person detected, stop if target lost.

Example first goal:

```text
Find a red box. Move slowly while searching. When the red box is detected and centered, stop and report complete.
```

Closed-loop phases:

1. `Search`: slowly rotate or scan until a candidate appears.
2. `Acquire`: use YOLO/ByteTrack and color sampling around the detection box to identify the target.
3. `Approach`: issue low-speed forward/turn intents while keeping the target centered.
4. `Align`: reduce speed and center the target.
5. `Complete`: send neutral command and stop the loop.
6. `Unsafe`: send neutral/e-stop and require user confirmation.

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
