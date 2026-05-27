# TODO.md

## Current Status

The MVP is now a working Chrome-first local perception app:

- Camera/MJPG/RTSP/YouTube source selection is implemented.
- YOLO11n ONNX detection runs in Chrome through ONNX Runtime Web.
- ByteTrack runs in browser TypeScript and produces stable track IDs.
- Canvas overlay, object list, FPS, YOLO timing, and ByteTrack timing are visible.
- Gemma4-E2B runs in browser through Transformers.js/Web Worker with optional current-frame image context.
- Docker packaging is implemented with host-mounted model volumes.
- `stream-gateway` supports RTSP/YouTube conversion with MJPG/HLS/MP4 outputs.
- V7RC mock/Web Bluetooth transport exists.
- Robot card supports Vehicle, Mecanum, and Tank modes.
- Robot command preview uses 4-channel `SRT` PWM frames.
- Connected robot transports send the current `SRT` command every 30ms.

Next planning target:

- Convert the workspace to a three-column robot task console.
- Add `Autopilot` and `Mission` task modes.
- Add OpenCV.js lane detection and bird's-eye view visualization.
- Add LLM JSON action plans that are expanded into 30ms V7RC `SRT` frames.

## Phase 0 - Project Bootstrap

- [x] Create Next.js TypeScript app structure.
- [x] Add ESLint setup.
- [x] Add Dockerfile.
- [x] Add Docker Compose example.
- [x] Add `.env.example`.
- [x] Add README with local, Docker, model, stream gateway, and robot notes.
- [x] Add AGENTS.md, MVP.md, TODO.md, and technical docs.

## Phase 1 - UI Shell

- [x] Build app layout with top menu, video area, object list, Gemma transcript, and right-side control cards.
- [x] Add responsive layout for desktop and laptop screens.
- [x] Keep video at the top in a 16:9 stage with Gemma transcript below it.
- [x] Add status indicators for camera, gateway, YOLO, tracker, LLM, and robot.
- [x] Add browser model download progress UI.
- [x] Add metrics widgets for FPS, YOLO time, and ByteTrack time.
- [x] Move camera/source controls into a dedicated Camera card above the YOLO11n card.
- [x] Add Camera settings modal with Camera/MJPG/RTSP/YouTube source options.
- [x] Add Gemma settings modal with include-frame, system prompt, and fixed prompt settings.
- [x] Cache Gemma prompt/settings in browser `localStorage`.
- [x] Convert workspace from two columns to three columns.
- [x] Reduce camera/video column width while preserving 16:9.
- [x] Add a middle card column with the same width as the right control cards.
- [x] Move robot task/autopilot cards into the middle column.
- [ ] Add a compact robot safety/command-rate metric strip.

## Phase 2 - Camera And Sources

- [x] Implement camera permission request.
- [x] Implement `enumerateDevices()` camera listing.
- [x] Implement camera selector.
- [x] Implement start/stop camera.
- [x] Handle camera switching without page reload.
- [x] Add mirror mode for MacBook/iPhone front cameras.
- [x] Add iPhone Continuity Camera label detection and selection support when macOS exposes it.
- [x] Add clear errors for blocked permission, missing camera, or insecure origin.
- [x] Add source selector for Camera, MJPG, RTSP, and YouTube.
- [x] Add MJPG URL input and raw `<img>` stream surface.
- [x] Add RTSP URL input and gateway guidance.
- [x] Add YouTube URL input and gateway guidance.
- [x] Route camera, video URL, HLS/MP4, and MJPG image sources through the same YOLO/ByteTrack pipeline.
- [x] Capture active source frames for Gemma multimodal prompts.
- [x] Add source deep links with optional autostart.
- [x] Validate YouTube gateway URL in Chrome with Docker web, YOLO11n, and ByteTrack.
- [ ] Validate CORS behavior for external MJPG/HLS streams because canvas capture requires readable media.
- [ ] Add manual validation notes for Camera, MJPG, HLS, MP4, RTSP gateway, and YouTube gateway paths.

## Phase 3 - Stream Gateway

- [x] Add `stream-gateway` service to Docker Compose.
- [x] Implement gateway API for creating, stopping, and inspecting stream sessions.
- [x] Add RTSP input support through ffmpeg.
- [x] Add YouTube input support through `yt-dlp` plus ffmpeg.
- [x] Add YouTube URL preflight check.
- [x] Add configurable YouTube resolver format, timeout, cookies file, and user-agent.
- [x] Add MJPG output.
- [x] Add HLS output.
- [x] Add MP4 output for YouTube playback.
- [x] Add frontend HLS playback through `hls.js`.
- [x] Add YouTube MP4/HLS output selector.
- [x] Store temporary stream artifacts outside the app image.
- [x] Add stream lifecycle cleanup for stopped or stale sessions.
- [x] Add gateway healthcheck endpoint and Docker healthcheck.
- [x] Add gateway session status endpoint with masked input URL and recent logs.
- [x] Poll active gateway session status from the frontend.
- [x] Document WebRTC as the later low-latency path.
- [ ] Benchmark RTSP->MJPG latency and CPU load.
- [ ] Benchmark RTSP->HLS latency and CPU load.
- [ ] Decide whether the first production robot camera target should use MJPG, HLS, MP4, or WebRTC.

## Phase 4 - Video Overlay

- [x] Add video/image source surface and overlay canvas.
- [x] Synchronize canvas size with displayed video/image size.
- [x] Implement coordinate scaling from source/model space to displayed video space.
- [x] Draw bounding boxes.
- [x] Draw class labels, confidence, and track IDs.
- [x] Use matching label background and box colors.
- [x] Keep label background width aligned with the box width.
- [x] Use black label text.
- [x] Ensure overlay remains aligned after window resize.
- [x] Remove unnecessary overlay-ready visual noise.

## Phase 5 - YOLO Browser Inference

- [x] Use `YOLO11n` detection as the MVP model.
- [x] Add YOLO11n ONNX preparation script.
- [x] Keep `YOLOv8n` as a compatibility fallback path only.
- [x] Add model and label loading flow.
- [x] Add ONNX Runtime Web.
- [x] Configure WebGPU execution provider when available.
- [x] Configure WASM fallback.
- [x] Implement frame preprocessing.
- [x] Implement YOLO output decoding.
- [x] Implement confidence filtering.
- [x] Implement non-maximum suppression.
- [x] Add detection loop with configurable frame interval.
- [x] Record YOLO inference time.
- [x] Keep YOLO active while Gemma is generating.
- [ ] Add unit tests for detection decoding and NMS.
- [ ] Add benchmark task for YOLO11n TFLite/LiteRT Web.
- [ ] Add benchmark task for YOLO11s ONNX quality/performance comparison.
- [ ] Add safety event rules derived directly from YOLO/ByteTrack, independent of LLM output.

## Phase 6 - ByteTrack

- [x] Implement bounding box and detection types.
- [x] Implement IoU calculation.
- [x] Implement track state model.
- [x] Implement high-score matching.
- [x] Implement low-score matching.
- [x] Implement lost-track buffer.
- [x] Implement removed-track cleanup.
- [x] Improve matching to reduce ID switches during object crossing.
- [x] Preserve stable IDs across frames as much as the current IoU/velocity model allows.
- [x] Record ByteTrack processing time.
- [ ] Add unit tests for IoU, matching, and track lifecycle.
- [ ] Evaluate Kalman-filter based prediction if ID switches remain too frequent.

## Phase 7 - Object List

- [x] Render current active tracks.
- [x] Show object name.
- [x] Show track ID.
- [x] Show confidence.
- [x] Show last seen age/missed frames.
- [x] Remove stale objects when tracks are removed.
- [ ] Sort by most recently updated or highest safety priority.
- [ ] Add optional class filters for robot-relevant objects.

## Phase 8 - Browser-Local Gemma4-E2B

- [x] Add Gemma transcript panel below the video.
- [x] Replace manual prompt send UI with Start/Stop loop controls.
- [x] Add response count and last inference time in the transcript footer.
- [x] Auto-scroll transcript to the latest response.
- [x] Add include-current-frame option.
- [x] Capture current source frame as image data when requested.
- [x] Use browser-local Transformers.js Gemma4-E2B ONNX as the default runtime.
- [x] Keep WebLLM/MLC path as an optional experimental runtime.
- [x] Choose browser-ready Gemma4-E2B-it ONNX artifact.
- [x] Implement browser-side model loader.
- [x] Implement model download progress and cache status.
- [x] Implement WebGPU capability detection.
- [x] Implement memory/error diagnostics for unsupported devices where available.
- [x] Make `NEXT_PUBLIC_LLM_RUNTIME`, `NEXT_PUBLIC_LLM_DEVICE`, `NEXT_PUBLIC_LLM_MODEL_ID`, `NEXT_PUBLIC_LLM_MODEL_URL`, token, temperature, frame-size, and loop-delay settings configurable.
- [x] Add safety-first scheduling rule: YOLO/ByteTrack stay active even when Gemma is slow.
- [x] Default Gemma to WASM worker mode so the LLM does not compete with YOLO for WebGPU.
- [x] Confirm chat requests do not call a server-side LLM API.
- [x] Remove temporary Gemma test prompt buttons.
- [ ] Verify Gemma4-E2B WASM worker behavior on the production Chrome machine.
- [ ] Add cancellable/interruptible Gemma generation.
- [ ] Add streaming partial Gemma output if the selected runtime supports it reliably.

## Phase 9 - External Model Storage

- [x] Decide host model directory layout.
- [x] Add scripts/docs for downloading YOLO model outside the app image.
- [x] Add scripts/docs for downloading browser-ready Gemma4-E2B artifacts outside the app image.
- [x] Add Docker volume mapping from `./models` to `/app/public/models`.
- [x] Verify container can serve YOLO public model files through mounted models.
- [x] Document that model files belong in browser cache/host volumes, not JavaScript `localStorage`.
- [ ] Verify production host can serve the full Gemma4-E2B artifact from mounted static files.
- [ ] Add checksum/version manifest for model artifacts.

## Phase 10 - Docker Production

- [x] Build Next.js standalone output.
- [x] Create production Docker image.
- [x] Create Docker Compose example.
- [x] Add optional stream-gateway container to Docker Compose.
- [x] Add gateway environment variables for allowed inputs, output mode, resolver settings, and segment directory.
- [x] Add stream-gateway healthcheck.
- [x] Verify app starts in container.
- [x] Verify gateway starts in container.
- [x] Verify YouTube HLS gateway source works against container-hosted app when policy allows.
- [x] Verify YouTube MP4 gateway source works against container-hosted app when policy allows.
- [ ] Add web app healthcheck.
- [ ] Verify camera works from Chrome against container-hosted app on the production host.
- [ ] Verify MJPG source works against container-hosted app on the production host.
- [ ] Verify RTSP gateway source works against container-hosted app with a real RTSP source.
- [ ] Verify Gemma4-E2B runs in Chrome against container-hosted static model artifacts.

## Phase 11 - Validation

- [x] Run lint after major changes.
- [x] Run production build after major changes.
- [x] Smoke-test Docker web at `localhost:3003`.
- [x] Test with built-in Mac camera.
- [x] Test with gateway-converted YouTube HLS URL.
- [x] Test with gateway-converted YouTube MP4 URL.
- [x] Test YOLO11n ONNX detection in Chrome.
- [x] Test ByteTrack overlay and object list in Chrome.
- [x] Test Gemma response loop enough to identify runtime/contention issues.
- [ ] Add unit test runner and pure-logic tests.
- [ ] Test with iPhone Continuity Camera if available in Chrome.
- [ ] Test with direct MJPG URL.
- [ ] Test with gateway-converted RTSP URL.
- [ ] Test with detection enabled for at least 10 minutes.
- [ ] Check memory growth during long-running detection and Gemma loops.
- [ ] Confirm no remote network calls are required for inference after models are installed.

## Phase 12 - V7RC Protocol

- [x] Collect the current V7RC firmware/protocol specification.
- [x] Confirm BLE service UUID, RX command characteristic UUID, TX notification characteristic UUID, and write mode.
- [x] Document V7RC command rules: 20-byte maximum, 3-character command code, `#` ending marker, and padding behavior.
- [x] Document `HEX` 16-channel PWM format and `pwm_us = value * 10` conversion.
- [x] Document `DEG`, `SRV`, `SR2`, `SRT`, and `CMD` command options.
- [x] Create TypeScript V7RC protocol encoder helpers.
- [x] Add `SRT` encoder path for 4-channel PWM control.
- [x] Add vehicle, mecanum, and tank SRT channel mappings.
- [x] Define current drivetrain channel semantics:
  - Vehicle: `CH0` steering, `CH1` throttle.
  - Tank: `CH0` turn, `CH1` throttle.
  - Mecanum: `CH0` strafe, `CH1` throttle, `CH2` turn.
- [x] Add neutral PWM behavior for unused channels.
- [x] Add mock V7RC transport for UI and loop testing without hardware.
- [ ] Confirm effective MTU and command pacing limits on target firmware.
- [ ] Confirm target firmware expects `SRT` for drivetrain control.
- [ ] Confirm channel calibration for the first robot chassis and arm hardware.
- [ ] Define command timeout behavior in firmware and UI.
- [ ] Define slew-rate limits for safe acceleration.
- [ ] Define how firmware reports low battery, failsafe, or command rejection if supported.
- [ ] Add V7RC decoder/parsing support if TX notifications expose structured replies.
- [ ] Add unit tests for `HEX`, `DEG`, `SRV`, `SR2`, `SRT`, `CMD`, packet length validation, and channel clamping.

## Phase 13 - Web Bluetooth Robot Link

- [x] Add Robot status card to the UI.
- [x] Add Mock transport connection path.
- [x] Add Chrome Web Bluetooth connection path.
- [x] Add `Disconnect`, `Neutral`, and `E-stop` controls.
- [x] Show device name, connection state, last packet, and last protocol error.
- [x] Document V7RC BLE UART UUIDs for service, RX, and TX characteristics.
- [x] Implement Bluetooth device filtering with service UUID `6E400001-B5A3-F393-E0A9-E50E24DCCA9E`.
- [x] Implement command characteristic writer.
- [x] Implement notification reader scaffold for acknowledgements or telemetry.
- [x] Send neutral frame when disconnecting.
- [x] Send current `SRT` command every 30ms while connected.
- [x] Throttle UI status updates during the 30ms command loop.
- [ ] Test 30ms BLE command loop with real V7RC hardware.
- [ ] Send neutral frame when the tab unloads or the perception loop stops.
- [ ] Add reconnection and stale connection error handling.
- [ ] Add manual service/characteristic configuration fallback for firmware variants.
- [ ] Add Chrome secure-context guidance for production HTTPS deployment.

## Phase 14 - Gemma Robot Action Loop

- [ ] Define `RobotGoal`, `PerceptionState`, `GemmaAction`, and `RobotCommand` TypeScript types.
- [ ] Add a goal editor for target object, target color, success condition, and safety constraints.
- [ ] Keep suggestion mode as the default: Gemma proposes actions, but hardware control remains gated.
- [ ] Update Gemma system prompt to require strict mission action-plan JSON.
- [ ] Use JSON-only prompt instructions in Chinese and reject Markdown responses.
- [ ] Add mission payload schema with `version`, `message`, `missionStatus`, `planDurationMs`, and `actions`.
- [ ] Support action moves: `forward`, `backward`, `turn_left`, `turn_right`, `strafe_left`, `strafe_right`, and `stop`.
- [ ] Convert LLM action `ms` durations into repeated 30ms V7RC `SRT` command frames.
- [ ] Add JSON parsing, schema validation, and fallback-to-neutral behavior for invalid Gemma output.
- [ ] Add current track summary, target color hints, recent command state, drive mode, bird's-eye state, and task goal to the Gemma prompt.
- [ ] Add color sampling inside tracked bounding boxes for target colors such as red, blue, green, yellow, black, and white.
- [ ] Translate `GemmaAction.intent` into normalized V7RC channel values through a safety controller.
- [ ] Clamp linear, turn, strafe, speed scale, and arm values before protocol encoding.
- [ ] Display the last mission `message`, including target-visible messages and `任務完成` / `mission complete`.
- [ ] Add loop metrics: Gemma inference time, command validation time, Bluetooth write time, command rate, and last stop reason.
- [ ] Store goal/control settings in browser local storage.

## Phase 15 - Task Console And Bird's-Eye View

- [x] Add Robot Task card.
- [x] Add mode buttons for `Autopilot` and `Mission`.
- [x] Add Start/Stop icon button on the Robot Task card.
- [x] Add task status states: idle, running, complete, blocked, unsafe, error.
- [x] Add task message display.
- [ ] Add command-plan preview for Mission mode.
- [x] Add Bird's-Eye View card below Robot Task.
- [ ] Implement perspective transform calibration settings.
- [ ] Render current frame as bird's-eye view.
- [x] Project YOLO/ByteTrack detections into bird's-eye view using box bottom-center points.
- [x] Draw projected object labels and track IDs on the bird's-eye view.
- [ ] Draw current motion vector or steering/throttle cue.
- [ ] Cache bird's-eye transform settings in browser local storage.

## Phase 16 - OpenCV.js Autopilot

- [ ] Choose OpenCV.js delivery method: self-hosted official `opencv.js`/WASM artifact under `public/vendor/opencv/`.
- [ ] Add OpenCV.js loader with ready/error status.
- [ ] Run OpenCV.js lane detection in a Web Worker if possible.
- [ ] Add ROI selection for lane/ground area.
- [ ] Add perspective transform from camera view to bird's-eye view.
- [ ] Add lane color and/or grayscale thresholding.
- [ ] Add blur + Canny edge detection.
- [ ] Add HoughLinesP or contour/sliding-window lane extraction.
- [ ] Estimate lane center and heading.
- [ ] Draw lane overlay on the camera view.
- [ ] Draw lane overlay on the bird's-eye view.
- [ ] Generate proportional steering from lane center offset.
- [ ] In clear-lane conditions, command 50% forward throttle.
- [ ] Stop/neutral when lane detection is missing, stale, or low confidence.
- [ ] Ensure YOLO/ByteTrack obstacle rules override lane-follow output.
- [ ] Add OpenCV heap cleanup discipline for all `cv.Mat` allocations.

## Phase 17 - Safety Controller

- [ ] Implement a safety layer that can stop the robot without waiting for LLM output.
- [ ] Convert YOLO/ByteTrack observations into deterministic hazard states.
- [ ] Stop or slow down when a person is detected near the path.
- [ ] Stop when target confidence drops below threshold for multiple frames.
- [ ] Add obstacle distance/size heuristics from bounding boxes.
- [ ] Add manual override that immediately disables autonomy and sends neutral.
- [ ] Add command timeout watchdog.
- [ ] Add max-speed, max-turn, and max-strafe limits by drive mode.
- [ ] Add slew-rate limiting for all motion channels.
- [ ] Add explicit armed/autonomy gate before non-neutral commands can reach hardware.
- [ ] Add emergency stop state that requires user reset.

## Phase 18 - Closed-Loop Goal Execution

- [ ] Implement first goal template: find a colored box.
- [ ] Implement search state: slow rotate/scan until a candidate target appears.
- [ ] Implement acquire state: verify target class, color, confidence, and ByteTrack stability.
- [ ] Implement approach state: keep target centered with low-speed forward/turn commands.
- [ ] Implement align state: reduce speed and center target before stopping.
- [ ] Implement complete state: send neutral, stop autonomy, and report goal complete.
- [ ] Implement unsafe state: send neutral/e-stop and require user confirmation.
- [ ] Test closed-loop logic entirely in mock transport mode.
- [ ] Test Bluetooth command transmission with wheels lifted or motors disabled.
- [ ] Test first low-speed hardware run with a physical colored box target.

## Later Enhancements

- [ ] Move YOLO inference loop to Web Worker.
- [ ] Move ByteTrack to Web Worker.
- [ ] Add WebGPU capability diagnostics panel.
- [ ] Add model selection UI.
- [ ] Add detection class filters.
- [ ] Add per-class confidence thresholds.
- [ ] Add track history trails.
- [ ] Add snapshot export.
- [ ] Add optional video recording.
- [ ] Add alert rules.
- [ ] Add persistent chat/session history.
- [ ] Add benchmark page for different YOLO models and runtimes.
- [ ] Add benchmark page for Gemma4-E2B browser artifacts and quantization levels.
- [ ] Add WebRTC stream gateway mode for lower-latency robot control.
- [ ] Add authenticated stream source support.
- [ ] Add per-source presets for robot camera endpoints.
- [ ] Add world memory for persistent target locations.
- [ ] Add arm inverse kinematics and grasp planning after basic channel control is safe.
- [ ] Add telemetry recording for perception, Gemma actions, protocol frames, and robot motion.
