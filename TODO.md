# TODO.md

## Phase 0 - Project Bootstrap

- [ ] Create Next.js TypeScript app structure.
- [ ] Add ESLint and formatting setup.
- [ ] Add Dockerfile.
- [ ] Add Docker Compose example.
- [ ] Add `.env.example`.
- [ ] Add basic README with local and Docker startup commands.

## Phase 1 - UI Shell

- [ ] Build app layout with top menu, camera area, object list, and chat area.
- [ ] Add responsive layout for desktop and laptop screens.
- [ ] Add status indicators for camera, YOLO, tracker, and LLM.
- [ ] Add browser model download progress UI.
- [ ] Add metrics widgets for FPS, YOLO time, and ByteTrack time.
- [x] Move camera/source controls from the top menu into a dedicated Camera card above the YOLO11n card in the right panel.
- [x] Show the active camera/source name and Start/Stop action directly on the Camera card.
- [x] Add a Camera card settings icon that opens a settings panel.
- [x] In the Camera settings panel, support source selection for Camera, MJPG, RTSP, and YouTube.
- [x] In the Camera settings panel, support detailed per-source settings such as camera device, mirror mode, MJPG URL, RTSP URL, YouTube URL, YouTube output mode, gateway status, refresh devices, and preflight checks.
- [x] Keep the top menu focused on global status and metrics after moving camera/source controls into the Camera card.

## Phase 2 - Camera

- [ ] Implement camera permission request.
- [ ] Implement `enumerateDevices()` camera listing.
- [ ] Implement camera selector.
- [ ] Implement start/stop camera.
- [ ] Handle camera switching without page reload.
- [ ] Confirm Mac built-in camera support.
- [ ] Confirm iPhone Continuity Camera appears as a selectable camera on macOS when configured.
- [ ] Add clear errors for blocked permission, missing camera, or insecure origin.

## Phase 2.5 - Stream Sources

- [x] Add source selector for Camera, MJPG, RTSP, and YouTube.
- [x] Keep Camera mode on `getUserMedia()`.
- [x] Add MJPG URL input and raw `<img>` stream surface.
- [x] Add RTSP URL input and document that native `rtsp://` requires gateway conversion for Chrome.
- [x] Add YouTube URL input and document that watch URLs require gateway conversion for Chrome/canvas access.
- [x] Route camera, video URL, and MJPG image sources through the same YOLO/ByteTrack pipeline.
- [x] Capture the active source frame for Gemma multimodal prompts.
- [x] Add source-specific startup and error messages.
- [x] Add source deep links with optional autostart for unattended Chrome tests and robot launch flows.
- [ ] Validate CORS behavior for MJPG/HLS streams because canvas capture requires readable media.
- [ ] Add tests or manual validation notes for camera, MJPG, HLS URL, and RTSP gateway URL.
- [x] Validate YouTube gateway URL in Chrome with Docker web, YOLO11n WebGPU, and ByteTrack.

## Phase 2.6 - Stream Gateway MVP

- [x] Add `stream-gateway` service to Docker Compose.
- [x] Choose first gateway backend: ffmpeg, GStreamer, or MediaMTX.
- [x] Define gateway API contract for creating/stopping streams.
- [x] Add RTSP input support.
- [x] Add YouTube input support through `yt-dlp` plus ffmpeg when allowed by deployment policy.
- [x] Add YouTube URL preflight check through `yt-dlp`.
- [x] Add configurable YouTube resolver format, timeout, cookies file, and user-agent.
- [x] Output MJPG endpoint for fastest YOLO/canvas integration.
- [x] Output HLS endpoint for lower bandwidth and better browser compatibility.
- [x] Add frontend HLS playback for gateway sources.
- [x] Add YouTube MP4 gateway output for faster Chrome playback.
- [x] Add YouTube MP4/HLS output selector in the source controls.
- [x] Store temporary HLS segments outside the app image.
- [x] Add stream lifecycle cleanup for stopped or stale sessions.
- [x] Add healthcheck endpoint for the gateway.
- [x] Add clear UI errors when conversion fails, URL is unreachable, or the stream codec is unsupported.
- [x] Add gateway session status endpoint with masked input URL and recent logs.
- [x] Poll active gateway session status from the frontend.
- [x] Document that WebRTC is the later low-latency path for robot closed-loop control.
- [ ] Benchmark RTSP->MJPG latency and CPU load.
- [ ] Benchmark RTSP->HLS latency and CPU load.
- [ ] Decide whether the first production robot target should use MJPG, HLS, or WebRTC.

## Phase 3 - Video Overlay

- [ ] Add video element and overlay canvas.
- [ ] Synchronize canvas size with displayed video size.
- [ ] Implement coordinate scaling from model input space to displayed video space.
- [ ] Draw bounding boxes.
- [ ] Draw class labels, confidence, and track IDs.
- [ ] Ensure overlay remains aligned after window resize.

## Phase 4 - YOLO Browser Inference

- [ ] Use `YOLO11n` detection as the MVP model.
- [ ] Export `yolo11n.pt` to ONNX with fixed 640 input.
- [ ] Keep `YOLOv8n` as a compatibility fallback only.
- [ ] Add model and labels loading flow.
- [ ] Add ONNX Runtime Web.
- [ ] Configure WebGPU execution provider when available.
- [ ] Configure WASM fallback.
- [ ] Implement frame preprocessing.
- [ ] Implement YOLO output decoding.
- [ ] Implement confidence filtering.
- [ ] Implement non-maximum suppression.
- [ ] Add detection loop with configurable frame interval.
- [ ] Record YOLO inference time.
- [ ] Add tests for detection decoding and NMS.
- [ ] Add a later benchmark task for YOLO11n TFLite/LiteRT Web.
- [ ] Add a later benchmark task for YOLO11s ONNX quality/performance comparison.

## Phase 5 - ByteTrack

- [ ] Implement bounding box and detection types.
- [ ] Implement IoU calculation.
- [ ] Implement track state model.
- [ ] Implement high-score matching.
- [ ] Implement low-score matching.
- [ ] Implement lost-track buffer.
- [ ] Implement removed-track cleanup.
- [ ] Preserve stable IDs across frames.
- [ ] Record ByteTrack processing time.
- [ ] Add tests for IoU, matching, and track lifecycle.

## Phase 6 - Object List

- [ ] Render current active tracks.
- [ ] Show object name.
- [ ] Show track ID.
- [ ] Show confidence.
- [ ] Show last seen age.
- [ ] Sort by most recently updated.
- [ ] Remove stale objects when tracks are removed.

## Phase 7 - Browser-Local Gemma4-E2B Chat

- [ ] Add chat transcript component.
- [ ] Add text input and send action.
- [ ] Add "include current frame" option.
- [ ] Capture current video frame as image data when requested.
- [ ] Choose browser runtime: Transformers.js, WebLLM, MLC WebLLM, or another WebGPU-capable runtime.
- [ ] Choose browser-ready Gemma4-E2B-it artifact.
- [ ] Implement browser-side model loader.
- [ ] Implement model download progress and cache status.
- [ ] Implement WebGPU capability detection.
- [ ] Implement memory/error diagnostics for unsupported devices.
- [ ] Make `NEXT_PUBLIC_LLM_RUNTIME`, `NEXT_PUBLIC_LLM_DEVICE`, `NEXT_PUBLIC_LLM_MODEL_ID`, `NEXT_PUBLIC_LLM_MODEL_URL`, and generation settings configurable.
- [ ] Add a safety-first scheduler that keeps YOLO/ByteTrack active even when Gemma is slow.
- [ ] Show loading and error states.
- [ ] Verify `google/gemma-4-E2B-it` or its browser-ready quantized derivative inside Chrome.
- [ ] Confirm chat requests do not call a server-side LLM API.

## Phase 8 - External Model Storage

- [ ] Decide host model directory layout.
- [ ] Add scripts or docs for downloading YOLO model outside the app image.
- [ ] Add scripts or docs for downloading browser-ready `google/gemma-4-E2B-it` artifacts outside the app image.
- [ ] Add Docker volume mappings.
- [ ] Verify container can access YOLO public model files.
- [ ] Verify container can serve Gemma4-E2B browser artifacts as static files when using host-mounted models.
- [ ] Verify Chrome can download/cache model artifacts from the app.

## Phase 9 - Docker Production

- [ ] Build Next.js standalone output.
- [ ] Create production Docker image.
- [ ] Create Docker Compose production example.
- [x] Add optional stream-gateway container to Docker Compose.
- [x] Add gateway environment variables for allowed inputs, output mode, and segment directory.
- [ ] Add healthcheck endpoint.
- [x] Add stream-gateway healthcheck.
- [ ] Verify app starts in container.
- [ ] Verify gateway starts in container.
- [ ] Verify camera works from Chrome against container-hosted app.
- [ ] Verify MJPG source works against container-hosted app.
- [ ] Verify RTSP gateway source works against container-hosted app.
- [x] Verify YouTube HLS gateway source works against container-hosted app when policy allows.
- [x] Verify YouTube MP4 gateway source works against container-hosted app when policy allows.
- [ ] Verify Gemma4-E2B runs in Chrome against container-hosted static model artifacts.

## Phase 10 - Validation

- [ ] Run lint.
- [ ] Run unit tests.
- [ ] Run production build.
- [ ] Test in Chrome on localhost.
- [ ] Test with built-in Mac camera.
- [ ] Test with iPhone Continuity Camera if available.
- [ ] Test with direct MJPG URL.
- [ ] Test with gateway-converted RTSP URL.
- [x] Test with gateway-converted YouTube HLS URL.
- [x] Test with gateway-converted YouTube MP4 URL.
- [ ] Test with detection enabled for at least 10 minutes.
- [ ] Check memory growth during long-running detection.
- [ ] Confirm no remote network calls are required for inference after models are installed.
- [ ] Confirm chat inference does not call a backend API.

## Phase 11 - V7RC Protocol Planning

- [x] Collect the current V7RC firmware/protocol specification.
- [x] Confirm BLE service UUID, command characteristic UUID, notification characteristic UUID, and write mode.
- [ ] Confirm effective MTU and command pacing limits on the target robot firmware.
- [x] Document V7RC command rules: 20-byte maximum, 3-character command code, `#` ending marker, and padding behavior.
- [x] Document `HEX` 16-channel PWM format and `pwm_us = value * 10` conversion.
- [x] Document `DEG`, `SRV`, `SR2`, `SRT`, and `CMD` command options.
- [ ] Confirm whether the robot firmware expects `HEX`, `SRV`, or `SRT` for drivetrain control.
- [ ] Define normalized channel value ranges for motion, arm servos, binary modes, neutral, and emergency stop.
- [ ] Confirm channel calibration for the first robot chassis and arm hardware.
- [ ] Document final channel semantics for drivetrain, speed scale, arm joints, gripper/tool, autonomy enable, neutral, and e-stop.
- [ ] Define neutral frame and command timeout behavior.
- [ ] Define command rate limit and slew-rate limits for safe acceleration.
- [ ] Define how firmware reports low battery, failsafe, or command rejection if supported.
- [x] Create a TypeScript `V7rcProtocol` encoder module.
- [ ] Add V7RC decoder/parsing support if TX notifications expose structured firmware replies.
- [ ] Add unit tests for `HEX`, `DEG`, `SRV`, `SR2`, `SRT`, `CMD`, packet length validation, and channel clamping.
- [x] Add a mock V7RC transport for UI and Gemma loop testing without hardware.

## Phase 12 - Web Bluetooth Robot Link

- [x] Add Robot status card to the UI.
- [x] Add `Connect Robot` path using Chrome Web Bluetooth.
- [x] Add `Disconnect`, `Neutral`, and `E-stop` controls.
- [x] Show device name, connection state, last packet, and last protocol error.
- [x] Document V7RC BLE UART UUIDs for service, RX, and TX characteristics.
- [x] Implement Bluetooth device filtering with service UUID `6E400001-B5A3-F393-E0A9-E50E24DCCA9E`.
- [ ] Implement manual service/characteristic configuration fallback for firmware variants.
- [x] Implement command characteristic writer.
- [x] Implement notification reader scaffold for acknowledgements or telemetry.
- [x] Send neutral frame when disconnecting.
- [ ] Send neutral frame when the tab unloads or the perception loop stops.
- [ ] Add reconnection and stale connection error handling.
- [ ] Add Chrome secure-context guidance for production HTTPS deployment.

## Phase 13 - Gemma Robot Action Loop

- [ ] Define `RobotGoal`, `PerceptionState`, `GemmaAction`, and `RobotCommand` TypeScript types.
- [ ] Add a goal editor for target object, target color, success condition, and safety constraints.
- [ ] Add "suggestion mode" where Gemma proposes commands but Bluetooth transmission is disabled.
- [ ] Update the Gemma system prompt to require strict action JSON.
- [ ] Add JSON parsing, schema validation, and fallback-to-neutral behavior for invalid Gemma output.
- [ ] Add current track summary, target color hints, and recent command state to the Gemma prompt.
- [ ] Add color sampling inside tracked bounding boxes for target colors such as red, blue, green, yellow, black, and white.
- [ ] Translate `GemmaAction.intent` into normalized V7RC channel values through a safety controller.
- [ ] Clamp linear, turn, strafe, speed scale, and arm values before protocol encoding.
- [ ] Add command preview UI showing proposed channel values before hardware transmission.
- [ ] Add loop metrics: Gemma inference time, command validation time, Bluetooth write time, command rate, and last stop reason.
- [ ] Store goal/prompt/control settings in browser local storage.

## Phase 14 - Closed-Loop Goal Execution

- [ ] Implement first goal template: find a colored box.
- [ ] Implement search state: slow rotate/scan until a candidate target appears.
- [ ] Implement acquire state: verify target class, color, confidence, and ByteTrack stability.
- [ ] Implement approach state: keep target centered with low-speed forward/turn commands.
- [ ] Implement align state: reduce speed and center target before stopping.
- [ ] Implement complete state: send neutral, stop autonomy, and report goal complete.
- [ ] Implement unsafe state: send neutral/e-stop and require user confirmation.
- [ ] Stop or slow down when a person is detected near the path.
- [ ] Stop when target confidence drops below threshold for multiple rounds.
- [ ] Add manual override that immediately disables autonomy and sends neutral.
- [ ] Test closed-loop logic entirely in mock transport mode.
- [ ] Test Bluetooth command transmission with wheels lifted or motors disabled.
- [ ] Test first low-speed hardware run with a physical colored box target.

## Later Enhancements

- [ ] Move YOLO inference loop to Web Worker.
- [ ] Move ByteTrack to Web Worker.
- [ ] Add WebGPU capability diagnostics.
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
