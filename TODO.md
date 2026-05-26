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
- [ ] Validate CORS behavior for MJPG/HLS streams because canvas capture requires readable media.
- [ ] Add tests or manual validation notes for camera, MJPG, HLS URL, RTSP gateway URL, and YouTube gateway URL.

## Phase 2.6 - Stream Gateway MVP

- [x] Add `stream-gateway` service to Docker Compose.
- [x] Choose first gateway backend: ffmpeg, GStreamer, or MediaMTX.
- [x] Define gateway API contract for creating/stopping streams.
- [x] Add RTSP input support.
- [x] Add YouTube input support through `yt-dlp` plus ffmpeg when allowed by deployment policy.
- [x] Output MJPG endpoint for fastest YOLO/canvas integration.
- [x] Output HLS endpoint for lower bandwidth and better browser compatibility.
- [ ] Store temporary HLS segments outside the app image.
- [x] Add stream lifecycle cleanup for stopped or stale sessions.
- [x] Add healthcheck endpoint for the gateway.
- [ ] Add clear UI errors when conversion fails, URL is unreachable, or the stream codec is unsupported.
- [ ] Document that WebRTC is the later low-latency path for robot closed-loop control.
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
- [ ] Make `NEXT_PUBLIC_LLM_RUNTIME`, `NEXT_PUBLIC_LLM_MODEL_ID`, `NEXT_PUBLIC_LLM_MODEL_URL`, and generation settings configurable.
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
- [ ] Add optional stream-gateway container to Docker Compose.
- [ ] Add gateway environment variables for allowed inputs, output mode, and segment directory.
- [ ] Add healthcheck endpoint.
- [ ] Add stream-gateway healthcheck.
- [ ] Verify app starts in container.
- [ ] Verify gateway starts in container.
- [ ] Verify camera works from Chrome against container-hosted app.
- [ ] Verify MJPG source works against container-hosted app.
- [ ] Verify RTSP gateway source works against container-hosted app.
- [ ] Verify YouTube gateway source works against container-hosted app when policy allows.
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
- [ ] Test with gateway-converted YouTube URL.
- [ ] Test with detection enabled for at least 10 minutes.
- [ ] Check memory growth during long-running detection.
- [ ] Confirm no remote network calls are required for inference after models are installed.
- [ ] Confirm chat inference does not call a backend API.

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
