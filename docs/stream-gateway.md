# Stream Gateway Plan

## Goal

Add an optional `stream-gateway` service that converts sources Chrome cannot use directly into browser-compatible streams for the existing YOLO, ByteTrack, and Gemma4 perception loop.

The gateway does not run object detection or LLM inference. It only normalizes video transport.

## Why It Is Needed

Chrome can use:

- `getUserMedia()` camera devices.
- MJPG image streams when CORS permits canvas reads.
- Browser-compatible video URLs.

Chrome cannot reliably use:

- Native `rtsp://` URLs.
- YouTube watch pages as canvas-readable video sources.

For robot perception, every source must eventually become a canvas-readable image/video surface so YOLO and Gemma can capture frames.

## MVP Architecture

```text
Robot / remote source
  ├─ RTSP camera
  ├─ MJPG camera
  └─ YouTube live/video source

stream-gateway
  ├─ input: rtsp://...
  ├─ input: YouTube URL
  ├─ converter: ffmpeg, GStreamer, or MediaMTX
  ├─ output: /streams/{id}.mjpg
  └─ output: /streams/{id}/index.m3u8

Chrome app
  ├─ Source mode: MJPG / RTSP / YouTube
  ├─ Browser-compatible URL from gateway
  ├─ YOLO11n ONNX Runtime Web
  ├─ ByteTrack
  └─ Gemma4-E2B frame capture
```

## Recommended Implementation Order

1. Add Docker Compose `stream-gateway` service.
2. Start with ffmpeg because it is easy to package and debug.
3. Support RTSP input first.
4. Output MJPG first for fast integration with the existing `<img>` path.
5. Add HLS output for lower bandwidth.
6. Add YouTube support through `yt-dlp` plus ffmpeg when allowed by the source and deployment policy.
7. Add cleanup for stale streams and temporary files.
8. Benchmark latency and CPU use.
9. Evaluate WebRTC mode for low-latency robot control.

## Gateway API Sketch

Create stream:

```http
POST /api/streams
Content-Type: application/json

{
  "sourceType": "rtsp",
  "url": "rtsp://robot.local/live",
  "output": "mjpg"
}
```

Response:

```json
{
  "id": "robot-front",
  "status": "starting",
  "url": "/streams/robot-front.mjpg"
}
```

Stop stream:

```http
DELETE /api/streams/robot-front
```

Health:

```http
GET /health
```

## Output Options

### MJPG

Best first MVP target.

Pros:

- Very simple in the browser.
- Works with the existing image stream surface.
- Easy frame capture for YOLO and Gemma.

Cons:

- Higher bandwidth.
- CPU cost can be significant at high resolution or FPS.

### HLS

Good general browser-compatible output.

Pros:

- Lower bandwidth than MJPG.
- Works well over HTTP.
- Good for monitoring and non-critical latency.

Cons:

- Latency is usually higher than MJPG or WebRTC.
- Chrome often needs an HLS JavaScript player for `.m3u8` playback.

### WebRTC

Best long-term target for robot closed-loop control.

Pros:

- Low latency.
- Designed for real-time media.

Cons:

- Requires signaling.
- Requires ICE/STUN/TURN planning for remote networks.
- More lifecycle and error handling.

## Docker Compose Shape

Target shape:

```yaml
services:
  web:
    build: .
    ports:
      - "${APP_PORT:-3000}:3000"
    volumes:
      - ./models:/app/public/models:ro

  stream-gateway:
    build:
      context: ./stream-gateway
    ports:
      - "${STREAM_GATEWAY_PORT:-3001}:3001"
    volumes:
      - ./streams:/var/lib/v7rc-streams
    environment:
      STREAM_OUTPUT_DIR: /var/lib/v7rc-streams
      STREAM_DEFAULT_OUTPUT: mjpg
```

## Frontend Integration Plan

The current frontend already has source modes for Camera, MJPG, RTSP, and YouTube. The next integration step is:

- If mode is `MJPG`, keep direct URL support.
- If mode is `RTSP`, call the gateway API with the entered URL and use the returned browser URL.
- If mode is `YouTube`, call the gateway API with the entered URL and use the returned browser URL.
- Show gateway startup state in the source control area.
- Stop the gateway stream when the user presses Stop or switches source modes.

## Security and Policy Notes

- Do not expose an unrestricted gateway on a public network.
- Validate and restrict allowed URL schemes.
- Consider allowlists for robot subnets or known camera hosts.
- YouTube ingestion must respect source terms and deployment policy.
- Avoid logging credentials embedded in RTSP URLs.
- Add authentication before production deployment outside a trusted LAN.

## Validation Checklist

- RTSP camera converts to MJPG and appears in the app.
- RTSP camera converts to HLS and appears in the app.
- YOLO boxes align on gateway output.
- ByteTrack IDs remain stable on gateway output.
- Gemma receives the active source frame.
- Stop cleans up ffmpeg/GStreamer processes.
- Source switch cleans up previous gateway sessions.
- Long-running stream does not leak disk segments or zombie processes.
- Latency and CPU use are recorded for MJPG and HLS.
