# Robot Control Plan

This document plans the next stage after browser-local perception: connecting to a robot through Bluetooth, encoding V7RC protocol channel frames, and using Gemma4-E2B to drive a cautious closed loop.

Protocol reference: [V7RC IO Command Protocol](https://github.com/v7rc/V7RC-Protocol/blob/main/protocol.en.md).

## Design Principle

Gemma4-E2B should make high-level decisions, not write raw motor bytes.

```text
Gemma action JSON
  -> schema validation
  -> safety controller
  -> normalized channel values
  -> V7RC protocol encoder
  -> Web Bluetooth transport
```

The first hardware-facing version should support "suggestion mode" before live control. In suggestion mode, the app displays proposed channel values and stop reasons, but does not send movement frames.

## Web Bluetooth

Chrome Web Bluetooth is the first transport because the app is Chrome-first and robot control should stay local.

Planned connection states:

- `disconnected`
- `connecting`
- `connected`
- `armed`
- `autonomy_active`
- `error`

Required controls:

- Connect robot.
- Disconnect robot.
- Enable/disable autonomy.
- Send neutral.
- Emergency stop.

Known browser constraints:

- User gesture is required for pairing.
- `localhost` works during development.
- Production should use HTTPS.
- The app must handle disconnects and page unload by sending neutral when possible.

## V7RC BLE UART UUIDs

V7RC uses BLE UART-style characteristics:

| Direction | UUID | Property |
| --- | --- | --- |
| Service | `6E400001-B5A3-F393-E0A9-E50E24DCCA9E` | BLE service |
| RX | `6E400002-B5A3-F393-E0A9-E50E24DCCA9E` | Write / Write Without Response |
| TX | `6E400003-B5A3-F393-E0A9-E50E24DCCA9E` | Notify |

The Web Bluetooth transport should:

- Request devices with the service UUID filter.
- Open the service after GATT connection.
- Write encoded V7RC command packets to RX.
- Prefer Write Without Response for low-latency command streaming when available, while keeping Write as a fallback.
- Subscribe to TX notifications for acknowledgements, telemetry, or firmware messages.
- Send neutral before disconnect when possible.

## V7RC Protocol Adapter

The V7RC IO Command Protocol uses compact command strings for BLE-sized packets:

- Every command is 20 bytes or less.
- The first 3 characters are the command code.
- Every packet ends with `#`.
- Unused fields are padded with `0`, while `CMD` uses spaces.

Relevant commands:

- `HEX`: recommended 16-channel PWM command. Format: `HEX + 16 raw bytes + #`. Payload byte 0 maps to channel 0 and byte 15 maps to channel 15. PWM conversion is `pwm_us = value * 10`.
- `DEG`: 16-channel angle command. Format: `DEG + 16 raw bytes + #`. Angle conversion is `degree = value - 127`.
- `SRV`: basic 4-channel PWM text command, for example `SRV1500100018002000#`.
- `SR2`: second 4-channel PWM group for C5 to C8.
- `SRT`: tank mode based on basic PWM, where firmware converts CH1 and CH2 to tank-control signals.
- `CMD`: pass-through custom command with up to 16 characters, padded with spaces.

The BLE service and characteristic UUIDs are now known, but the transport should keep them configurable for firmware variants:

```ts
type V7rcChannelFrame = {
  command: "HEX" | "DEG" | "SRV" | "SR2" | "SRT" | "CMD";
  channels: number[];
  raw?: Uint8Array | string;
};

type V7rcTransport = {
  connect(): Promise<void>;
  disconnect(): Promise<void>;
  writeFrame(frame: V7rcChannelFrame): Promise<void>;
  sendNeutral(): Promise<void>;
};
```

The adapter should own:

- Channel clamping.
- Deadband.
- Slew-rate limits.
- Neutral frame generation.
- `HEX`, `DEG`, `SRV`, `SR2`, `SRT`, and `CMD` encoding.
- Exact packet length validation.
- Optional acknowledgement parsing if the firmware exposes notifications.

## Initial Channel Semantics

These are logical channel semantics mapped onto `HEX` channel indices. The first payload byte is channel 0.

| V7RC channel | Meaning | Range | Notes |
| --- | --- | --- | --- |
| `0` | Drive throttle | `-1..1` | Map to PWM around calibrated neutral, usually 1500 us. |
| `1` | Steering / yaw | `-1..1` | Differential turn or steering. |
| `2` | Strafe / lateral | `-1..1` | Optional; neutral for non-omni chassis. |
| `3` | Speed scale / mode | `0..1` | Browser-side limiter unless firmware exposes a mode channel. |
| `4` | Arm base yaw | `-1..1` | Optional manipulator. |
| `5` | Arm shoulder | `-1..1` | Optional manipulator. |
| `6` | Arm elbow | `-1..1` | Optional manipulator. |
| `7` | Wrist / gripper | `-1..1` | Split later if hardware exposes separate channels. |
| `8` | Tool / auxiliary | `0..1` | Optional actuator. |
| `9` | Autonomy enable | `0/1` | Only if firmware implements this as a channel. |
| `10` | Neutral / brake | `0/1` | Prefer browser-side neutral frame unless firmware provides a brake channel. |
| `11` | Emergency stop | `0/1` | Prefer a dedicated firmware stop path if available. |
| `12..15` | Reserved | neutral | Keep neutral until hardware mappings are confirmed. |

## Gemma Action Schema

Gemma should return strict JSON:

```json
{
  "goal_status": "searching",
  "observation": "A red box candidate is visible near the center-left.",
  "target": {
    "label": "box",
    "color": "red",
    "track_id": "T12",
    "confidence": 0.82
  },
  "intent": {
    "linear": 0.12,
    "turn": -0.18,
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

Allowed `goal_status` values:

- `searching`
- `approaching`
- `aligned`
- `complete`
- `blocked`
- `unsafe`

Invalid JSON, unknown status, NaN values, out-of-range commands, low confidence, or missing target state must fall back to neutral.

## First Goal Template

Goal:

```text
Find a red box. Move slowly while searching. When the red box is detected and centered, stop and report complete.
```

Goal fields:

- Target object text.
- Optional YOLO class filter.
- Target color.
- Minimum confidence.
- Centering tolerance.
- Maximum speed.
- Stop-if-person-detected flag.
- Target-lost timeout.

## Closed-Loop State Machine

```text
idle
  -> search
  -> acquire
  -> approach
  -> align
  -> complete

any state
  -> unsafe
  -> neutral/e-stop
```

State behavior:

- `search`: slowly rotate or scan for a candidate.
- `acquire`: verify class, color, confidence, and ByteTrack stability.
- `approach`: move slowly while keeping target centered.
- `align`: reduce speed and center target.
- `complete`: send neutral and stop autonomy.
- `unsafe`: send neutral or e-stop and require user confirmation.

## Safety Checklist

- Start in suggestion mode.
- Require explicit autonomy enable.
- Clamp every command.
- Enforce low default speed scale.
- Send neutral on timeout, disconnect, tab close, invalid Gemma output, target loss, or user stop.
- Keep emergency stop visible whenever Bluetooth is connected.
- Test mock transport first, then hardware with wheels lifted or motors disabled.
